//! DET-SINCOS-2 guard: the bits of `impact_normal` and of the arc-shifted
//! hard-sphere pairs must not depend on the build profile.
//!
//! In release, LLVM can fuse `sin` and `cos` of one argument into macOS
//! `___sincos_stret`, which differs from separate calls by up to 2 ulp.
//! `fring_model::real::sin_cos` stops that. The hash was recorded in a debug
//! build, where nothing is fused. CI runs this test in release too.

use fring_collide::hard_sphere::{HardSphereModel, Pair};
use fring_collide::pair::impact_normal;
use fring_model::Vec3;
use fring_model::config::Restitution;
use fring_model::rng::{Counter, Philox, Stream};

/// FNV-1a over the bits of the values.
struct Hash(u64);

impl Hash {
    fn f64(&mut self, x: f64) {
        for b in x.to_bits().to_le_bytes() {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    fn vec(&mut self, v: Vec3<f64>) {
        self.f64(v.x);
        self.f64(v.y);
        self.f64(v.z);
    }
}

/// Hash recorded in a debug build.
const RECORDED: u64 = 0xc816_c708_9199_7513;

fn draw(rng: &Philox, index: u32) -> [f64; 2] {
    rng.uniform_f64(Counter {
        index,
        step: 0,
        stream: Stream::TEST,
        draw: 0,
    })
}

#[test]
fn collide_sin_cos_users_bits_do_not_depend_on_the_profile() {
    let rng = Philox::from_seed(20_261_110);
    let mut h = Hash(0xcbf2_9ce4_8422_2325);
    // impact_normal: 2000 pairs of uniforms, a different g each time.
    for i in 0..200_000_u32 {
        let [u1, u2] = draw(&rng, i);
        let [a, b] = draw(&rng, 10_000 + i);
        let g = Vec3::new(a - 0.5, b - 0.5, 0.3 + 0.01 * f64::from(i % 7));
        h.vec(impact_normal(g, u1, u2));
    }
    // resolve_pairs with shifted pairs: head-on, with an arc width.
    let model = HardSphereModel::new(0.01, Restitution::Constant { epsilon: 0.5 });
    for i in 0..50_000_u32 {
        let [w, v] = draw(&rng, 20_000 + i);
        let width_rad = 0.05 + 0.5 * w;
        let mut vel = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.001 * v, 0.0, 0.0)];
        let shift: i8 = if i % 2 == 0 { 1 } else { -1 };
        let pairs = [Pair { i: 0, j: 1, shift }];
        let sep = |_: usize, _: usize, _: i8| Vec3::new(-0.015, 0.001 * w, 0.0);
        model.resolve_pairs(&pairs, 1.0e3, Some(width_rad), sep, &mut vel);
        h.vec(vel[0]);
        h.vec(vel[1]);
    }
    println!("collide hash {:#018x}", h.0);
    assert_eq!(h.0, RECORDED);
}
