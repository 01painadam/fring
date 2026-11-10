//! DET-SINCOS-2 guard: the bits of `fit_ring` must not depend on the build
//! profile (see `fring-collide/tests/profile_bits.rs` for the reason). The
//! hash was recorded in a debug build. CI runs this test in release too.

use fring_analysis::apse::fit_ring;
use fring_model::rng::{Counter, Philox, Stream};

/// Hash recorded in a debug build.
const RECORDED: u64 = 0x40c9_9563_58a6_1c6e;

#[test]
fn fit_ring_bits_do_not_depend_on_the_profile() {
    let rng = Philox::from_seed(20_261_110);
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    let mut put = |x: f64| {
        for b in x.to_bits().to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for run in 0..40_u32 {
        let (mut rho, mut phi) = (Vec::new(), Vec::new());
        for k in 0..200_u32 {
            let [u, v] = rng.uniform_f64(Counter {
                index: run * 1000 + k,
                step: 0,
                stream: Stream::TEST,
                draw: 0,
            });
            let p = std::f64::consts::TAU * u - std::f64::consts::PI;
            phi.push(p);
            rho.push(140_222.0 + 30.0 * (v - 0.5) + 15.0 * p.cos());
        }
        let (c0, c1, c2) = fit_ring(&rho, &phi);
        put(c0);
        put(c1);
        put(c2);
    }
    println!("analysis hash {h:#018x}");
    assert_eq!(h, RECORDED);
}
