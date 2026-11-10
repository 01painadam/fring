//! DET-SINCOS-2 guard: the bits of the `prm` uniform words (the sector table
//! of `bin::params`) must not depend on the build profile (see
//! `fring-collide/tests/profile_bits.rs` for the reason). The hash was
//! recorded in a debug build. CI runs this test in release too.

use fring_gpu::bin::{BinGeometry, params};

/// Hash recorded in a debug build.
const RECORDED: u64 = 0xf661_7f13_3014_2bf6;

#[test]
fn bin_params_bits_do_not_depend_on_the_profile() {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    for (k, a0) in [140_222.43_f64, 139_380.0, 141_000.0]
        .into_iter()
        .enumerate()
    {
        let g =
            BinGeometry::new(a0, 5.8e-4, [3.0, 40.0, 3.0], 2.0e5 + 1.0e4 * k as f64, 0.1).unwrap();
        for w in params(&g) {
            for b in w.to_bits().to_le_bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
    }
    println!("gpu hash {h:#018x}");
    assert_eq!(h, RECORDED);
}
