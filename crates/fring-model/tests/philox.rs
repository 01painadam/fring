//! Philox4x32 known-answer tests (task F2 acceptance) and basic statistics.
//!
//! The vectors are the `philox4x32` lines of the Random123 known-answer file
//! `tests/kat_vectors`, https://github.com/DEShawResearch/random123, commit
//! 9545ff6413f258be2f04c1d319d99aaef7521150. Column order in that file:
//! rounds, counter (4 words), key (2 words), expected output (4 words).
//! The third line of each set uses the hexadecimal digits of π.

use fring_model::rng::{
    Counter, Philox, Stream, box_muller, philox4x32, philox4x32_10, u01_f32, u01_f64,
};

/// (rounds, counter, key, expected)
type Vector = (u32, [u32; 4], [u32; 2], [u32; 4]);

const KAT: [Vector; 6] = [
    (
        7,
        [0x00000000, 0x00000000, 0x00000000, 0x00000000],
        [0x00000000, 0x00000000],
        [0x5f6fb709, 0x0d893f64, 0x4f121f81, 0x4f730a48],
    ),
    (
        7,
        [0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff],
        [0xffffffff, 0xffffffff],
        [0x5207ddc2, 0x45165e59, 0x4d8ee751, 0x8c52f662],
    ),
    (
        7,
        [0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344],
        [0xa4093822, 0x299f31d0],
        [0x4dfccaba, 0x190a87f0, 0xc47362ba, 0xb6b5242a],
    ),
    (
        10,
        [0x00000000, 0x00000000, 0x00000000, 0x00000000],
        [0x00000000, 0x00000000],
        [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8],
    ),
    (
        10,
        [0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff],
        [0xffffffff, 0xffffffff],
        [0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd],
    ),
    (
        10,
        [0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344],
        [0xa4093822, 0x299f31d0],
        [0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1],
    ),
];

#[test]
fn random123_known_answers() {
    for (rounds, counter, key, expected) in KAT {
        assert_eq!(
            philox4x32(rounds, counter, key),
            expected,
            "rounds {rounds}, counter {counter:08x?}, key {key:08x?}"
        );
    }
}

#[test]
fn philox4x32_10_matches_the_10_round_vectors() {
    for (_, counter, key, expected) in KAT.iter().filter(|v| v.0 == 10) {
        assert_eq!(philox4x32_10(*counter, *key), *expected);
    }
}

#[test]
fn philox_struct_uses_seed_and_counter_layout() {
    // Key [0xa4093822, 0x299f31d0] is seed 0x299f31d0_a4093822.
    // Counter words [0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344] are
    // index 0x243f6a88, step 0x13198a2e_85a308d3, stream 0x0370, draw 0x7344.
    let rng = Philox::from_seed(0x299f_31d0_a409_3822);
    let counter = Counter {
        index: 0x243f_6a88,
        step: 0x1319_8a2e_85a3_08d3,
        stream: Stream(0x0370),
        draw: 0x7344,
    };
    assert_eq!(
        rng.block(counter),
        [0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1]
    );
}

#[test]
fn different_streams_give_different_numbers() {
    let rng = Philox::from_seed(1);
    let c = |stream| Counter {
        index: 5,
        step: 9,
        stream,
        draw: 0,
    };
    assert_ne!(
        rng.block(c(Stream::INITIAL_CONDITIONS)),
        rng.block(c(Stream::COLLISIONS))
    );
}

/// Mean and variance of `values`.
fn mean_and_variance(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, var)
}

/// 10^5 blocks. With N = 2 × 10^5 uniforms, the standard error of the mean is
/// sqrt(1/12/N) = 6.5e-4 and of the variance sqrt((1/80 - 1/144)/N) = 1.7e-4.
/// The limits are 5 standard errors. The test is deterministic (fixed seed).
#[test]
fn uniform_and_normal_moments() {
    let rng = Philox::from_seed(20_261_101);
    let n_blocks: u32 = 100_000;
    let mut u64s = Vec::with_capacity(2 * n_blocks as usize);
    let mut u32s = Vec::with_capacity(4 * n_blocks as usize);
    let mut normals = Vec::with_capacity(2 * n_blocks as usize);
    for index in 0..n_blocks {
        let counter = Counter {
            index,
            step: 0,
            stream: Stream::TEST,
            draw: 0,
        };
        let [a, b] = rng.uniform_f64(counter);
        u64s.extend([a, b]);
        u32s.extend(rng.uniform_f32(counter).map(f64::from));
        let (z1, z2) = box_muller(a, b);
        normals.extend([z1, z2]);
    }

    let (mean, var) = mean_and_variance(&u64s);
    assert!((mean - 0.5).abs() < 5.0 * 6.5e-4, "f64 uniform mean {mean}");
    assert!(
        (var - 1.0 / 12.0).abs() < 5.0 * 1.7e-4,
        "f64 uniform variance {var}"
    );

    // N = 4 × 10^5: standard errors 4.6e-4 (mean) and 1.2e-4 (variance).
    let (mean, var) = mean_and_variance(&u32s);
    assert!((mean - 0.5).abs() < 5.0 * 4.6e-4, "f32 uniform mean {mean}");
    assert!(
        (var - 1.0 / 12.0).abs() < 5.0 * 1.2e-4,
        "f32 uniform variance {var}"
    );

    // N = 2 × 10^5 normals: standard errors 1/sqrt(N) = 2.2e-3 (mean) and
    // sqrt(2/N) = 3.2e-3 (variance).
    let (mean, var) = mean_and_variance(&normals);
    assert!(mean.abs() < 5.0 * 2.2e-3, "normal mean {mean}");
    assert!((var - 1.0).abs() < 5.0 * 3.2e-3, "normal variance {var}");
}

#[test]
fn conversions_agree_between_f32_and_f64() {
    // The f32 value uses the top 23 bits of the word; the f64 value uses the
    // top 52 bits of (word, next word). They differ by less than 2^-23.
    for word in [0_u32, 1, 0x8000_0000, 0xDEAD_BEEF, u32::MAX] {
        let a = f64::from(u01_f32(word));
        let b = u01_f64(word, 0);
        assert!((a - b).abs() < 2f64.powi(-23), "word {word:#x}: {a} {b}");
    }
}
