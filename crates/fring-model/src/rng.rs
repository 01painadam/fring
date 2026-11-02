//! Philox4x32-10: the only random number generator in `fring`.
//!
//! Philox is counter-based (Salmon, Moraes, Dror and Shaw 2011, "Parallel random
//! numbers: as easy as 1, 2, 3", SC11, doi:10.1145/2063384.2063405). It is a
//! pure function: the same (key, counter) always gives the same four 32-bit
//! words. There is no state to save or share between threads, so the CPU and
//! the GPU give the same numbers for the same particle and step.
//!
//! # Key and counter layout
//!
//! This layout is a shared interface. The WGSL version must use the same one.
//!
//! - **Key** (2 words): the run seed from the config. `key = [seed low 32 bits,
//!   seed high 32 bits]`.
//! - **Counter** (4 words), from [`Counter::words`]:
//!   - word 0: `index`, the item that needs the numbers (particle index,
//!     collision cell or pair).
//!   - words 1 and 2: `step`, the time step number (low and high 32 bits).
//!   - word 3: `stream` in the high 16 bits, `draw` in the low 16 bits.
//!
//! A [`Stream`] separates independent uses of the RNG (initial conditions,
//! collisions). `draw` numbers the blocks of four words that one item needs in
//! one step and stream: draw 0 gives words 0 to 3, draw 1 gives words 4 to 7,
//! and so on. So no two uses can share a counter.
//!
//! # Conversion to floats
//!
//! [`u01_f64`] and [`u01_f32`] map words to the open interval (0, 1), never 0
//! or 1, so `ln(u)` is always finite. The f32 version uses 23 bits, so every
//! result is exact in f32 and the WGSL twin gives the same bits.
//!
//! ```
//! use fring_model::rng::{Counter, Philox, Stream, u01_f64};
//!
//! let rng = Philox::from_seed(42);
//! let words = rng.block(Counter {
//!     index: 7,
//!     step: 0,
//!     stream: Stream::INITIAL_CONDITIONS,
//!     draw: 0,
//! });
//! let u = u01_f64(words[0], words[1]);
//! assert!(u > 0.0 && u < 1.0);
//! ```

use crate::Real;

/// Multiplier for words 0 and 1 (Random123 `PHILOX_M4x32_0`).
const M0: u32 = 0xD251_1F53;
/// Multiplier for words 2 and 3 (Random123 `PHILOX_M4x32_1`).
const M1: u32 = 0xCD9E_8D57;
/// Weyl increment for key word 0 (golden ratio, Random123 `PHILOX_W32_0`).
const W0: u32 = 0x9E37_79B9;
/// Weyl increment for key word 1 (√3 − 1, Random123 `PHILOX_W32_1`).
const W1: u32 = 0xBB67_AE85;

/// High and low 32-bit words of the 64-bit product `a * b`.
///
/// WGSL has no 64-bit integers, so the WGSL twin must build this from 16-bit
/// halves. Its result must be bit-identical to this function.
#[inline]
pub fn mulhilo(a: u32, b: u32) -> (u32, u32) {
    let product = u64::from(a) * u64::from(b);
    ((product >> 32) as u32, product as u32)
}

#[inline]
fn round(ctr: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let (hi0, lo0) = mulhilo(M0, ctr[0]);
    let (hi1, lo1) = mulhilo(M1, ctr[2]);
    [hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0]
}

/// Philox4x32 with a given number of rounds.
///
/// Use [`philox4x32_10`] in simulation code. Other round counts exist only to
/// test against more published vectors.
pub fn philox4x32(rounds: u32, counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let mut ctr = counter;
    let mut key = key;
    for r in 0..rounds {
        if r > 0 {
            key[0] = key[0].wrapping_add(W0);
            key[1] = key[1].wrapping_add(W1);
        }
        ctr = round(ctr, key);
    }
    ctr
}

/// Philox4x32-10: four random 32-bit words for one (counter, key).
#[inline]
pub fn philox4x32_10(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    philox4x32(10, counter, key)
}

/// An independent use of the RNG (high 16 bits of counter word 3).
///
/// Add a new stream here (A0) when a new part of the code needs random
/// numbers. Never reuse a number.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Stream(pub u16);

impl Stream {
    /// Tests and examples only. Simulation code never uses it.
    pub const TEST: Stream = Stream(0);
    /// Initial positions and velocities of particles.
    pub const INITIAL_CONDITIONS: Stream = Stream(1);
    /// Collision candidates, acceptance and impact geometry.
    pub const COLLISIONS: Stream = Stream(2);
    /// Direction of the initial shadow offset of the chaos indicator
    /// (ICR-003, ADR 0008). Counter: index = particle id, step = 0, draw = 0.
    pub const CHAOS_SHADOW: Stream = Stream(3);
}

/// The counter for one block of four random words.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Counter {
    /// The item that needs the numbers: particle index, cell or pair.
    pub index: u32,
    /// Time step number. Use 0 for initial conditions.
    pub step: u64,
    /// The use of the RNG.
    pub stream: Stream,
    /// Block number when one item needs more than four words in one step.
    pub draw: u16,
}

impl Counter {
    /// The four counter words, in the layout of the module documentation.
    pub const fn words(self) -> [u32; 4] {
        [
            self.index,
            self.step as u32,
            (self.step >> 32) as u32,
            ((self.stream.0 as u32) << 16) | self.draw as u32,
        ]
    }
}

/// Philox4x32-10 with the key made from the run seed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Philox {
    key: [u32; 2],
}

impl Philox {
    /// Makes the generator for a run seed (`[run] seed` in the config).
    pub const fn from_seed(seed: u64) -> Self {
        Self {
            key: [seed as u32, (seed >> 32) as u32],
        }
    }

    /// The two key words: `[seed low 32 bits, seed high 32 bits]`.
    pub const fn key(&self) -> [u32; 2] {
        self.key
    }

    /// Four random 32-bit words for one counter.
    #[inline]
    pub fn block(&self, counter: Counter) -> [u32; 4] {
        philox4x32_10(counter.words(), self.key)
    }

    /// Two uniform f64 values in (0, 1) for one counter.
    #[inline]
    pub fn uniform_f64(&self, counter: Counter) -> [f64; 2] {
        let w = self.block(counter);
        [u01_f64(w[0], w[1]), u01_f64(w[2], w[3])]
    }

    /// Four uniform f32 values in (0, 1) for one counter.
    #[inline]
    pub fn uniform_f32(&self, counter: Counter) -> [f32; 4] {
        self.block(counter).map(u01_f32)
    }
}

/// 2^-52 (exact).
const TWO_POW_M52: f64 = 1.0 / (1_u64 << 52) as f64;
/// 2^-23 (exact).
const TWO_POW_M23: f32 = 1.0 / (1_u32 << 23) as f32;

/// Uniform f64 in (0, 1) from two words. Uses the top 52 of the 64 bits.
///
/// The result is `(k + 0.5) / 2^52` with `k` in 0 to 2^52 − 1, so it is in
/// [2^-53, 1 − 2^-53]. Every value is exact in f64.
#[inline]
pub fn u01_f64(hi: u32, lo: u32) -> f64 {
    let bits = ((u64::from(hi) << 32) | u64::from(lo)) >> 12;
    (bits as f64 + 0.5) * TWO_POW_M52
}

/// Uniform f32 in (0, 1) from one word. Uses the top 23 of the 32 bits.
///
/// The result is `(k + 0.5) / 2^23` with `k` in 0 to 2^23 − 1, so it is in
/// [2^-24, 1 − 2^-24]. Every value is exact in f32.
#[inline]
pub fn u01_f32(word: u32) -> f32 {
    ((word >> 9) as f32 + 0.5) * TWO_POW_M23
}

/// Two independent standard normal values from two uniforms in (0, 1)
/// (Box-Muller transform).
///
/// `u1` must be in (0, 1]; [`u01_f64`] and [`u01_f32`] give such values.
pub fn box_muller<T: Real>(u1: T, u2: T) -> (T, T) {
    let radius = (T::from_f64(-2.0) * u1.ln()).sqrt();
    let angle_rad = T::from_f64(2.0) * T::PI * u2;
    (radius * angle_rad.cos(), radius * angle_rad.sin())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mulhilo_extremes() {
        assert_eq!(mulhilo(0, 0xFFFF_FFFF), (0, 0));
        // (2^32 − 1)^2 = 2^64 − 2^33 + 1
        assert_eq!(
            mulhilo(0xFFFF_FFFF, 0xFFFF_FFFF),
            (0xFFFF_FFFE, 0x0000_0001)
        );
    }

    #[test]
    fn counter_layout() {
        let c = Counter {
            index: 0x0102_0304,
            step: 0x0A0B_0C0D_1112_1314,
            stream: Stream(0xBEEF),
            draw: 0x0042,
        };
        assert_eq!(
            c.words(),
            [0x0102_0304, 0x1112_1314, 0x0A0B_0C0D, 0xBEEF_0042]
        );
    }

    #[test]
    fn seed_to_key() {
        let rng = Philox::from_seed(0x1234_5678_9ABC_DEF0);
        assert_eq!(rng.key(), [0x9ABC_DEF0, 0x1234_5678]);
    }

    #[test]
    fn u01_f64_is_open_interval() {
        assert_eq!(u01_f64(0, 0), 2f64.powi(-53));
        assert_eq!(u01_f64(u32::MAX, u32::MAX), 1.0 - 2f64.powi(-53));
        assert!(u01_f64(u32::MAX, u32::MAX) < 1.0);
    }

    #[test]
    fn u01_f32_is_open_interval() {
        assert_eq!(u01_f32(0), 2f32.powi(-24));
        assert_eq!(u01_f32(u32::MAX), 1.0 - 2f32.powi(-24));
        assert!(u01_f32(u32::MAX) < 1.0);
    }

    #[test]
    fn box_muller_is_finite_at_the_ends() {
        for u1 in [u01_f64(0, 0), u01_f64(u32::MAX, u32::MAX)] {
            let (a, b) = box_muller(u1, 0.3);
            assert!(a.is_finite() && b.is_finite());
        }
        let (a, b) = box_muller(u01_f32(0), 0.3_f32);
        assert!(a.is_finite() && b.is_finite());
    }
}
