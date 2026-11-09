//! Timing of one backend through the [`Backend`] trait.

use fring_model::Backend;
use std::time::Instant;

/// Steps per frame read-back for the "with read-back" column. This is the
/// `frame_every_steps` of the example config (about 6 days at 360 s).
pub const FRAME_STEPS: u64 = 1460;

/// Smallest and largest number of steps per measurement.
pub const MIN_STEPS: u64 = 10;
/// Largest number of steps per measurement (the ephemeris must cover
/// warm-up, probes and 3 repeats of this).
pub const MAX_STEPS: u64 = 30_000;

/// Result of one measurement point.
#[derive(Clone, Debug, PartialEq)]
pub struct Timing {
    /// Steps in each timed call.
    pub steps_per_call: u64,
    /// Best of 3, particle-steps per second.
    pub rate_best: f64,
    /// Slowest of 3.
    pub rate_min: f64,
    /// Fastest of 3 (the same as `rate_best`).
    pub rate_max: f64,
    /// Time of one `particles()` read-back (plus `chaos()` if on), seconds.
    pub readback_s: f64,
    /// Rate if a frame is read back every [`FRAME_STEPS`] steps.
    pub rate_with_readback: f64,
    /// The rate of every timed call, in order (B2 writes one CSV row each).
    pub raw_rates: Vec<f64>,
    /// Median of `raw_rates`.
    pub rate_median: f64,
    /// Collision statistics summed over the timed calls (collision rows only).
    pub coll: Option<CollTotals>,
    /// The statistics of each repeat when the repeats ran on fresh backends
    /// ([`measure_fresh`]); one entry per repeat. One entry (all repeats
    /// together) from [`measure_with`]; empty without collisions.
    pub repeat_coll: Vec<CollTotals>,
}

/// Collision statistics summed over the timed calls of one point.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CollTotals {
    /// Number of collision steps in the timed calls.
    pub coll_steps: u64,
    /// Accepted pairs.
    pub collisions: u64,
    /// Candidate pairs tested.
    pub candidates: u64,
    /// Sum over collision steps of the occupied cells.
    pub occupied_cells: u64,
    /// Sum of N_c.
    pub occupancy_sum: u64,
    /// Sum of N_c squared.
    pub occupancy_sq_sum: u64,
    /// Particles with no partner in their cell.
    pub lonely_particles: u64,
    /// Overfull cells (SPEC 5.6).
    pub overfull_cells: u64,
}

impl CollTotals {
    /// Candidates per particle per collision step.
    pub fn candidates_per_particle(&self, n: usize) -> f64 {
        self.candidates as f64 / (n as f64 * self.coll_steps.max(1) as f64)
    }

    /// Accepted pairs per particle per collision step.
    pub fn collisions_per_particle(&self, n: usize) -> f64 {
        self.collisions as f64 / (n as f64 * self.coll_steps.max(1) as f64)
    }

    /// Particle-weighted occupancy, sum N_c^2 / sum N_c.
    pub fn weighted_occupancy(&self) -> f64 {
        self.occupancy_sq_sum as f64 / (self.occupancy_sum.max(1) as f64)
    }
}

/// What to measure besides the rate (B2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Options {
    /// Timed calls (3 for B1).
    pub repeats: usize,
    /// `Some(every_steps)` if the backend runs collisions: the step counts
    /// are multiples of it and the statistics are summed.
    pub collisions_every: Option<u64>,
}

/// B1 options: 3 repeats, no collisions.
pub const B1_OPTIONS: Options = Options {
    repeats: 3,
    collisions_every: None,
};

/// Median of a non-empty slice.
pub fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let k = s.len();
    if k % 2 == 1 {
        s[k / 2]
    } else {
        0.5 * (s[k / 2 - 1] + s[k / 2])
    }
}

/// Relative spread (max - min) / median of the raw rates.
pub fn spread(v: &[f64]) -> f64 {
    let max = v.iter().copied().fold(f64::MIN, f64::max);
    let min = v.iter().copied().fold(f64::MAX, f64::min);
    (max - min) / median(v)
}

/// Rounds a step count down to a multiple of `every` (at least `every`).
pub fn align_steps(steps: u64, every: u64) -> u64 {
    ((steps / every) * every).max(every)
}

/// Picks the number of steps so that one call takes about `target_s`, from a
/// probe of `probe_steps` that took `probe_s`. The result is a multiple of
/// [`MIN_STEPS`] within [`MIN_STEPS`, `max_steps`].
pub fn choose_steps(probe_steps: u64, probe_s: f64, target_s: f64, max_steps: u64) -> u64 {
    let per_step = (probe_s / probe_steps as f64).max(1e-12);
    let raw = (target_s / per_step).round();
    let raw = if raw.is_finite() {
        raw as u64
    } else {
        max_steps
    };
    let rounded = (raw / MIN_STEPS) * MIN_STEPS;
    rounded.clamp(MIN_STEPS, max_steps)
}

/// Rate with a read-back of `readback_s` after every `frame_steps` steps.
pub fn rate_with_readback(n: usize, rate: f64, frame_steps: u64, readback_s: f64) -> f64 {
    let work = n as f64 * frame_steps as f64;
    work / (work / rate + readback_s)
}

fn time_step(b: &mut dyn Backend, steps: u64) -> Result<f64, String> {
    let t = Instant::now();
    b.step(steps).map_err(|e| e.to_string())?;
    Ok(t.elapsed().as_secs_f64())
}

/// Warms up, picks the step count, times 3 calls, and times the read-back.
/// `chaos_on` adds `chaos()` to the read-back. `target_s` is the time of one
/// call.
pub fn measure(
    b: &mut dyn Backend,
    n: usize,
    chaos_on: bool,
    target_s: f64,
    max_steps: u64,
) -> Result<Timing, String> {
    measure_with(b, n, chaos_on, target_s, max_steps, B1_OPTIONS)
}

/// Warm-up and calibration: grow the probe until it takes at least a quarter
/// of the target (short probes are dominated by launch costs), then pick the
/// step count of one timed call (a multiple of `every`).
fn calibrate(
    b: &mut dyn Backend,
    target_s: f64,
    max_steps: u64,
    every: u64,
) -> Result<u64, String> {
    let mut steps = MIN_STEPS;
    let mut t = time_step(b, steps)?;
    for _ in 0..4 {
        if t >= 0.25 * target_s || steps >= max_steps {
            break;
        }
        steps = choose_steps(steps, t, target_s, max_steps).max(steps + MIN_STEPS);
        t = time_step(b, steps)?;
    }
    Ok(align_steps(
        choose_steps(steps, t, target_s, max_steps),
        every,
    ))
}

/// The collision statistics since the previous `stats()` call, for
/// `coll_steps` collision steps.
fn take_coll(b: &mut dyn Backend, coll_steps: u64) -> Result<CollTotals, String> {
    let st = b.stats().map_err(|e| e.to_string())?;
    let c = st.collision;
    Ok(CollTotals {
        coll_steps,
        collisions: st.collisions_since_last,
        candidates: c.candidates,
        occupied_cells: c.occupied_cells,
        occupancy_sum: c.occupancy_sum,
        occupancy_sq_sum: c.occupancy_sq_sum,
        lonely_particles: c.lonely_particles,
        overfull_cells: c.overfull_cells,
    })
}

/// Read-back: step a little (so the host copy is stale), then time the copy.
fn time_readback(b: &mut dyn Backend, chaos_on: bool, every: u64) -> Result<f64, String> {
    let mut readback_s = f64::MAX;
    for _ in 0..2 {
        b.step(align_steps(MIN_STEPS, every))
            .map_err(|e| e.to_string())?;
        let t0 = Instant::now();
        {
            let v = b.particles().map_err(|e| e.to_string())?;
            std::hint::black_box(v.particles.x_km[0]);
        }
        if chaos_on {
            std::hint::black_box(b.chaos().map_err(|e| e.to_string())?);
        }
        readback_s = readback_s.min(t0.elapsed().as_secs_f64());
    }
    Ok(readback_s)
}

fn sum_coll(parts: &[CollTotals]) -> CollTotals {
    parts.iter().fold(CollTotals::default(), |a, c| CollTotals {
        coll_steps: a.coll_steps + c.coll_steps,
        collisions: a.collisions + c.collisions,
        candidates: a.candidates + c.candidates,
        occupied_cells: a.occupied_cells + c.occupied_cells,
        occupancy_sum: a.occupancy_sum + c.occupancy_sum,
        occupancy_sq_sum: a.occupancy_sq_sum + c.occupancy_sq_sum,
        lonely_particles: a.lonely_particles + c.lonely_particles,
        overfull_cells: a.overfull_cells + c.overfull_cells,
    })
}

fn finish(
    n: usize,
    steps: u64,
    rates: Vec<f64>,
    readback_s: f64,
    repeat_coll: Vec<CollTotals>,
) -> Timing {
    let rate_best = rates.iter().copied().fold(f64::MIN, f64::max);
    let rate_min = rates.iter().copied().fold(f64::MAX, f64::min);
    Timing {
        steps_per_call: steps,
        rate_best,
        rate_min,
        rate_max: rate_best,
        readback_s,
        rate_with_readback: rate_with_readback(n, rate_best, FRAME_STEPS, readback_s),
        rate_median: median(&rates),
        raw_rates: rates,
        coll: (!repeat_coll.is_empty()).then(|| sum_coll(&repeat_coll)),
        repeat_coll,
    }
}

/// [`measure`] with options (repeat count, collision statistics). All repeats
/// run on the same backend, one after the other.
pub fn measure_with(
    b: &mut dyn Backend,
    n: usize,
    chaos_on: bool,
    target_s: f64,
    max_steps: u64,
    opt: Options,
) -> Result<Timing, String> {
    // Step counts are multiples of 10 and of `every` (collision steps fall on
    // the end of a call, as in a real run).
    let every = opt.collisions_every.unwrap_or(1).max(1);
    let steps = calibrate(b, target_s, max_steps, every)?;
    if opt.collisions_every.is_some() {
        // Clear the statistics of the calibration.
        b.stats().map_err(|e| e.to_string())?;
    }
    let mut rates = Vec::with_capacity(opt.repeats);
    for _ in 0..opt.repeats {
        let s = time_step(b, steps)?;
        rates.push(n as f64 * steps as f64 / s);
    }
    let repeat_coll = if opt.collisions_every.is_some() {
        vec![take_coll(b, opt.repeats as u64 * steps / every)?]
    } else {
        Vec::new()
    };
    let readback_s = time_readback(b, chaos_on, every)?;
    Ok(finish(n, steps, rates, readback_s, repeat_coll))
}

/// A way to make a backend in its initial state.
pub type MakeBackend<'a> = dyn FnMut() -> Result<Box<dyn Backend>, String> + 'a;

/// Like [`measure_with`] for collision rows: every repeat runs on a new
/// backend in the initial state, so that the repeats measure the same state
/// (collisions cool the ring, so repeats on one backend would not). One
/// backend is made for the calibration and the read-back, then one per
/// repeat, each warmed up with one untimed call of the smallest step count.
/// The old backend is dropped before the next is made. The statistics of every
/// repeat are kept in `Timing::repeat_coll`.
pub fn measure_fresh(
    make: &mut MakeBackend<'_>,
    n: usize,
    chaos_on: bool,
    target_s: f64,
    max_steps: u64,
    opt: Options,
) -> Result<Timing, String> {
    let every = opt.collisions_every.unwrap_or(1).max(1);
    let (steps, readback_s) = {
        let mut b = make()?;
        let steps = calibrate(b.as_mut(), target_s, max_steps, every)?;
        (steps, time_readback(b.as_mut(), chaos_on, every)?)
    };
    let mut rates = Vec::with_capacity(opt.repeats);
    let mut repeat_coll = Vec::with_capacity(opt.repeats);
    for _ in 0..opt.repeats {
        let mut b = make()?;
        // Untimed warm-up of the new backend (pipelines, caches); its
        // statistics are cleared.
        b.step(align_steps(MIN_STEPS, every))
            .map_err(|e| e.to_string())?;
        b.stats().map_err(|e| e.to_string())?;
        let s = time_step(b.as_mut(), steps)?;
        rates.push(n as f64 * steps as f64 / s);
        repeat_coll.push(take_coll(b.as_mut(), steps / every)?);
    }
    Ok(finish(n, steps, rates, readback_s, repeat_coll))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steps_follow_target() {
        // 10 steps took 0.1 s: 0.01 s per step, so 1.5 s needs 150 steps.
        assert_eq!(choose_steps(10, 0.1, 1.5, 30_000), 150);
        // Slow: never below the minimum.
        assert_eq!(choose_steps(10, 100.0, 1.5, 30_000), MIN_STEPS);
        // Fast: capped.
        assert_eq!(choose_steps(10, 1e-9, 1.5, 30_000), 30_000);
        // Multiple of 10.
        assert_eq!(choose_steps(10, 0.0333, 1.0, 30_000) % MIN_STEPS, 0);
    }

    #[test]
    fn median_spread_and_alignment() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 2.0, 3.0]), 2.5);
        // (max - min) / median = 0.2 / 1.0.
        assert!((spread(&[0.9, 1.0, 1.1]) - 0.2).abs() < 1e-12);
        assert_eq!(align_steps(1234, 10), 1230);
        assert_eq!(align_steps(3, 10), 10);
        assert_eq!(align_steps(7, 1), 7);
        let c = CollTotals {
            coll_steps: 10,
            candidates: 500,
            collisions: 100,
            occupancy_sum: 20,
            occupancy_sq_sum: 60,
            ..CollTotals::default()
        };
        assert_eq!(c.candidates_per_particle(100), 0.5);
        assert_eq!(c.collisions_per_particle(100), 0.1);
        assert_eq!(c.weighted_occupancy(), 3.0);
    }

    #[test]
    fn readback_lowers_rate() {
        // N = 1e6, 1460 steps at 1e9/s take 1.46 s; a 0.54 s read-back gives 0.73 of it.
        let r = rate_with_readback(1_000_000, 1e9, 1460, 0.54);
        assert!((r / 1e9 - 0.73).abs() < 1e-3, "{r}");
        assert_eq!(rate_with_readback(10, 5.0, 100, 0.0), 5.0);
    }
}
