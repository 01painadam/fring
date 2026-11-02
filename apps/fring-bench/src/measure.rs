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
    // Warm-up and calibration: grow the probe until it takes at least a
    // quarter of the target (short probes are dominated by launch costs).
    let mut steps = MIN_STEPS;
    let mut t = time_step(b, steps)?;
    for _ in 0..4 {
        if t >= 0.25 * target_s || steps >= max_steps {
            break;
        }
        steps = choose_steps(steps, t, target_s, max_steps).max(steps + MIN_STEPS);
        t = time_step(b, steps)?;
    }
    let steps = choose_steps(steps, t, target_s, max_steps);
    let mut rates = Vec::with_capacity(3);
    for _ in 0..3 {
        let s = time_step(b, steps)?;
        rates.push(n as f64 * steps as f64 / s);
    }
    let rate_best = rates.iter().copied().fold(f64::MIN, f64::max);
    let rate_min = rates.iter().copied().fold(f64::MAX, f64::min);
    // Read-back: step a little (so the host copy is stale), then time the copy.
    let mut readback_s = f64::MAX;
    for _ in 0..2 {
        b.step(MIN_STEPS).map_err(|e| e.to_string())?;
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
    Ok(Timing {
        steps_per_call: steps,
        rate_best,
        rate_min,
        rate_max: rate_best,
        readback_s,
        rate_with_readback: rate_with_readback(n, rate_best, FRAME_STEPS, readback_s),
    })
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
    fn readback_lowers_rate() {
        // N = 1e6, 1460 steps at 1e9/s take 1.46 s; a 0.54 s read-back gives 0.73 of it.
        let r = rate_with_readback(1_000_000, 1e9, 1460, 0.54);
        assert!((r / 1e9 - 0.73).abs() < 1e-3, "{r}");
        assert_eq!(rate_with_readback(10, 5.0, 100, 0.0), 5.0);
    }
}
