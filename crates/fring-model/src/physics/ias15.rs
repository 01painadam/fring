//! IAS15: a 15th-order integrator with adaptive step, for the moon ephemeris.
//!
//! Written from the paper: H. Rein and D. S. Spiegel, "IAS15: a fast, adaptive,
//! high-order integrator for gravitational dynamics, accurate to machine
//! precision over a billion orbits", MNRAS 446, 1424 (2015), arXiv:1409.4779.
//! No code of other implementations was used.
//!
//! The method, in short. The system is `x'' = F(x)` (the force does not depend
//! on velocity or time here). Inside one step of length `dt`, with `tau` in
//! [0, 1], the acceleration is a polynomial of degree 7:
//! `F(tau) = F0 + b0 tau + b1 tau² + ... + b6 tau⁷`. It is fixed by the
//! accelerations at the 7 Gauss-Radau nodes `h_1 .. h_7` (plus `F0` at `h_0 = 0`).
//! Two integrations give position and velocity as polynomials in `tau`. The
//! values at the nodes depend on the coefficients, so a predictor-corrector
//! iteration solves for them. The new step size comes from the size of `b6`
//! relative to the acceleration: `dt_new = dt (epsilon / (|b6| / |F|))^(1/7)`.
//! A step with `dt_new < 0.25 dt` is rejected and repeated.
//!
//! Round-off control: positions, velocities and the time are added with
//! compensated (Kahan or two-sum) summation, so the rounding error of the
//! many small increments does not accumulate as a random walk.
//!
//! The state is a flat array of components (for moons: `x, y, z` of each).

// Index loops over several parallel arrays are clearer than iterator chains here.
#![allow(clippy::needless_range_loop)]

use crate::{Error, Result};

const NODES: usize = 8;
const NB: usize = 7;
/// Largest number of corrector iterations per step.
const MAX_ITERATIONS: usize = 12;
/// Convergence limit on the change of b6 relative to the largest acceleration.
const B6_TOLERANCE: f64 = 1.0e-16;
/// A step whose allowed size is below this fraction of the size used is repeated.
const REJECT_BELOW: f64 = 0.25;
/// The step size grows by at most this factor per step.
const MAX_GROWTH: f64 = 4.0;

/// Starting values for the Gauss-Radau nodes (fractions of the step). They are
/// refined by Newton's method in [`radau_nodes`].
const NODE_GUESS: [f64; 7] = [
    0.056_262_560_536_922_1,
    0.180_240_691_736_892_4,
    0.352_624_717_113_169_6,
    0.547_153_626_330_555_4,
    0.734_210_177_215_410_5,
    0.885_320_946_839_095_8,
    0.977_520_613_561_287_5,
];

/// Legendre polynomial `P_n(x)` and `P_(n-1)(x)`.
fn legendre(n: usize, x: f64) -> (f64, f64) {
    let (mut p_prev, mut p) = (1.0, x);
    for k in 2..=n {
        let kf = k as f64;
        let next = ((2.0 * kf - 1.0) * x * p - (kf - 1.0) * p_prev) / kf;
        p_prev = p;
        p = next;
    }
    (p, p_prev)
}

/// The 8 Gauss-Radau nodes on [0, 1]: `h_0 = 0` and the 7 roots in (0, 1) of
/// `P_7(x) + P_8(x)` mapped from [-1, 1] by `h = (x + 1) / 2`.
pub fn radau_nodes() -> [f64; NODES] {
    let f = |x: f64| {
        let (p8, p7) = legendre(8, x);
        p7 + p8
    };
    let mut h = [0.0; NODES];
    for (i, g) in NODE_GUESS.iter().enumerate() {
        let mut x = 2.0 * g - 1.0;
        for _ in 0..6 {
            let dx = 1.0e-7;
            let slope = (f(x + dx) - f(x - dx)) / (2.0 * dx);
            x -= f(x) / slope;
        }
        h[i + 1] = 0.5 * (x + 1.0);
    }
    h
}

/// An IAS15 integration of `x'' = F(x)`.
#[derive(Clone, Debug)]
pub struct Ias15 {
    epsilon: f64,
    dim: usize,
    h: [f64; NODES],
    /// `b_j = sum_k cmat[j][k] g_(k+1)`: Newton form to power form.
    cmat: [[f64; NB]; NB],
    /// Binomial coefficients `binom[n][k]`, for the shift of the polynomial.
    binom: [[f64; NB + 2]; NB + 2],
    t_hi: f64,
    t_lo: f64,
    x: Vec<f64>,
    x_comp: Vec<f64>,
    v: Vec<f64>,
    v_comp: Vec<f64>,
    dt: f64,
    b: Vec<[f64; NB]>,
    // Scratch.
    f0: Vec<f64>,
    fnode: Vec<Vec<f64>>,
    xs: Vec<f64>,
    // Last accepted step, for dense output.
    last: Option<LastStep>,
    steps: u64,
    rejected: u64,
    force_evaluations: u64,
}

#[derive(Clone, Debug)]
struct LastStep {
    t0_hi: f64,
    t0_lo: f64,
    dt: f64,
    x0: Vec<f64>,
    v0: Vec<f64>,
    f0: Vec<f64>,
    b: Vec<[f64; NB]>,
}

/// Compensated addition `sum += inc` (Kahan). `comp` holds the rounding error.
#[inline]
fn kahan_add(sum: &mut f64, comp: &mut f64, inc: f64) {
    let y = inc - *comp;
    let t = *sum + y;
    *comp = (t - *sum) - y;
    *sum = t;
}

impl Ias15 {
    /// Starts an integration at time `t0_s` with state `x0` and `v0` (same
    /// length). `epsilon` is the accuracy parameter (the paper uses 1e-9);
    /// `initial_dt_s` is the first trial step.
    pub fn new(epsilon: f64, t0_s: f64, x0: &[f64], v0: &[f64], initial_dt_s: f64) -> Result<Self> {
        if x0.len() != v0.len() || x0.is_empty() {
            return Err(Error::Input(
                "ias15: x and v need the same length > 0".into(),
            ));
        }
        if !(epsilon.is_finite() && epsilon > 0.0 && initial_dt_s.is_finite() && initial_dt_s > 0.0)
        {
            return Err(Error::Input(format!(
                "ias15: epsilon {epsilon} and initial step {initial_dt_s} must be positive"
            )));
        }
        let dim = x0.len();
        let h = radau_nodes();
        // Newton basis N_k(tau) = tau (tau - h_1) ... (tau - h_(k-1)), k = 1..7,
        // as power series; cmat[j][k-1] = coefficient of tau^(j+1) in N_k.
        let mut cmat = [[0.0; NB]; NB];
        let mut poly = [0.0; NODES];
        poly[1] = 1.0;
        for k in 1..=NB {
            for j in 0..NB {
                cmat[j][k - 1] = poly[j + 1];
            }
            // Multiply by (tau - h_k).
            let mut next = [0.0; NODES];
            for d in 0..NODES - 1 {
                next[d + 1] += poly[d];
                next[d] -= h[k] * poly[d];
            }
            poly = next;
        }
        let mut binom = [[0.0_f64; NB + 2]; NB + 2];
        for n in 0..NB + 2 {
            binom[n][0] = 1.0;
            for k in 1..=n {
                binom[n][k] = binom[n - 1][k - 1] + if k < n { binom[n - 1][k] } else { 0.0 };
            }
        }
        Ok(Self {
            epsilon,
            dim,
            h,
            cmat,
            binom,
            t_hi: t0_s,
            t_lo: 0.0,
            x: x0.to_vec(),
            x_comp: vec![0.0; dim],
            v: v0.to_vec(),
            v_comp: vec![0.0; dim],
            dt: initial_dt_s,
            b: vec![[0.0; NB]; dim],
            f0: vec![0.0; dim],
            fnode: vec![vec![0.0; dim]; NODES],
            xs: vec![0.0; dim],
            last: None,
            steps: 0,
            rejected: 0,
            force_evaluations: 0,
        })
    }

    /// Current time in seconds.
    pub fn time_s(&self) -> f64 {
        self.t_hi + self.t_lo
    }

    /// Current position components.
    pub fn x(&self) -> &[f64] {
        &self.x
    }

    /// Current velocity components.
    pub fn v(&self) -> &[f64] {
        &self.v
    }

    /// Accepted steps so far.
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Rejected (repeated) steps so far.
    pub fn rejected_steps(&self) -> u64 {
        self.rejected
    }

    /// Calls of the force function so far.
    pub fn force_evaluations(&self) -> u64 {
        self.force_evaluations
    }

    /// The step size that the next step will try, in seconds.
    pub fn next_dt_s(&self) -> f64 {
        self.dt
    }

    /// Start time of the last accepted step.
    pub fn last_step_start_s(&self) -> Option<f64> {
        self.last.as_ref().map(|l| l.t0_hi + l.t0_lo)
    }

    /// Advances one adaptive step. `force(x, a)` writes the accelerations at
    /// positions `x` into `a`. Returns the step taken in seconds.
    pub fn step<F: FnMut(&[f64], &mut [f64])>(&mut self, force: &mut F) -> Result<f64> {
        let dim = self.dim;
        // Positions and velocities with the compensation applied.
        let x0: Vec<f64> = (0..dim).map(|i| self.x[i] - self.x_comp[i]).collect();
        let v0: Vec<f64> = (0..dim).map(|i| self.v[i] - self.v_comp[i]).collect();
        force(&x0, &mut self.f0);
        self.force_evaluations += 1;
        if !self.f0.iter().all(|a| a.is_finite()) {
            return Err(Error::Backend("ias15: acceleration is not finite".into()));
        }

        let mut retries = 0;
        let (dt, ratio) = loop {
            let dt = self.dt;
            self.corrector(force, &x0, &v0, dt)?;
            // Error estimate: largest |b6| over largest |F| at the last node.
            let b6 = self.b.iter().map(|b| b[NB - 1].abs()).fold(0.0, f64::max);
            let fmax = self.fnode[NODES - 1]
                .iter()
                .map(|a| a.abs())
                .fold(0.0, f64::max);
            if !(b6.is_finite() && fmax.is_finite()) {
                return Err(Error::Backend("ias15: step diverged".into()));
            }
            let eb = if fmax > 0.0 { b6 / fmax } else { 0.0 };
            let ratio = if eb > 0.0 {
                (self.epsilon / eb).powf(1.0 / 7.0)
            } else {
                MAX_GROWTH
            };
            if ratio < REJECT_BELOW && retries < 30 {
                self.rejected += 1;
                retries += 1;
                // Repeat with the smaller step; the polynomial scales as q^(k+1).
                let q = ratio;
                for b in &mut self.b {
                    let mut qk = q;
                    for bk in b.iter_mut() {
                        *bk *= qk;
                        qk *= q;
                    }
                }
                self.dt = dt * q;
                continue;
            }
            break (dt, ratio.min(MAX_GROWTH));
        };

        // Accept: advance the state.
        let dt2 = dt * dt;
        let last = LastStep {
            t0_hi: self.t_hi,
            t0_lo: self.t_lo,
            dt,
            x0: x0.clone(),
            v0: v0.clone(),
            f0: self.f0.clone(),
            b: self.b.clone(),
        };
        for i in 0..dim {
            let b = &self.b[i];
            let xinc = v0[i] * dt
                + dt2
                    * (self.f0[i] * 0.5
                        + b[0] / 6.0
                        + b[1] / 12.0
                        + b[2] / 20.0
                        + b[3] / 30.0
                        + b[4] / 42.0
                        + b[5] / 56.0
                        + b[6] / 72.0);
            let vinc = dt
                * (self.f0[i]
                    + b[0] / 2.0
                    + b[1] / 3.0
                    + b[2] / 4.0
                    + b[3] / 5.0
                    + b[4] / 6.0
                    + b[5] / 7.0
                    + b[6] / 8.0);
            kahan_add(&mut self.x[i], &mut self.x_comp[i], xinc);
            kahan_add(&mut self.v[i], &mut self.v_comp[i], vinc);
        }
        // Time: two-sum.
        let s = self.t_hi + dt;
        let bb = s - self.t_hi;
        self.t_lo += (self.t_hi - (s - bb)) + (dt - bb);
        self.t_hi = s;
        self.last = Some(last);
        self.steps += 1;

        // Next step size, and the polynomial seen from the new step:
        // F_new(t') = F_old(1 + q t'), so b'_j = q^(j+1) sum_(k >= j) C(k+1, j+1) b_k.
        let q = ratio;
        self.dt = dt * q;
        let binom = self.binom;
        for b in &mut self.b {
            let old = *b;
            let mut qj = q;
            for j in 0..NB {
                let mut sum = 0.0;
                for (k, ok) in old.iter().enumerate().skip(j) {
                    sum += binom[k + 1][j + 1] * ok;
                }
                b[j] = qj * sum;
                qj *= q;
            }
        }
        Ok(dt)
    }

    /// Predictor-corrector iteration for the step `dt` from `(x0, v0)`; leaves
    /// the converged coefficients in `self.b` and the node accelerations in
    /// `self.fnode`.
    fn corrector<F: FnMut(&[f64], &mut [f64])>(
        &mut self,
        force: &mut F,
        x0: &[f64],
        v0: &[f64],
        dt: f64,
    ) -> Result<()> {
        let dim = self.dim;
        let mut prev_delta = f64::INFINITY;
        for iteration in 0..MAX_ITERATIONS {
            for n in 1..NODES {
                let hn = self.h[n];
                for i in 0..dim {
                    let b = &self.b[i];
                    let poly = self.f0[i] * 0.5
                        + hn * (b[0] / 6.0
                            + hn * (b[1] / 12.0
                                + hn * (b[2] / 20.0
                                    + hn * (b[3] / 30.0
                                        + hn * (b[4] / 42.0
                                            + hn * (b[5] / 56.0 + hn * b[6] / 72.0))))));
                    self.xs[i] = x0[i] + hn * dt * v0[i] + dt * dt * hn * hn * poly;
                }
                let (xs, fnode) = (&self.xs, &mut self.fnode[n]);
                force(xs, fnode);
                self.force_evaluations += 1;
            }
            self.fnode[0].copy_from_slice(&self.f0);
            // Divided differences and new coefficients.
            let mut delta: f64 = 0.0;
            for i in 0..dim {
                let mut dd = [0.0; NODES];
                for n in 0..NODES {
                    dd[n] = self.fnode[n][i];
                }
                for level in 1..NODES {
                    for n in (level..NODES).rev() {
                        dd[n] = (dd[n] - dd[n - 1]) / (self.h[n] - self.h[n - level]);
                    }
                }
                let mut newb = [0.0; NB];
                for (j, nb) in newb.iter_mut().enumerate() {
                    let mut sum = 0.0;
                    for k in j..NB {
                        sum += self.cmat[j][k] * dd[k + 1];
                    }
                    *nb = sum;
                }
                delta = delta.max((newb[NB - 1] - self.b[i][NB - 1]).abs());
                self.b[i] = newb;
            }
            let fmax = self.fnode[NODES - 1]
                .iter()
                .map(|a| a.abs())
                .fold(0.0, f64::max);
            if !delta.is_finite() {
                return Err(Error::Backend("ias15: iteration diverged".into()));
            }
            if iteration >= 1 && (delta <= B6_TOLERANCE * fmax || delta >= prev_delta) {
                break;
            }
            prev_delta = delta;
        }
        Ok(())
    }

    /// State at time `t_s` inside the last accepted step (positions and
    /// velocities written to `x_out` and `v_out`). The time should lie between
    /// the start and the end of that step.
    ///
    /// # Panics
    /// If no step has been taken, or the slices have the wrong length.
    pub fn dense_output(&self, t_s: f64, x_out: &mut [f64], v_out: &mut [f64]) {
        let l = self.last.as_ref().expect("dense_output needs a step");
        let tau = ((t_s - l.t0_hi) - l.t0_lo) / l.dt;
        let (dt, dt2) = (l.dt, l.dt * l.dt);
        for i in 0..self.dim {
            let b = &l.b[i];
            let xp = l.f0[i] * 0.5
                + tau
                    * (b[0] / 6.0
                        + tau
                            * (b[1] / 12.0
                                + tau
                                    * (b[2] / 20.0
                                        + tau
                                            * (b[3] / 30.0
                                                + tau
                                                    * (b[4] / 42.0
                                                        + tau
                                                            * (b[5] / 56.0
                                                                + tau * b[6] / 72.0))))));
            x_out[i] = l.x0[i] + tau * dt * l.v0[i] + dt2 * tau * tau * xp;
            let vp = l.f0[i]
                + tau
                    * (b[0] / 2.0
                        + tau
                            * (b[1] / 3.0
                                + tau
                                    * (b[2] / 4.0
                                        + tau
                                            * (b[3] / 5.0
                                                + tau
                                                    * (b[4] / 6.0
                                                        + tau
                                                            * (b[5] / 7.0 + tau * b[6] / 8.0))))));
            v_out[i] = l.v0[i] + dt * tau * vp;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radau_nodes_match_the_known_values() {
        let h = radau_nodes();
        for i in 0..7 {
            assert!((h[i + 1] - NODE_GUESS[i]).abs() < 1e-14, "{i}");
        }
        // Each interior node is a root of P7 + P8 (mapped to [-1, 1]).
        for hn in &h[1..] {
            let (p8, p7) = legendre(8, 2.0 * hn - 1.0);
            assert!((p7 + p8).abs() < 1e-13);
        }
        assert!(h.windows(2).all(|w| w[0] < w[1]));
        assert!(h[7] < 1.0);
    }

    #[test]
    fn harmonic_oscillator_over_many_periods() {
        // x'' = -x, period 2 pi.
        let mut f = |x: &[f64], a: &mut [f64]| a[0] = -x[0];
        let mut s = Ias15::new(1e-9, 0.0, &[1.0], &[0.0], 0.05).unwrap();
        let t_end = 2.0 * std::f64::consts::PI * 1000.0;
        while s.time_s() < t_end {
            s.step(&mut f).unwrap();
        }
        let t = s.time_s();
        assert!(
            (s.x()[0] - t.cos()).abs() < 1e-11,
            "{:e}",
            s.x()[0] - t.cos()
        );
        assert!((s.v()[0] + t.sin()).abs() < 1e-11);
    }
}
