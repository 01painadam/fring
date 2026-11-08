//! The moons' part of the Wisdom-Holman kick (ADR 0003, ADR 0004).
//!
//! A massless particle at `p` (Saturn-centred, inertial) feels, besides
//! Saturn's point mass (the Kepler drift) and the J terms,
//!
//! ```text
//! a_moons(p) = sum_j GM_j [ (r_j - p)/|r_j - p|³  -  r_j/|r_j|³ ]
//! ```
//!
//! The first part is the direct pull of moon j. The second part is the
//! indirect term: Saturn itself is pulled toward moon j, and the frame is
//! centred on Saturn. The indirect term is the same for every particle, so a
//! [`KickTable`] computes it once per kick, together with the moon positions.
//!
//! The table holds one entry per kick of a block of steps. The kick of step
//! `s` (counted from the start of the run) is at the middle of the step,
//! `t = start + (s + 1/2) h`, because the step is drift-kick-drift.

use super::gravity::SaturnField;
use super::{interpolate_moons_into, kepler::kepler_drift};
use crate::{BodyState, Error, MoonEphemeris, Real, Result, Vec3};

/// The indirect acceleration `-sum_j GM_j r_j/|r_j|³` in km s⁻²: the same for
/// every particle at one time.
pub fn indirect_acceleration<T: Real>(gm_km3_s2: &[T], moon_pos_km: &[Vec3<T>]) -> Vec3<T> {
    let mut a = Vec3::new(T::ZERO, T::ZERO, T::ZERO);
    for (&gm, &r) in gm_km3_s2.iter().zip(moon_pos_km) {
        let r2 = r.norm_squared();
        a -= r * (gm / (r2 * r2.sqrt()));
    }
    a
}

/// The direct pull `sum_j GM_j (r_j - p)/|r_j - p|³` of the moons at the
/// position `p_km`, in km s⁻².
#[inline]
pub fn moon_direct_acceleration<T: Real>(
    gm_km3_s2: &[T],
    moon_pos_km: &[Vec3<T>],
    p_km: Vec3<T>,
) -> Vec3<T> {
    let mut a = Vec3::new(T::ZERO, T::ZERO, T::ZERO);
    for (&gm, &r) in gm_km3_s2.iter().zip(moon_pos_km) {
        let d = r - p_km;
        let d2 = d.norm_squared();
        a += d * (gm / (d2 * d2.sqrt()));
    }
    a
}

/// The whole kick acceleration of a particle at `p_km`: the J terms of
/// Saturn's field, the direct pull of the moons and the indirect term (given,
/// from [`indirect_acceleration`]). Without the point mass, which is in the
/// Kepler drift.
#[inline]
pub fn kick_acceleration<T: Real>(
    field: &SaturnField<T>,
    gm_km3_s2: &[T],
    moon_pos_km: &[Vec3<T>],
    indirect_km_s2: Vec3<T>,
    p_km: Vec3<T>,
) -> Vec3<T> {
    field.acceleration_without_point_mass(p_km)
        + moon_direct_acceleration(gm_km3_s2, moon_pos_km, p_km)
        + indirect_km_s2
}

/// Moon positions and the indirect acceleration at the kick times of a block
/// of steps. Filled once, then read by every particle.
#[derive(Clone, Debug)]
pub struct KickTable<T: Real> {
    gm_km3_s2: Vec<T>,
    /// `n_steps × n_moons` positions, step-major.
    pos_km: Vec<Vec3<T>>,
    indirect_km_s2: Vec<Vec3<T>>,
    n_steps: usize,
    /// Extended rows (ICR-004), filled only by [`KickTable::fill_ext`].
    ext: ExtRows<T>,
}

/// A table row for the Hermite sub-kicks: time from the block start, positions, velocities.
type Node<'a, T> = (T, &'a [Vec3<T>], &'a [Vec3<T>]);

/// Moon velocities at the kick rows and the two sync rows (block start and
/// block end), for the sub-kicks of ICR-004.
#[derive(Clone, Debug)]
struct ExtRows<T: Real> {
    filled: bool,
    /// `n_steps × n_moons` velocities, step-major.
    vel_km_s: Vec<Vec3<T>>,
    /// Row 0: block start `t_0`. Row 1: block end `t_n`. Each `n_moons` long.
    sync_pos_km: [Vec<Vec3<T>>; 2],
    sync_vel_km_s: [Vec<Vec3<T>>; 2],
    /// False when the sync time is outside the ephemeris.
    sync_ok: [bool; 2],
}

impl<T: Real> ExtRows<T> {
    fn empty() -> Self {
        Self {
            filled: false,
            vel_km_s: Vec::new(),
            sync_pos_km: [Vec::new(), Vec::new()],
            sync_vel_km_s: [Vec::new(), Vec::new()],
            sync_ok: [false; 2],
        }
    }
}

impl<T: Real> KickTable<T> {
    /// An empty table for moons with these GM values (km³ s⁻²).
    pub fn new(gm_km3_s2: Vec<T>) -> Self {
        Self {
            gm_km3_s2,
            pos_km: Vec::new(),
            indirect_km_s2: Vec::new(),
            n_steps: 0,
            ext: ExtRows::empty(),
        }
    }

    /// A table for the moons of an ephemeris (their GM values), still empty.
    pub fn for_ephemeris(eph: &MoonEphemeris) -> Self {
        Self::new(
            eph.tracks
                .iter()
                .map(|t| T::from_f64(t.gm_km3_s2))
                .collect(),
        )
    }

    /// Number of moons.
    pub fn n_moons(&self) -> usize {
        self.gm_km3_s2.len()
    }

    /// Number of kicks in the table.
    pub fn len(&self) -> usize {
        self.n_steps
    }

    /// True if the table has no kicks.
    pub fn is_empty(&self) -> bool {
        self.n_steps == 0
    }

    /// GM of each moon in km³ s⁻².
    pub fn gm_km3_s2(&self) -> &[T] {
        &self.gm_km3_s2
    }

    /// Moon positions at kick `k`, in km.
    pub fn moon_positions_km(&self, k: usize) -> &[Vec3<T>] {
        let m = self.n_moons();
        &self.pos_km[k * m..(k + 1) * m]
    }

    /// The indirect acceleration at kick `k`, in km s⁻².
    pub fn indirect_km_s2(&self, k: usize) -> Vec3<T> {
        self.indirect_km_s2[k]
    }

    /// Fills the table for steps `first_step` to `first_step + n_steps - 1`
    /// of a run that started at `run_start_s`, with the step `h_s`. Kick `k`
    /// is at `run_start_s + (first_step + k + 1/2) h_s`.
    ///
    /// Returns [`Error::Input`] if a kick time is outside the ephemeris, or
    /// if the number of moons differs from the ephemeris.
    pub fn fill(
        &mut self,
        eph: &MoonEphemeris,
        run_start_s: f64,
        h_s: f64,
        first_step: u64,
        n_steps: usize,
    ) -> Result<()> {
        self.fill_inner(eph, run_start_s, h_s, first_step, n_steps, false)
    }

    /// As [`fill`](Self::fill), and also the moon velocities at the kick rows
    /// and the moon states at the two sync times `t_0 = start + first_step h`
    /// and `t_n = start + (first_step + n_steps) h` (ICR-004 section 3). The
    /// rows of `fill` are the same bits. A sync time outside the ephemeris is
    /// not an error: [`sync_ok`](Self::sync_ok) is then false for that row.
    pub fn fill_ext(
        &mut self,
        eph: &MoonEphemeris,
        run_start_s: f64,
        h_s: f64,
        first_step: u64,
        n_steps: usize,
    ) -> Result<()> {
        self.fill_inner(eph, run_start_s, h_s, first_step, n_steps, true)
    }

    fn fill_inner(
        &mut self,
        eph: &MoonEphemeris,
        run_start_s: f64,
        h_s: f64,
        first_step: u64,
        n_steps: usize,
        ext: bool,
    ) -> Result<()> {
        let m = self.n_moons();
        if eph.tracks.len() != m {
            return Err(Error::Input(format!(
                "kick table has {m} moons, the ephemeris has {}",
                eph.tracks.len()
            )));
        }
        self.pos_km.clear();
        self.indirect_km_s2.clear();
        self.n_steps = 0;
        self.ext = ExtRows::empty();
        let mut states = vec![BodyState::default(); m];
        let mut pos = vec![Vec3::new(T::ZERO, T::ZERO, T::ZERO); m];
        for k in 0..n_steps {
            let t = run_start_s + ((first_step + k as u64) as f64 + 0.5) * h_s;
            interpolate_moons_into(eph, t, &mut states)?;
            for (p, s) in pos.iter_mut().zip(&states) {
                *p = s.position_km.cast::<T>();
            }
            self.indirect_km_s2
                .push(indirect_acceleration(&self.gm_km3_s2, &pos));
            self.pos_km.extend_from_slice(&pos);
            if ext {
                self.ext
                    .vel_km_s
                    .extend(states.iter().map(|s| s.velocity_km_s.cast::<T>()));
            }
        }
        if ext {
            for (row, step) in [(0, first_step), (1, first_step + n_steps as u64)] {
                let t = run_start_s + (step as f64) * h_s;
                if interpolate_moons_into(eph, t, &mut states).is_ok() {
                    self.ext.sync_ok[row] = true;
                    self.ext.sync_pos_km[row] =
                        states.iter().map(|s| s.position_km.cast::<T>()).collect();
                    self.ext.sync_vel_km_s[row] =
                        states.iter().map(|s| s.velocity_km_s.cast::<T>()).collect();
                }
            }
            self.ext.filled = true;
        }
        self.n_steps = n_steps;
        Ok(())
    }

    /// True if the table was made by [`fill_ext`](Self::fill_ext).
    pub fn has_ext(&self) -> bool {
        self.ext.filled
    }

    /// Moon velocities at kick `k`, in km/s (needs `fill_ext`).
    pub fn moon_velocities_km_s(&self, k: usize) -> &[Vec3<T>] {
        let m = self.n_moons();
        &self.ext.vel_km_s[k * m..(k + 1) * m]
    }

    /// True if sync row `row` (0: block start, 1: block end) has moon states.
    pub fn sync_ok(&self, row: usize) -> bool {
        self.ext.sync_ok[row]
    }

    /// Moon positions at sync row `row`, km (empty if not `sync_ok`).
    pub fn sync_positions_km(&self, row: usize) -> &[Vec3<T>] {
        &self.ext.sync_pos_km[row]
    }

    /// Moon velocities at sync row `row`, km/s (empty if not `sync_ok`).
    pub fn sync_velocities_km_s(&self, row: usize) -> &[Vec3<T>] {
        &self.ext.sync_vel_km_s[row]
    }

    /// Moon states at sub-kick `j` of step `k` of the block, when the step is
    /// done as `n_sub` sub-steps (ICR-004 section 3): the kick is at
    /// `t_k + (j + 1/2) h/n_sub`. `pos` and `vel` have `n_moons` entries.
    ///
    /// For `n_sub = 1` the rows of kick `k` are copied (no interpolation).
    /// Otherwise it is the cubic Hermite polynomial between the two nearest
    /// rows of the sequence S0, K_0, ..., K_{n-1}, S1. If a sync row is not
    /// available, the two nearest kick rows are used instead (a short
    /// extrapolation); with one kick only, the kick row itself.
    pub fn moons_at_subkick(
        &self,
        k: usize,
        j: u32,
        n_sub: u32,
        h_s: T,
        pos: &mut [Vec3<T>],
        vel: &mut [Vec3<T>],
    ) {
        let m = self.n_moons();
        if n_sub == 1 {
            pos[..m].copy_from_slice(self.moon_positions_km(k));
            vel[..m].copy_from_slice(self.moon_velocities_km_s(k));
            return;
        }
        let half = T::from_f64(0.5);
        let n = self.n_steps;
        // Times from the block start t_0, s.
        let t = (T::from_f64(k as f64)
            + (T::from_f64(f64::from(j)) + half) / T::from_f64(f64::from(n_sub)))
            * h_s;
        let tk = (T::from_f64(k as f64) + half) * h_s;
        // The two nodes: (time, positions, velocities).
        let node = |i: i64| -> Option<Node<'_, T>> {
            if i < 0 {
                self.ext.sync_ok[0].then(|| {
                    (
                        T::ZERO,
                        &self.ext.sync_pos_km[0][..],
                        &self.ext.sync_vel_km_s[0][..],
                    )
                })
            } else if i as usize >= n {
                self.ext.sync_ok[1].then(|| {
                    (
                        T::from_f64(n as f64) * h_s,
                        &self.ext.sync_pos_km[1][..],
                        &self.ext.sync_vel_km_s[1][..],
                    )
                })
            } else {
                let i = i as usize;
                Some((
                    (T::from_f64(i as f64) + half) * h_s,
                    self.moon_positions_km(i),
                    self.moon_velocities_km_s(i),
                ))
            }
        };
        let k = k as i64;
        let (a, b) = if t < tk {
            match (node(k - 1), node(k)) {
                (Some(a), Some(b)) => (a, b),
                // No S0: use K_k and K_{k+1}.
                _ => (
                    node(k).unwrap(),
                    node(k + 1).unwrap_or_else(|| node(k).unwrap()),
                ),
            }
        } else {
            match (node(k), node(k + 1)) {
                (Some(a), Some(b)) => (a, b),
                // No S1: use K_{k-1} and K_k.
                _ => (
                    node(k - 1).unwrap_or_else(|| node(k).unwrap()),
                    node(k).unwrap(),
                ),
            }
        };
        if a.0 == b.0 {
            pos[..m].copy_from_slice(a.1);
            vel[..m].copy_from_slice(a.2);
            return;
        }
        for i in 0..m {
            let (p, v) =
                super::moon_radius::hermite_moon(t, a.0, b.0, a.1[i], a.2[i], b.1[i], b.2[i]);
            pos[i] = p;
            vel[i] = v;
        }
    }
}

/// Wisdom-Holman steps of one particle with the moons: as
/// [`wh_steps`](super::wh_steps), but the kick of step `k` also has the moon
/// terms from `table` (one step per entry of the table). Consecutive half
/// drifts are merged.
///
/// Returns `None` and leaves the state unchanged if an orbit is not bound.
pub fn wh_steps_moons<T: Real>(
    field: &SaturnField<T>,
    table: &KickTable<T>,
    pos_km: &mut Vec3<T>,
    vel_km_s: &mut Vec3<T>,
    h_s: T,
) -> Option<()> {
    let n_steps = table.len();
    if n_steps == 0 {
        return Some(());
    }
    let half = T::from_f64(0.5) * h_s;
    let (mut p, mut v) = kepler_drift(field.gm_km3_s2, *pos_km, *vel_km_s, half)?;
    for k in 0..n_steps {
        let a = kick_acceleration(
            field,
            table.gm_km3_s2(),
            table.moon_positions_km(k),
            table.indirect_km_s2(k),
            p,
        );
        v = Vec3::new(v.x + a.x * h_s, v.y + a.y * h_s, v.z + a.z * h_s);
        let dt = if k + 1 == n_steps { half } else { h_s };
        (p, v) = kepler_drift(field.gm_km3_s2, p, v, dt)?;
    }
    *pos_km = p;
    *vel_km_s = v;
    Some(())
}

#[cfg(test)]
mod tests {
    use super::super::{MoonInit, MoonSystem, build_ephemeris, wh_steps};
    use super::*;
    use crate::params::{moons_from_toml_str, saturn_from_toml_str};

    fn saturn_and_moons() -> (crate::SaturnParams, Vec<MoonInit>) {
        let s = saturn_from_toml_str(include_str!("../../../../params/saturn.toml")).unwrap();
        let m = moons_from_toml_str(include_str!("../../../../params/moons.toml")).unwrap();
        (s, m)
    }

    /// The kick of a particle equals the equation of motion of ADR 0004 for a
    /// moon with GM = 0 (a massless third body in `MoonSystem`).
    #[test]
    fn kick_matches_the_moon_system_for_a_massless_body() {
        let (saturn, moons) = saturn_and_moons();
        let gm = vec![moons[0].gm_km3_s2, moons[1].gm_km3_s2, 0.0];
        let sys = MoonSystem::new(&saturn, gm.clone());
        let p = Vec3::new(-49420.7, 131165.1, -1.94);
        let m0 = moons[0].state.position_km;
        let m1 = moons[1].state.position_km;
        let flat = [m0.x, m0.y, m0.z, m1.x, m1.y, m1.z, p.x, p.y, p.z];
        let mut acc = [0.0; 9];
        sys.accelerations(&flat, &mut acc);

        let pos = [m0, m1];
        let ind = indirect_acceleration(&gm[..2], &pos);
        let kick = kick_acceleration(&sys.field, &gm[..2], &pos, ind, p);
        let r = p.norm();
        let point = p * (-saturn.gm_km3_s2 / (r * r * r));
        let total = kick + point;
        let expected = Vec3::new(acc[6], acc[7], acc[8]);
        let err = (total - expected).norm() / expected.norm();
        assert!(err < 1e-14, "relative error {err:e}");
    }

    /// With no moons, `wh_steps_moons` is `wh_steps`, bit for bit.
    #[test]
    fn without_moons_it_is_the_plain_wisdom_holman_loop() {
        let (saturn, _) = saturn_and_moons();
        let field = SaturnField::<f64>::from_params(&saturn);
        let eph = MoonEphemeris::none();
        let mut table = KickTable::<f64>::for_ephemeris(&eph);
        // `fill` without moons needs no ephemeris range.
        table.fill(&eph, 0.0, 356.0, 0, 200).unwrap();
        assert_eq!(table.len(), 200);
        let p0 = Vec3::new(140222.43, 0.0, 0.0);
        let v0 = Vec3::new(0.0, super::super::circular_speed(&field, 140222.43), 0.1);
        let (mut p1, mut v1) = (p0, v0);
        let (mut p2, mut v2) = (p0, v0);
        wh_steps(&field, &mut p1, &mut v1, 356.0, 200).unwrap();
        wh_steps_moons(&field, &table, &mut p2, &mut v2, 356.0).unwrap();
        assert_eq!((p1, v1), (p2, v2));
    }

    #[test]
    fn fill_uses_the_middle_of_each_step_and_checks_the_range() {
        let (saturn, moons) = saturn_and_moons();
        let eph = build_ephemeris(&saturn, &moons, 20_000.0, 1800.0).unwrap();
        let mut table = KickTable::<f64>::for_ephemeris(&eph);
        let (start, h) = (100.0, 360.0);
        table.fill(&eph, start, h, 7, 5).unwrap();
        let mut st = vec![BodyState::default(); 2];
        interpolate_moons_into(&eph, start + 9.5 * h, &mut st).unwrap();
        assert_eq!(table.moon_positions_km(2)[1], st[1].position_km);
        let ind = indirect_acceleration(table.gm_km3_s2(), table.moon_positions_km(2));
        assert_eq!(table.indirect_km_s2(2), ind);
        // Past the end of the table: an error, not a wrong value.
        let r = table.fill(&eph, start, h, 1000, 5);
        assert!(matches!(r, Err(Error::Input(_))));
    }

    /// `fill_ext` gives the same rows as `fill`, plus velocities and sync rows;
    /// sub-kicks are the row itself for n = 1 and close to the ephemeris else.
    #[test]
    fn fill_ext_rows_and_subkicks() {
        let (saturn, moons) = saturn_and_moons();
        let h = 356.25;
        let eph = build_ephemeris(&saturn, &moons, 40.0 * h + 3600.0, 1800.0).unwrap();
        let mut a = KickTable::<f64>::for_ephemeris(&eph);
        let mut b = KickTable::<f64>::for_ephemeris(&eph);
        let (start, first, n) = (1800.0, 3u64, 10usize);
        a.fill(&eph, start, h, first, n).unwrap();
        b.fill_ext(&eph, start, h, first, n).unwrap();
        assert!(!a.has_ext() && b.has_ext());
        for k in 0..n {
            assert_eq!(a.moon_positions_km(k), b.moon_positions_km(k));
            assert_eq!(a.indirect_km_s2(k), b.indirect_km_s2(k));
        }
        assert!(b.sync_ok(0) && b.sync_ok(1));
        let mut st = vec![BodyState::default(); 2];
        interpolate_moons_into(&eph, start + (first + n as u64) as f64 * h, &mut st).unwrap();
        assert_eq!(b.sync_positions_km(1)[1], st[1].position_km);
        assert_eq!(b.moon_velocities_km_s(4)[0], {
            interpolate_moons_into(&eph, start + (first + 4) as f64 * h + 0.5 * h, &mut st)
                .unwrap();
            st[0].velocity_km_s
        });
        let (mut p, mut v) = (vec![Vec3::zero(); 2], vec![Vec3::zero(); 2]);
        b.moons_at_subkick(4, 0, 1, h, &mut p, &mut v);
        assert_eq!(p, b.moon_positions_km(4));
        // n = 4 sub-kicks of steps 0, 5 and 9 (edges of the block included).
        let mut worst: f64 = 0.0;
        for k in [0usize, 5, 9] {
            for j in 0..4u32 {
                b.moons_at_subkick(k, j, 4, h, &mut p, &mut v);
                let t = start + (first as f64 + k as f64 + (f64::from(j) + 0.5) / 4.0) * h;
                interpolate_moons_into(&eph, t, &mut st).unwrap();
                for m in 0..2 {
                    worst = worst.max((p[m] - st[m].position_km).norm());
                }
            }
        }
        assert!(worst < 5e-3, "sub-kick position error {worst} km");
        // Without a block-end row the last sub-kicks still work.
        let mut c = KickTable::<f64>::for_ephemeris(&eph);
        c.fill_ext(&eph, start, h, 0, 40).unwrap();
        let ok = c.sync_ok(1);
        assert!(ok);
    }
}
