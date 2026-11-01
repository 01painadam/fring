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
}

impl<T: Real> KickTable<T> {
    /// An empty table for moons with these GM values (km³ s⁻²).
    pub fn new(gm_km3_s2: Vec<T>) -> Self {
        Self {
            gm_km3_s2,
            pos_km: Vec::new(),
            indirect_km_s2: Vec::new(),
            n_steps: 0,
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
        }
        self.n_steps = n_steps;
        Ok(())
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
}
