//! The extended particle loop of ICR-004: lost particles (freeze), the
//! moon-radius rule and per-particle sub-stepping.
//!
//! [`advance_ext`] is used only when a key of `[loss]` or `[substep]` changes
//! the behaviour. Without them the backend uses [`crate::integrate::advance_chunk`],
//! which this module does not touch, so "off means off" holds by construction.
//!
//! One call advances one particle (and its chaos shadow, if any) through one
//! block. The result depends only on the particle, its stored level and the
//! shared [`KickTable`], so it does not depend on threads or chunks.

use crate::integrate::State;
use fring_model::physics::moon_radius::{MoonsAtKick, drift_hits_moon, particle_level};
use fring_model::physics::{
    KickTable, SaturnField, indirect_acceleration, kepler_drift, kick_acceleration,
};
use fring_model::{DROP_UNBOUND, Vec3, drop_inside_moon};

/// Value of the stored level when it is not known yet (start of the run).
pub const LEVEL_UNKNOWN: u8 = 255;

/// Most moons that the extended loop handles (reasons `2 + k`, k = 0 to 5).
pub const MAX_MOONS: usize = 6;

/// What a block of the extended loop needs besides the particle.
pub struct ExtParams<'a> {
    /// Saturn's field.
    pub field: &'a SaturnField<f64>,
    /// The block table, made by `KickTable::fill_ext`.
    pub table: &'a KickTable<f64>,
    /// Step, s.
    pub h_s: f64,
    /// Global 0-based number of step 0 of the block.
    pub first_step: u64,
    /// Moon radii, km, in `System` order (empty if neither key needs them).
    pub radii_km: &'a [f64],
    /// `moon_radius = "params"`.
    pub radius_rule: bool,
    /// `Some((accuracy, max_level))` with `[substep]`.
    pub substep: Option<(f64, u8)>,
}

/// One body (main particle or shadow) in the loop.
#[derive(Clone, Copy)]
struct Body {
    p: Vec3<f64>,
    v: Vec3<f64>,
}

fn body(s: &State) -> Body {
    Body {
        p: Vec3::new(s[0], s[1], s[2]),
        v: Vec3::new(s[3], s[4], s[5]),
    }
}

fn state(b: &Body) -> State {
    [b.p.x, b.p.y, b.p.z, b.v.x, b.v.y, b.v.z]
}

fn moons_at<'a>(
    prm: &'a ExtParams<'_>,
    gm: &'a [f64],
    pos_km: &'a [Vec3<f64>],
    vel_km_s: &'a [Vec3<f64>],
) -> MoonsAtKick<'a, f64> {
    MoonsAtKick {
        pos_km,
        vel_km_s,
        gm_km3_s2: gm,
        radius_km: prm.radii_km,
    }
}

/// True if the level rule gives 0 for every moon, decided with squares only
/// (no square root). It is a sufficient condition, so a `false` means
/// "compute the rule". For each moon, with `d` the distance and `u` the
/// relative speed, the conditions
///
/// ```text
/// d >= Dc,   4 u dt <= d,   h² u² <= acc² d² / 4,   Dc³ = 8.6 GM h² / acc²
/// ```
///
/// give `d_eff >= 3d/4` and `h² v_eff² <= h² u² + 8 GM h² / (3 d) <=
/// (1/4 + 5/16) acc² d² = 9/16 acc² d² <= acc² d_eff²`, which is the rule for
/// L = 0 (`8.6 > 8.53` is the margin of the second step). `far_d2` is `Dc²`
/// for the largest GM.
#[allow(clippy::too_many_arguments)]
fn level_zero_certain(
    h: f64,
    dt: f64,
    acc: f64,
    far_d2: f64,
    p: Vec3<f64>,
    w: Vec3<f64>,
    moon_pos: &[Vec3<f64>],
    moon_vel: &[Vec3<f64>],
) -> bool {
    let (h2, acc2, dt2) = (h * h, acc * acc, dt * dt);
    moon_pos.iter().zip(moon_vel).all(|(mp, mv)| {
        let d2 = (p - *mp).norm_squared();
        let u2 = (w - *mv).norm_squared();
        d2 >= far_d2 && 16.0 * u2 * dt2 <= d2 && 4.0 * h2 * u2 <= acc2 * d2
    })
}

/// Advances one particle (and `shadow`) through the block of `prm.table`.
///
/// A live particle ends the block at the synchronised time and gets the level
/// for the first step of the next block in `level`. A particle that is lost
/// keeps its state at the start of the block, gets `reason` (1 unbound,
/// `2 + k` inside moon `k`) and `drop_step` (the global step of the loss). A
/// particle with `reason != 0` is not touched.
///
/// A loss of the shadow (a Kepler drift that fails) counts as a loss of the
/// pair with reason 1. The shadow has no radius test and uses the levels of
/// its main particle.
pub fn advance_ext(
    prm: &ExtParams<'_>,
    s: &mut State,
    shadow: Option<&mut State>,
    reason: &mut u8,
    drop_step: &mut u64,
    level: &mut u8,
) {
    advance_ext_hooked(prm, s, shadow, reason, drop_step, level, LevelHook::Off);
}

/// A way to look at the levels of [`advance_ext_hooked`] (for tests).
pub enum LevelHook<'a> {
    /// Nothing; the rule chooses the levels.
    Off,
    /// Push the level of every step of the block to the vector.
    Record(&'a mut Vec<u8>),
    /// Use these levels instead of the rule: `levels[k]` for step `k` of the
    /// block (the entry for the first step of the next block is `levels[len]`
    /// if present, else 0).
    Replay(&'a [u8]),
}

/// [`advance_ext`] with a [`LevelHook`].
pub fn advance_ext_hooked(
    prm: &ExtParams<'_>,
    s: &mut State,
    shadow: Option<&mut State>,
    reason: &mut u8,
    drop_step: &mut u64,
    level: &mut u8,
    mut hook: LevelHook<'_>,
) {
    let table = prm.table;
    let n_steps = table.len();
    if *reason != 0 || n_steps == 0 {
        return;
    }
    let m = table.n_moons();
    debug_assert!(m <= MAX_MOONS && table.has_ext());
    let (h, gm_s) = (prm.h_s, prm.field.drift_gm_km3_s2);
    let gm = table.gm_km3_s2();
    let mut bodies = [body(s), shadow.as_deref().map_or(body(s), body)];
    let nb = 1 + usize::from(shadow.is_some());

    // Level of the first step.
    let mut lev: u8 = match prm.substep {
        _ if matches!(hook, LevelHook::Replay(_)) => match &hook {
            LevelHook::Replay(r) => r.first().copied().unwrap_or(0),
            _ => 0,
        },
        None => 0,
        Some((acc, max)) => {
            if *level != LEVEL_UNKNOWN {
                *level
            } else if table.sync_ok(0) {
                let at = moons_at(
                    prm,
                    gm,
                    table.sync_positions_km(0),
                    table.sync_velocities_km_s(0),
                );
                particle_level(h, h, bodies[0].p, bodies[0].v, &at, acc, max, None)
            } else {
                0
            }
        }
    };

    // Position relative to each moon at the start of the drift, for the radius rule.
    let mut prev = [Vec3::zero(); MAX_MOONS];
    let mut prev_ok = false;
    if prm.radius_rule && table.sync_ok(0) {
        for (j, mp) in table.sync_positions_km(0).iter().enumerate() {
            prev[j] = bodies[0].p - *mp;
        }
        prev_ok = true;
    }

    // Squared distance beyond which a particle is far from every moon (see
    // `level_zero_certain`).
    let far_d2 = match prm.substep {
        Some((acc, _)) => {
            let gm_max = gm.iter().cloned().fold(0.0f64, f64::max);
            (8.6 * gm_max * h * h / (acc * acc)).cbrt().powi(2)
        }
        None => 0.0,
    };
    let mut mpos = [Vec3::zero(); MAX_MOONS];
    let mut mvel = [Vec3::zero(); MAX_MOONS];
    let mut dt = h / f64::from(1u32 << lev) * 0.5;
    let mut attr = 0usize; // step to which the next drift is attributed
    let mut next_level = lev;

    // A drift of length `dt`, then the radius test with the moons `end`.
    // Returns the reason of a loss, or 0.
    let drift = |bodies: &mut [Body; 2],
                 dt: f64,
                 end: Option<(&[Vec3<f64>], bool)>,
                 prev: &mut [Vec3<f64>; MAX_MOONS],
                 prev_ok: &mut bool|
     -> u8 {
        for b in bodies.iter_mut().take(nb) {
            match kepler_drift(gm_s, b.p, b.v, dt) {
                Some((p, v)) => (b.p, b.v) = (p, v),
                None => return DROP_UNBOUND,
            }
        }
        if !prm.radius_rule {
            return 0;
        }
        let Some((end, true)) = end else {
            *prev_ok = false;
            return 0;
        };
        let mut hit = 0u8;
        for (j, mp) in end.iter().enumerate().take(m) {
            let rel = bodies[0].p - *mp;
            if *prev_ok
                && hit == 0
                && drift_hits_moon(prev[j].to_array(), rel.to_array(), prm.radii_km[j])
            {
                hit = drop_inside_moon(j);
            }
            prev[j] = rel;
        }
        *prev_ok = true;
        hit
    };

    let lost = |r: u8, attr: usize, reason: &mut u8, drop_step: &mut u64| {
        *reason = r;
        *drop_step = prm.first_step + attr as u64;
    };

    for k in 0..n_steps {
        let n_sub = 1u32 << lev;
        let hs = h / f64::from(n_sub);
        if let LevelHook::Record(v) = &mut hook {
            v.push(lev);
        }
        for j in 0..n_sub {
            let (mp, mv): (&[Vec3<f64>], &[Vec3<f64>]) = if n_sub == 1 {
                (table.moon_positions_km(k), table.moon_velocities_km_s(k))
            } else {
                table.moons_at_subkick(k, j, n_sub, h, &mut mpos, &mut mvel);
                (&mpos[..m], &mvel[..m])
            };
            let r = drift(&mut bodies, dt, Some((mp, true)), &mut prev, &mut prev_ok);
            if r != 0 {
                lost(r, attr, reason, drop_step);
                return;
            }
            let ind = if n_sub == 1 {
                table.indirect_km_s2(k)
            } else {
                indirect_acceleration(gm, mp)
            };
            for b in bodies.iter_mut().take(nb) {
                let a = kick_acceleration(prm.field, gm, mp, ind, b.p);
                b.v = Vec3::new(b.v.x + a.x * hs, b.v.y + a.y * hs, b.v.z + a.z * hs);
            }
            attr = k;
            if j + 1 < n_sub {
                dt = hs;
                continue;
            }
            // Last kick of step k: choose the level of step k + 1.
            next_level = match prm.substep {
                _ if matches!(hook, LevelHook::Replay(_)) => match &hook {
                    LevelHook::Replay(r) => r.get(k + 1).copied().unwrap_or(0),
                    _ => 0,
                },
                None => 0,
                Some((acc, max)) => {
                    let dt_next = h + hs * 0.5;
                    if level_zero_certain(h, dt_next, acc, far_d2, bodies[0].p, bodies[0].v, mp, mv)
                    {
                        0
                    } else {
                        let at = moons_at(prm, gm, mp, mv);
                        particle_level(h, dt_next, bodies[0].p, bodies[0].v, &at, acc, max, None)
                    }
                }
            };
            dt = if k + 1 == n_steps {
                hs * 0.5
            } else if lev == 0 && next_level == 0 {
                h
            } else {
                hs * 0.5 + h / f64::from(1u32 << next_level) * 0.5
            };
            lev = next_level;
        }
    }
    // The half drift to the end of the block (the loop left `dt` for it).
    let end = (table.sync_positions_km(1), table.sync_ok(1));
    let r = drift(&mut bodies, dt, Some(end), &mut prev, &mut prev_ok);
    if r != 0 {
        lost(r, attr, reason, drop_step);
        return;
    }
    *s = state(&bodies[0]);
    if let Some(sh) = shadow {
        *sh = state(&bodies[1]);
    }
    *level = next_level;
}

#[cfg(test)]
mod tests {
    use super::*;
    use fring_model::physics::moon_radius::substep_level;

    /// The shortcut never says "level 0" where the rule gives more.
    #[test]
    fn level_zero_shortcut_is_sound() {
        let (h, acc, gm): (f64, f64, f64) = (356.25, 0.1, 0.01071);
        let far_d2 = (8.6 * gm * h * h / (acc * acc)).cbrt().powi(2);
        let mut x = 0x9e37_79b9_7f4a_7c15u64;
        let mut u = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 11) as f64 / (1u64 << 53) as f64
        };
        let (mut certain, mut total) = (0, 0);
        for _ in 0..2_000_000 {
            let d = 10f64.powf(1.0 + 3.0 * u()); // 10 to 10,000 km
            let sp = 10f64.powf(-3.0 + 2.5 * u()); // 0.001 to 0.3 km/s
            let (p, w) = (Vec3::new(d, 0.0, 0.0), Vec3::new(0.0, sp, 0.0));
            let zero = Vec3::zero();
            let dt = 1.5 * h;
            let fast = level_zero_certain(h, dt, acc, far_d2, p, w, &[zero], &[zero]);
            let rule = substep_level(h, d, sp, dt, 43.09, gm, acc, 4);
            total += 1;
            if fast {
                certain += 1;
                assert_eq!(rule, 0, "d {d} u {sp}");
            }
        }
        assert!(certain > total / 10, "the shortcut is never used");
    }
}
