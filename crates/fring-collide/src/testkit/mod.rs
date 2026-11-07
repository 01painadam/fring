//! Shared statistical tests of the collision models (`SPEC.md` section 11).
//!
//! A model implements [`BoxModel`] for a periodic cube with no forces. The
//! drivers [`t4_rate`], [`t5_energy_loss`], [`t6_budget`], [`t7_haff`] and
//! [`t9_determinism`] build the box, run the model and compare the result with
//! kinetic theory. The models of K2, K3 and K5 call the same drivers, so all
//! models are measured with the same setups and numbers.
//!
//! [`reference`] has two slow models that follow `SPEC.md` directly. They pass
//! these tests, so a failure of a new model points to the model, not the test.
//!
//! In a box, positions are the local coordinates of the spec and velocities are
//! the peculiar velocities.

pub mod reference;

use std::f64::consts::PI;

use fring_model::Vec3;
use fring_model::config::Restitution;
use fring_model::rng::{Counter, Philox, Stream, box_muller, u01_f64};

use crate::pair;

/// One-dimensional velocity dispersion c of each box setup at the start, in
/// km/s (1 cm/s).
pub const DISPERSION_KM_S: f64 = 1.0e-5;
/// Seed of the box positions and velocities. Models get their own seed.
pub const SETUP_SEED: u64 = 0x4B31_5345_5455_5000;
/// Tolerance of the collision rate (PLAN section 10).
pub const TOL_RATE: f64 = 0.02;
/// Tolerance of the mean energy loss per collision.
pub const TOL_ENERGY_LOSS: f64 = 0.02;
/// Tolerance of the temperature at t = t₀ in the cooling test.
pub const TOL_HAFF: f64 = 0.03;

/// Cell edge of the stochastic setups in km.
const STOCHASTIC_CELL_KM: f64 = 1.0;
/// Step of the stochastic setups at the start, in s: c dt = 0.1 cell.
const STOCHASTIC_DT_S: f64 = 1.0e4;
/// Particle radius of the stochastic setups in km (only W σ matters).
const STOCHASTIC_RADIUS_KM: f64 = 1.0e-6;
/// Particle radius of the hard-sphere setups in km.
const HARD_SPHERE_RADIUS_KM: f64 = 1.0e-3;
/// Hard-sphere step: ⟨g⟩ dt = this fraction of the radius s (so 0.05 × 2s).
const HARD_SPHERE_STEP_FRACTION: f64 = 0.1;
/// Upper limit of steps or trials in one driver.
const MAX_ITERATIONS: u64 = 2_000_000;

/// Parameters of a model in a box test.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModelParams {
    /// Radius s of one real particle in km.
    pub particle_radius_km: f64,
    /// Real particles per superparticle W (1 for hard spheres).
    pub weight: f64,
    /// Restitution law.
    pub restitution: Restitution,
    /// Seed of the model's Philox stream.
    pub seed: u64,
}

impl ModelParams {
    /// σ = π (2s)² in km².
    pub fn cross_section_km2(&self) -> f64 {
        PI * (2.0 * self.particle_radius_km).powi(2)
    }
}

pub use crate::tally::Tally;

/// A periodic cube divided into cubic cells.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoxSpec {
    /// Edge of the cube in km.
    pub side_km: f64,
    /// Cells along each edge (stochastic models use them as their cells).
    pub cells_per_side: u32,
}

impl BoxSpec {
    /// Edge of one cell in km.
    pub fn cell_km(&self) -> f64 {
        self.side_km / f64::from(self.cells_per_side)
    }

    /// Volume of the cube in km³.
    pub fn volume_km3(&self) -> f64 {
        self.side_km.powi(3)
    }

    /// Volume of one cell in km³.
    pub fn cell_volume_km3(&self) -> f64 {
        self.cell_km().powi(3)
    }

    /// Number of cells.
    pub fn cell_count(&self) -> usize {
        (self.cells_per_side as usize).pow(3)
    }

    /// Flat index of the cell that holds `p` (a position in [0, side)³).
    pub fn cell_of(&self, p: Vec3<f64>) -> usize {
        let k = self.cells_per_side as i64;
        let cell = self.cell_km();
        let f = |x: f64| ((x / cell).floor() as i64).clamp(0, k - 1) as usize;
        let k = k as usize;
        (f(p.x) * k + f(p.y)) * k + f(p.z)
    }

    /// `p` wrapped into [0, side)³.
    pub fn wrap(&self, p: Vec3<f64>) -> Vec3<f64> {
        let s = self.side_km;
        let w = |x: f64| {
            let y = x.rem_euclid(s);
            if y >= s { 0.0 } else { y }
        };
        Vec3::new(w(p.x), w(p.y), w(p.z))
    }

    /// Shortest periodic image of a position difference.
    pub fn min_image(&self, d: Vec3<f64>) -> Vec3<f64> {
        let s = self.side_km;
        let m = |x: f64| x - s * (x / s).round();
        Vec3::new(m(d.x), m(d.y), m(d.z))
    }
}

/// A collision model that can run in the box.
pub trait BoxModel {
    /// One collision step of length `dt_s` at the end of integrator step
    /// `step`. `pos_km` are the positions after streaming; the model changes
    /// `vel_km_s` and does not change the positions.
    fn collide(
        &mut self,
        cells: &BoxSpec,
        pos_km: &[Vec3<f64>],
        vel_km_s: &mut [Vec3<f64>],
        step: u64,
        dt_s: f64,
    ) -> Tally;
}

/// Which theory a driver uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelKind {
    /// Superparticles in cells; rate from the pairs in each cell.
    Stochastic,
    /// Real hard spheres; rate with the Enskog factor χ.
    HardSphere,
}

/// How large a sample a driver takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sample {
    /// The definitive test: SE ≤ tolerance / 5, and the error within the
    /// tolerance (`SPEC.md` section 11, sample-size rule). Use `--release`.
    Full,
    /// A quick check for every `cargo test`: SE ≤ tolerance / 2, and the error
    /// within 5 SE. It finds large mistakes only.
    Quick,
}

impl Sample {
    fn target_se(self, tolerance: f64) -> f64 {
        match self {
            Sample::Full => tolerance / 5.0,
            Sample::Quick => tolerance / 2.0,
        }
    }
}

/// The result of one statistical test.
#[derive(Clone, Debug, PartialEq)]
pub struct Check {
    /// Test name, for example "K1-T4 rate".
    pub name: String,
    /// Measured value.
    pub measured: f64,
    /// Theory value.
    pub theory: f64,
    /// Standard error of the measured value.
    pub standard_error: f64,
    /// Number of collisions in the sample.
    pub samples: u64,
    /// Relative tolerance.
    pub tolerance: f64,
    /// Sample size class.
    pub sample: Sample,
}

impl Check {
    /// Relative error (measured − theory) / theory.
    pub fn relative_error(&self) -> f64 {
        (self.measured - self.theory) / self.theory
    }

    /// True if the sample is large enough and the error is inside the bound.
    pub fn passed(&self) -> bool {
        let scale = self.theory.abs();
        let se_ok = self.standard_error <= self.sample.target_se(self.tolerance) * scale * 1.0001;
        let bound = match self.sample {
            Sample::Full => self.tolerance * scale,
            Sample::Quick => 5.0 * self.standard_error,
        };
        se_ok && (self.measured - self.theory).abs() <= bound
    }

    /// Prints the result and panics if [`Check::passed`] is false.
    pub fn assert_ok(&self) {
        println!("{self}");
        assert!(self.passed(), "failed: {self}");
    }
}

impl std::fmt::Display for Check {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.theory == 0.0 {
            return write!(
                f,
                "{} ({:?}): measured {:e}, theory 0 (must be exact), collisions {}",
                self.name, self.sample, self.measured, self.samples
            );
        }
        write!(
            f,
            "{} ({:?}): measured {:.6e}, theory {:.6e}, error {:+.3} %, SE {:.3} %, \
             tolerance {:.1} %, collisions {}",
            self.name,
            self.sample,
            self.measured,
            self.theory,
            100.0 * self.relative_error(),
            100.0 * self.standard_error / self.theory.abs(),
            100.0 * self.tolerance,
            self.samples
        )
    }
}

// ---------------------------------------------------------------- theory --

/// Mean relative speed ⟨g⟩ = 4c/√π of two Maxwellian particles with
/// one-dimensional dispersion `c_km_s`.
pub fn mean_relative_speed(c_km_s: f64) -> f64 {
    4.0 * c_km_s / PI.sqrt()
}

/// Enskog factor χ = (1 − D/2) / (1 − D)³ (Carnahan and Starling 1969) for
/// the volume filling factor D.
pub fn enskog_chi(filling: f64) -> f64 {
    (1.0 - 0.5 * filling) / (1.0 - filling).powi(3)
}

/// Mean energy loss per collision, per unit mass of one particle, in km² s⁻²,
/// for Maxwellian particles with dispersion `c_km_s` (`SPEC.md` appendix A).
///
/// Constant ε: (1 − ε²) c². Otherwise a quadrature over the collision-weighted
/// relative speed g and μ² = cos² θ (uniform).
pub fn mean_energy_loss_per_collision(law: &Restitution, c_km_s: f64) -> f64 {
    if let Restitution::Constant { epsilon } = *law {
        return (1.0 - epsilon * epsilon) * c_km_s * c_km_s;
    }
    // g has the Maxwell distribution with σ_g = √2 c; collisions weight it by g,
    // so its density is g³ exp(−g²/2σ_g²) / (2 σ_g⁴). Midpoint rule.
    let sg = 2.0_f64.sqrt() * c_km_s;
    let (ng, nm) = (4000, 2000);
    let g_max = 12.0 * sg;
    let dg = g_max / ng as f64;
    let dm = 1.0 / nm as f64;
    let mut sum = 0.0;
    for a in 0..ng {
        let g = (a as f64 + 0.5) * dg;
        let weight = g.powi(3) * (-0.5 * g * g / (sg * sg)).exp() / (2.0 * sg.powi(4)) * dg;
        let mut inner = 0.0;
        for b in 0..nm {
            let mu2 = (b as f64 + 0.5) * dm;
            let vn = g * mu2.sqrt();
            let eps = pair::epsilon(law, vn);
            inner += 0.25 * (1.0 - eps * eps) * vn * vn * dm;
        }
        sum += weight * inner;
    }
    sum
}

/// Σ over cells of N_c (N_c − 1) for the positions `pos_km`.
pub fn pair_sum(cells: &BoxSpec, pos_km: &[Vec3<f64>]) -> f64 {
    let mut count = vec![0_u64; cells.cell_count()];
    for &p in pos_km {
        count[cells.cell_of(p)] += 1;
    }
    count
        .iter()
        .map(|&n| (n * n.saturating_sub(1)) as f64)
        .sum()
}

/// One-dimensional dispersion √(Σ |u − ū|² / 3N) in km/s.
pub fn dispersion(vel_km_s: &[Vec3<f64>]) -> f64 {
    let n = vel_km_s.len() as f64;
    let mean = momentum(vel_km_s) / n;
    let s: f64 = vel_km_s.iter().map(|&u| (u - mean).norm_squared()).sum();
    (s / (3.0 * n)).sqrt()
}

/// Σ u in km/s (momentum per unit mass of one particle).
pub fn momentum(vel_km_s: &[Vec3<f64>]) -> Vec3<f64> {
    vel_km_s.iter().fold(Vec3::zero(), |a, &u| a + u)
}

/// ½ Σ |u|² in km² s⁻² (kinetic energy per unit mass of one particle).
pub fn kinetic_energy(vel_km_s: &[Vec3<f64>]) -> f64 {
    0.5 * vel_km_s.iter().map(|&u| u.norm_squared()).sum::<f64>()
}

// ----------------------------------------------------------------- setups --

/// A box, its particles and the step of one driver.
#[derive(Clone, Debug)]
pub struct Setup {
    /// Theory to use.
    pub kind: ModelKind,
    /// The box.
    pub cells: BoxSpec,
    /// Number of particles.
    pub n: usize,
    /// Model parameters.
    pub params: ModelParams,
    /// Step at the start in s.
    pub dt_s: f64,
    /// Volume filling factor D (hard spheres; 0 for stochastic setups).
    pub filling: f64,
}

/// Particle positions and velocities of a box.
#[derive(Clone, Debug, PartialEq)]
pub struct BoxState {
    /// Positions in [0, side)³, km.
    pub pos_km: Vec<Vec3<f64>>,
    /// Velocities, km/s.
    pub vel_km_s: Vec<Vec3<f64>>,
}

impl Setup {
    /// Stochastic setup: `cells_per_side`³ cells of 1 km, mean occupancy
    /// `occupancy`, and W such that a particle collides with probability
    /// `p_step` per step at the start.
    pub fn stochastic(
        cells_per_side: u32,
        occupancy: f64,
        p_step: f64,
        law: Restitution,
        seed: u64,
    ) -> Self {
        let cells = BoxSpec {
            side_km: STOCHASTIC_CELL_KM * f64::from(cells_per_side),
            cells_per_side,
        };
        let n = (cells.cell_count() as f64 * occupancy).round() as usize;
        let w_sigma = p_step * cells.cell_volume_km3()
            / (occupancy * mean_relative_speed(DISPERSION_KM_S) * STOCHASTIC_DT_S);
        let sigma = PI * (2.0 * STOCHASTIC_RADIUS_KM).powi(2);
        Self {
            kind: ModelKind::Stochastic,
            cells,
            n,
            params: ModelParams {
                particle_radius_km: STOCHASTIC_RADIUS_KM,
                weight: w_sigma / sigma,
                restitution: law,
                seed,
            },
            dt_s: STOCHASTIC_DT_S,
            filling: 0.0,
        }
    }

    /// Hard-sphere setup: `n` spheres of radius 1 m at filling factor
    /// `filling`, step ⟨g⟩ dt = 0.1 s.
    pub fn hard_sphere(n: usize, filling: f64, law: Restitution, seed: u64) -> Self {
        let s = HARD_SPHERE_RADIUS_KM;
        let volume = n as f64 * 4.0 / 3.0 * PI * s.powi(3) / filling;
        Self {
            kind: ModelKind::HardSphere,
            cells: BoxSpec {
                side_km: volume.cbrt(),
                cells_per_side: 1,
            },
            n,
            params: ModelParams {
                particle_radius_km: s,
                weight: 1.0,
                restitution: law,
                seed,
            },
            dt_s: HARD_SPHERE_STEP_FRACTION * s / mean_relative_speed(DISPERSION_KM_S),
            filling,
        }
    }

    /// Fresh positions (uniform; without overlaps for hard spheres) and
    /// Maxwellian velocities with dispersion [`DISPERSION_KM_S`], from the
    /// setup seed and `trial`.
    pub fn initial_state(&self, trial: u64) -> BoxState {
        let rng = Philox::from_seed(SETUP_SEED);
        let block = |i: usize, draw: u16| {
            rng.block(Counter {
                index: i as u32,
                step: trial,
                stream: Stream::TEST,
                draw,
            })
        };
        let vel_km_s = (0..self.n)
            .map(|i| {
                let a = block(i, 0);
                let b = block(i, 1);
                let (x, y) = box_muller(u01_f64(a[0], a[1]), u01_f64(a[2], a[3]));
                let (z, _) = box_muller(u01_f64(b[0], b[1]), u01_f64(b[2], b[3]));
                Vec3::new(x, y, z) * DISPERSION_KM_S
            })
            .collect();
        let side = self.cells.side_km;
        let point = |i: usize, attempt: u16| {
            let a = block(i, 2 + 2 * attempt);
            let b = block(i, 3 + 2 * attempt);
            Vec3::new(
                u01_f64(a[0], a[1]),
                u01_f64(a[2], a[3]),
                u01_f64(b[0], b[1]),
            ) * side
        };
        let pos_km = match self.kind {
            ModelKind::Stochastic => (0..self.n).map(|i| point(i, 0)).collect(),
            ModelKind::HardSphere => {
                let contact = 2.0 * self.params.particle_radius_km;
                let mut grid = reference::Grid::new(&self.cells, contact, self.n);
                let mut pos: Vec<Vec3<f64>> = Vec::with_capacity(self.n);
                for i in 0..self.n {
                    let mut placed = false;
                    for attempt in 0..1000_u16 {
                        let p = point(i, attempt);
                        let mut clear = true;
                        grid.for_neighbours(p, |j| {
                            let d = self.cells.min_image(p - pos[j as usize]);
                            if d.norm_squared() < contact * contact {
                                clear = false;
                            }
                        });
                        if clear {
                            grid.insert(i as u32, p);
                            pos.push(p);
                            placed = true;
                            break;
                        }
                    }
                    assert!(
                        placed,
                        "no free place for particle {i}: filling factor too high"
                    );
                }
                pos
            }
        };
        BoxState { pos_km, vel_km_s }
    }

    /// Moves the positions by u dt and wraps them.
    pub fn stream(&self, state: &mut BoxState, dt_s: f64) {
        for (p, &u) in state.pos_km.iter_mut().zip(&state.vel_km_s) {
            *p = self.cells.wrap(*p + u * dt_s);
        }
    }

    /// B = ω / c, the collision frequency of a particle per unit dispersion, in
    /// km⁻¹, for the current positions (`SPEC.md` appendix A).
    fn frequency_per_dispersion(&self, pos_km: &[Vec3<f64>]) -> f64 {
        let w_sigma = self.params.weight * self.params.cross_section_km2();
        let n = self.n as f64;
        let k = 4.0 / PI.sqrt();
        match self.kind {
            ModelKind::Stochastic => {
                w_sigma * k * pair_sum(&self.cells, pos_km) / (n * self.cells.cell_volume_km3())
            }
            ModelKind::HardSphere => {
                enskog_chi(self.filling) * w_sigma * k * (n - 1.0) / self.cells.volume_km3()
            }
        }
    }

    /// B of a uniform box: W σ (4/√π) (N − 1) / V, times χ for hard spheres.
    fn uniform_frequency_per_dispersion(&self) -> f64 {
        let chi = match self.kind {
            ModelKind::Stochastic => 1.0,
            ModelKind::HardSphere => enskog_chi(self.filling),
        };
        chi * self.params.weight * self.params.cross_section_km2() * 4.0 / PI.sqrt()
            * (self.n as f64 - 1.0)
            / self.cells.volume_km3()
    }

    /// Expected collision events in one step of `dt_s` for the dispersion
    /// `c_km_s`: ½ N ω dt.
    fn expected_collisions(&self, pos_km: &[Vec3<f64>], c_km_s: f64, dt_s: f64) -> f64 {
        0.5 * self.n as f64 * self.frequency_per_dispersion(pos_km) * c_km_s * dt_s
    }
}

// ---------------------------------------------------------------- drivers --

/// K1-T4: collision rate in a box with ε = 1 against kinetic theory.
///
/// Stochastic: 8³ cells with mean occupancy `occupancy`; the theory uses the
/// pairs of each cell at each step. Hard spheres: 8,000 spheres at D = 0.005
/// with the Enskog factor (`occupancy` is not used).
pub fn t4_rate<M: BoxModel>(
    kind: ModelKind,
    occupancy: f64,
    sample: Sample,
    mut make: impl FnMut(ModelParams) -> M,
) -> Check {
    let law = Restitution::Constant { epsilon: 1.0 };
    let setup = match kind {
        ModelKind::Stochastic => Setup::stochastic(8, occupancy, 0.02, law, 41),
        ModelKind::HardSphere => Setup::hard_sphere(8000, 0.005, law, 41),
    };
    let mut state = setup.initial_state(0);
    let c = dispersion(&state.vel_km_s);
    let target = sample.target_se(TOL_RATE);
    let mut model = make(setup.params);
    let (mut measured, mut expected) = (0_u64, 0.0);
    let mut step = 0;
    // Poisson count: relative SE = 1 / √expected.
    while expected < 1.0 / (target * target) && step < MAX_ITERATIONS {
        step += 1;
        setup.stream(&mut state, setup.dt_s);
        expected += setup.expected_collisions(&state.pos_km, c, setup.dt_s);
        let t = model.collide(
            &setup.cells,
            &state.pos_km,
            &mut state.vel_km_s,
            step,
            setup.dt_s,
        );
        assert_eq!(t.stats.overfull_cells, 0, "overfull cells in the rate test");
        measured += t.collisions;
    }
    Check {
        name: format!("K1-T4 rate {kind:?} occupancy {occupancy}"),
        measured: measured as f64,
        theory: expected,
        standard_error: expected.sqrt(),
        samples: measured,
        tolerance: TOL_RATE,
        sample,
    }
}

/// K1-T5: mean energy loss per collision against theory.
///
/// Each trial takes fresh positions and Maxwellian velocities and does one
/// step in which at most about 1 % of the particles collide, so repeated
/// collisions in a step are rare. For ε = 1 the loss must be exactly 0.
pub fn t5_energy_loss<M: BoxModel>(
    kind: ModelKind,
    law: Restitution,
    sample: Sample,
    mut make: impl FnMut(ModelParams) -> M,
) -> Check {
    let setup = match kind {
        ModelKind::Stochastic => Setup::stochastic(8, 5.0, 0.01, law, 51),
        ModelKind::HardSphere => Setup::hard_sphere(8000, 0.05, law, 51),
    };
    let theory = mean_energy_loss_per_collision(&law, DISPERSION_KM_S);
    let mut model = make(setup.params);
    let mut tally = Tally::default();
    let target = sample.target_se(TOL_ENERGY_LOSS);
    let mut trial = 0;
    loop {
        let mut state = setup.initial_state(trial);
        setup.stream(&mut state, setup.dt_s);
        tally += model.collide(
            &setup.cells,
            &state.pos_km,
            &mut state.vel_km_s,
            trial,
            setup.dt_s,
        );
        trial += 1;
        let n = tally.collisions as f64;
        if n >= 2.0 {
            let mean = tally.stats.energy_loss_km2_s2 / n;
            let var = (tally.energy_loss_sq_km4_s4 / n - mean * mean).max(0.0);
            let se = (var / n).sqrt();
            let enough = if theory == 0.0 {
                tally.collisions >= 2000
            } else {
                se <= target * theory
            };
            if enough || trial >= MAX_ITERATIONS {
                if theory == 0.0 {
                    assert_eq!(
                        tally.stats.energy_loss_km2_s2, 0.0,
                        "ε = 1 must lose no energy"
                    );
                }
                return Check {
                    name: format!("K1-T5 energy loss {kind:?} {law:?}"),
                    measured: mean,
                    theory,
                    // For ε = 1 the result is exact; give a tiny SE so that the check passes.
                    standard_error: if theory == 0.0 { 0.0 } else { se },
                    samples: tally.collisions,
                    tolerance: TOL_ENERGY_LOSS,
                    sample,
                };
            }
        }
    }
}

/// Result of K1-T6.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Budget {
    /// Collisions in the run.
    pub collisions: u64,
    /// Largest change of a component of the total momentum, km/s.
    pub momentum_error_km_s: f64,
    /// Allowed: 4 ε_mach × max |u| × collisions.
    pub momentum_bound_km_s: f64,
    /// |K₀ − K_end − Σ loss| / K₀ (inelastic) or |K_end − K₀| / K₀ (ε = 1).
    pub energy_error_rel: f64,
    /// Allowed: 10⁻¹⁰ (inelastic) or 10⁻¹² (ε = 1).
    pub energy_bound_rel: f64,
}

impl Budget {
    /// Prints the result and panics if a bound is not met.
    pub fn assert_ok(&self) {
        println!("K1-T6 budget: {self:?}");
        assert!(self.collisions > 0, "no collisions");
        assert!(
            self.momentum_error_km_s <= self.momentum_bound_km_s,
            "momentum: {self:?}"
        );
        assert!(
            self.energy_error_rel <= self.energy_bound_rel,
            "energy: {self:?}"
        );
    }
}

/// K1-T6: momentum and energy over 1,000 steps with constant `epsilon`.
pub fn t6_budget<M: BoxModel>(
    kind: ModelKind,
    epsilon: f64,
    mut make: impl FnMut(ModelParams) -> M,
) -> Budget {
    let law = Restitution::Constant { epsilon };
    let setup = match kind {
        ModelKind::Stochastic => Setup::stochastic(8, 5.0, 0.02, law, 61),
        ModelKind::HardSphere => Setup::hard_sphere(8000, 0.005, law, 61),
    };
    let mut state = setup.initial_state(0);
    let mut model = make(setup.params);
    let p0 = momentum(&state.vel_km_s);
    let k0 = kinetic_energy(&state.vel_km_s);
    let max_speed = |v: &[Vec3<f64>]| v.iter().map(|u| u.norm()).fold(0.0, f64::max);
    let mut umax = max_speed(&state.vel_km_s);
    let mut tally = Tally::default();
    for step in 1..=1000 {
        setup.stream(&mut state, setup.dt_s);
        tally += model.collide(
            &setup.cells,
            &state.pos_km,
            &mut state.vel_km_s,
            step,
            setup.dt_s,
        );
        umax = umax.max(max_speed(&state.vel_km_s));
    }
    let dp = momentum(&state.vel_km_s) - p0;
    let k1 = kinetic_energy(&state.vel_km_s);
    let elastic = epsilon == 1.0;
    Budget {
        collisions: tally.collisions,
        momentum_error_km_s: dp.x.abs().max(dp.y.abs()).max(dp.z.abs()),
        momentum_bound_km_s: 4.0 * f64::EPSILON * umax * tally.collisions as f64,
        energy_error_rel: if elastic {
            (k1 - k0).abs() / k0
        } else {
            (k0 - k1 - tally.stats.energy_loss_km2_s2).abs() / k0
        },
        energy_bound_rel: if elastic { 1e-12 } else { 1e-10 },
    }
}

/// K1-T7: homogeneous cooling with ε = 0.9 against Haff's law, at t = t₀.
///
/// The step grows as the gas cools (dt ∝ 1/c), so that each step has about the
/// same number of collisions. The theory integrates Haff's equation step by
/// step with the same steps (stochastic: with the pairs of each cell).
pub fn t7_haff<M: BoxModel>(
    kind: ModelKind,
    sample: Sample,
    mut make: impl FnMut(ModelParams) -> M,
) -> Check {
    let epsilon = 0.9;
    let law = Restitution::Constant { epsilon };
    let setup = match (kind, sample) {
        // Full: about 7.2 × 10⁴ particles, so SE ≈ 0.51 % (limit 0.6 %).
        (ModelKind::Stochastic, Sample::Full) => Setup::stochastic(33, 2.0, 0.02, law, 71),
        (ModelKind::Stochastic, Sample::Quick) => Setup::stochastic(16, 2.5, 0.02, law, 71),
        (ModelKind::HardSphere, Sample::Full) => Setup::hard_sphere(72_000, 0.01, law, 71),
        (ModelKind::HardSphere, Sample::Quick) => Setup::hard_sphere(9000, 0.01, law, 71),
    };
    let mut state = setup.initial_state(0);
    let mut model = make(setup.params);
    let c0 = dispersion(&state.vel_km_s);
    let k0 = 1.5 * setup.n as f64 * c0 * c0;
    let loss_factor = 1.0 - epsilon * epsilon;
    // t₀ = 6 / ((1 − ε²) ω₀), with ω₀ of a uniform box (it only sets the end time).
    let t0 = 6.0 / (loss_factor * setup.uniform_frequency_per_dispersion() * c0);
    let mut c_theory = c0;
    let (mut t, mut step) = (0.0, 0);
    let mut tally = Tally::default();
    while t < t0 && step < MAX_ITERATIONS {
        step += 1;
        let dt = setup.dt_s * c0 / c_theory;
        setup.stream(&mut state, dt);
        let b = setup.frequency_per_dispersion(&state.pos_km);
        c_theory /= 1.0 + loss_factor * b * c_theory * dt / 6.0;
        tally += model.collide(&setup.cells, &state.pos_km, &mut state.vel_km_s, step, dt);
        t += dt;
    }
    let c = dispersion(&state.vel_km_s);
    Check {
        name: format!("K1-T7 Haff {kind:?} T/T0 at t0"),
        measured: (c / c0).powi(2),
        theory: (c_theory / c0).powi(2),
        standard_error: tally.energy_loss_sq_km4_s4.sqrt() / k0,
        samples: tally.collisions,
        tolerance: TOL_HAFF,
        sample,
    }
}

/// K1-T9: the same seed gives the same velocities, bit for bit. For a
/// stochastic model, another seed gives other velocities. Panics on failure.
pub fn t9_determinism<M: BoxModel>(kind: ModelKind, mut make: impl FnMut(ModelParams) -> M) {
    let law = Restitution::Constant { epsilon: 0.5 };
    let setup = match kind {
        ModelKind::Stochastic => Setup::stochastic(4, 5.0, 0.05, law, 91),
        ModelKind::HardSphere => Setup::hard_sphere(1000, 0.01, law, 91),
    };
    let mut run = |seed: u64| {
        let mut params = setup.params;
        params.seed = seed;
        let mut model = make(params);
        let mut state = setup.initial_state(0);
        let mut collisions = 0;
        for step in 1..=200 {
            setup.stream(&mut state, setup.dt_s);
            collisions += model
                .collide(
                    &setup.cells,
                    &state.pos_km,
                    &mut state.vel_km_s,
                    step,
                    setup.dt_s,
                )
                .collisions;
        }
        assert!(collisions > 0, "no collisions in the determinism run");
        state.vel_km_s
    };
    let bits = |v: &[Vec3<f64>]| {
        v.iter()
            .flat_map(|u| u.to_array().map(f64::to_bits))
            .collect::<Vec<u64>>()
    };
    let a = bits(&run(91));
    let b = bits(&run(91));
    assert_eq!(a, b, "K1-T9: the same seed gave different velocities");
    if kind == ModelKind::Stochastic {
        let c = bits(&run(92));
        assert_ne!(a, c, "K1-T9: another seed gave the same velocities");
    }
    println!("K1-T9 determinism {kind:?}: ok");
}
