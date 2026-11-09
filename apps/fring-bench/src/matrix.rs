//! The benchmark matrix: which backend, how many particles, chaos on or off.

/// A backend variant under test.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    /// `CpuBackend` (f64, all threads).
    Cpu,
    /// GPU option D: fast, does not meet PLAN section 10.
    GpuD,
    /// GPU option D+: production.
    GpuDp,
    /// GPU option C: reference.
    GpuC,
}

impl Variant {
    /// Short name for CSV and tables.
    pub fn name(self) -> &'static str {
        match self {
            Variant::Cpu => "cpu_f64",
            Variant::GpuD => "gpu_d",
            Variant::GpuDp => "gpu_dp",
            Variant::GpuC => "gpu_c",
        }
    }

    /// Label for the Markdown table.
    pub fn label(self) -> &'static str {
        match self {
            Variant::Cpu => "CPU f64",
            Variant::GpuD => "GPU D (not for science)",
            Variant::GpuDp => "GPU D+ (production)",
            Variant::GpuC => "GPU C (reference)",
        }
    }

    /// True for the GPU variants.
    pub fn is_gpu(self) -> bool {
        self != Variant::Cpu
    }
}

/// A feature set switched on in a B2 point (task B2). B1 points are `Base`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Feature {
    /// Test particles only (the B1 workload).
    Base,
    /// `[loss] on_unbound = "freeze"`.
    Freeze,
    /// Freeze and the moon radius rule (`moon_radius = "params"`).
    Loss,
    /// Stochastic collisions (K5).
    Coll,
    /// Stochastic collisions, freeze and the radius rule.
    CollLoss,
    /// `[substep]` (accuracy 0.1, max_level 4) with freeze and the rule. CPU only.
    Substep,
}

impl Feature {
    /// Name for CSV and the `--features` argument.
    pub fn name(self) -> &'static str {
        match self {
            Feature::Base => "base",
            Feature::Freeze => "freeze",
            Feature::Loss => "loss",
            Feature::Coll => "coll",
            Feature::CollLoss => "coll+loss",
            Feature::Substep => "substep",
        }
    }

    /// Parses a name.
    pub fn parse(s: &str) -> Option<Feature> {
        [
            Feature::Base,
            Feature::Freeze,
            Feature::Loss,
            Feature::Coll,
            Feature::CollLoss,
            Feature::Substep,
        ]
        .into_iter()
        .find(|f| f.name() == s)
    }

    /// Stochastic collisions on.
    pub fn collisions(self) -> bool {
        matches!(self, Feature::Coll | Feature::CollLoss)
    }

    /// The freeze rule on (alone or with the radius rule).
    pub fn freeze(self) -> bool {
        matches!(
            self,
            Feature::Freeze | Feature::Loss | Feature::CollLoss | Feature::Substep
        )
    }

    /// The moon radius rule on.
    pub fn radius_rule(self) -> bool {
        matches!(self, Feature::Loss | Feature::CollLoss | Feature::Substep)
    }
}

/// The particle state of a point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    /// The cold core of B1 (sigma = 5 km, e = 0.0024, seed 1).
    Core,
    /// The band of the A3-ANALYSIS runs (139,700 to 141,100 km, circular
    /// orbits, seed 20261102), generated directly.
    Band,
    /// A dense core for collision cost: like `Core` but sigma = 0.3 km (the
    /// resolved-core width of SPEC section 10), seed 1.
    Dense,
}

impl State {
    /// Name for CSV and the `--states` argument.
    pub fn name(self) -> &'static str {
        match self {
            State::Core => "core",
            State::Band => "band",
            State::Dense => "dense",
        }
    }

    /// Parses a name.
    pub fn parse(s: &str) -> Option<State> {
        [State::Core, State::Band, State::Dense]
            .into_iter()
            .find(|x| x.name() == s)
    }
}

/// One measurement point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Point {
    /// Backend variant.
    pub variant: Variant,
    /// Number of particles.
    pub n: usize,
    /// Chaos indicator on (ICR-003: d0 = 1 km, K = 100).
    pub chaos: bool,
    /// Feature set (B2); `Base` for B1.
    pub feature: Feature,
    /// Particle state (B2); `Core` for B1.
    pub state: State,
    /// Integrator steps per collision step (only with collisions).
    pub every_steps: u64,
    /// Sensitivity point: radial cell size 0.2 km instead of 0.05 km.
    pub wide_cells: bool,
}

impl Point {
    /// A B1 point: no features, cold core.
    pub fn b1(variant: Variant, n: usize, chaos: bool) -> Point {
        Point {
            variant,
            n,
            chaos,
            feature: Feature::Base,
            state: State::Core,
            every_steps: 1,
            wide_cells: false,
        }
    }

    /// True for a B1 point (the B1 table layout applies).
    pub fn is_b1(&self) -> bool {
        self.feature == Feature::Base && self.state == State::Core
    }
}

/// The CPU runs the chaos indicator only up to this N (it is slow there).
pub const CPU_CHAOS_MAX_N: usize = 100_000;

/// Builds the matrix, ordered by N so that one particle set serves all
/// points of a size.
pub fn build_matrix(sizes: &[usize], with_mode_d: bool) -> Vec<Point> {
    let mut variants = vec![Variant::Cpu];
    if with_mode_d {
        variants.push(Variant::GpuD);
    }
    variants.extend([Variant::GpuDp, Variant::GpuC]);
    let mut m = Vec::new();
    for &n in sizes {
        for &variant in &variants {
            for chaos in [false, true] {
                if chaos && !variant.is_gpu() && n > CPU_CHAOS_MAX_N {
                    continue;
                }
                m.push(Point::b1(variant, n, chaos));
            }
        }
    }
    m
}

/// The CPU runs the B2 feature rows only up to this N (the collision step
/// of the CPU is a few hundred ms at 10^7).
pub const CPU_B2_MAX_N: usize = 1_000_000;

/// Options of the B2 matrix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct B2Spec {
    /// Feature sets.
    pub features: Vec<Feature>,
    /// Particle states.
    pub states: Vec<State>,
    /// Values of `every_steps` for the collision rows.
    pub every: Vec<u64>,
    /// Add the cell-size sensitivity point (GPU D+, 10^6 if in `sizes`).
    pub sensitivity: bool,
}

/// Builds the B2 matrix, ordered by state and N so that one particle set
/// serves all points of a size. Backends: GPU D+ and CPU (D and C are not
/// measured: C refuses collisions and `[loss]`). `Substep` is CPU only.
pub fn build_matrix_b2(sizes: &[usize], spec: &B2Spec) -> Vec<Point> {
    let mut m = Vec::new();
    for &state in &spec.states {
        for &n in sizes {
            for variant in [Variant::GpuDp, Variant::Cpu] {
                if !variant.is_gpu() && n > CPU_B2_MAX_N {
                    continue;
                }
                for &feature in &spec.features {
                    if feature == Feature::Substep && variant.is_gpu() {
                        continue;
                    }
                    let everys: &[u64] = if feature.collisions() {
                        &spec.every
                    } else {
                        &[1]
                    };
                    for &every_steps in everys {
                        m.push(Point {
                            variant,
                            n,
                            chaos: false,
                            feature,
                            state,
                            every_steps,
                            wide_cells: false,
                        });
                    }
                }
            }
            if spec.sensitivity && n == 1_000_000 {
                m.push(Point {
                    variant: Variant::GpuDp,
                    n,
                    chaos: false,
                    feature: Feature::Coll,
                    state,
                    every_steps: 10,
                    wide_cells: true,
                });
            }
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matrix_has_expected_points() {
        let m = build_matrix(&[10_000, 100_000, 1_000_000, 10_000_000], true);
        // GPU: 3 modes x 4 sizes x 2 = 24. CPU: 4 off + 2 on = 6.
        assert_eq!(m.len(), 30);
        let cpu_chaos = m
            .iter()
            .filter(|p| p.variant == Variant::Cpu && p.chaos)
            .count();
        assert_eq!(cpu_chaos, 2);
        assert!(
            !m.iter()
                .any(|p| p.variant == Variant::Cpu && p.chaos && p.n > CPU_CHAOS_MAX_N)
        );
    }

    #[test]
    fn mode_d_is_optional() {
        let m = build_matrix(&[10_000], false);
        assert!(m.iter().all(|p| p.variant != Variant::GpuD));
        assert_eq!(m.len(), 2 + 2 + 2);
    }

    #[test]
    fn b2_matrix_rules() {
        let spec = B2Spec {
            features: vec![
                Feature::Base,
                Feature::Coll,
                Feature::CollLoss,
                Feature::Substep,
            ],
            states: vec![State::Band],
            every: vec![1, 10],
            sensitivity: true,
        };
        let m = build_matrix_b2(&[100_000, 1_000_000, 10_000_000], &spec);
        // GPU: (1 + 2 + 2) x 3 sizes + 1 sensitivity = 16.
        // CPU (N <= 10^6): (1 + 2 + 2 + 1) x 2 sizes = 12.
        assert_eq!(m.len(), 28);
        assert!(m.iter().all(|p| !p.chaos && p.state == State::Band));
        assert!(
            !m.iter()
                .any(|p| p.variant.is_gpu() && p.feature == Feature::Substep)
        );
        assert!(
            !m.iter()
                .any(|p| p.variant == Variant::Cpu && p.n > CPU_B2_MAX_N)
        );
        assert_eq!(m.iter().filter(|p| p.wide_cells).count(), 1);
        assert!(
            m.iter()
                .filter(|p| !p.feature.collisions())
                .all(|p| p.every_steps == 1)
        );
    }

    #[test]
    fn feature_names_round_trip() {
        for f in [
            Feature::Base,
            Feature::Freeze,
            Feature::Loss,
            Feature::Coll,
            Feature::CollLoss,
            Feature::Substep,
        ] {
            assert_eq!(Feature::parse(f.name()), Some(f));
        }
        assert!(Feature::CollLoss.collisions() && Feature::CollLoss.radius_rule());
        assert!(Feature::Freeze.freeze() && !Feature::Freeze.radius_rule());
        assert_eq!(Feature::parse("x"), None);
    }
}
