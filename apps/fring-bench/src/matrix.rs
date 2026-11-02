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

/// One measurement point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Point {
    /// Backend variant.
    pub variant: Variant,
    /// Number of particles.
    pub n: usize,
    /// Chaos indicator on (ICR-003: d0 = 1 km, K = 100).
    pub chaos: bool,
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
                m.push(Point { variant, n, chaos });
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
}
