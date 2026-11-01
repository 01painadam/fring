//! The precision options of [`crate::spike`] on the GPU (`shaders/spike.wgsl`).
//!
//! [`GpuSpike::new`] runs the double-single self-test first and fails closed
//! (ADR 0006).

use crate::ds::Ds;
use crate::gpu::{Gpu, Kernel, Words, from_words, to_words};
use crate::probe::self_test;
use crate::spike::{Consts, Opt, State, Tables};
use fring_model::{Real, Result};
use std::time::Duration;

/// The WGSL source of the spike kernels.
pub const SPIKE_WGSL: &str = concat!(
    include_str!("../shaders/ds.wgsl"),
    include_str!("../shaders/trig.wgsl"),
    include_str!("../shaders/spike.wgsl")
);

/// Compiled kernels of all options.
pub struct GpuSpike<'g> {
    kernels: Vec<(Opt, Kernel<'g>)>,
}

impl<'g> GpuSpike<'g> {
    /// Runs [`self_test`] and compiles the kernels of every option.
    ///
    /// # Errors
    /// [`fring_model::Error::Backend`] if double-single arithmetic does not
    /// work on this GPU.
    pub fn new(gpu: &'g Gpu) -> Result<Self> {
        self_test(gpu)?;
        let kernels = Opt::ALL
            .iter()
            .map(|&o| (o, gpu.kernel(SPIKE_WGSL, o.entry())))
            .collect();
        Ok(Self { kernels })
    }

    /// The uniform parameters (6 × vec4) for the constants `c` and table `tab`.
    pub fn params<const M: u8>(c: &Consts<M>, tab: &Tables<M>, n_particles: usize) -> Words {
        let f = &c.field_d;
        let h = Ds::<M>::split(c.h_s);
        let half = Ds::<M>::split(0.5 * c.h_s);
        let gm_moons = tab.d.gm_km3_s2();
        let gm_m = |j: usize| gm_moons.get(j).copied().unwrap_or(Ds::ZERO);
        let ds2 = |d: Ds<M>| [d.hi, d.lo];
        let mut v: Vec<f32> = Vec::with_capacity(24);
        v.extend(ds2(f.gm_km3_s2));
        v.extend(ds2(h));
        v.extend(ds2(c.inv_gm_d));
        v.extend(ds2(f.reference_radius_km));
        v.extend(ds2(f.j2));
        v.extend(ds2(f.j4));
        v.extend(ds2(f.j6));
        v.extend(ds2(gm_m(0)));
        v.extend(ds2(gm_m(1)));
        v.push(1.0); // the guard ds_one
        v.push(f32::from_bits(tab.d.len() as u32));
        v.push(f32::from_bits(tab.d.n_moons() as u32));
        v.push(f32::from_bits(n_particles as u32));
        v.extend(ds2(half));
        to_words(&v)
    }

    /// Advances `states` by the steps of `tab` with option `opt` on the GPU.
    /// Returns the GPU time of the dispatch.
    ///
    /// # Panics
    /// If the table has more than two moons.
    pub fn advance<const M: u8>(
        &self,
        opt: Opt,
        c: &Consts<M>,
        tab: &Tables<M>,
        states: &mut [State<M>],
    ) -> Duration {
        assert!(
            tab.d.n_moons() <= 2,
            "the spike kernels take at most 2 moons"
        );
        let kernel = &self
            .kernels
            .iter()
            .find(|(o, _)| *o == opt)
            .expect("kernel")
            .1;
        let flat: Vec<f32> = states
            .iter()
            .flat_map(|s| s.iter().flat_map(|d| [d.hi, d.lo]))
            .collect();
        let mut table = tab.pack();
        if table.is_empty() {
            table.push(0.0);
        }
        let mut bufs = vec![to_words(&flat), to_words(&table)];
        let prm = Self::params(c, tab, states.len());
        let t = kernel.run(&mut bufs, Some(&prm), states.len().div_ceil(64) as u32, 1);
        let out = from_words(&bufs[0]);
        for (s, w) in states.iter_mut().zip(out.as_chunks::<12>().0) {
            for (k, d) in s.iter_mut().enumerate() {
                *d = Ds {
                    hi: w[2 * k],
                    lo: w[2 * k + 1],
                };
            }
        }
        t
    }
}
