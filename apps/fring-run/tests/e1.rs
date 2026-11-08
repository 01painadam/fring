//! Tests of the E1-REF runner (`fring_run::e1`) on the real reference files.
//! Small cases only; the full runs are in the handoff `docs/handoffs/E1-RUN.md`.

use fring_model::Particles;
use fring_run::e1::{self, L1Row, Variant};
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(format!("{}/../..", env!("CARGO_MANIFEST_DIR")))
}

fn read(name: &str) -> String {
    std::fs::read_to_string(root().join("validation/reference").join(name)).unwrap()
}

/// Five encounters from at least two different epochs.
fn five() -> Vec<L1Row> {
    let all = e1::parse_l1_states(&read("e1_l1_states.csv")).unwrap();
    assert_eq!(all.len(), 2000);
    let first = all[0].t0_idx;
    let other = all.iter().find(|r| r.t0_idx != first).unwrap().clone();
    let mut v: Vec<L1Row> = all.iter().take(2).cloned().collect();
    v.push(other.clone());
    v.extend(
        all.iter()
            .filter(|r| r.t0_idx == other.t0_idx)
            .skip(1)
            .take(2)
            .cloned(),
    );
    assert!(v.iter().any(|r| r.t0_idx != first));
    v
}

fn cover(rows: &[L1Row]) -> f64 {
    rows.iter().map(|r| r.t0_s).fold(0.0, f64::max) + e1::WINDOW_S + 3600.0
}

#[test]
fn layer1_group_and_repeat_are_bit_identical() {
    let rows = five();
    let s = e1::build_setup(&root().join("params"), cover(&rows)).unwrap();
    let run = |rows: &[L1Row]| {
        e1::layer1(
            rows,
            &s.system,
            Variant::Plain,
            e1::cpu_backend(s.radii_km.clone()),
        )
        .unwrap()
    };
    let a = run(&rows);
    let b = run(&rows);
    assert_eq!(a.finals, b.finals, "two runs differ");
    assert!(a.frozen.is_empty());
    assert_eq!(a.finals.len(), 5);
    // One id alone (its own backend) equals the same id inside its group.
    let one = run(&rows[2..3]);
    let in_group = a.finals.iter().find(|f| f.0 == rows[2].id).unwrap();
    assert_eq!(one.finals[0], *in_group);
    // The state moved by roughly 16 km/s x 172,800 s along the orbit, not by NaN.
    for (id, f) in &a.finals {
        let r = rows.iter().find(|r| r.id == *id).unwrap();
        let d = ((f[0] - r.state[0]).powi(2) + (f[1] - r.state[1]).powi(2)).sqrt();
        assert!(d > 1e5 && d < 4e6, "id {id}: moved {d} km");
    }
}

#[test]
fn substep_variant_runs_and_agrees_roughly() {
    let rows = five();
    let s = e1::build_setup(&root().join("params"), cover(&rows)).unwrap();
    let go = |v| e1::layer1(&rows, &s.system, v, e1::cpu_backend(s.radii_km.clone())).unwrap();
    let (p, q) = (go(Variant::Plain), go(Variant::Substep));
    for (a, b) in p.finals.iter().zip(&q.finals) {
        assert_eq!(a.0, b.0);
        assert!((a.1[0] - b.1[0]).abs() < 50.0, "{} vs {}", a.1[0], b.1[0]);
    }
}

#[test]
fn moons_agree_with_e1_ref() {
    let s = e1::build_setup(&root().join("params"), 352.0 * 86_400.0).unwrap();
    let (text, dr, dv) = e1::moon_difference(&s.system.moons, &read("e1_moons_t0.csv")).unwrap();
    assert_eq!(text.lines().filter(|l| !l.starts_with('#')).count(), 36);
    assert!(dr < 0.01, "max moon position difference {dr} km");
    assert!(dv < 1e-5, "max moon velocity difference {dv} km/s");
}

#[test]
fn layer2_smoke() {
    let all = e1::parse_l2_states(&read("e1_l2_states.csv")).unwrap();
    assert_eq!(all.len(), 300);
    let mut p = Particles::default();
    for i in 0..3 {
        p.push(all.id[i], all.position_km(i), all.velocity_km_s(i));
    }
    let days = 20.0;
    let s = e1::build_setup(&root().join("params"), days * 86_400.0 + 3600.0).unwrap();
    let r = e1::layer2(
        p.clone(),
        days,
        e1::NOMINAL_STEP_S,
        1,
        &s.system,
        Variant::Plain,
        e1::cpu_backend(s.radii_km.clone()),
    )
    .unwrap();
    assert_eq!(r.rows.len(), 3);
    assert_eq!(r.n_steps, 4850);
    for (i, row) in r.rows.iter().enumerate() {
        let p0 = p.position_km(i);
        let m0 = s.system.moons.tracks[0].samples[0].position_km;
        let d0 = ((p0.x - m0.x).powi(2) + (p0.y - m0.y).powi(2) + (p0.z - m0.z).powi(2)).sqrt();
        assert!(row.min_prometheus_km <= d0 + 1e-9);
        assert!(row.min_prometheus_km > 0.0);
        assert!(row.megno_mean.is_finite());
    }
    let r2 = e1::layer2(
        p,
        days,
        e1::NOMINAL_STEP_S,
        1,
        &s.system,
        Variant::Plain,
        e1::cpu_backend(s.radii_km.clone()),
    )
    .unwrap();
    assert_eq!(r.rows, r2.rows, "two runs differ");
}
