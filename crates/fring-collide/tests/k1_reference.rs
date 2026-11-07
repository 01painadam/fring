//! K1-T4 to K1-T7 and K1-T9 on the two reference models (`SPEC.md` section
//! 11). They show that the shared tests are right: a model that follows the
//! spec passes them.
//!
//! The quick tests run in every `cargo test`. The full tests (the sample-size
//! rule of SPEC 11) are long; run them with
//! `cargo test -p fring-collide --release -- --ignored`.

use fring_collide::testkit::reference::{AllPairs, HardSphere};
use fring_collide::testkit::{
    ModelKind, Sample, t4_rate, t5_energy_loss, t6_budget, t7_haff, t9_determinism,
};
use fring_model::config::Restitution;

const S: ModelKind = ModelKind::Stochastic;
const H: ModelKind = ModelKind::HardSphere;

fn power_law() -> Restitution {
    // ε from about 0.5 to 0.7 over the speeds of the box (c = 1 cm/s), so the
    // test sees whether ε uses vₙ or |g|.
    Restitution::PowerLaw {
        reference_speed_km_s: 1e-6,
        exponent: 0.234,
    }
}

// ---- quick ----

#[test]
fn quick_t4_rate_stochastic() {
    for occupancy in [2.0, 5.0] {
        t4_rate(S, occupancy, Sample::Quick, AllPairs::new).assert_ok();
    }
}

#[test]
fn quick_t5_energy_loss_stochastic() {
    t5_energy_loss(
        S,
        Restitution::Constant { epsilon: 0.5 },
        Sample::Quick,
        AllPairs::new,
    )
    .assert_ok();
    t5_energy_loss(
        S,
        Restitution::Constant { epsilon: 1.0 },
        Sample::Quick,
        AllPairs::new,
    )
    .assert_ok();
}

#[test]
fn quick_t6_budget_stochastic() {
    t6_budget(S, 0.5, AllPairs::new).assert_ok();
    t6_budget(S, 1.0, AllPairs::new).assert_ok();
}

#[test]
fn quick_t7_haff_stochastic() {
    t7_haff(S, Sample::Quick, AllPairs::new).assert_ok();
}

#[test]
fn t9_determinism_both() {
    t9_determinism(S, AllPairs::new);
    t9_determinism(H, HardSphere::new);
}

// ---- full (long) ----

#[test]
#[ignore = "long: run with --release -- --ignored"]
fn full_t4_rate_stochastic() {
    for occupancy in [2.0, 5.0, 20.0] {
        t4_rate(S, occupancy, Sample::Full, AllPairs::new).assert_ok();
    }
}

#[test]
#[ignore = "long: run with --release -- --ignored"]
fn full_t5_energy_loss_stochastic() {
    for e in [0.0, 0.5, 0.9, 1.0] {
        t5_energy_loss(
            S,
            Restitution::Constant { epsilon: e },
            Sample::Full,
            AllPairs::new,
        )
        .assert_ok();
    }
    t5_energy_loss(S, power_law(), Sample::Full, AllPairs::new).assert_ok();
}

#[test]
#[ignore = "long: run with --release -- --ignored"]
fn full_t7_haff_stochastic() {
    t7_haff(S, Sample::Full, AllPairs::new).assert_ok();
}

#[test]
#[ignore = "long: run with --release -- --ignored"]
fn full_t4_rate_hard_sphere() {
    t4_rate(H, 0.0, Sample::Full, HardSphere::new).assert_ok();
}

#[test]
#[ignore = "long: run with --release -- --ignored"]
fn full_t5_energy_loss_hard_sphere() {
    t5_energy_loss(
        H,
        Restitution::Constant { epsilon: 0.5 },
        Sample::Full,
        HardSphere::new,
    )
    .assert_ok();
    t5_energy_loss(H, power_law(), Sample::Full, HardSphere::new).assert_ok();
}

#[test]
#[ignore = "long: run with --release -- --ignored"]
fn full_t6_budget_hard_sphere() {
    t6_budget(H, 0.5, HardSphere::new).assert_ok();
    t6_budget(H, 1.0, HardSphere::new).assert_ok();
}

#[test]
#[ignore = "long: run with --release -- --ignored"]
fn full_t7_haff_hard_sphere() {
    t7_haff(H, Sample::Full, HardSphere::new).assert_ok();
}
