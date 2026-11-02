//! ICR-005 item 4: the two-phase `SelfTest` gives the result of `self_test`.
//! Needs a GPU, so `#[ignore]`:
//! `cargo test -p fring-gpu --release --test gpu_selftest -- --ignored --nocapture`.

use fring_gpu::gpu::Gpu;
use fring_gpu::probe::{SelfTest, self_test};
use std::time::{Duration, Instant};

#[test]
#[ignore = "needs a GPU"]
fn submit_and_poll_give_the_result_of_the_blocking_self_test() {
    let Some(gpu) = Gpu::new() else {
        eprintln!("no GPU: skipped");
        return;
    };
    let blocking = self_test(&gpu);
    let mut t = SelfTest::submit(&gpu);
    let t0 = Instant::now();
    let two_phase = loop {
        if let Some(r) = t.poll() {
            break r;
        }
        assert!(
            t0.elapsed() < Duration::from_secs(60),
            "self-test timed out"
        );
        std::thread::sleep(Duration::from_millis(1));
    };
    assert_eq!(
        blocking.map_err(|e| e.to_string()),
        two_phase.map_err(|e| e.to_string())
    );
    // The result is given once.
    assert!(matches!(t.poll(), Some(Err(_))));
}
