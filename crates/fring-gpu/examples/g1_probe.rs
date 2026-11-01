//! Precision probe of the GPU for task G1 (ADR 0006).
//!
//! `cargo run -p fring-gpu --release --example g1_probe > docs/adr/0006/probe_m4.txt`

fn main() {
    let Some(gpu) = fring_gpu::gpu::Gpu::new() else {
        eprintln!("no GPU adapter found");
        std::process::exit(2);
    };
    let r = fring_gpu::probe::run_probe(&gpu);
    println!("{r:#?}");
    println!("guarded double-single works: {}", r.double_single_works());
    match fring_gpu::probe::self_test(&gpu) {
        Ok(()) => println!("self-test: passed"),
        Err(e) => println!("self-test: {e}"),
    }
}
