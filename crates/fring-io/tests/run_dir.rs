//! Tests that touch the disk: a few hundred particles only, in a temp directory.

use fring_io::{
    Frame, MoonRecord, MoonsFile, RunDir, RunMeta, frame_file_name, parse_frame_file_name,
};
use fring_model::{BodyState, StepStats, Vec3};
use std::path::PathBuf;

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!("fring-io-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn frame(n: u32) -> Frame {
    Frame {
        step: 5,
        time_s: 3600.0,
        a0_km: 140_221.0,
        n0_rad_s: 1.17e-4,
        t_ref_s: 0.0,
        id: (0..n).collect(),
        x_km: (0..n).map(|i| i as f32 * 0.5).collect(),
        y_km: (0..n).map(|i| -(i as f32)).collect(),
        z_km: vec![-0.0; n as usize],
        velocities: None,
    }
}

#[test]
fn file_names() {
    assert_eq!(frame_file_name(42), "000042.bin");
    assert_eq!(frame_file_name(1_000_000), "1000000.bin");
    assert_eq!(parse_frame_file_name("000042.bin"), Some(42));
    assert_eq!(parse_frame_file_name("1000000.bin"), Some(1_000_000));
    assert_eq!(parse_frame_file_name("42.bin"), None);
    assert_eq!(parse_frame_file_name("000042.bin.tmp"), None);
    assert_eq!(parse_frame_file_name("00004a.bin"), None);
}

#[test]
fn disk_run_dir_roundtrip() {
    let t = TempDir::new("roundtrip");
    let dir = RunDir::new(t.0.join("run"));
    dir.create().unwrap();

    let f = frame(300);
    dir.write_frame_file(2, &f).unwrap();
    dir.write_frame_file(10, &frame(3)).unwrap();
    dir.write_frame_file(1_000_000, &frame(1)).unwrap();
    assert_eq!(dir.list_frames().unwrap(), vec![2, 10, 1_000_000]);
    assert!(f.bit_identical(&dir.read_frame_file(2).unwrap()));

    let moons = MoonsFile {
        names: vec!["Prometheus".into()],
        records: vec![MoonRecord {
            time_s: 0.0,
            bodies: vec![BodyState {
                position_km: Vec3::new(1.0, 2.0, 3.0),
                velocity_km_s: Vec3::new(4.0, 5.0, 6.0),
            }],
        }],
    };
    dir.write_moons_file(&moons).unwrap();
    assert_eq!(dir.read_moons_file().unwrap(), moons);

    let rows = vec![StepStats {
        time_s: 1.0,
        steps_total: 2,
        collisions_since_last: 3,
        max_rel_energy_error: Some(1e-12),
    }];
    dir.write_stats_file(&rows).unwrap();
    assert_eq!(dir.read_stats_file().unwrap(), rows);

    let meta = RunMeta::new("0.1.0", "abc123", "test machine", 12.5);
    dir.write_meta_file(&meta).unwrap();
    assert_eq!(dir.read_meta_file().unwrap(), meta);
    assert_eq!(meta.frame_format_version, 1);
}

#[test]
fn config_copy_is_byte_identical() {
    let t = TempDir::new("config");
    let dir = RunDir::new(t.0.join("run"));
    dir.create().unwrap();
    // Odd spacing, comments and CRLF must survive.
    let bytes = b"# comment\r\n[run]\r\nseed   =  1 \r\n\xff\x00end";
    dir.write_config_copy(bytes).unwrap();
    assert_eq!(dir.read_config_copy().unwrap(), bytes);
}

#[test]
fn atomic_write_leaves_no_tmp_file_and_ignores_stale_tmp() {
    let t = TempDir::new("atomic");
    let dir = RunDir::new(t.0.join("run"));
    dir.create().unwrap();
    // A stale tmp file from a stopped run is ignored by the listing.
    std::fs::write(dir.frames_dir().join("000007.bin.tmp"), b"partial").unwrap();
    dir.write_frame_file(0, &frame(4)).unwrap();
    assert_eq!(dir.list_frames().unwrap(), vec![0]);
    assert!(!dir.frames_dir().join("000000.bin.tmp").exists());
}

#[test]
fn failed_write_removes_tmp_and_keeps_old_file() {
    let t = TempDir::new("fail");
    let dir = RunDir::new(t.0.join("run"));
    dir.create().unwrap();
    dir.write_frame_file(0, &frame(4)).unwrap();
    let mut bad = frame(4);
    bad.z_km.pop();
    assert!(dir.write_frame_file(0, &bad).is_err());
    assert!(!dir.frames_dir().join("000000.bin.tmp").exists());
    assert_eq!(dir.read_frame_file(0).unwrap().len(), 4);
}

#[test]
fn read_missing_file_is_io_error() {
    let t = TempDir::new("missing");
    let dir = RunDir::new(t.0.join("nothing"));
    assert!(dir.read_frame_file(0).is_err());
}

#[test]
fn meta_json_has_expected_keys() {
    let meta = RunMeta::new("0.1.0", "abc", "m", 1.0);
    let mut b = Vec::new();
    fring_io::write_meta(&mut b, &meta).unwrap();
    let v: serde_json::Value = serde_json::from_slice(&b).unwrap();
    for k in [
        "code_version",
        "git_hash",
        "machine",
        "wall_time_s",
        "frame_format_version",
        "moons_format_version",
        "stats_format_version",
    ] {
        assert!(v.get(k).is_some(), "missing {k}");
    }
}
