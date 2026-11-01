//! Frame round-trip, golden-byte and damaged-file tests. All in memory.

use fring_io::{
    FRAME_TYPE_ROTATING_F32, FRAME_VERSION, Frame, RotatingVelocities, read_frame,
    read_frame_header, write_frame,
};
use fring_model::rng::philox4x32_10;

/// Frame whose float values are random bit patterns: NaN with payloads,
/// infinities, subnormals and -0.0 all occur.
fn random_bits_frame(n: usize, with_velocities: bool) -> Frame {
    let key = [0xdead_beef, 0x0123_4567];
    let mut f = Frame {
        step: 123_456_789_012,
        time_s: 1.5e9,
        a0_km: 140_221.3,
        n0_rad_s: 1.17e-4,
        t_ref_s: 0.0,
        ..Frame::default()
    };
    let mut vel = RotatingVelocities::default();
    for i in 0..n {
        let w = philox4x32_10([i as u32, 0, 0, 0], key);
        let w2 = philox4x32_10([i as u32, 1, 0, 0], key);
        f.id.push(w[3]);
        f.x_km.push(f32::from_bits(w[0]));
        f.y_km.push(f32::from_bits(w[1]));
        f.z_km.push(f32::from_bits(w[2]));
        vel.vx_km_s.push(f32::from_bits(w2[0]));
        vel.vy_km_s.push(f32::from_bits(w2[1]));
        vel.vz_km_s.push(f32::from_bits(w2[2]));
    }
    if with_velocities {
        f.velocities = Some(vel);
    }
    f
}

fn to_bytes(f: &Frame) -> Vec<u8> {
    let mut b = Vec::new();
    write_frame(&mut b, f).unwrap();
    b
}

fn err_text(bytes: &[u8]) -> String {
    read_frame(&mut &bytes[..]).unwrap_err().to_string()
}

fn small_frame() -> Frame {
    Frame {
        step: 7,
        time_s: 100.0,
        a0_km: 140_000.0,
        n0_rad_s: 1.0e-4,
        t_ref_s: 0.0,
        id: vec![10, 11],
        x_km: vec![1.0, 2.0],
        y_km: vec![3.0, 4.0],
        z_km: vec![5.0, 6.0],
        velocities: None,
    }
}

#[test]
fn frame_roundtrip_small_with_and_without_velocities() {
    for with_v in [false, true] {
        let f = random_bits_frame(1000, with_v);
        let g = read_frame(&mut to_bytes(&f).as_slice()).unwrap();
        assert!(f.bit_identical(&g));
        assert_eq!(g.velocities.is_some(), with_v);
    }
}

#[test]
fn frame_roundtrip_million_bit_exact() {
    let n = 1_000_000;
    for with_v in [false, true] {
        let f = random_bits_frame(n, with_v);
        let bytes = to_bytes(&f);
        let per_particle = if with_v { 28 } else { 16 };
        assert_eq!(bytes.len(), 72 + n * per_particle);
        let g = read_frame(&mut bytes.as_slice()).unwrap();
        assert_eq!(g.len(), n);
        assert!(f.bit_identical(&g), "values differ bitwise");
        // Writing the result again gives the same bytes.
        assert_eq!(to_bytes(&g), bytes);
    }
}

#[test]
fn special_values_keep_their_bits() {
    let mut f = small_frame();
    f.x_km = vec![-0.0, f32::from_bits(0x7fa0_0001)]; // signalling NaN with payload
    f.y_km = vec![f32::INFINITY, f32::from_bits(1)]; // subnormal
    let g = read_frame(&mut to_bytes(&f).as_slice()).unwrap();
    assert_eq!(g.x_km[0].to_bits(), 0x8000_0000);
    assert_eq!(g.x_km[1].to_bits(), 0x7fa0_0001);
    assert_eq!(g.y_km[1].to_bits(), 1);
    assert!(f.bit_identical(&g));
}

#[test]
fn frame_byte_layout_golden() {
    let mut f = small_frame();
    f.velocities = None;
    let b = to_bytes(&f);
    assert_eq!(&b[0..8], b"FRINGFRM");
    assert_eq!(b[8..12], 1u32.to_le_bytes());
    assert_eq!(b[12..16], 72u32.to_le_bytes());
    assert_eq!(b[16..20], 1u32.to_le_bytes());
    assert_eq!(b[20..24], 0u32.to_le_bytes());
    assert_eq!(b[24..32], 7u64.to_le_bytes());
    assert_eq!(b[32..40], 2u64.to_le_bytes());
    assert_eq!(b[40..48], 100.0f64.to_le_bytes());
    assert_eq!(b[48..56], 140_000.0f64.to_le_bytes());
    assert_eq!(b[56..64], 1.0e-4f64.to_le_bytes());
    assert_eq!(b[64..72], 0.0f64.to_le_bytes());
    // Body: ids, then x, y, z.
    assert_eq!(b[72..76], 10u32.to_le_bytes());
    assert_eq!(b[76..80], 11u32.to_le_bytes());
    assert_eq!(b[80..84], 1.0f32.to_le_bytes());
    assert_eq!(b[88..92], 3.0f32.to_le_bytes());
    assert_eq!(b[96..100], 5.0f32.to_le_bytes());
    assert_eq!(b.len(), 72 + 2 * 16);
    assert_eq!(FRAME_VERSION, 1);
    assert_eq!(FRAME_TYPE_ROTATING_F32, 1);
}

#[test]
fn header_only_read_gives_fields() {
    let mut f = small_frame();
    f.velocities = Some(RotatingVelocities {
        vx_km_s: vec![0.0; 2],
        vy_km_s: vec![0.0; 2],
        vz_km_s: vec![0.0; 2],
    });
    let h = read_frame_header(&mut to_bytes(&f).as_slice()).unwrap();
    assert_eq!(h.n_particles, 2);
    assert_eq!(h.step, 7);
    assert!(h.has_velocities());
    assert_eq!(h.header_len, 72);
}

#[test]
fn rejects_bad_magic() {
    let mut b = to_bytes(&small_frame());
    b[0] = b'X';
    assert!(err_text(&b).contains("bad magic"));
}

#[test]
fn rejects_unknown_version() {
    let mut b = to_bytes(&small_frame());
    b[8..12].copy_from_slice(&2u32.to_le_bytes());
    assert!(err_text(&b).contains("unsupported frame version 2"));
}

#[test]
fn rejects_unknown_flags_and_frame_type() {
    let mut b = to_bytes(&small_frame());
    b[20..24].copy_from_slice(&2u32.to_le_bytes());
    assert!(err_text(&b).contains("flag"));
    let mut b = to_bytes(&small_frame());
    b[16..20].copy_from_slice(&9u32.to_le_bytes());
    assert!(err_text(&b).contains("frame type"));
}

#[test]
fn rejects_bad_header_len() {
    for bad in [64u32, 73, 80_000] {
        let mut b = to_bytes(&small_frame());
        b[12..16].copy_from_slice(&bad.to_le_bytes());
        assert!(err_text(&b).contains("header_len"), "{bad}");
    }
}

#[test]
fn longer_header_is_skipped() {
    let f = small_frame();
    let b = to_bytes(&f);
    let mut longer = b[..72].to_vec();
    longer[12..16].copy_from_slice(&80u32.to_le_bytes());
    longer.extend_from_slice(&[0xAA; 8]); // fields from a future minor version
    longer.extend_from_slice(&b[72..]);
    let g = read_frame(&mut longer.as_slice()).unwrap();
    assert!(f.bit_identical(&g));
}

#[test]
fn rejects_truncated_at_every_cut() {
    let mut f = random_bits_frame(5, true);
    f.step = 1;
    let b = to_bytes(&f);
    for cut in 0..b.len() {
        let e = err_text(&b[..cut]);
        assert!(e.contains("truncated"), "cut {cut}: {e}");
    }
}

#[test]
fn rejects_trailing_data() {
    let mut b = to_bytes(&small_frame());
    b.push(0);
    assert!(err_text(&b).contains("trailing"));
}

#[test]
fn huge_count_fails_as_truncated_or_limit() {
    let mut b = to_bytes(&small_frame());
    b[32..40].copy_from_slice(&(1u64 << 32).to_le_bytes());
    assert!(err_text(&b).contains("truncated"));
    b[32..40].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(err_text(&b).contains("limit"));
}

#[test]
fn velocity_flag_with_missing_arrays_is_truncated() {
    let mut b = to_bytes(&small_frame());
    b[20..24].copy_from_slice(&1u32.to_le_bytes());
    assert!(err_text(&b).contains("truncated"));
}

#[test]
fn write_rejects_ragged_arrays() {
    let mut f = small_frame();
    f.z_km.pop();
    assert!(write_frame(&mut Vec::new(), &f).is_err());
}
