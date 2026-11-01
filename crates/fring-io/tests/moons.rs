use fring_io::{MoonRecord, MoonsFile, read_moons, write_moons};
use fring_model::{BodyState, Vec3};

fn body(k: f64) -> BodyState {
    BodyState {
        position_km: Vec3::new(k, k + 0.1, -k),
        velocity_km_s: Vec3::new(-k, 0.5 * k, f64::from_bits(0x7ff8_0000_0000_1234)),
    }
}

fn file(names: &[&str], n_records: usize) -> MoonsFile {
    MoonsFile {
        names: names.iter().map(|s| s.to_string()).collect(),
        records: (0..n_records)
            .map(|i| MoonRecord {
                time_s: i as f64 * 360.0,
                bodies: (0..names.len())
                    .map(|m| body((i * 10 + m) as f64))
                    .collect(),
            })
            .collect(),
    }
}

fn bits(f: &MoonsFile) -> Vec<u64> {
    f.records
        .iter()
        .flat_map(|r| {
            std::iter::once(r.time_s).chain(r.bodies.iter().flat_map(|b| {
                [
                    b.position_km.x,
                    b.position_km.y,
                    b.position_km.z,
                    b.velocity_km_s.x,
                    b.velocity_km_s.y,
                    b.velocity_km_s.z,
                ]
            }))
        })
        .map(f64::to_bits)
        .collect()
}

#[test]
fn roundtrip_zero_one_two_moons() {
    for names in [&[][..], &["Prometheus"], &["Prometheus", "Pandora"]] {
        let f = file(names, 5);
        let mut b = Vec::new();
        write_moons(&mut b, &f).unwrap();
        let g = read_moons(&mut b.as_slice()).unwrap();
        assert_eq!(g.names, f.names);
        assert_eq!(bits(&g), bits(&f));
        // Header is a multiple of 8 and records are 8 (1 + 6 m) bytes.
        let header_len = u32::from_le_bytes(b[12..16].try_into().unwrap()) as usize;
        assert_eq!(header_len % 8, 0);
        assert_eq!(b.len(), header_len + 5 * 8 * (1 + 6 * names.len()));
    }
}

#[test]
fn zero_records_and_unicode_names() {
    let f = file(&["Prométhée", "Пандора"], 0);
    let mut b = Vec::new();
    write_moons(&mut b, &f).unwrap();
    assert_eq!(read_moons(&mut b.as_slice()).unwrap(), f);
}

#[test]
fn rejects_damaged_files() {
    let mut b = Vec::new();
    write_moons(&mut b, &file(&["A", "B"], 3)).unwrap();
    for cut in 0..b.len() {
        let e = read_moons(&mut &b[..cut]).unwrap_err().to_string();
        assert!(e.contains("truncated"), "cut {cut}: {e}");
    }
    let mut t = b.clone();
    t.push(0);
    assert!(
        read_moons(&mut t.as_slice())
            .unwrap_err()
            .to_string()
            .contains("trailing")
    );
    let mut m = b.clone();
    m[0] = b'X';
    assert!(
        read_moons(&mut m.as_slice())
            .unwrap_err()
            .to_string()
            .contains("bad magic")
    );
    let mut v = b.clone();
    v[8] = 9;
    assert!(
        read_moons(&mut v.as_slice())
            .unwrap_err()
            .to_string()
            .contains("unsupported")
    );
    let mut fl = b.clone();
    fl[20] = 1;
    assert!(
        read_moons(&mut fl.as_slice())
            .unwrap_err()
            .to_string()
            .contains("flag")
    );
    // Name length that runs past the table.
    let mut n = b.clone();
    n[32..36].copy_from_slice(&1000u32.to_le_bytes());
    assert!(
        read_moons(&mut n.as_slice())
            .unwrap_err()
            .to_string()
            .contains("name table")
    );
    // Huge record count fails as truncated.
    let mut c = b.clone();
    c[24..32].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(
        read_moons(&mut c.as_slice())
            .unwrap_err()
            .to_string()
            .contains("truncated")
    );
}

#[test]
fn write_rejects_wrong_body_count() {
    let mut f = file(&["A"], 2);
    f.records[1].bodies.clear();
    assert!(write_moons(&mut Vec::new(), &f).is_err());
}
