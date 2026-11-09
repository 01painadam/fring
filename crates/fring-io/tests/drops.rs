//! `drops.csv` round trip, header, crash rule R1 and the run-directory wrappers.

use fring_io::{
    DROPS_COLUMNS, DropRow, RunDir, read_drops_csv, read_drops_csv_report, write_drops_csv,
};

fn rows() -> Vec<DropRow> {
    vec![
        DropRow {
            id: 3,
            reason: 1,
            drop_step: 17,
            time_s: 17.0 * 356.0,
            position_km: [1.0e-300, -0.0, 140_221.123_456_789],
            velocity_km_s: [-20.5, 0.1 + 0.2, 1.0e5],
        },
        DropRow {
            id: u32::MAX,
            reason: 2,
            drop_step: u64::MAX - 1,
            time_s: 1.0e18,
            position_km: [1.0, 2.0, 3.0],
            velocity_km_s: [4.0, 5.0, 6.0],
        },
    ]
}

fn text(rows: &[DropRow]) -> String {
    let mut v = Vec::new();
    write_drops_csv(&mut v, rows).unwrap();
    String::from_utf8(v).unwrap()
}

#[test]
fn header_is_the_icr_text() {
    assert_eq!(
        DROPS_COLUMNS.join(","),
        "id,reason,drop_step,time_s,x_km,y_km,z_km,vx_km_s,vy_km_s,vz_km_s"
    );
    let t = text(&[]);
    assert_eq!(t, format!("{}\n", DROPS_COLUMNS.join(",")));
    // A header-only file reads as no rows.
    let r = read_drops_csv_report(&mut t.as_bytes()).unwrap();
    assert!(r.rows.is_empty() && !r.dropped_partial);
}

#[test]
fn round_trip_is_exact() {
    let r = rows();
    let back = read_drops_csv(&mut text(&r).as_bytes()).unwrap();
    assert_eq!(back.len(), 2);
    for (a, b) in r.iter().zip(&back) {
        assert_eq!((a.id, a.reason, a.drop_step), (b.id, b.reason, b.drop_step));
        assert_eq!(a.time_s.to_bits(), b.time_s.to_bits());
        for k in 0..3 {
            assert_eq!(a.position_km[k].to_bits(), b.position_km[k].to_bits());
            assert_eq!(a.velocity_km_s[k].to_bits(), b.velocity_km_s[k].to_bits());
        }
    }
}

#[test]
fn crash_rule_drops_an_unfinished_last_line_at_every_cut() {
    let full = text(&rows());
    let head = DROPS_COLUMNS.join(",").len() + 1;
    let first_end = head + full[head..].find('\n').unwrap() + 1;
    for cut in head..full.len() {
        let r = read_drops_csv_report(&mut &full.as_bytes()[..cut]).unwrap();
        let complete = if cut >= first_end { 1 } else { 0 };
        let ends_on_newline = full.as_bytes()[cut - 1] == b'\n';
        let expect = if cut == full.len() {
            2
        } else if ends_on_newline && cut == first_end {
            1
        } else {
            complete
        };
        assert_eq!(r.rows.len(), expect, "cut {cut}");
        assert_eq!(
            r.dropped_partial,
            !ends_on_newline && cut > head,
            "cut {cut}"
        );
    }
    // A cut that still parses (the last number is shorter) is dropped too.
    let cut = full.trim_end().trim_end_matches('0').to_string();
    let r = read_drops_csv_report(&mut cut.as_bytes()).unwrap();
    assert_eq!(r.rows.len(), 1);
    assert!(r.dropped_partial);
}

#[test]
fn bad_files_are_errors() {
    let h = DROPS_COLUMNS.join(",");
    assert!(read_drops_csv(&mut "".as_bytes()).is_err());
    assert!(read_drops_csv(&mut "id,reason\n1,1\n".as_bytes()).is_err());
    // A finished ragged row is an error, not dropped.
    assert!(read_drops_csv(&mut format!("{h}\n1,1,2\n").as_bytes()).is_err());
    assert!(read_drops_csv(&mut format!("{h}\n1,1,2,x,0,0,0,0,0,0\n").as_bytes()).is_err());
    assert!(read_drops_csv(&mut format!("{h}\n1,300,2,0,0,0,0,0,0,0\n").as_bytes()).is_err());
    // Unknown columns are ignored and the order is free.
    let ok = "extra,drop_step,id,reason,time_s,x_km,y_km,z_km,vx_km_s,vy_km_s,vz_km_s\n\
              9,5,7,2,1.5,1,2,3,4,5,6\n";
    let r = read_drops_csv(&mut ok.as_bytes()).unwrap();
    assert_eq!((r[0].id, r[0].reason, r[0].drop_step), (7, 2, 5));
}

#[test]
fn run_dir_round_trip_leaves_no_tmp_file() {
    let dir = std::env::temp_dir().join(format!("fring-io-drops-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let rd = RunDir::new(&dir);
    rd.create().unwrap();
    rd.write_drops_file(&rows()).unwrap();
    assert_eq!(rd.read_drops_file().unwrap(), rows());
    rd.write_drops_file(&[]).unwrap();
    assert!(rd.read_drops_file().unwrap().is_empty());
    assert!(!dir.join("drops.csv.tmp").exists());
    let _ = std::fs::remove_dir_all(&dir);
}
