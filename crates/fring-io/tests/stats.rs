use fring_io::{read_stats_csv, write_stats_csv};
use fring_model::StepStats;

fn rows() -> Vec<StepStats> {
    vec![
        StepStats {
            time_s: 0.0,
            steps_total: 0,
            collisions_since_last: 0,
            max_rel_energy_error: None,
        },
        StepStats {
            time_s: 0.1 + 0.2,
            steps_total: u64::MAX,
            collisions_since_last: 12345,
            max_rel_energy_error: Some(1.2345678901234567e-13),
        },
        StepStats {
            time_s: 1e300,
            steps_total: 5,
            collisions_since_last: 1,
            max_rel_energy_error: Some(f64::NAN),
        },
        StepStats {
            time_s: -5e-324,
            steps_total: 6,
            collisions_since_last: 2,
            max_rel_energy_error: Some(f64::INFINITY),
        },
    ]
}

fn same(a: &StepStats, b: &StepStats) -> bool {
    a.time_s.to_bits() == b.time_s.to_bits()
        && a.steps_total == b.steps_total
        && a.collisions_since_last == b.collisions_since_last
        && a.max_rel_energy_error.map(f64::to_bits) == b.max_rel_energy_error.map(f64::to_bits)
}

#[test]
fn stats_roundtrip_exact_f64() {
    let r = rows();
    let mut b = Vec::new();
    write_stats_csv(&mut b, &r).unwrap();
    let g = read_stats_csv(&mut b.as_slice()).unwrap();
    assert_eq!(g.len(), r.len());
    for (x, y) in r.iter().zip(&g) {
        // NaN: the text round trip keeps NaN but not its payload; compare class.
        if x.max_rel_energy_error.is_some_and(f64::is_nan) {
            assert!(y.max_rel_energy_error.unwrap().is_nan());
            assert_eq!(x.time_s.to_bits(), y.time_s.to_bits());
        } else {
            assert!(same(x, y), "{x:?} vs {y:?}");
        }
    }
}

#[test]
fn stats_plain_csv_header() {
    let mut b = Vec::new();
    write_stats_csv(&mut b, &rows()).unwrap();
    let text = String::from_utf8(b).unwrap();
    assert_eq!(
        text.lines().next().unwrap(),
        "time_s,steps_total,collisions_since_last,max_rel_energy_error"
    );
    assert!(!text.contains('#'));
    assert!(
        text.lines().nth(1).unwrap().ends_with(','),
        "empty field = None"
    );
}

#[test]
fn stats_unknown_and_reordered_columns() {
    let text = "extra,steps_total,time_s,max_rel_energy_error\nx,7,2.5,1e-9\n";
    let g = read_stats_csv(&mut text.as_bytes()).unwrap();
    assert_eq!(g.len(), 1);
    assert_eq!(g[0].time_s, 2.5);
    assert_eq!(g[0].steps_total, 7);
    assert_eq!(g[0].collisions_since_last, 0);
    assert_eq!(g[0].max_rel_energy_error, Some(1e-9));
}

#[test]
fn stats_missing_optional_columns() {
    let g = read_stats_csv(&mut "time_s\n1.0\n2.0\n".as_bytes()).unwrap();
    assert_eq!(g.len(), 2);
    assert_eq!(g[1].max_rel_energy_error, None);
}

#[test]
fn stats_rejects_bad_input() {
    for (text, needle) in [
        ("", "empty"),
        ("steps_total\n1\n", "time_s"),
        ("time_s,steps_total\n1.0\n", "fields"),
        ("time_s\nabc\n", "not a number"),
        ("time_s,steps_total\n1.0,-3\n", "not an integer"),
    ] {
        let e = read_stats_csv(&mut text.as_bytes())
            .unwrap_err()
            .to_string();
        assert!(e.contains(needle), "{text:?}: {e}");
    }
}

#[test]
fn stats_drops_unfinished_last_line() {
    use fring_io::read_stats_csv_report;
    // A crash during a write leaves a line without a newline. It is dropped
    // even when it parses (here "2.5" could be a cut "2.5e9").
    for tail in ["2.5", "2.", "", "2.5,7,0"] {
        let text = format!("time_s,steps_total,collisions_since_last\n1.0,3,0\n{tail}");
        let got = read_stats_csv_report(&mut text.as_bytes()).unwrap();
        assert_eq!(got.rows.len(), 1, "{tail:?}");
        assert_eq!(got.rows[0].time_s, 1.0);
        assert_eq!(got.dropped_partial, !tail.is_empty(), "{tail:?}");
    }
    // A finished but bad row is still an error.
    assert!(read_stats_csv(&mut "time_s,steps_total\n1.0\n".as_bytes()).is_err());
    // A header without a newline is an empty file.
    let e = read_stats_csv(&mut "time_s,steps".as_bytes()).unwrap_err();
    assert!(e.to_string().contains("empty"), "{e}");
}
