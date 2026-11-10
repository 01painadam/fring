//! Guard: no WGSL source of this crate holds a constant that is not finite.
//!
//! The WGSL spec makes a const-expression that evaluates to NaN or infinity a
//! shader-creation error. Chrome (Tint) enforces it; native wgpu (naga) does
//! not, so native tests pass while the browser rejects the whole module
//! (WGSL-NAN, 2026-11-09). This test scans the sources without a GPU.
//!
//! It looks for `bitcast<f32>(` applied to an integer literal, and for a
//! division by a literal zero. Comments are skipped. A failure marker must be
//! a finite sentinel instead, for example x = -1.0e30.

use std::fs;
use std::path::{Path, PathBuf};

fn wgsl_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            wgsl_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "wgsl") {
            out.push(p);
        }
    }
}

/// Problems in one line of code (comment already removed).
fn problems(line: &str) -> Vec<String> {
    let mut found = Vec::new();
    let mut rest = line;
    while let Some(k) = rest.find("bitcast<f32>(") {
        let after = rest[k + "bitcast<f32>(".len()..].trim_start();
        if after.starts_with(|c: char| c.is_ascii_digit()) {
            found.push("bitcast<f32> of an integer literal (may be NaN or inf)".to_string());
        }
        rest = &rest[k + 1..];
    }
    let b = line.as_bytes();
    for (i, &c) in b.iter().enumerate() {
        if c != b'/' {
            continue;
        }
        let t = line[i + 1..].trim_start().as_bytes();
        if t.first() != Some(&b'0') {
            continue;
        }
        // A literal that is all zeros: "0", "0.0", "0.", "0f", "0.0f", "0u".
        let mut j = 1;
        while j < t.len() && (t[j] == b'0' || t[j] == b'.') {
            j += 1;
        }
        if j < t.len() && (t[j] == b'f' || t[j] == b'u' || t[j] == b'i') {
            j += 1;
        }
        let ends = j >= t.len() || !(t[j].is_ascii_alphanumeric() || t[j] == b'_');
        if ends {
            found.push("division by a literal zero".to_string());
        }
    }
    found
}

#[test]
fn problems_finds_the_known_bad_forms() {
    assert!(!problems("x = bitcast<f32>(0x7fc00000u);").is_empty());
    assert!(!problems("x = bitcast<f32>( 0x7f800000u );").is_empty());
    assert!(!problems("y = a / 0.0;").is_empty());
    assert!(!problems("y = a/0;").is_empty());
    assert!(problems("y = bitcast<f32>(bits_var);").is_empty());
    assert!(problems("y = a / 0.5 + b / 10.0 + c / 0.01;").is_empty());
}

#[test]
fn no_wgsl_constant_is_non_finite() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders");
    let mut files = Vec::new();
    wgsl_files(&root, &mut files);
    assert!(files.len() >= 8, "found only {} WGSL files", files.len());
    let mut bad = Vec::new();
    for f in &files {
        let text = fs::read_to_string(f).unwrap();
        for (n, raw) in text.lines().enumerate() {
            let code = raw.split("//").next().unwrap();
            for p in problems(code) {
                bad.push(format!("{}:{}: {p}: {}", f.display(), n + 1, raw.trim()));
            }
        }
    }
    assert!(
        bad.is_empty(),
        "non-finite WGSL constants:\n{}",
        bad.join("\n")
    );
}
