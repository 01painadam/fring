// Double-single (DS) arithmetic: a value is hi + lo, two f32 with
// |lo| <= ulp(hi)/2, about 48 significant bits.
//
// Rust twin: crates/fring-gpu/src/ds.rs (same operations in the same order).
// Algorithms:
// - two_sum: Knuth, TAOCP vol. 2, 4.2.2 (6 operations, any order of |a|, |b|).
// - fast_two_sum: Dekker 1971 (needs |a| >= |b| or a = 0).
// - two_prod: exact product error with a fused multiply-add.
// - two_prod_dekker: Dekker 1971 with Veltkamp's split (no FMA needed).
// - ds_add: Joldes, Muller, Popescu 2017, Algorithm 6 (AccurateDWPlusDW).
// - ds_mul: Joldes, Muller, Popescu 2017, Algorithm 12 (DWTimesDW3, with FMA).
// - ds_div: Joldes, Muller, Popescu 2017, Algorithm 17 (DWDivDW2).
// These rely on correctly rounded +, -, * (WGSL guarantees them for f32) and
// on the compiler not re-ordering them. The probe (probe.wgsl) checks both.

fn two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = a + b;
    let bb = s - a;
    let e = (a - (s - bb)) + (b - bb);
    return vec2<f32>(s, e);
}

fn fast_two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = a + b;
    let e = b - (s - a);
    return vec2<f32>(s, e);
}

fn two_prod(a: f32, b: f32) -> vec2<f32> {
    let p = a * b;
    let e = fma(a, b, -p);
    return vec2<f32>(p, e);
}

// Veltkamp split: a = hi + lo with 12 significant bits in each part.
fn split(a: f32) -> vec2<f32> {
    let c = 4097.0 * a;
    let hi = c - (c - a);
    return vec2<f32>(hi, a - hi);
}

fn two_prod_dekker(a: f32, b: f32) -> vec2<f32> {
    let p = a * b;
    let sa = split(a);
    let sb = split(b);
    let e = ((sa.x * sb.x - p) + sa.x * sb.y + sa.y * sb.x) + sa.y * sb.y;
    return vec2<f32>(p, e);
}

fn ds_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let s = two_sum(a.x, b.x);
    let t = two_sum(a.y, b.y);
    let c = s.y + t.x;
    let v = fast_two_sum(s.x, c);
    let w = t.y + v.y;
    return fast_two_sum(v.x, w);
}

fn ds_neg(a: vec2<f32>) -> vec2<f32> {
    return -a;
}

fn ds_sub(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return ds_add(a, -b);
}

fn ds_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let c = two_prod(a.x, b.x);
    let tl0 = a.y * b.y;
    let tl1 = fma(a.x, b.y, tl0);
    let cl2 = fma(a.y, b.x, tl1);
    let cl3 = c.y + cl2;
    return fast_two_sum(c.x, cl3);
}

// DS times f32 (Joldes et al. Algorithm 9, DWTimesFP3).
fn ds_mul_f(a: vec2<f32>, b: f32) -> vec2<f32> {
    let c = two_prod(a.x, b);
    let cl3 = fma(a.y, b, c.y);
    return fast_two_sum(c.x, cl3);
}

fn ds_div(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let th = 1.0 / b.x;
    let rh = fma(-b.x, th, 1.0);
    let rl = -(b.y * th);
    let e = fast_two_sum(rh, rl);
    let d = ds_mul_f(e, th);
    let m = ds_add(d, vec2<f32>(th, 0.0));
    return ds_mul(a, m);
}
