// Double-single (DS) arithmetic: a value is vec2(hi, lo), two f32 with
// |lo| <= ulp(hi)/2, about 48 significant bits (relative precision 2^-48).
//
// Rust twin: crates/fring-gpu/src/ds.rs (same operations in the same order).
// Algorithms:
// - fast_two_sum: Dekker 1971 (needs |a| >= |b|); two_sum: fast_two_sum
//   after a sort by magnitude; two_prod: exact product error with an fma.
// - ds_add: Joldes, Muller, Popescu 2017, Algorithm 6 (bound 3u^2, u = 2^-24);
//   ds_add_f: Algorithm 4; ds_mul: Algorithm 12 (bound 5u^2); ds_mul_f:
//   Algorithm 9; ds_div: Algorithm 17 with TwoSum in line 4.
// - ds_sqrt: one Newton correction of the f32 root.
// - ds_sin_cos_small: Taylor series for |x| <= 0.25.
//
// TwoSum is Fast2Sum after a branch-free sort by magnitude. Knuth's 6-operation
// TwoSum with the same guards passed alone but failed inside ds_add on the M4
// (the compiler re-associated it after inlining), so it is not used.
//
// GUARD (ADR 0006). wgpu compiles Metal shaders with fast math on, and the
// Metal compiler then re-associates (a + b) - a to b, which removes the error
// terms. The functions below multiply those sums by `ds_one`, a runtime 1.0
// from a uniform buffer that the compiler cannot see. Every kernel that uses
// this file must set `ds_one` from its uniform first. A multiplication by
// 1.0 is exact, so the results are those of the plain algorithms. The probe
// (src/probe.rs, self_test) checks on the GPU that this still holds.

var<private> ds_one: f32;

fn fast_two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = (a + b) * ds_one;
    let z = (s - a) * ds_one;
    return vec2<f32>(s, b - z);
}

fn two_sum(a: f32, b: f32) -> vec2<f32> {
    let swap = abs(b) > abs(a);
    return fast_two_sum(select(a, b, swap), select(b, a, swap));
}

fn two_prod(a: f32, b: f32) -> vec2<f32> {
    let p = a * b;
    return vec2<f32>(p, fma(a, b, -p));
}

fn ds(x: f32) -> vec2<f32> {
    return vec2<f32>(x, 0.0);
}

fn ds_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let s = two_sum(a.x, b.x);
    let t = two_sum(a.y, b.y);
    let c = s.y + t.x;
    let v = fast_two_sum(s.x, c);
    let w = t.y + v.y;
    return fast_two_sum(v.x, w);
}

fn ds_add_f(a: vec2<f32>, b: f32) -> vec2<f32> {
    let s = two_sum(a.x, b);
    let v = a.y + s.y;
    return fast_two_sum(s.x, v);
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

fn ds_mul_f(a: vec2<f32>, b: f32) -> vec2<f32> {
    let c = two_prod(a.x, b);
    let cl3 = fma(a.y, b, c.y);
    return fast_two_sum(c.x, cl3);
}

fn ds_div(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let th = 1.0 / b.x;
    let rh = fma(-b.x, th, 1.0);
    let rl = -(b.y * th);
    let e = two_sum(rh, rl);
    let d = ds_mul_f(e, th);
    let m = ds_add_f(d, th);
    return ds_mul(a, m);
}

fn ds_sqrt(a: vec2<f32>) -> vec2<f32> {
    if (a.x <= 0.0) {
        return vec2<f32>(0.0, 0.0);
    }
    let s = sqrt(a.x);
    let p = two_prod(s, s);
    let e = (((a.x - p.x) * ds_one) - p.y) + a.y;
    let c = e / (2.0 * s);
    return fast_two_sum(s, c);
}

struct DsTrig {
    s: vec2<f32>,
    c: vec2<f32>,
    omc: vec2<f32>,
}

// Sine, cosine and 1 - cosine of |x| <= 0.25 (Taylor series to x^13 and
// x^14; the terms from x^7 on are summed in f32).
fn ds_sin_cos_small(x: vec2<f32>) -> DsTrig {
    let t = ds_mul(x, x);
    let th = t.x;
    let ps = fma(fma(th, fma(th, -1.6059044e-10f, 2.5052108e-08f), -2.7557319e-06f), th, 1.984127e-04f);
    let s_in = ds_add(ds_mul(ds_add(ds_mul_f(t, -ps), vec2<f32>(8.333334e-03f, -4.346172e-10f)), t),
        vec2<f32>(-1.6666667e-01f, 4.967054e-09f));
    let s = ds_add(x, ds_mul(ds_mul(x, t), s_in));
    let pc = fma(fma(th, fma(th, -1.1470745e-11f, 2.0876756e-09f), -2.755732e-07f), th, 2.4801588e-05f);
    let c_in = ds_add(ds_mul(ds_add(ds_mul_f(t, -pc), vec2<f32>(1.3888889e-03f, -3.3631094e-11f)), t),
        vec2<f32>(-4.1666668e-02f, 1.2417635e-09f));
    let omc = ds_mul(t, ds_add(ds(0.5), ds_mul(t, c_in)));
    return DsTrig(s, ds_sub(ds(1.0), omc), omc);
}
