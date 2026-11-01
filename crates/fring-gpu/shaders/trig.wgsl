// Sine and cosine in f32 (needs ds.wgsl for the guard `ds_one`).
// Rust twin: sin_cos_f32 in crates/fring-gpu/src/wf32.rs.
//
// Cody-Waite reduction (pi/2 = P1 + P2 + P3, P1 and P2 with 12 significant
// bits) to r in [-pi/4, pi/4], then minimax polynomials in t = r^2 from
// docs/adr/0006/remez_sincos.py (Remez exchange): relative error of sin
// 8.4e-9, absolute error of cos 7.2e-10 before rounding. The WGSL built-ins
// are only bounded by 2^-11 absolute, so they are not used.

fn sin_cos_f32(x: f32) -> vec2<f32> {
    let k = round(x * 6.3661975e-01f);
    let r = fma(-k, -8.7055163e-10f, ((x - k * 1.5708008e+00f) * ds_one - k * -4.4535846e-06f) * ds_one);
    let t = r * r;
    let sp = fma(r * t, fma(t, fma(t, -1.9503963e-04f, 8.332101e-03f), -1.6666655e-01f), r);
    let cp = fma(t * t, fma(t, fma(t, 2.4463838e-05f, -1.3887654e-03f), 4.1666653e-02f), fma(-0.5, t, 1.0));
    let q = i32(k) & 3;
    if (q == 0) {
        return vec2<f32>(sp, cp);
    } else if (q == 1) {
        return vec2<f32>(cp, -sp);
    } else if (q == 2) {
        return vec2<f32>(-sp, -cp);
    }
    return vec2<f32>(-cp, sp);
}
