// The pair operator, the impact geometry and the restitution laws in f32
// (SPEC 4, 5.4, 6). Rust twin: crates/fring-collide/src/pair.rs, run with
// f32 (`pair::collide::<f32>`, `pair::impact_normal::<f32>`); the same
// operations in the same order. Restitution: `pair::epsilon` in f64, rounded
// to f32 (the WGSL uses pow in f32; the test measures the difference).

const PAIR_PI: f32 = 3.1415927;

struct Impulse {
    // Change of the peculiar velocity of particle i (particle j gets minus this).
    d: vec3<f32>,
    // Energy loss 1/4 (1 - eps^2) gn^2 in km^2 s^-2.
    loss: f32,
    eps: f32,
    // false: the pair receded (gn >= 0); nothing changes.
    hit: bool,
}

// kind 0: constant eps; kind 1: power law (vn / vc)^-p above vc.
fn restitution(vn: f32, kind: u32, eps: f32, vc: f32, pexp: f32) -> f32 {
    if (kind == 0u) { return eps; }
    if (vn > vc) { return pow(vn / vc, -pexp); }
    return 1.0;
}

fn collide_pair(g: vec3<f32>, n_hat: vec3<f32>, kind: u32, eps0: f32, vc: f32, pexp: f32) -> Impulse {
    let gn = dot(g, n_hat);
    if (!(gn < 0.0)) { return Impulse(vec3<f32>(0.0), 0.0, 1.0, false); }
    let eps = restitution(abs(gn), kind, eps0, vc, pexp);
    let d = n_hat * (-(0.5 * (1.0 + eps) * gn));
    return Impulse(d, 0.25 * (1.0 - eps * eps) * gn * gn, eps, true);
}

fn impact_normal(g: vec3<f32>, u1: f32, u2: f32) -> vec3<f32> {
    let gh = g / sqrt(dot(g, g));
    var axis = vec3<f32>(1.0, 0.0, 0.0);
    if (abs(gh.z) < 0.9) { axis = vec3<f32>(0.0, 0.0, 1.0); }
    let c = cross(gh, axis);
    let e1 = c / sqrt(dot(c, c));
    let e2 = cross(gh, e1);
    let cos_t = sqrt(u1);
    let sin_t = sqrt(1.0 - u1);
    let psi = 2.0 * PAIR_PI * u2;
    return (e1 * cos(psi) + e2 * sin(psi)) * sin_t - gh * cos_t;
}
