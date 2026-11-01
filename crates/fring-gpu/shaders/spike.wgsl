// The precision options of task G1 (ADR 0006): one Wisdom-Holman block per
// particle and dispatch. Append to ds.wgsl and trig.wgsl.
// Rust twins: crates/fring-gpu/src/spike.rs (options B, B+, D) and the
// generic fring_model::physics code with Wf32 (option A) or Ds (option C).
//
// Bindings:
// 0: state, 12 f32 per particle: x, y, z (km), vx, vy, vz (km/s), each as
//    (hi, lo). Option A uses only the hi parts.
// 1: table, 24 f32 per step (Tables::pack): moon positions DS (moon j,
//    component c at 6j + 2c), indirect term DS at 12, indirect term f32 at 18.
// 2: parameters, see prm_* below.

@group(0) @binding(0) var<storage, read_write> st: array<f32>;
@group(0) @binding(1) var<storage, read> tab: array<f32>;
@group(0) @binding(2) var<uniform> prm: array<vec4<f32>, 6>;

fn prm_gm() -> vec2<f32> { return prm[0].xy; }
fn prm_h() -> vec2<f32> { return prm[0].zw; }
fn prm_inv_gm() -> vec2<f32> { return prm[1].xy; }
fn prm_radius() -> vec2<f32> { return prm[1].zw; }
fn prm_j2() -> vec2<f32> { return prm[2].xy; }
fn prm_j4() -> vec2<f32> { return prm[2].zw; }
fn prm_j6() -> vec2<f32> { return prm[3].xy; }
fn prm_gm_moon(j: u32) -> vec2<f32> {
    if (j == 0u) { return prm[3].zw; }
    return prm[4].xy;
}
fn prm_n_steps() -> u32 { return bitcast<u32>(prm[4].w); }
fn prm_n_moons() -> u32 { return bitcast<u32>(prm[5].x); }
fn prm_n_particles() -> u32 { return bitcast<u32>(prm[5].y); }
fn prm_half() -> vec2<f32> { return prm[5].zw; }

// ---- DS vectors ----

struct V3 {
    x: vec2<f32>,
    y: vec2<f32>,
    z: vec2<f32>,
}

fn v3_add(a: V3, b: V3) -> V3 { return V3(ds_add(a.x, b.x), ds_add(a.y, b.y), ds_add(a.z, b.z)); }
fn v3_sub(a: V3, b: V3) -> V3 { return V3(ds_sub(a.x, b.x), ds_sub(a.y, b.y), ds_sub(a.z, b.z)); }
fn v3_mul(a: V3, s: vec2<f32>) -> V3 { return V3(ds_mul(a.x, s), ds_mul(a.y, s), ds_mul(a.z, s)); }
fn v3_mul_f(a: V3, s: f32) -> V3 { return V3(ds_mul_f(a.x, s), ds_mul_f(a.y, s), ds_mul_f(a.z, s)); }
fn v3_add_w(a: V3, w: vec3<f32>) -> V3 { return V3(ds_add_f(a.x, w.x), ds_add_f(a.y, w.y), ds_add_f(a.z, w.z)); }
fn v3_dot(a: V3, b: V3) -> vec2<f32> { return ds_add(ds_add(ds_mul(a.x, b.x), ds_mul(a.y, b.y)), ds_mul(a.z, b.z)); }
fn v3_hi(a: V3) -> vec3<f32> { return vec3<f32>(a.x.x, a.y.x, a.z.x); }
fn dot_f(a: vec3<f32>, b: vec3<f32>) -> f32 { return a.x * b.x + a.y * b.y + a.z * b.z; }

fn ds_gt(a: vec2<f32>, b: vec2<f32>) -> bool { return a.x > b.x || (a.x == b.x && a.y > b.y); }
fn ds_abs(a: vec2<f32>) -> vec2<f32> {
    if (a.x < 0.0) { return -a; }
    return a;
}

fn load_p(i: u32) -> V3 {
    let b = 12u * i;
    return V3(vec2<f32>(st[b], st[b + 1u]), vec2<f32>(st[b + 2u], st[b + 3u]), vec2<f32>(st[b + 4u], st[b + 5u]));
}
fn load_v(i: u32) -> V3 {
    let b = 12u * i + 6u;
    return V3(vec2<f32>(st[b], st[b + 1u]), vec2<f32>(st[b + 2u], st[b + 3u]), vec2<f32>(st[b + 4u], st[b + 5u]));
}
fn store(i: u32, p: V3, v: V3) {
    let b = 12u * i;
    st[b] = p.x.x; st[b + 1u] = p.x.y; st[b + 2u] = p.y.x; st[b + 3u] = p.y.y; st[b + 4u] = p.z.x; st[b + 5u] = p.z.y;
    st[b + 6u] = v.x.x; st[b + 7u] = v.x.y; st[b + 8u] = v.y.x; st[b + 9u] = v.y.y; st[b + 10u] = v.z.x; st[b + 11u] = v.z.y;
}

fn moon_ds(k: u32, j: u32) -> V3 {
    let b = 24u * k + 6u * j;
    return V3(vec2<f32>(tab[b], tab[b + 1u]), vec2<f32>(tab[b + 2u], tab[b + 3u]), vec2<f32>(tab[b + 4u], tab[b + 5u]));
}
fn indirect_ds(k: u32) -> V3 {
    let b = 24u * k + 12u;
    return V3(vec2<f32>(tab[b], tab[b + 1u]), vec2<f32>(tab[b + 2u], tab[b + 3u]), vec2<f32>(tab[b + 4u], tab[b + 5u]));
}
fn indirect_f(k: u32) -> vec3<f32> {
    let b = 24u * k + 18u;
    return vec3<f32>(tab[b], tab[b + 1u], tab[b + 2u]);
}

// ---- f32 physics (twins of the generic code with Wf32) ----

struct KepF {
    p: vec3<f32>,
    v: vec3<f32>,
    ok: bool,
}

// Newton's method for x = dE (as fring_model::physics::kepler_drift).
fn newton_f32(ecos0: f32, esin0: f32, m: f32) -> f32 {
    var x = m;
    for (var it = 0; it < 40; it++) {
        let sc = sin_cos_f32(x);
        let f = x - ecos0 * sc.x + esin0 * (1.0 - sc.y) - m;
        let fp = 1.0 - ecos0 * sc.y + esin0 * sc.x;
        let dx = f / fp;
        x = x - dx;
        if (abs(dx) <= 4.0 * 1.1920929e-07f * max(1.0, abs(x))) {
            break;
        }
    }
    return x;
}

// kepler_drift::<Wf32>.
fn kepler_f(gm: f32, p: vec3<f32>, v: vec3<f32>, dt: f32) -> KepF {
    let r0 = sqrt(dot_f(p, p));
    let v2 = dot_f(v, v);
    let alpha = 2.0 / r0 - v2 / gm;
    if (!(alpha > 0.0)) {
        return KepF(p, v, false);
    }
    let a = 1.0 / alpha;
    let n = sqrt(gm / (a * a * a));
    let sqrt_gma = sqrt(gm * a);
    let ecos0 = 1.0 - r0 / a;
    let esin0 = dot_f(p, v) / sqrt_gma;
    let m = n * dt;
    let x = newton_f32(ecos0, esin0, m);
    let sc = sin_cos_f32(x);
    let sh = sin_cos_f32(0.5 * x).x;
    let omc = 2.0 * sh * sh;
    let r = a * (1.0 - ecos0 * sc.y + esin0 * sc.x);
    let f = 1.0 - (a / r0) * omc;
    let g = dt - (x - sc.x) / n;
    let fdot = -(sqrt_gma / (r * r0)) * sc.x;
    let gdot = 1.0 - (a / r) * omc;
    return KepF(f * p + g * v, fdot * p + gdot * v, true);
}

// SaturnField::acceleration_without_point_mass in f32.
fn accel_j_f(p: vec3<f32>) -> vec3<f32> {
    let r2 = dot_f(p, p);
    let r = sqrt(r2);
    let u = p.z / r;
    let u2 = u * u;
    let q = prm_radius().x / r;
    let q2 = q * q;
    let q4 = q2 * q2;
    let q6 = q4 * q2;
    let p2 = (3.0 * u2 - 1.0) * 0.5;
    let p4 = ((35.0 * u2 - 30.0) * u2 + 3.0) * 0.125;
    let p6 = (((231.0 * u2 - 315.0) * u2 + 105.0) * u2 - 5.0) * 0.0625;
    let d2 = 3.0 * u;
    let d4 = ((35.0 * u2 - 15.0) * u) * 0.5;
    let d6 = (((1386.0 * u2 - 1260.0) * u2 + 210.0) * u) * 0.0625;
    let j2q = prm_j2().x * q2;
    let j4q = prm_j4().x * q4;
    let j6q = prm_j6().x * q6;
    let aa = 3.0 * j2q * p2 + 5.0 * j4q * p4 + 7.0 * j6q * p6;
    let b = j2q * d2 + j4q * d4 + j6q * d6;
    let radial = 0.0 - aa - u * b;
    let k = -prm_gm().x / (r2 * r);
    return vec3<f32>(k * radial * p.x, k * radial * p.y, k * (radial * p.z + b * r));
}

// kick_acceleration::<Wf32> with the f32 moon positions (hi parts).
fn kick_acc_f(k: u32, p: vec3<f32>) -> vec3<f32> {
    var am = vec3<f32>(0.0, 0.0, 0.0);
    for (var j = 0u; j < prm_n_moons(); j++) {
        let mp = v3_hi(moon_ds(k, j));
        let d = mp - p;
        let d2 = dot_f(d, d);
        am = am + d * (prm_gm_moon(j).x / (d2 * sqrt(d2)));
    }
    return accel_j_f(p) + am + indirect_f(k);
}

// ---- DS physics (twins of the generic code with Ds: option C) ----

struct KepD {
    p: V3,
    v: V3,
    ok: bool,
}

fn kepler_c(gm: vec2<f32>, p: V3, v: V3, dt: vec2<f32>) -> KepD {
    let one = ds(1.0);
    let r0 = ds_sqrt(v3_dot(p, p));
    let v2 = v3_dot(v, v);
    let alpha = ds_sub(ds_div(ds(2.0), r0), ds_div(v2, gm));
    if (!(alpha.x > 0.0)) {
        return KepD(p, v, false);
    }
    let a = ds_div(one, alpha);
    let n = ds_sqrt(ds_div(gm, ds_mul(ds_mul(a, a), a)));
    let sqrt_gma = ds_sqrt(ds_mul(gm, a));
    let ecos0 = ds_sub(one, ds_div(r0, a));
    let esin0 = ds_div(v3_dot(p, v), sqrt_gma);
    let m = ds_mul(n, dt);
    var x = m;
    for (var it = 0; it < 40; it++) {
        let t = ds_sin_cos_small(x);
        let f = ds_sub(ds_add(ds_sub(x, ds_mul(ecos0, t.s)), ds_mul(esin0, ds_sub(one, t.c))), m);
        let fp = ds_add(ds_sub(one, ds_mul(ecos0, t.c)), ds_mul(esin0, t.s));
        let dx = ds_div(f, fp);
        x = ds_sub(x, dx);
        var mx = one;
        if (ds_gt(ds_abs(x), one)) { mx = ds_abs(x); }
        let tol = ds_mul(vec2<f32>(1.4210855e-14f, 0.0), mx);
        if (!ds_gt(ds_abs(dx), tol)) {
            break;
        }
    }
    let t = ds_sin_cos_small(x);
    let sh = ds_sin_cos_small(ds_mul(ds(0.5), x)).s;
    let omc = ds_mul(ds_mul(ds(2.0), sh), sh);
    let r = ds_mul(a, ds_add(ds_sub(one, ds_mul(ecos0, t.c)), ds_mul(esin0, t.s)));
    let f = ds_sub(one, ds_mul(ds_div(a, r0), omc));
    let g = ds_sub(dt, ds_div(ds_sub(x, t.s), n));
    let fdot = ds_mul(-ds_div(sqrt_gma, ds_mul(r, r0)), t.s);
    let gdot = ds_sub(one, ds_mul(ds_div(a, r), omc));
    let p1 = V3(ds_add(ds_mul(f, p.x), ds_mul(g, v.x)), ds_add(ds_mul(f, p.y), ds_mul(g, v.y)), ds_add(ds_mul(f, p.z), ds_mul(g, v.z)));
    let v1 = V3(ds_add(ds_mul(fdot, p.x), ds_mul(gdot, v.x)), ds_add(ds_mul(fdot, p.y), ds_mul(gdot, v.y)), ds_add(ds_mul(fdot, p.z), ds_mul(gdot, v.z)));
    return KepD(p1, v1, true);
}

fn accel_j_c(p: V3) -> V3 {
    let r2 = v3_dot(p, p);
    let r = ds_sqrt(r2);
    let u = ds_div(p.z, r);
    let u2 = ds_mul(u, u);
    let q = ds_div(prm_radius(), r);
    let q2 = ds_mul(q, q);
    let q4 = ds_mul(q2, q2);
    let q6 = ds_mul(q4, q2);
    let p2 = ds_mul(ds_sub(ds_mul(ds(3.0), u2), ds(1.0)), ds(0.5));
    let p4 = ds_mul(ds_add(ds_mul(ds_sub(ds_mul(ds(35.0), u2), ds(30.0)), u2), ds(3.0)), ds(0.125));
    let p6 = ds_mul(ds_sub(ds_mul(ds_add(ds_mul(ds_sub(ds_mul(ds(231.0), u2), ds(315.0)), u2), ds(105.0)), u2), ds(5.0)), ds(0.0625));
    let d2 = ds_mul(ds(3.0), u);
    let d4 = ds_mul(ds_mul(ds_sub(ds_mul(ds(35.0), u2), ds(15.0)), u), ds(0.5));
    let d6 = ds_mul(ds_mul(ds_add(ds_mul(ds_sub(ds_mul(ds(1386.0), u2), ds(1260.0)), u2), ds(210.0)), u), ds(0.0625));
    let j2q = ds_mul(prm_j2(), q2);
    let j4q = ds_mul(prm_j4(), q4);
    let j6q = ds_mul(prm_j6(), q6);
    let aa = ds_add(ds_add(ds_mul(ds_mul(ds(3.0), j2q), p2), ds_mul(ds_mul(ds(5.0), j4q), p4)), ds_mul(ds_mul(ds(7.0), j6q), p6));
    let b = ds_add(ds_add(ds_mul(j2q, d2), ds_mul(j4q, d4)), ds_mul(j6q, d6));
    let radial = ds_sub(ds_sub(ds(0.0), aa), ds_mul(u, b));
    let k = ds_div(-prm_gm(), ds_mul(r2, r));
    let kr = ds_mul(k, radial);
    return V3(ds_mul(kr, p.x), ds_mul(kr, p.y), ds_mul(k, ds_add(ds_mul(radial, p.z), ds_mul(b, r))));
}

fn kick_acc_c(k: u32, p: V3) -> V3 {
    var am = V3(ds(0.0), ds(0.0), ds(0.0));
    for (var j = 0u; j < prm_n_moons(); j++) {
        let d = v3_sub(moon_ds(k, j), p);
        let d2 = v3_dot(d, d);
        am = v3_add(am, v3_mul(d, ds_div(prm_gm_moon(j), ds_mul(d2, ds_sqrt(d2)))));
    }
    return v3_add(v3_add(accel_j_c(p), am), indirect_ds(k));
}

// ---- Mixed options (twins of spike.rs) ----

// Option B drift: f32 Kepler drift of the hi parts, increments added in DS.
fn drift_b(p: V3, v: V3, dt: f32) -> KepD {
    let pw = v3_hi(p);
    let vw = v3_hi(v);
    let gm = prm_gm().x;
    let r0 = sqrt(dot_f(pw, pw));
    let v2 = dot_f(vw, vw);
    let alpha = 2.0 / r0 - v2 / gm;
    if (!(alpha > 0.0)) {
        return KepD(p, v, false);
    }
    let a = 1.0 / alpha;
    let n = sqrt(gm / (a * a * a));
    let sqrt_gma = sqrt(gm * a);
    let ecos0 = 1.0 - r0 / a;
    let esin0 = dot_f(pw, vw) / sqrt_gma;
    let m = n * dt;
    let x = newton_f32(ecos0, esin0, m);
    let sc = sin_cos_f32(x);
    let sh = sin_cos_f32(0.5 * x).x;
    let omc = 2.0 * sh * sh;
    let r = a * (1.0 - ecos0 * sc.y + esin0 * sc.x);
    let fm1 = -((a / r0) * omc);
    let g = dt - (x - sc.x) / n;
    let fdot = -(sqrt_gma / (r * r0)) * sc.x;
    let gm1 = -((a / r) * omc);
    return KepD(v3_add_w(p, fm1 * pw + g * vw), v3_add_w(v, fdot * pw + gm1 * vw), true);
}

struct Scal {
    r0: vec2<f32>,
    ir0: vec2<f32>,
    a: vec2<f32>,
    sqrt_gma: vec2<f32>,
    n: vec2<f32>,
    ecos0: vec2<f32>,
    esin0: vec2<f32>,
    m: vec2<f32>,
    ok: bool,
}

fn ds_scalars(p: V3, v: V3, dt: vec2<f32>) -> Scal {
    let one = ds(1.0);
    let r0 = ds_sqrt(v3_dot(p, p));
    let ir0 = ds_div(one, r0);
    let alpha = ds_sub(ds_mul_f(ir0, 2.0), ds_mul(v3_dot(v, v), prm_inv_gm()));
    let z = ds(0.0);
    if (!(alpha.x > 0.0)) {
        return Scal(z, z, z, z, z, z, z, z, false);
    }
    let a = ds_div(one, alpha);
    let sqrt_gma = ds_sqrt(ds_mul(prm_gm(), a));
    let n = ds_mul(ds_mul(sqrt_gma, alpha), alpha);
    let ecos0 = ds_sub(one, ds_mul(r0, alpha));
    let esin0 = ds_mul(ds_mul(ds_mul(v3_dot(p, v), n), a), prm_inv_gm());
    return Scal(r0, ir0, a, sqrt_gma, n, ecos0, esin0, ds_mul(n, dt), true);
}

// Option B+ drift.
fn drift_bp(p: V3, v: V3, dt: vec2<f32>) -> KepD {
    let k = ds_scalars(p, v, dt);
    if (!k.ok) {
        return KepD(p, v, false);
    }
    let a = k.a.x;
    let r0 = k.r0.x;
    let n = k.n.x;
    let sqrt_gma = k.sqrt_gma.x;
    let ecos0 = k.ecos0.x;
    let esin0 = k.esin0.x;
    let x = newton_f32(ecos0, esin0, k.m.x);
    let sc = sin_cos_f32(x);
    let sh = sin_cos_f32(0.5 * x).x;
    let omc = 2.0 * sh * sh;
    let r = a * (1.0 - ecos0 * sc.y + esin0 * sc.x);
    let fm1 = -((a / r0) * omc);
    let g = dt.x - (x - sc.x) / n;
    let fdot = -(sqrt_gma / (r * r0)) * sc.x;
    let gm1 = -((a / r) * omc);
    let p1 = v3_add(p, v3_add(v3_mul_f(p, fm1), v3_mul_f(v, g)));
    let v1 = v3_add(v, v3_add(v3_mul_f(p, fdot), v3_mul_f(v, gm1)));
    return KepD(p1, v1, true);
}

// Option D drift.
fn drift_d(p: V3, v: V3, dt: vec2<f32>) -> KepD {
    let k = ds_scalars(p, v, dt);
    if (!k.ok) {
        return KepD(p, v, false);
    }
    let one = ds(1.0);
    let ec = k.ecos0.x;
    let es = k.esin0.x;
    let x32 = newton_f32(ec, es, k.m.x);
    let xd = ds(x32);
    let t = ds_sin_cos_small(xd);
    let f = ds_sub(ds_add(ds_sub(xd, ds_mul(k.ecos0, t.s)), ds_mul(k.esin0, t.omc)), k.m);
    let fp = 1.0 - ec * t.c.x + es * t.s.x;
    let d = f.x / fp;
    let x = ds_add_f(xd, -d);
    let s = ds_sub(t.s, ds_mul_f(t.c, d));
    let cs = ds_add(t.c, ds_mul_f(t.s, d));
    let omc = ds_sub(t.omc, ds_mul_f(t.s, d));
    let r = ds_mul(k.a, ds_add(ds_sub(one, ds_mul(k.ecos0, cs)), ds_mul(k.esin0, s)));
    let ir = ds_div(one, r);
    let fm1 = -ds_mul(ds_mul(omc, k.a), k.ir0);
    let g = ds_sub(dt, ds_div(ds_sub(x, s), k.n));
    let fdot = -ds_mul(ds_mul(ds_mul(k.sqrt_gma, ir), k.ir0), s);
    let gm1 = -ds_mul(ds_mul(omc, k.a), ir);
    let p1 = v3_add(p, v3_add(v3_mul(p, fm1), v3_mul(v, g)));
    let v1 = v3_add(v, v3_add(v3_mul(p, fdot), v3_mul(v, gm1)));
    return KepD(p1, v1, true);
}

// Kick of B+ and D: f32 magnitudes along the DS position, DS moon separations.
fn kick_dir(k: u32, p: V3, v: V3) -> V3 {
    let pw = v3_hi(p);
    let r2 = dot_f(pw, pw);
    let r = sqrt(r2);
    let u = pw.z / r;
    let u2 = u * u;
    let q = prm_radius().x / r;
    let q2 = q * q;
    let q4 = q2 * q2;
    let q6 = q4 * q2;
    let p2 = (3.0 * u2 - 1.0) * 0.5;
    let p4 = ((35.0 * u2 - 30.0) * u2 + 3.0) * 0.125;
    let p6 = (((231.0 * u2 - 315.0) * u2 + 105.0) * u2 - 5.0) * 0.0625;
    let d2 = 3.0 * u;
    let d4 = ((35.0 * u2 - 15.0) * u) * 0.5;
    let d6 = (((1386.0 * u2 - 1260.0) * u2 + 210.0) * u) * 0.0625;
    let j2q = prm_j2().x * q2;
    let j4q = prm_j4().x * q4;
    let j6q = prm_j6().x * q6;
    let aa = 3.0 * j2q * p2 + 5.0 * j4q * p4 + 7.0 * j6q * p6;
    let b = j2q * d2 + j4q * d4 + j6q * d6;
    let radial = -aa - u * b;
    let kk = -prm_gm().x / (r2 * r);
    let h = prm_h().x;
    let sxy = kk * radial * h;
    let sz = kk * b * r * h;
    var am = vec3<f32>(0.0, 0.0, 0.0);
    for (var j = 0u; j < prm_n_moons(); j++) {
        let m = moon_ds(k, j);
        let d = vec3<f32>(ds_sub(m.x, p.x).x, ds_sub(m.y, p.y).x, ds_sub(m.z, p.z).x);
        let dd2 = dot_f(d, d);
        am = am + d * (prm_gm_moon(j).x / (dd2 * sqrt(dd2)));
    }
    let small = (am + indirect_f(k)) * h;
    let dv = V3(ds_mul_f(p.x, sxy), ds_mul_f(p.y, sxy), ds_add_f(ds_mul_f(p.z, sxy), sz));
    return v3_add_w(v3_add(v, dv), small);
}

// ---- Entry points: one block of prm_n_steps() steps per particle ----

@compute @workgroup_size(64)
fn step_a(@builtin(global_invocation_id) id: vec3<u32>) {
    ds_one = prm[4].z;
    let i = id.x;
    if (i >= prm_n_particles()) { return; }
    let n = prm_n_steps();
    let gm = prm_gm().x;
    let h = prm_h().x;
    let half = 0.5 * h;
    let pd = load_p(i);
    let vd = load_v(i);
    var o = kepler_f(gm, v3_hi(pd), v3_hi(vd), half);
    var ok = o.ok;
    for (var k = 0u; k < n; k++) {
        let a = kick_acc_f(k, o.p);
        let v = o.v + a * h;
        var dt = h;
        if (k + 1u == n) { dt = half; }
        o = kepler_f(gm, o.p, v, dt);
        ok = ok && o.ok;
    }
    if (ok) {
        let z = 0.0;
        store(i, V3(vec2<f32>(o.p.x, z), vec2<f32>(o.p.y, z), vec2<f32>(o.p.z, z)), V3(vec2<f32>(o.v.x, z), vec2<f32>(o.v.y, z), vec2<f32>(o.v.z, z)));
    }
}

@compute @workgroup_size(64)
fn step_c(@builtin(global_invocation_id) id: vec3<u32>) {
    ds_one = prm[4].z;
    let i = id.x;
    if (i >= prm_n_particles()) { return; }
    let n = prm_n_steps();
    let h = prm_h();
    let half = ds_mul(ds(0.5), h);
    var o = kepler_c(prm_gm(), load_p(i), load_v(i), half);
    var ok = o.ok;
    for (var k = 0u; k < n; k++) {
        let a = kick_acc_c(k, o.p);
        let v = V3(ds_add(o.v.x, ds_mul(a.x, h)), ds_add(o.v.y, ds_mul(a.y, h)), ds_add(o.v.z, ds_mul(a.z, h)));
        var dt = h;
        if (k + 1u == n) { dt = half; }
        o = kepler_c(prm_gm(), o.p, v, dt);
        ok = ok && o.ok;
    }
    if (ok) { store(i, o.p, o.v); }
}

@compute @workgroup_size(64)
fn step_b(@builtin(global_invocation_id) id: vec3<u32>) {
    ds_one = prm[4].z;
    let i = id.x;
    if (i >= prm_n_particles()) { return; }
    let n = prm_n_steps();
    let h = prm_h();
    let half = prm_half();
    var o = drift_b(load_p(i), load_v(i), half.x);
    var ok = o.ok;
    for (var k = 0u; k < n; k++) {
        let a = kick_acc_f(k, v3_hi(o.p));
        let v = v3_add_w(o.v, a * h.x);
        var dt = h;
        if (k + 1u == n) { dt = half; }
        o = drift_b(o.p, v, dt.x);
        ok = ok && o.ok;
    }
    if (ok) { store(i, o.p, o.v); }
}

@compute @workgroup_size(64)
fn step_bp(@builtin(global_invocation_id) id: vec3<u32>) {
    ds_one = prm[4].z;
    let i = id.x;
    if (i >= prm_n_particles()) { return; }
    let n = prm_n_steps();
    let h = prm_h();
    let half = prm_half();
    var o = drift_bp(load_p(i), load_v(i), half);
    var ok = o.ok;
    for (var k = 0u; k < n; k++) {
        let v = kick_dir(k, o.p, o.v);
        var dt = h;
        if (k + 1u == n) { dt = half; }
        o = drift_bp(o.p, v, dt);
        ok = ok && o.ok;
    }
    if (ok) { store(i, o.p, o.v); }
}

@compute @workgroup_size(64)
fn step_d(@builtin(global_invocation_id) id: vec3<u32>) {
    ds_one = prm[4].z;
    let i = id.x;
    if (i >= prm_n_particles()) { return; }
    let n = prm_n_steps();
    let h = prm_h();
    let half = prm_half();
    var o = drift_d(load_p(i), load_v(i), half);
    var ok = o.ok;
    for (var k = 0u; k < n; k++) {
        let v = kick_dir(k, o.p, o.v);
        var dt = h;
        if (k + 1u == n) { dt = half; }
        o = drift_d(o.p, v, dt);
        ok = ok && o.ok;
    }
    if (ok) { store(i, o.p, o.v); }
}
