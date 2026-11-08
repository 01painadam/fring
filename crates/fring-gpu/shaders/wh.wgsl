// The production Wisdom-Holman kernels (task G2, ADR 0006 and 0007): option D
// (science runs) and option C (reference mode). Append to ds.wgsl and trig.wgsl.
// The kernels are copies of the G1 kernels in spike.wgsl with the layout of
// the backend; a test checks that they give the same bits.
// Rust twins: crates/fring-gpu/src/wh.rs (with spike.rs for the drift).
//
// Bindings (group 0):
// 0: state, 6 planes of cap DS values (plane c = x, y, z, vx, vy, vz), each
//    vec2(hi, lo); particle i is at c * cap + i.
// 1: kick table, ROW (36) f32 per step (wh::pack_table): moon j at 6 j (DS
//    x, y, z), indirect term DS at 24, indirect term f32 (scaled by h/f32(h),
//    for option D) at 30.
// 2: run constants, see prm_* below (11 vec4; ICR-006 radii in 9, flag in 10).
// 3: per-dispatch control, 2 vec4<u32> (32 bytes): n_steps, n_particles, cap,
//    n_moons; then first_step (the global step number of step 0 of the block;
//    E1-GPU), 3 words unused.
// 4: status (ICR-006, ICR-004), 8 + 2 cap words: status[0] = smallest index of a dropped
//    particle (0xFFFFFFFF if none), status[1] = number dropped for an unbound
//    or invalid orbit, status[2 + k] = number dropped inside moon k (k < 4;
//    words 6 and 7 are reserved), status[8 + i] = reason of particle i (0
//    live, 1 unbound or invalid orbit, 2 + k inside moon k), and
//    status[8 + cap + i] = drop_step of particle i (the 0-based global step in
//    which the loss was found, valid when the reason is not 0; E1-GPU). The
//    arc log (arc.wgsl) follows at 8 + 2 cap.

@group(0) @binding(0) var<storage, read_write> st: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> tab: array<f32>;
@group(0) @binding(2) var<uniform> prm: array<vec4<f32>, 11>;
@group(0) @binding(3) var<uniform> ctl: array<vec4<u32>, 2>;
@group(0) @binding(4) var<storage, read_write> status: array<atomic<u32>>;

fn prm_gm() -> vec2<f32> { return prm[0].xy; }
fn prm_h() -> vec2<f32> { return prm[0].zw; }
fn prm_inv_gm() -> vec2<f32> { return prm[1].xy; }
fn prm_radius() -> vec2<f32> { return prm[1].zw; }
fn prm_j2() -> vec2<f32> { return prm[2].xy; }
fn prm_j4() -> vec2<f32> { return prm[2].zw; }
fn prm_j6() -> vec2<f32> { return prm[3].xy; }
fn prm_half() -> vec2<f32> { return prm[3].zw; }
fn prm_one() -> f32 { return prm[4].x; }
// The f32 constants of the D kick, scaled by h / f32(h) (ADR 0007).
fn prm_j2s() -> f32 { return prm[5].x; }
fn prm_j4s() -> f32 { return prm[5].y; }
fn prm_j6s() -> f32 { return prm[5].z; }
fn prm_gm_moon_s(j: u32) -> f32 { return prm[6][j]; }
fn prm_gm_moon_ds(j: u32) -> vec2<f32> {
    let q = prm[7u + (j >> 1u)];
    if ((j & 1u) == 0u) { return q.xy; }
    return q.zw;
}
// Moon radius squared, km^2, of moon j (0 = not checked); the rule flag (ICR-006).
fn prm_r2(j: u32) -> f32 { return prm[9][j]; }
fn prm_rule() -> f32 { return prm[10].x; }
fn prm_n_steps() -> u32 { return ctl[0].x; }
fn prm_n_particles() -> u32 { return ctl[0].y; }
fn prm_cap() -> u32 { return ctl[0].z; }
fn prm_n_moons() -> u32 { return ctl[0].w; }
fn prm_first_step() -> u32 { return ctl[1].x; }

const ROW: u32 = 36u;

fn load_p(i: u32) -> V3 {
    let c = prm_cap();
    return V3(st[i], st[c + i], st[2u * c + i]);
}
fn load_v(i: u32) -> V3 {
    let c = prm_cap();
    return V3(st[3u * c + i], st[4u * c + i], st[5u * c + i]);
}
fn store(i: u32, p: V3, v: V3) {
    let c = prm_cap();
    st[i] = p.x; st[c + i] = p.y; st[2u * c + i] = p.z;
    st[3u * c + i] = v.x; st[4u * c + i] = v.y; st[5u * c + i] = v.z;
}

fn moon_ds(k: u32, j: u32) -> V3 {
    let b = ROW * k + 6u * j;
    return V3(vec2<f32>(tab[b], tab[b + 1u]), vec2<f32>(tab[b + 2u], tab[b + 3u]), vec2<f32>(tab[b + 4u], tab[b + 5u]));
}
fn indirect_ds(k: u32) -> V3 {
    let b = ROW * k + 24u;
    return V3(vec2<f32>(tab[b], tab[b + 1u]), vec2<f32>(tab[b + 2u], tab[b + 3u]), vec2<f32>(tab[b + 4u], tab[b + 5u]));
}
fn indirect_f(k: u32) -> vec3<f32> {
    let b = ROW * k + 30u;
    return vec3<f32>(tab[b], tab[b + 1u], tab[b + 2u]);
}

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
        am = v3_add(am, v3_mul(d, ds_div(prm_gm_moon_ds(j), ds_mul(d2, ds_sqrt(d2)))));
    }
    return v3_add(v3_add(accel_j_c(p), am), indirect_ds(k));
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
    let j2q = prm_j2s() * q2;
    let j4q = prm_j4s() * q4;
    let j6q = prm_j6s() * q6;
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
        am = am + d * (prm_gm_moon_s(j) / (dd2 * sqrt(dd2)));
    }
    let small = (am + indirect_f(k)) * h;
    let dv = V3(ds_mul_f(p.x, sxy), ds_mul_f(p.y, sxy), ds_add_f(ds_mul_f(p.z, sxy), sz));
    return v3_add_w(v3_add(v, dv), small);
}

// Kick of option D+ (G2, ADR 0006 addendum): as kick_dir, but the J2 term is
// computed in DS with h in DS (the f32 magnitudes of D made a mean-motion
// error that grew to 1e-4 rad in the tail). J4, J6 and the moons as in D.
fn kick_dir_plus(k: u32, p: V3, v: V3) -> V3 {
    let r2d = v3_dot(p, p);
    let rd = ds_sqrt(r2d);
    let ud = ds_div(p.z, rd);
    let u2d = ds_mul(ud, ud);
    let qd = ds_div(prm_radius(), rd);
    let q2d = ds_mul(qd, qd);
    let j2q = ds_mul(prm_j2(), q2d);
    let p2d = ds_mul(ds_sub(ds_mul(ds(3.0), u2d), ds(1.0)), ds(0.5));
    let d2d = ds_mul(ds(3.0), ud);
    let aa2 = ds_mul(ds_mul(ds(3.0), j2q), p2d);
    let b2 = ds_mul(j2q, d2d);
    let radial2 = ds_sub(ds_sub(ds(0.0), aa2), ds_mul(ud, b2));
    let kkd = ds_div(-prm_gm(), ds_mul(r2d, rd));
    let hd = prm_h();
    let sxy2 = ds_mul(ds_mul(kkd, radial2), hd);
    let sz2 = ds_mul(ds_mul(ds_mul(kkd, b2), rd), hd);

    let pw = v3_hi(p);
    let r2 = dot_f(pw, pw);
    let r = sqrt(r2);
    let u = pw.z / r;
    let u2 = u * u;
    let q = prm_radius().x / r;
    let q2 = q * q;
    let q4 = q2 * q2;
    let q6 = q4 * q2;
    let p4 = ((35.0 * u2 - 30.0) * u2 + 3.0) * 0.125;
    let p6 = (((231.0 * u2 - 315.0) * u2 + 105.0) * u2 - 5.0) * 0.0625;
    let d4 = ((35.0 * u2 - 15.0) * u) * 0.5;
    let d6 = (((1386.0 * u2 - 1260.0) * u2 + 210.0) * u) * 0.0625;
    let j4q = prm_j4s() * q4;
    let j6q = prm_j6s() * q6;
    let aa = 5.0 * j4q * p4 + 7.0 * j6q * p6;
    let b = j4q * d4 + j6q * d6;
    let radial = -aa - u * b;
    let kk = -prm_gm().x / (r2 * r);
    let h = prm_h().x;
    let sxy = ds_add_f(sxy2, kk * radial * h);
    let sz = ds_add_f(sz2, kk * b * r * h);
    var am = vec3<f32>(0.0, 0.0, 0.0);
    for (var j = 0u; j < prm_n_moons(); j++) {
        let m = moon_ds(k, j);
        let d = vec3<f32>(ds_sub(m.x, p.x).x, ds_sub(m.y, p.y).x, ds_sub(m.z, p.z).x);
        let dd2 = dot_f(d, d);
        am = am + d * (prm_gm_moon_s(j) / (dd2 * sqrt(dd2)));
    }
    let small = (am + indirect_f(k)) * h;
    let dv = V3(ds_mul(p.x, sxy), ds_mul(p.y, sxy), ds_add(ds_mul(p.z, sxy), sz));
    return v3_add_w(v3_add(v, dv), small);
}

// ---- ICR-006: moon-radius rule. Rust twins: wh::moon_rel, wh::segment_min_d2 ----

// Position of the particle relative to moon j in table row `row`, p - m, f32 of
// the DS difference (the large equal parts cancel in DS first).
fn moon_rel(row: u32, j: u32, p: V3) -> vec3<f32> {
    let m = moon_ds(row, j);
    return vec3<f32>(ds_sub(p.x, m.x).x, ds_sub(p.y, m.y).x, ds_sub(p.z, m.z).x);
}

// True if the row holds a moon position (the host writes 1e30 into x of moon 0
// for a time outside the ephemeris).
fn row_valid(row: u32) -> bool { return tab[ROW * row] < 1.0e20; }

// Smallest squared distance from the origin to the segment a -> b.
fn seg_min_d2(a: vec3<f32>, b: vec3<f32>) -> f32 {
    let d = b - a;
    let dd = dot_f(d, d);
    var s = 0.0;
    if (dd > 0.0) { s = clamp(-dot_f(a, d) / dd, 0.0, 1.0); }
    let c = a + d * s;
    return dot_f(c, c);
}

// Relative positions to each moon at the end of the last checked drift.
var<private> rprev: array<vec3<f32>, 4>;
var<private> rvalid: bool;

// Starts the rule at the sync time at the start of the block (table row `row`).
fn rule_start(row: u32, p: V3) {
    rvalid = row_valid(row);
    if (rvalid) {
        for (var j = 0u; j < prm_n_moons(); j++) { rprev[j] = moon_rel(row, j, p); }
    }
}

// The drift that ended at p, where the moons are at table row `row`. Returns 0
// if no moon was hit, else 1 + the lowest moon index hit. A drift with no
// moon position at either end is skipped.
fn rule_step(row: u32, p: V3) -> u32 {
    let now_valid = row_valid(row);
    var hit = 0u;
    if (now_valid) {
        for (var j = 0u; j < prm_n_moons(); j++) {
            let b = moon_rel(row, j, p);
            let r2 = prm_r2(j);
            if (rvalid && hit == 0u && r2 > 0.0 && seg_min_d2(rprev[j], b) < r2) { hit = j + 1u; }
            rprev[j] = b;
        }
    }
    rvalid = now_valid;
    return hit;
}

// ---- Entry points: one block of prm_n_steps() steps per particle ----

// ICR-006: a particle with a reason other than 0 is frozen. Its thread
// returns at once (one u32 load per particle per block).
fn is_dropped(i: u32) -> bool { return atomicLoad(&status[8u + i]) != 0u; }

// Sets the reason of particle i (1 unbound or invalid orbit, 2 + k inside
// moon k), counts it in word `reason`, and lowers word 0. Each thread calls
// it once for its own particle, so the counts are exact.
//
// `seg` is the drift in which the loss was found: 0 for the half drift before
// kick 0, k + 1 for the drift after kick k (the last one is the half drift to
// the end of the block). The drop_step word is first_step + max(seg - 1, 0):
// the drift after kick k belongs to step k. The GPU convention: the end half
// drift of a block belongs to the last step of the block (first + n - 1); the
// CPU, which merges the end half drift with the start half drift of the next
// block, can give one step later (E1-CPU handoff). Written only here, so a
// live particle costs nothing.
fn drop_particle(i: u32, reason: u32, seg: u32) {
    atomicStore(&status[8u + prm_cap() + i], prm_first_step() + max(seg, 1u) - 1u);
    atomicStore(&status[8u + i], reason);
    atomicAdd(&status[reason], 1u);
    atomicMin(&status[0], i);
}

@compute @workgroup_size(WG_SIZE)
fn step_d(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    ds_one = prm_one();
    let i = id.y * nw.x * u32(WG_SIZE) + id.x;
    if (i >= prm_n_particles()) { return; }
    if (is_dropped(i)) { return; }
    let n = prm_n_steps();
    let h = prm_h();
    let half = prm_half();
    var o = drift_d(load_p(i), load_v(i), half);
    var ok = o.ok;
    var seg = 0u;
    for (var k = 0u; k < n; k++) {
        if (!ok) { break; }
        seg = k + 1u;
        let v = kick_dir(k, o.p, o.v);
        var dt = h;
        if (k + 1u == n) { dt = half; }
        o = drift_d(o.p, v, dt);
        ok = ok && o.ok;
    }
    if (ok) { store(i, o.p, o.v); } else { drop_particle(i, 1u, seg); }
}

@compute @workgroup_size(WG_SIZE)
fn step_dp(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    ds_one = prm_one();
    let i = id.y * nw.x * u32(WG_SIZE) + id.x;
    if (i >= prm_n_particles()) { return; }
    if (is_dropped(i)) { return; }
    let n = prm_n_steps();
    let h = prm_h();
    let half = prm_half();
    var o = drift_d(load_p(i), load_v(i), half);
    var ok = o.ok;
    var seg = 0u;
    for (var k = 0u; k < n; k++) {
        if (!ok) { break; }
        seg = k + 1u;
        let v = kick_dir_plus(k, o.p, o.v);
        var dt = h;
        if (k + 1u == n) { dt = half; }
        o = drift_d(o.p, v, dt);
        ok = ok && o.ok;
    }
    if (ok) { store(i, o.p, o.v); } else { drop_particle(i, 1u, seg); }
}

// As step_d, with the moon-radius rule (ICR-006): a particle whose path comes
// within the radius of moon k, in any drift of the block, is dropped with
// reason 2 + k and not stored. Table rows n and n + 1 hold the moons at the
// sync times at the start and at the end of the block.
@compute @workgroup_size(WG_SIZE)
fn step_d_r(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    ds_one = prm_one();
    let i = id.y * nw.x * u32(WG_SIZE) + id.x;
    if (i >= prm_n_particles()) { return; }
    if (is_dropped(i)) { return; }
    let n = prm_n_steps();
    let h = prm_h();
    let half = prm_half();
    let p0 = load_p(i);
    rule_start(n, p0);
    var o = drift_d(p0, load_v(i), half);
    var ok = o.ok;
    var reason = 0u;
    var seg = 0u;
    if (ok) {
        let hit = rule_step(0u, o.p);
        if (hit != 0u) { reason = hit + 1u; }
    }
    for (var k = 0u; k < n; k++) {
        if (!ok || reason != 0u) { break; }
        seg = k + 1u;
        let v = kick_dir(k, o.p, o.v);
        var dt = h;
        var row = k + 1u;
        if (k + 1u == n) { dt = half; row = n + 1u; }
        o = drift_d(o.p, v, dt);
        ok = ok && o.ok;
        if (ok) {
            let hit = rule_step(row, o.p);
            if (hit != 0u) { reason = hit + 1u; }
        }
    }
    if (!ok) {
        drop_particle(i, 1u, seg);
    } else if (reason != 0u) {
        drop_particle(i, reason, seg);
    } else {
        store(i, o.p, o.v);
    }
}

// As step_dp, with the moon-radius rule (ICR-006): a particle whose path comes
// within the radius of moon k, in any drift of the block, is dropped with
// reason 2 + k and not stored. Table rows n and n + 1 hold the moons at the
// sync times at the start and at the end of the block.
@compute @workgroup_size(WG_SIZE)
fn step_dp_r(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    ds_one = prm_one();
    let i = id.y * nw.x * u32(WG_SIZE) + id.x;
    if (i >= prm_n_particles()) { return; }
    if (is_dropped(i)) { return; }
    let n = prm_n_steps();
    let h = prm_h();
    let half = prm_half();
    let p0 = load_p(i);
    rule_start(n, p0);
    var o = drift_d(p0, load_v(i), half);
    var ok = o.ok;
    var reason = 0u;
    var seg = 0u;
    if (ok) {
        let hit = rule_step(0u, o.p);
        if (hit != 0u) { reason = hit + 1u; }
    }
    for (var k = 0u; k < n; k++) {
        if (!ok || reason != 0u) { break; }
        seg = k + 1u;
        let v = kick_dir_plus(k, o.p, o.v);
        var dt = h;
        var row = k + 1u;
        if (k + 1u == n) { dt = half; row = n + 1u; }
        o = drift_d(o.p, v, dt);
        ok = ok && o.ok;
        if (ok) {
            let hit = rule_step(row, o.p);
            if (hit != 0u) { reason = hit + 1u; }
        }
    }
    if (!ok) {
        drop_particle(i, 1u, seg);
    } else if (reason != 0u) {
        drop_particle(i, reason, seg);
    } else {
        store(i, o.p, o.v);
    }
}

@compute @workgroup_size(WG_SIZE)
fn step_c(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    ds_one = prm_one();
    let i = id.y * nw.x * u32(WG_SIZE) + id.x;
    if (i >= prm_n_particles()) { return; }
    let n = prm_n_steps();
    let h = prm_h();
    let half = prm_half();
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
    if (ok) { store(i, o.p, o.v); } else { atomicMin(&status[0], i); }
}

// ---- Test entry points: one function applied to the state of each particle
// (state in, state out), for the twin tests of tests/gpu_wh.rs. dt = half step.

fn z2() -> vec2<f32> { return vec2<f32>(0.0, 0.0); }

@compute @workgroup_size(WG_SIZE)
fn t_ds_scalars(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    ds_one = prm_one();
    let i = id.y * nw.x * u32(WG_SIZE) + id.x;
    if (i >= prm_n_particles()) { return; }
    let s = ds_scalars(load_p(i), load_v(i), prm_half());
    store(i, V3(s.r0, s.a, s.n), V3(s.ecos0, s.esin0, s.m));
}

// x = newton_f32(ecos0, esin0, m) from the hi parts of x, y, z; result in vx.
@compute @workgroup_size(WG_SIZE)
fn t_newton_f32(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    ds_one = prm_one();
    let i = id.y * nw.x * u32(WG_SIZE) + id.x;
    if (i >= prm_n_particles()) { return; }
    let p = load_p(i);
    let x = newton_f32(p.x.x, p.y.x, p.z.x);
    store(i, p, V3(vec2<f32>(x, 0.0), z2(), z2()));
}

@compute @workgroup_size(WG_SIZE)
fn t_drift_d(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    ds_one = prm_one();
    let i = id.y * nw.x * u32(WG_SIZE) + id.x;
    if (i >= prm_n_particles()) { return; }
    let o = drift_d(load_p(i), load_v(i), prm_half());
    store(i, o.p, o.v);
}

@compute @workgroup_size(WG_SIZE)
fn t_kepler_c(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    ds_one = prm_one();
    let i = id.y * nw.x * u32(WG_SIZE) + id.x;
    if (i >= prm_n_particles()) { return; }
    let o = kepler_c(prm_gm(), load_p(i), load_v(i), prm_half());
    store(i, o.p, o.v);
}

@compute @workgroup_size(WG_SIZE)
fn t_accel_j_c(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    ds_one = prm_one();
    let i = id.y * nw.x * u32(WG_SIZE) + id.x;
    if (i >= prm_n_particles()) { return; }
    let p = load_p(i);
    store(i, p, accel_j_c(p));
}

@compute @workgroup_size(WG_SIZE)
fn t_kick_acc_c(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    ds_one = prm_one();
    let i = id.y * nw.x * u32(WG_SIZE) + id.x;
    if (i >= prm_n_particles()) { return; }
    let p = load_p(i);
    store(i, p, kick_acc_c(0u, p));
}

@compute @workgroup_size(WG_SIZE)
fn t_kick_dir(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    ds_one = prm_one();
    let i = id.y * nw.x * u32(WG_SIZE) + id.x;
    if (i >= prm_n_particles()) { return; }
    let p = load_p(i);
    store(i, p, kick_dir(0u, p, load_v(i)));
}

@compute @workgroup_size(WG_SIZE)
fn t_kick_dir_plus(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    ds_one = prm_one();
    let i = id.y * nw.x * u32(WG_SIZE) + id.x;
    if (i >= prm_n_particles()) { return; }
    let p = load_p(i);
    store(i, p, kick_dir_plus(0u, p, load_v(i)));
}


// ICR-006 test entry points. t_moon_rel: p - m for moon 0 of table row 0, in vx, vy, vz (hi).
@compute @workgroup_size(WG_SIZE)
fn t_moon_rel(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    ds_one = prm_one();
    let i = id.y * nw.x * u32(WG_SIZE) + id.x;
    if (i >= prm_n_particles()) { return; }
    let p = load_p(i);
    let r = moon_rel(0u, 0u, p);
    store(i, p, V3(vec2<f32>(r.x, 0.0), vec2<f32>(r.y, 0.0), vec2<f32>(r.z, 0.0)));
}

// t_segment_min: a = hi of (x, y, z), b = hi of (vx, vy, vz); result in vx.
@compute @workgroup_size(WG_SIZE)
fn t_segment_min(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    ds_one = prm_one();
    let i = id.y * nw.x * u32(WG_SIZE) + id.x;
    if (i >= prm_n_particles()) { return; }
    let p = load_p(i);
    let v = load_v(i);
    let d2 = seg_min_d2(v3_hi(p), v3_hi(v));
    store(i, p, V3(vec2<f32>(d2, 0.0), z2(), z2()));
}
