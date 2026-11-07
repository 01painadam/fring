// Task D1: the co-moving arc wrap (docs/PLAN.md 7.6). Appended after wh.wgsl.
// Rust twin: crates/fring-gpu/src/arc.rs (`wrap_ds`); the f64 definition is
// fring_model::physics::arc.
//
// Arc data in the kick table (layout in arc.rs): in words 33 and 34 of every
// kick row k and of the extra row n: cos and sin of the inertial longitude of
// the arc centre at the time of that row. In the extra row n + 1: cos(dtheta)
// and sin(dtheta) as DS (words 0 to 3), cos and sin of dtheta / 2 (4, 5), and
// cos and sin of the arc centre at the sync time at the start of the block
// (6, 7).
//
// A wrap site is applied at the start of the block (site 0), before each kick
// k (site k + 1) and at the end of the block (site n + 1). A particle outside
// the wedge is rotated about z by -/+ dtheta in position and velocity (DS).
// For chaos, the parent logs its rotations in the status buffer (after the
// 8 + cap words: ARC_LW words per particle: the number of events, then events
// (site << 2) | code, code 1 = rotated by -dtheta, 2 = by +dtheta) and the
// shadow replays them, so both always take the same k.

const ARC_LW: u32 = 8u;

struct ArcOut {
    p: V3,
    v: V3,
    d0: i32,
    d1: i32,
    ok: bool,
}

fn arc_cw(n: u32) -> vec2<f32> {
    let b = ROW * (n + 1u);
    return vec2<f32>(tab[b], tab[b + 1u]);
}

fn arc_sw(n: u32) -> vec2<f32> {
    let b = ROW * (n + 1u);
    return vec2<f32>(tab[b + 2u], tab[b + 3u]);
}

// The vector (x, y) rotated by the angle with cosine c and sine s (DS).
fn arc_rot(a: V3, c: vec2<f32>, s: vec2<f32>) -> V3 {
    return V3(ds_sub(ds_mul(a.x, c), ds_mul(a.y, s)), ds_add(ds_mul(a.x, s), ds_mul(a.y, c)), a.z);
}

// Wraps (p, v) into the wedge whose centre has cos, sin = psi. `n` is the
// number of steps of the block (the constants are in table row n + 1).
fn arc_wrap(p: V3, v: V3, psi: vec2<f32>, n: u32) -> ArcOut {
    let b = ROW * (n + 1u);
    let cw = arc_cw(n);
    let sw = arc_sw(n);
    let ch = tab[b + 4u];
    let sh = tab[b + 5u];
    var q = p;
    var u = v;
    var d0 = 0;
    var d1 = 0;
    for (var it = 0u; it < 3u; it++) {
        let x = q.x.x * psi.x + q.y.x * psi.y;
        let y = q.y.x * psi.x - q.x.x * psi.y;
        // 1e-6 = fring_model::physics::arc::EDGE_TOL (see there; the angular slack is EDGE_TOL * (cos h + sin h)).
        if (x * sh - abs(y) * ch >= -1.0e-6 * (abs(x) + abs(y))) {
            return ArcOut(q, u, d0, d1, true);
        }
        if (it == 2u) { break; }
        var dir = 1;
        var s = sw;
        if (y >= 0.0) { dir = -1; s = -sw; }
        q = arc_rot(q, cw, s);
        u = arc_rot(u, cw, s);
        if (it == 0u) { d0 = dir; } else { d1 = dir; }
    }
    return ArcOut(q, u, d0, d1, false);
}

fn arc_psi(row: u32) -> vec2<f32> {
    return vec2<f32>(tab[ROW * row + 33u], tab[ROW * row + 34u]);
}

fn arc_site_psi(site: u32, n: u32) -> vec2<f32> {
    if (site == 0u) { return vec2<f32>(tab[ROW * (n + 1u) + 6u], tab[ROW * (n + 1u) + 7u]); }
    if (site == n + 1u) { return arc_psi(n); }
    return arc_psi(site - 1u);
}

fn arc_code(d: i32) -> u32 {
    if (d < 0) { return 1u; }
    return 2u;
}

// One wrap site. role 0: wrap; role 1: wrap and log; role 2: replay the log
// of the parent. `cnt` is the number of logged events (role 1) or the index of
// the next event to replay (role 2). Returns false if the particle needs more
// than 2 rotations or the log is full.
fn arc_site(i: u32, role: u32, site: u32, n: u32, p: ptr<function, V3>, v: ptr<function, V3>, cnt: ptr<function, u32>) -> bool {
    let lb = 8u + prm_cap() + i * ARC_LW;
    if (role == 2u) {
        let total = atomicLoad(&status[lb]);
        loop {
            if (*cnt >= total) { break; }
            let e = atomicLoad(&status[lb + 1u + *cnt]);
            if ((e >> 2u) != site) { break; }
            var s = arc_sw(n);
            if ((e & 3u) == 1u) { s = -s; }
            *p = arc_rot(*p, arc_cw(n), s);
            *v = arc_rot(*v, arc_cw(n), s);
            *cnt = *cnt + 1u;
        }
        return true;
    }
    let w = arc_wrap(*p, *v, arc_site_psi(site, n), n);
    if (!w.ok) { return false; }
    *p = w.p;
    *v = w.v;
    if (role == 1u) {
        if (w.d0 != 0) {
            if (*cnt >= ARC_LW - 1u) { return false; }
            atomicStore(&status[lb + 1u + *cnt], (site << 2u) | arc_code(w.d0));
            *cnt = *cnt + 1u;
        }
        if (w.d1 != 0) {
            if (*cnt >= ARC_LW - 1u) { return false; }
            atomicStore(&status[lb + 1u + *cnt], (site << 2u) | arc_code(w.d1));
            *cnt = *cnt + 1u;
        }
    }
    return true;
}

// The block kernel with the arc wrap. role: 0 plain, 1 parent with log
// (chaos), 2 shadow replaying the log. plus: option D+ kick.
fn arc_run(id: vec3<u32>, nw: vec3<u32>, plus: bool, role: u32) {
    ds_one = prm_one();
    let i = id.y * nw.x * u32(WG_SIZE) + id.x;
    if (i >= prm_n_particles()) { return; }
    if (is_dropped(i)) { return; }
    let n = prm_n_steps();
    let h = prm_h();
    let half = prm_half();
    var p = load_p(i);
    var v = load_v(i);
    var cnt = 0u;
    var ok = arc_site(i, role, 0u, n, &p, &v, &cnt);
    var o = KepD(p, v, ok);
    if (ok) { o = drift_d(p, v, half); ok = o.ok; }
    for (var k = 0u; k < n; k++) {
        if (!ok) { break; }
        var pp = o.p;
        var vv = o.v;
        if (!arc_site(i, role, k + 1u, n, &pp, &vv, &cnt)) { ok = false; break; }
        var kv: V3;
        if (plus) { kv = kick_dir_plus(k, pp, vv); } else { kv = kick_dir(k, pp, vv); }
        var dt = h;
        if (k + 1u == n) { dt = half; }
        o = drift_d(pp, kv, dt);
        ok = ok && o.ok;
    }
    if (ok) {
        var pe = o.p;
        var ve = o.v;
        ok = arc_site(i, role, n + 1u, n, &pe, &ve, &cnt);
        if (ok) {
            store(i, pe, ve);
            if (role == 1u) { atomicStore(&status[8u + prm_cap() + i * ARC_LW], cnt); }
        }
    }
    if (!ok) { drop_particle(i, 1u); }
}

@compute @workgroup_size(WG_SIZE)
fn step_d_arc(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    arc_run(id, nw, false, 0u);
}

@compute @workgroup_size(WG_SIZE)
fn step_d_arc_log(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    arc_run(id, nw, false, 1u);
}

@compute @workgroup_size(WG_SIZE)
fn step_d_arc_replay(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    arc_run(id, nw, false, 2u);
}

@compute @workgroup_size(WG_SIZE)
fn step_dp_arc(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    arc_run(id, nw, true, 0u);
}

@compute @workgroup_size(WG_SIZE)
fn step_dp_arc_log(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    arc_run(id, nw, true, 1u);
}

@compute @workgroup_size(WG_SIZE)
fn step_dp_arc_replay(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    arc_run(id, nw, true, 2u);
}

// Test entry point: one wrap site 1 (kick row 0 angle) of the state of each
// particle; the table has n steps and the constants in row n + 1 (as a block
// of n steps).
@compute @workgroup_size(WG_SIZE)
fn t_arc_wrap(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    ds_one = prm_one();
    let i = id.y * nw.x * u32(WG_SIZE) + id.x;
    if (i >= prm_n_particles()) { return; }
    var p = load_p(i);
    var v = load_v(i);
    var cnt = 0u;
    if (arc_site(i, 0u, 1u, prm_n_steps(), &p, &v, &cnt)) {
        store(i, p, v);
    } else {
        // Marks a failure: NaN in x (the test reads it).
        store(i, V3(vec2<f32>(bitcast<f32>(0x7fc00000u), 0.0), p.y, p.z), v);
    }
}
