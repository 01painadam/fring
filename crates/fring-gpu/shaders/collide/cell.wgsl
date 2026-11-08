// Stochastic collisions on the cells of K5-BIN (task K5): one thread per cell.
// Specification: crates/fring-collide/SPEC.md 5.2 to 5.5, 9, 12. Rust twin of
// the per-cell algorithm: `collide::cell_twin` in crates/fring-gpu/src/collide.rs
// (the CPU model is fring-collide `StochasticModel::process_cell`).
//
// Input: the sorted ids and keys of GpuBinner (or the same layout made by a
// test): sorted[p] is a particle id (0xFFFFFFFF after the last binned one),
// skeys[p] its cell key; a cell is a run of equal keys, its members in id
// order. The thread of the first position of a run does the whole cell:
//  1. u = v - v_circ phi_hat in DS from the state (or u = v in box mode); the
//     mean ubar in DS; w = u - ubar in f32 (SPEC 12) into the scratch uw[p+q].
//  2. g_max = 2 max |w|; M = N (N-1)/2 k_rate g_max; floor(M) candidates plus
//     one with probability frac(M); Philox draws as in SPEC 5.5, with the f32
//     uniform u01_f32 of one word (word 0: rounding of M; draw 1+2k: words 0, 1
//     give i, j and word 2 the acceptance; draw 2+2k: words 0 and 2 give u1, u2).
//  3. An accepted pair gets the impulse of pair.wgsl: w_i, w_j and the DS state
//     velocity of both members change by the same f32 delta.
// The order of the cells does not matter (the RNG is indexed by the smallest id).
// The tally has integer sums (atomics, so exact in any order) and the energy
// loss: one f32 per cell in en[p], then summed in DS in a fixed order by
// col_sum1 and col_sum2 (so the same input gives the same bits).
//
// A cell that needs 32,767 or more candidates is skipped and counted in
// tot[T_TOOFULL]; the host returns the error. (The CPU stops at the first such
// cell; here the other cells are processed first. The run stops either way.)
//
// Bindings (group 0):
// 0: st      state planes (6 planes of cap DS values: x y z vx vy vz), read_write
// 1: sorted  ids, read
// 2: skeys   keys, read
// 3: uw      scratch vec4<f32> per sorted position
// 4: en      2 n entries (loss, then sum of squared losses of the cell that starts at p), then 4 floats per chunk of 256
// 5: tot     16 atomic words, see T_*
// 6: cp      uniform 5 vec4<f32>: [0] one, -, eps, vc  [1] p, -, -, -  [2] GM, R (DS)  [3] J2, J4 (DS)  [4] J6 (DS)
// 7: ctl     uniform vec4<u32>: n, cap, flags (bit 0: box mode, bit 1: power law), number of chunks
// 8: dyn     uniform 2 vec4<u32>: [0] key low, key high, step low, step high  [1] k_rate bits, ...
// k_rate = W sigma dt / V_c.

@group(0) @binding(0) var<storage, read_write> st: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> sorted: array<u32>;
@group(0) @binding(2) var<storage, read> skeys: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> uw: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> en: array<f32>;
@group(0) @binding(5) var<storage, read_write> tot: array<atomic<u32>>;
@group(0) @binding(6) var<uniform> cp: array<vec4<f32>, 5>;
@group(0) @binding(7) var<uniform> ctl: vec4<u32>;
@group(0) @binding(8) var<uniform> dyn_: array<vec4<u32>, 2>;

const NONE: u32 = 0xFFFFFFFFu;
const COL_WG: u32 = 64u;
const CHUNK: u32 = 256u;
const MAX_CANDIDATES: u32 = 32767u;
// Words of tot.
const T_CAND: u32 = 0u;
const T_COLL: u32 = 1u;
const T_GMAX: u32 = 2u;
const T_OCC: u32 = 3u;
const T_OCC_SUM: u32 = 4u;
const T_OCC_SQ_LO: u32 = 5u;
const T_OCC_SQ_HI: u32 = 6u;
const T_LONELY: u32 = 7u;
const T_OVERFULL: u32 = 8u;
const T_TOOFULL: u32 = 9u;
const T_LOSS_HI: u32 = 10u;
const T_LOSS_LO: u32 = 11u;
const T_LOSS2_HI: u32 = 12u;
const T_LOSS2_LO: u32 = 13u;

fn index_of(id: vec3<u32>, nw: vec3<u32>) -> u32 {
    return id.y * nw.x * COL_WG + id.x;
}

fn load_u(pid: u32) -> U3 {
    let cap = ctl.y;
    let vx = st[3u * cap + pid];
    let vy = st[4u * cap + pid];
    let vz = st[5u * cap + pid];
    if ((ctl.z & 1u) != 0u) { return U3(vx, vy, vz); }
    return peculiar_ds(st[pid], st[cap + pid], vx, vy, vz, cp[2].xy, cp[2].zw, cp[3].xy, cp[3].zw, cp[4].xy);
}

fn add_velocity(pid: u32, d: vec3<f32>) {
    let cap = ctl.y;
    st[3u * cap + pid] = ds_add_f(st[3u * cap + pid], d.x);
    st[4u * cap + pid] = ds_add_f(st[4u * cap + pid], d.y);
    st[5u * cap + pid] = ds_add_f(st[5u * cap + pid], d.z);
}

@compute @workgroup_size(64)
fn col_clear(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < 16u) { atomicStore(&tot[id.x], 0u); }
}

@compute @workgroup_size(64)
fn col_cells(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    ds_one = cp[0].x;
    let p = index_of(id, nw);
    let n = ctl.x;
    if (p >= n) { return; }
    en[p] = 0.0;
    en[n + p] = 0.0;
    let first = sorted[p];
    if (first == NONE) { return; }
    let key = skeys[p];
    if (p > 0u) {
        let prev = skeys[p - 1u];
        if (prev.x == key.x && prev.y == key.y) { return; }
    }
    var e = p + 1u;
    while (e < n && sorted[e] != NONE) {
        let k2 = skeys[e];
        if (k2.x != key.x || k2.y != key.y) { break; }
        e = e + 1u;
    }
    let nc = e - p;
    atomicAdd(&tot[T_OCC], 1u);
    atomicAdd(&tot[T_OCC_SUM], nc);
    let sq = mulhilo(nc, nc);
    let old = atomicAdd(&tot[T_OCC_SQ_LO], sq.y);
    atomicAdd(&tot[T_OCC_SQ_HI], sq.x + select(0u, 1u, old + sq.y < old));
    if (nc < 2u) {
        atomicAdd(&tot[T_LONELY], nc);
        return;
    }
    // Mean peculiar velocity in DS.
    var sx = vec2<f32>(0.0);
    var sy = vec2<f32>(0.0);
    var sz = vec2<f32>(0.0);
    for (var q = 0u; q < nc; q = q + 1u) {
        let u = load_u(sorted[p + q]);
        sx = ds_add(sx, u.x);
        sy = ds_add(sy, u.y);
        sz = ds_add(sz, u.z);
    }
    let cnt = ds(f32(nc));
    let mx = ds_div(sx, cnt);
    let my = ds_div(sy, cnt);
    let mz = ds_div(sz, cnt);
    // w = u - ubar in f32; g_max = 2 max |w|.
    var wmax2 = 0.0;
    for (var q = 0u; q < nc; q = q + 1u) {
        let u = load_u(sorted[p + q]);
        let w = vec3<f32>(ds_sub(u.x, mx).x, ds_sub(u.y, my).x, ds_sub(u.z, mz).x);
        uw[p + q] = vec4<f32>(w, 0.0);
        wmax2 = max(wmax2, dot(w, w));
    }
    let g_max = 2.0 * sqrt(wmax2);
    if (!(g_max > 0.0)) { return; }
    let k_rate = bitcast<f32>(dyn_[1].x);
    let per_pair = k_rate * g_max;
    if (per_pair > 1.0) { atomicAdd(&tot[T_OVERFULL], 1u); }
    let m = 0.5 * f32(nc) * f32(nc - 1u) * per_pair;
    if (m >= f32(MAX_CANDIDATES)) {
        atomicAdd(&tot[T_TOOFULL], 1u);
        return;
    }
    let rkey = vec2<u32>(dyn_[0].x, dyn_[0].y);
    let base = vec4<u32>(first, dyn_[0].z, dyn_[0].w, STREAM_COLLISIONS << 16u);
    let w0 = philox4x32_10(base, rkey);
    let whole = floor(m);
    var cands = u32(whole);
    if (m - whole > u01_f32(w0.x)) { cands = cands + 1u; }
    var kind = 0u;
    if ((ctl.z & 2u) != 0u) { kind = 1u; }
    var collisions = 0u;
    var exceeded = 0u;
    var loss = 0.0;
    var loss2 = 0.0;
    for (var k = 0u; k < cands; k = k + 1u) {
        let w = philox4x32_10(vec4<u32>(base.x, base.y, base.z, base.w | (1u + 2u * k)), rkey);
        let i = mulhilo(w.x, nc).x;
        let j0 = mulhilo(w.y, nc - 1u).x;
        let j = select(j0, j0 + 1u, j0 >= i);
        let g = uw[p + i].xyz - uw[p + j].xyz;
        let g_norm = sqrt(dot(g, g));
        if (g_norm <= 0.0) { continue; }
        if (g_norm > g_max) {
            exceeded = exceeded + 1u;
        } else if (u01_f32(w.z) >= g_norm / g_max) {
            continue;
        }
        let v = philox4x32_10(vec4<u32>(base.x, base.y, base.z, base.w | (2u + 2u * k)), rkey);
        let n_hat = impact_normal(g, u01_f32(v.x), u01_f32(v.z));
        let imp = collide_pair(g, n_hat, kind, cp[0].z, cp[0].w, cp[1].x);
        if (imp.hit) {
            uw[p + i] = uw[p + i] + vec4<f32>(imp.d, 0.0);
            uw[p + j] = uw[p + j] - vec4<f32>(imp.d, 0.0);
            add_velocity(sorted[p + i], imp.d);
            add_velocity(sorted[p + j], -imp.d);
            collisions = collisions + 1u;
            loss = loss + imp.loss;
            loss2 = loss2 + imp.loss * imp.loss;
        }
    }
    atomicAdd(&tot[T_CAND], cands);
    atomicAdd(&tot[T_COLL], collisions);
    atomicAdd(&tot[T_GMAX], exceeded);
    en[p] = loss;
    en[n + p] = loss2;
}

// Partial sums of 256 cells in DS, in index order: the energy loss of each
// cell and the sum of its squared single losses. Partials: 4 floats per chunk
// from en[2 n].
@compute @workgroup_size(64)
fn col_sum1(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    ds_one = cp[0].x;
    let c = index_of(id, nw);
    let n = ctl.x;
    if (c >= ctl.w) { return; }
    var a = vec2<f32>(0.0);
    var b = vec2<f32>(0.0);
    let hi = min(n, (c + 1u) * CHUNK);
    for (var k = c * CHUNK; k < hi; k = k + 1u) {
        a = ds_add_f(a, en[k]);
        b = ds_add_f(b, en[n + k]);
    }
    let o = 2u * n + 4u * c;
    en[o] = a.x;
    en[o + 1u] = a.y;
    en[o + 2u] = b.x;
    en[o + 3u] = b.y;
}

@compute @workgroup_size(1)
fn col_sum2() {
    ds_one = cp[0].x;
    let n = ctl.x;
    var a = vec2<f32>(0.0);
    var b = vec2<f32>(0.0);
    for (var c = 0u; c < ctl.w; c = c + 1u) {
        let o = 2u * n + 4u * c;
        a = ds_add(a, vec2<f32>(en[o], en[o + 1u]));
        b = ds_add(b, vec2<f32>(en[o + 2u], en[o + 3u]));
    }
    atomicStore(&tot[T_LOSS_HI], bitcast<u32>(a.x));
    atomicStore(&tot[T_LOSS_LO], bitcast<u32>(a.y));
    atomicStore(&tot[T_LOSS2_HI], bitcast<u32>(b.x));
    atomicStore(&tot[T_LOSS2_LO], bitcast<u32>(b.y));
}
