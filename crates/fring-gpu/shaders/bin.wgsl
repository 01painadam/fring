// Binning by cell (task K5-BIN): counting sort of the particles into hashed
// buckets, then a ranking of the members of each bucket by (cell key, id). Append to
// ds.wgsl. Rust twin: crates/fring-gpu/src/bin.rs (same operations in the same
// order; the integer parts are exact).
//
// The cell of a particle (crates/fring-collide/SPEC.md 5.1) is
// (floor(x / Lx), floor(y_c / Ly), floor(z / Lz)) with x = rho - a0,
// y_c = a0 ((phi - phi0) mod 2 pi) and z the height, from the DS state:
// rho = sqrt(X^2 + Y^2); phi (inertial) = k pi/32 + atan(y'/x'), with k the
// sector of an f32 atan2 and (x', y') the position turned by -k pi/32 (DS
// cos and sin from a table), so that |y'/x'| < 0.0493; the series of atan is
// summed in DS. alpha = n0 t + phi0 (mod 2 pi, from f64 on the host) turns
// the inertial longitude into the rotating one.
//
// Key of a cell, 64 bits in two u32 words, ordered like (ix, iy, iz):
//   hi = (ix + 2^21) << 10 | iy >> 11      (ix: 22 bits)
//   lo = (iy & 2047) << 21 | (iz + 2^20)   (iy: 21 bits, iz: 21 bits)
// The key (0xFFFFFFFF, 0xFFFFFFFF) is not a cell: it marks a particle that
// takes no part (dropped, or outside the range of the key).
//
// Bindings (group 0):
// 0: state (the planes of GpuStepper, cap DS values per plane), read.
// 1: status (8 + cap words of GpuStepper; word 8 + i is the drop reason), read.
// 2: prm uniform, 135 vec4<f32>: [0] one, arc flag (1 = arc); [4] D/2 (DS), edge slack km; [1] a0, Lx; [2] Ly, Lz; [3] 2 pi, D
//    (each a DS pair); [5 + 2 u] = DS cos, DS sin of (u - 32) pi/32 (u = 0..64);
//    [6 + 2 u] = DS of that angle.
// 3: ctl uniform, 2 vec4<u32>: [0] n, cap, mask (M - 1), n_y; [1] mpad, nb, M.
// 4: tdyn uniform, vec4<f32>: alpha as a DS pair (hi, lo).
// 5: counts: M buckets, zero up to mpad; then 8 info words at mpad:
//    +0 particles binned, +1 particles not binned (dropped or bad state),
//    +2 particles out of range of the key.
// 6: start: the exclusive prefix sums of the counts (mpad words), then nb block
//    sums. start[M] = number of binned particles.
// 7: keys: key of particle i (vec2<u32>), cap entries.
// 8: sorted: particle ids in bucket order and, inside a bucket, in (key, id)
//    order; entries from the number of binned particles on are 0xFFFFFFFF.
// 9: skeys: key of sorted[p]. Before bin_emit its .x holds the ids in the
//    order of the atomic scatter (scratch).

@group(0) @binding(0) var<storage, read> st: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> status: array<u32>;
@group(0) @binding(2) var<uniform> prm: array<vec4<f32>, 135>;
@group(0) @binding(3) var<uniform> ctl: array<vec4<u32>, 2>;
@group(0) @binding(4) var<uniform> tdyn: vec4<f32>;
@group(0) @binding(5) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> start: array<u32>;
@group(0) @binding(7) var<storage, read_write> keys: array<vec2<u32>>;
@group(0) @binding(8) var<storage, read_write> sorted: array<u32>;
@group(0) @binding(9) var<storage, read_write> skeys: array<vec2<u32>>;

const NONE: u32 = 0xFFFFFFFFu;
const BIN_WG: u32 = 64u;
const SCAN_WG: u32 = 256u;
const SCAN_BLOCK: u32 = 1024u;
const IX_OFF: i32 = 2097152;
const IZ_OFF: i32 = 1048576;
const IX_MAX: f32 = 2097150.0;
const IZ_MAX: f32 = 1048574.0;
// 1/5, 1/7, ... 1/15 as f32 (twins: bin::ATAN_C5 ...).
const C5: f32 = 0.2;
const C7: f32 = 0.14285715;
const C9: f32 = 0.11111111;
const C11: f32 = 0.09090909;
const C13: f32 = 0.07692308;
const C15: f32 = 0.06666667;
const THIRD: vec2<f32> = vec2<f32>(0.33333334, -9.934108e-09);

fn n_particles() -> u32 { return ctl[0].x; }
fn cap() -> u32 { return ctl[0].y; }
fn mask() -> u32 { return ctl[0].z; }
fn n_y() -> u32 { return ctl[0].w; }
fn mpad() -> u32 { return ctl[1].x; }
fn n_blocks() -> u32 { return ctl[1].y; }
fn n_buckets() -> u32 { return ctl[1].z; }

fn ds_negp(a: vec2<f32>) -> bool {
    return a.x < 0.0 || (a.x == 0.0 && a.y < 0.0);
}

// floor of a DS value that is far below 2^23 in size.
fn ds_floor(q: vec2<f32>) -> f32 {
    let f = floor(q.x);
    if (q.x == f && q.y < 0.0) { return f - 1.0; }
    return f;
}

// atan t for |t| < 0.0493 (truncation below 1e-18): t (1 - u/3 + u^2 P(u)).
fn ds_atan_small(t: vec2<f32>) -> vec2<f32> {
    let u = ds_mul(t, t);
    let uh = u.x;
    let p = fma(uh, fma(uh, fma(uh, fma(uh, fma(uh, -C15, C13), -C11), C9), -C7), C5);
    let tail = (uh * uh) * p;
    let s = ds_add_f(ds_sub(vec2<f32>(1.0, 0.0), ds_mul(u, THIRD)), tail);
    return ds_mul(t, s);
}

fn pack_key(ix: i32, iy: u32, iz: i32) -> vec2<u32> {
    let hi = (u32(ix + IX_OFF) << 10u) | (iy >> 11u);
    let lo = ((iy & 2047u) << 21u) | u32(iz + IZ_OFF);
    return vec2<u32>(hi, lo);
}

fn hash_key(k: vec2<u32>) -> u32 {
    var h = k.y ^ (k.x * 0x85EBCA6Bu);
    h = h ^ (h >> 16u);
    h = h * 0x7FEB352Du;
    h = h ^ (h >> 15u);
    h = h * 0x846CA68Bu;
    h = h ^ (h >> 16u);
    return h;
}

// 0 = cell (in `key`), 1 = not binned (dropped or bad state), 2 = out of range.
struct Binned {
    kind: u32,
    key: vec2<u32>,
}

fn cell_of(i: u32) -> Binned {
    let c = cap();
    let bad = Binned(1u, vec2<u32>(NONE, NONE));
    if (status[8u + i] != 0u) { return bad; }
    let px = st[i];
    let py = st[c + i];
    let pz = st[2u * c + i];
    let r2 = ds_add(ds_mul(px, px), ds_mul(py, py));
    if (!(r2.x > 0.0 && r2.x < 1.0e18)) { return bad; }
    let rho = ds_sqrt(r2);
    // Sector of the longitude.
    let coarse = atan2(py.x, px.x);
    let k = clamp(i32(floor(coarse * 10.185916 + 0.5)), -32, 32);
    let row = 5u + 2u * u32(k + 32);
    let cs = prm[row];
    let ck = cs.xy;
    let sk = cs.zw;
    let phik = prm[row + 1u].xy;
    let xp = ds_add(ds_mul(px, ck), ds_mul(py, sk));
    let yp = ds_sub(ds_mul(py, ck), ds_mul(px, sk));
    let t = ds_div(yp, xp);
    let phi = ds_add(phik, ds_atan_small(t));
    // Rotating longitude minus phi0, in [0, 2 pi).
    let twopi = prm[3].xy;
    var psi = ds_sub(phi, tdyn.xy);
    if (ds_negp(psi)) { psi = ds_add(psi, twopi); }
    if (ds_negp(psi)) { psi = ds_add(psi, twopi); }
    if (!ds_negp(ds_sub(psi, twopi))) { psi = ds_sub(psi, twopi); }
    let a0 = prm[1].xy;
    // Arc (prm[0].y = 1): the signed angle from the arc centre in (-pi, pi]
    // plus half the width, so y runs 0..D across the arc; tdyn holds the
    // centre longitude then.
    let arc = prm[0].y != 0.0;
    if (arc) {
        let over_pi = ds_sub(psi, vec2<f32>(0.5 * twopi.x, 0.5 * twopi.y));
        if (over_pi.x > 0.0 || (over_pi.x == 0.0 && over_pi.y > 0.0)) { psi = ds_sub(psi, twopi); }
    }
    var y = ds_mul(a0, psi);
    if (arc) {
        // y runs 0..D (D = 2 (D/2) is exact). Within the slack outside an edge
        // the particle is taken at its periodic image (y -/+ D).
        let half = prm[4].xy;
        let d = ds_add(half, half);
        y = ds_add(y, half);
        if (y.x < 0.0 && y.x > -prm[4].z) {
            y = ds_add(y, d);
        } else if (y.x >= 0.0 && !ds_negp(ds_sub(y, d)) && ds_sub(y, d).x < prm[4].z) {
            y = ds_sub(y, d);
        }
    }
    let x = ds_sub(rho, a0);
    let qx = ds_div(x, prm[1].zw);
    let qy = ds_div(y, prm[2].xy);
    let qz = ds_div(pz, prm[2].zw);
    let fx = ds_floor(qx);
    var fy = ds_floor(qy);
    let fz = ds_floor(qz);
    let ny = f32(n_y());
    if (fy >= ny) {
        // Rounding at the end of the domain: y - D below 1e-6 km is the last cell.
        let over = ds_sub(y, prm[3].zw);
        if (over.x < 1.0e-6) { fy = ny - 1.0; }
    }
    if (!(abs(fx) <= IX_MAX && abs(fz) <= IZ_MAX && fy >= 0.0 && fy < ny)) {
        return Binned(2u, vec2<u32>(NONE, NONE));
    }
    return Binned(0u, pack_key(i32(fx), u32(fy), i32(fz)));
}

fn index_of(id: vec3<u32>, nw: vec3<u32>) -> u32 {
    return id.y * nw.x * BIN_WG + id.x;
}

@compute @workgroup_size(64)
fn bin_clear(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let j = index_of(id, nw);
    if (j < mpad() + 8u) { atomicStore(&counts[j], 0u); }
}

@compute @workgroup_size(64)
fn bin_keys(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    ds_one = prm[0].x;
    let i = index_of(id, nw);
    if (i >= n_particles()) { return; }
    let b = cell_of(i);
    keys[i] = b.key;
    if (b.kind == 0u) {
        atomicAdd(&counts[hash_key(b.key) & mask()], 1u);
    } else {
        atomicAdd(&counts[mpad() + b.kind], 1u);
    }
}

var<workgroup> sh: array<u32, 256>;

// Exclusive scan of 1024 counts per workgroup (4 per thread); the block
// total goes to start[mpad + block].
@compute @workgroup_size(256)
fn scan_local(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let base = wg.x * SCAN_BLOCK + li * 4u;
    let a = atomicLoad(&counts[base]);
    let b = atomicLoad(&counts[base + 1u]);
    let c = atomicLoad(&counts[base + 2u]);
    let d = atomicLoad(&counts[base + 3u]);
    let t = a + b + c + d;
    sh[li] = t;
    workgroupBarrier();
    for (var off = 1u; off < SCAN_WG; off = off << 1u) {
        var v = 0u;
        if (li >= off) { v = sh[li - off]; }
        workgroupBarrier();
        sh[li] = sh[li] + v;
        workgroupBarrier();
    }
    let e = sh[li] - t;
    start[base] = e;
    start[base + 1u] = e + a;
    start[base + 2u] = e + a + b;
    start[base + 3u] = e + a + b + c;
    if (li == SCAN_WG - 1u) { start[mpad() + wg.x] = sh[li]; }
}

// One workgroup: exclusive scan of the nb block totals.
@compute @workgroup_size(256)
fn scan_sums(@builtin(local_invocation_index) li: u32) {
    let nb = n_blocks();
    let chunk = (nb + SCAN_WG - 1u) / SCAN_WG;
    let lo = min(li * chunk, nb);
    let hi = min(lo + chunk, nb);
    var s = 0u;
    for (var j = lo; j < hi; j++) { s += start[mpad() + j]; }
    sh[li] = s;
    workgroupBarrier();
    for (var off = 1u; off < SCAN_WG; off = off << 1u) {
        var v = 0u;
        if (li >= off) { v = sh[li - off]; }
        workgroupBarrier();
        sh[li] = sh[li] + v;
        workgroupBarrier();
    }
    var run = sh[li] - s;
    for (var j = lo; j < hi; j++) {
        let v = start[mpad() + j];
        start[mpad() + j] = run;
        run += v;
    }
}

@compute @workgroup_size(64)
fn scan_add(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let j = index_of(id, nw);
    if (j < mpad()) { start[j] = start[j] + start[mpad() + j / SCAN_BLOCK]; }
}

// Places each particle in its bucket with a count that runs down. The order
// inside a bucket is not fixed (atomics); bin_rank fixes it. The ids go to
// skeys[pos].x, which bin_emit overwrites at the end.
@compute @workgroup_size(64)
fn bin_scatter(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = index_of(id, nw);
    if (i >= n_particles()) { return; }
    let k = keys[i];
    if (k.x == NONE && k.y == NONE) { return; }
    let b = hash_key(k) & mask();
    let pos = start[b] + atomicSub(&counts[b], 1u) - 1u;
    skeys[pos].x = i;
}

fn key_less(a: u32, b: u32) -> bool {
    let ka = keys[a];
    let kb = keys[b];
    if (ka.x != kb.x) { return ka.x < kb.x; }
    if (ka.y != kb.y) { return ka.y < kb.y; }
    return a < b;
}

// One thread per binned particle: its place in the bucket is the number of
// members of the bucket that come before it in the order (key, id). The
// order is total (ids differ), so the places are 0, 1, ... and the result does
// not depend on the order of the atomics. The cost is quadratic in the size of
// the bucket but spread over one thread per member (a single thread that
// insertion-sorts a bucket of 16384 took 17 s).
@compute @workgroup_size(64)
fn bin_rank(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let p = index_of(id, nw);
    if (p >= n_particles()) { return; }
    if (p >= start[n_buckets()]) { return; }
    let e = skeys[p].x;
    let b = hash_key(keys[e]) & mask();
    let s = start[b];
    let last = start[b + 1u];
    var rank = 0u;
    for (var j = s; j < last; j++) {
        if (key_less(skeys[j].x, e)) { rank++; }
    }
    sorted[s + rank] = e;
}

@compute @workgroup_size(64)
fn bin_emit(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let p = index_of(id, nw);
    if (p >= n_particles()) { return; }
    let total = start[n_buckets()];
    if (p == 0u) { atomicStore(&counts[mpad()], total); }
    if (p < total) {
        skeys[p] = keys[sorted[p]];
    } else {
        sorted[p] = NONE;
        skeys[p] = vec2<u32>(NONE, NONE);
    }
}
