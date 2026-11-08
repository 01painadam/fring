// Counter-based RNG for the stochastic collisions (task K5): Philox4x32-10
// and the f32 uniform, bit-identical to crates/fring-model/src/rng.rs
// (`philox4x32_10`, `mulhilo`, `u01_f32`). The Rust functions are the twin;
// crates/fring-gpu/tests/gpu_collide.rs compares them word for word.
//
// Counter layout (SPEC 5.5): [index, step low, step high, stream << 16 | draw].
// Key: [seed low, seed high].

const PHILOX_M0: u32 = 0xD2511F53u;
const PHILOX_M1: u32 = 0xCD9E8D57u;
const PHILOX_W0: u32 = 0x9E3779B9u;
const PHILOX_W1: u32 = 0xBB67AE85u;
// Stream::COLLISIONS of rng.rs.
const STREAM_COLLISIONS: u32 = 2u;

// (high, low) 32-bit words of the 64-bit product a * b, from 16-bit halves.
fn mulhilo(a: u32, b: u32) -> vec2<u32> {
    let a0 = a & 0xFFFFu;
    let a1 = a >> 16u;
    let b0 = b & 0xFFFFu;
    let b1 = b >> 16u;
    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;
    let mid = (p00 >> 16u) + (p01 & 0xFFFFu) + (p10 & 0xFFFFu);
    let lo = (p00 & 0xFFFFu) | (mid << 16u);
    let hi = p11 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u);
    return vec2<u32>(hi, lo);
}

fn philox4x32_10(c: vec4<u32>, k: vec2<u32>) -> vec4<u32> {
    var ctr = c;
    var key = k;
    for (var r = 0u; r < 10u; r = r + 1u) {
        if (r > 0u) {
            key.x = key.x + PHILOX_W0;
            key.y = key.y + PHILOX_W1;
        }
        let a = mulhilo(PHILOX_M0, ctr.x);
        let b = mulhilo(PHILOX_M1, ctr.z);
        ctr = vec4<u32>(b.x ^ ctr.y ^ key.x, b.y, a.x ^ ctr.w ^ key.y, a.y);
    }
    return ctr;
}

// Uniform f32 in (0, 1) from the top 23 bits of one word: (k + 0.5) 2^-23.
fn u01_f32(word: u32) -> f32 {
    return (f32(word >> 9u) + 0.5) * 1.1920929e-7;
}
