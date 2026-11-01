//! Frame files (`frames/NNNNNN.bin`): particles in the rotating frame, f32.
//!
//! Layout: `docs/snapshot-format.md` section 5.

use crate::bytes::{
    expect_end, read_err, read_f32s, read_f64, read_u32, read_u32s, read_u64, skip, write_f32s,
    write_u32s,
};
use fring_model::{Error, Result};
use std::io::{Read, Write};

/// ASCII magic at the start of every frame file.
pub const FRAME_MAGIC: [u8; 8] = *b"FRINGFRM";
/// Frame format version written by this crate.
pub const FRAME_VERSION: u32 = 1;
/// `frame_type` 1: particle positions in the rotating frame, f32.
pub const FRAME_TYPE_ROTATING_F32: u32 = 1;
/// Header size in version 1, in bytes.
const HEADER_LEN_V1: u32 = 72;
/// Largest header that a reader accepts, in bytes.
const HEADER_LEN_MAX: u32 = 4096;
/// Flag bit 0: velocities are present.
const FLAG_VELOCITIES: u32 = 1;
/// Largest particle count (ids are u32).
const MAX_PARTICLES: u64 = 1 << 32;

/// Parsed frame header.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameHeader {
    /// Format version of the file.
    pub version: u32,
    /// Header length in bytes (the body starts here).
    pub header_len: u32,
    /// Frame type, see [`FRAME_TYPE_ROTATING_F32`].
    pub frame_type: u32,
    /// Flag bits. Bit 0: velocities present.
    pub flags: u32,
    /// Step number.
    pub step: u64,
    /// Number of particles.
    pub n_particles: u64,
    /// Simulation time in seconds.
    pub time_s: f64,
    /// Reference semi-major axis a0 in km.
    pub a0_km: f64,
    /// Rotation rate of the rotating frame in rad/s.
    pub n0_rad_s: f64,
    /// Epoch in seconds at which rotating and inertial axes coincide (0 by convention).
    pub t_ref_s: f64,
}

impl FrameHeader {
    /// True if the body holds velocities.
    pub fn has_velocities(&self) -> bool {
        self.flags & FLAG_VELOCITIES != 0
    }
}

/// Velocities in the rotating frame, in km/s.
#[derive(Clone, Debug, Default)]
pub struct RotatingVelocities {
    pub vx_km_s: Vec<f32>,
    pub vy_km_s: Vec<f32>,
    pub vz_km_s: Vec<f32>,
}

/// One frame. Positions (and velocities) are in the **rotating frame**, f32.
/// The caller converts from the inertial frame; this crate does not.
#[derive(Clone, Debug, Default)]
pub struct Frame {
    /// Step number.
    pub step: u64,
    /// Simulation time in seconds.
    pub time_s: f64,
    /// Reference semi-major axis a0 in km.
    pub a0_km: f64,
    /// Rotation rate of the rotating frame in rad/s.
    pub n0_rad_s: f64,
    /// Epoch at which the frames coincide, in seconds (0 by convention).
    pub t_ref_s: f64,
    /// Particle ids.
    pub id: Vec<u32>,
    pub x_km: Vec<f32>,
    pub y_km: Vec<f32>,
    pub z_km: Vec<f32>,
    /// Optional velocities.
    pub velocities: Option<RotatingVelocities>,
}

impl Frame {
    /// Number of particles.
    pub fn len(&self) -> usize {
        self.id.len()
    }

    /// True if there are no particles.
    pub fn is_empty(&self) -> bool {
        self.id.is_empty()
    }

    /// Checks that all arrays have the same length.
    pub fn validate(&self) -> Result<()> {
        let n = self.id.len();
        let mut lens = vec![
            ("x_km", self.x_km.len()),
            ("y_km", self.y_km.len()),
            ("z_km", self.z_km.len()),
        ];
        if let Some(v) = &self.velocities {
            lens.push(("vx_km_s", v.vx_km_s.len()));
            lens.push(("vy_km_s", v.vy_km_s.len()));
            lens.push(("vz_km_s", v.vz_km_s.len()));
        }
        for (name, len) in lens {
            if len != n {
                return Err(Error::Input(format!(
                    "frame: {name} has {len} values, id has {n}"
                )));
            }
        }
        Ok(())
    }

    /// True if every field is identical bit for bit (so NaN payloads and
    /// the sign of zero count). `==` on floats cannot tell these apart.
    pub fn bit_identical(&self, other: &Frame) -> bool {
        fn same(a: &[f32], b: &[f32]) -> bool {
            a.len() == b.len() && a.iter().zip(b).all(|(p, q)| p.to_bits() == q.to_bits())
        }
        let vel_same = match (&self.velocities, &other.velocities) {
            (None, None) => true,
            (Some(a), Some(b)) => {
                same(&a.vx_km_s, &b.vx_km_s)
                    && same(&a.vy_km_s, &b.vy_km_s)
                    && same(&a.vz_km_s, &b.vz_km_s)
            }
            _ => false,
        };
        self.step == other.step
            && self.time_s.to_bits() == other.time_s.to_bits()
            && self.a0_km.to_bits() == other.a0_km.to_bits()
            && self.n0_rad_s.to_bits() == other.n0_rad_s.to_bits()
            && self.t_ref_s.to_bits() == other.t_ref_s.to_bits()
            && self.id == other.id
            && same(&self.x_km, &other.x_km)
            && same(&self.y_km, &other.y_km)
            && same(&self.z_km, &other.z_km)
            && vel_same
    }
}

/// Writes one frame (version 1).
pub fn write_frame<W: Write>(w: &mut W, frame: &Frame) -> Result<()> {
    frame.validate()?;
    let flags = if frame.velocities.is_some() {
        FLAG_VELOCITIES
    } else {
        0
    };
    w.write_all(&FRAME_MAGIC)?;
    w.write_all(&FRAME_VERSION.to_le_bytes())?;
    w.write_all(&HEADER_LEN_V1.to_le_bytes())?;
    w.write_all(&FRAME_TYPE_ROTATING_F32.to_le_bytes())?;
    w.write_all(&flags.to_le_bytes())?;
    w.write_all(&frame.step.to_le_bytes())?;
    w.write_all(&(frame.id.len() as u64).to_le_bytes())?;
    w.write_all(&frame.time_s.to_le_bytes())?;
    w.write_all(&frame.a0_km.to_le_bytes())?;
    w.write_all(&frame.n0_rad_s.to_le_bytes())?;
    w.write_all(&frame.t_ref_s.to_le_bytes())?;
    write_u32s(w, &frame.id)?;
    write_f32s(w, &frame.x_km)?;
    write_f32s(w, &frame.y_km)?;
    write_f32s(w, &frame.z_km)?;
    if let Some(v) = &frame.velocities {
        write_f32s(w, &v.vx_km_s)?;
        write_f32s(w, &v.vy_km_s)?;
        write_f32s(w, &v.vz_km_s)?;
    }
    Ok(())
}

/// Reads and checks a frame header. Leaves the reader at the start of the body.
pub fn read_frame_header<R: Read>(r: &mut R) -> Result<FrameHeader> {
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)
        .map_err(|e| read_err(e, "frame magic"))?;
    if magic != FRAME_MAGIC {
        return Err(Error::Io(format!(
            "not a frame file: bad magic {magic:02x?}"
        )));
    }
    let version = read_u32(r, "frame version")?;
    if version != FRAME_VERSION {
        return Err(Error::Io(format!(
            "unsupported frame version {version} (this reader knows version {FRAME_VERSION})"
        )));
    }
    let header_len = read_u32(r, "frame header_len")?;
    if !(HEADER_LEN_V1..=HEADER_LEN_MAX).contains(&header_len) || header_len % 8 != 0 {
        return Err(Error::Io(format!(
            "bad frame header_len {header_len}: need a multiple of 8 from {HEADER_LEN_V1} to {HEADER_LEN_MAX}"
        )));
    }
    let frame_type = read_u32(r, "frame type")?;
    if frame_type != FRAME_TYPE_ROTATING_F32 {
        return Err(Error::Io(format!("unknown frame type {frame_type}")));
    }
    let flags = read_u32(r, "frame flags")?;
    if flags & !FLAG_VELOCITIES != 0 {
        return Err(Error::Io(format!("unknown frame flag bits {flags:#x}")));
    }
    let step = read_u64(r, "frame step")?;
    let n_particles = read_u64(r, "frame particle count")?;
    if n_particles > MAX_PARTICLES {
        return Err(Error::Io(format!(
            "particle count {n_particles} is above the limit {MAX_PARTICLES}"
        )));
    }
    let time_s = read_f64(r, "frame time_s")?;
    let a0_km = read_f64(r, "frame a0_km")?;
    let n0_rad_s = read_f64(r, "frame n0_rad_s")?;
    let t_ref_s = read_f64(r, "frame t_ref_s")?;
    skip(
        r,
        u64::from(header_len - HEADER_LEN_V1),
        "frame header extension",
    )?;
    Ok(FrameHeader {
        version,
        header_len,
        frame_type,
        flags,
        step,
        n_particles,
        time_s,
        a0_km,
        n0_rad_s,
        t_ref_s,
    })
}

/// Reads one whole frame. The reader must end exactly at the end of the frame.
pub fn read_frame<R: Read>(r: &mut R) -> Result<Frame> {
    let h = read_frame_header(r)?;
    let n = h.n_particles;
    let id = read_u32s(r, n, "ids")?;
    let x_km = read_f32s(r, n, "x_km")?;
    let y_km = read_f32s(r, n, "y_km")?;
    let z_km = read_f32s(r, n, "z_km")?;
    let velocities = if h.has_velocities() {
        Some(RotatingVelocities {
            vx_km_s: read_f32s(r, n, "vx_km_s")?,
            vy_km_s: read_f32s(r, n, "vy_km_s")?,
            vz_km_s: read_f32s(r, n, "vz_km_s")?,
        })
    } else {
        None
    };
    expect_end(r, "frame")?;
    Ok(Frame {
        step: h.step,
        time_s: h.time_s,
        a0_km: h.a0_km,
        n0_rad_s: h.n0_rad_s,
        t_ref_s: h.t_ref_s,
        id,
        x_km,
        y_km,
        z_km,
        velocities,
    })
}
