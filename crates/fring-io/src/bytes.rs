//! Little-endian helpers that move arrays through `Read` and `Write` in chunks.
//!
//! Reading allocates only as data arrives, so a corrupt count in a header
//! fails with "truncated" before it can allocate much memory.

use fring_model::{Error, Result};
use std::io::{Read, Write};

/// Elements per chunk (64 KiB of u32 or f32).
const CHUNK: usize = 16 * 1024;

/// Converts a read error to [`Error::Io`]. End of file reads as "truncated".
pub(crate) fn read_err(e: std::io::Error, what: &str) -> Error {
    if e.kind() == std::io::ErrorKind::UnexpectedEof {
        Error::Io(format!("truncated file: ended inside {what}"))
    } else {
        Error::Io(format!("{what}: {e}"))
    }
}

pub(crate) fn read_exact<R: Read, const N: usize>(r: &mut R, what: &str) -> Result<[u8; N]> {
    let mut buf = [0u8; N];
    r.read_exact(&mut buf).map_err(|e| read_err(e, what))?;
    Ok(buf)
}

pub(crate) fn read_u32<R: Read>(r: &mut R, what: &str) -> Result<u32> {
    Ok(u32::from_le_bytes(read_exact(r, what)?))
}

pub(crate) fn read_u64<R: Read>(r: &mut R, what: &str) -> Result<u64> {
    Ok(u64::from_le_bytes(read_exact(r, what)?))
}

pub(crate) fn read_f64<R: Read>(r: &mut R, what: &str) -> Result<f64> {
    Ok(f64::from_le_bytes(read_exact(r, what)?))
}

/// Reads and discards `n` bytes, in chunks.
pub(crate) fn skip<R: Read>(r: &mut R, mut n: u64, what: &str) -> Result<()> {
    let mut buf = [0u8; 4096];
    while n > 0 {
        let k = n.min(buf.len() as u64) as usize;
        r.read_exact(&mut buf[..k]).map_err(|e| read_err(e, what))?;
        n -= k as u64;
    }
    Ok(())
}

/// Fails if the reader has more data.
pub(crate) fn expect_end<R: Read>(r: &mut R, what: &str) -> Result<()> {
    let mut one = [0u8; 1];
    loop {
        match r.read(&mut one) {
            Ok(0) => return Ok(()),
            Ok(_) => return Err(Error::Io(format!("trailing data after {what}"))),
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(Error::Io(format!("{what}: {e}"))),
        }
    }
}

macro_rules! array_io {
    ($write:ident, $read:ident, $t:ty) => {
        pub(crate) fn $write<W: Write>(w: &mut W, values: &[$t]) -> Result<()> {
            const SIZE: usize = std::mem::size_of::<$t>();
            let mut buf = Vec::with_capacity(CHUNK.min(values.len()) * SIZE);
            for chunk in values.chunks(CHUNK) {
                buf.clear();
                for v in chunk {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                w.write_all(&buf)?;
            }
            Ok(())
        }

        /// Reads `n` values. Memory grows chunk by chunk as data arrives.
        pub(crate) fn $read<R: Read>(r: &mut R, n: u64, what: &str) -> Result<Vec<$t>> {
            const SIZE: usize = std::mem::size_of::<$t>();
            let mut out: Vec<$t> = Vec::with_capacity((n as usize).min(CHUNK));
            let mut buf = vec![0u8; CHUNK * SIZE];
            let mut left = n;
            while left > 0 {
                let k = left.min(CHUNK as u64) as usize;
                r.read_exact(&mut buf[..k * SIZE])
                    .map_err(|e| read_err(e, what))?;
                for b in buf[..k * SIZE].chunks_exact(SIZE) {
                    out.push(<$t>::from_le_bytes(b.try_into().expect("chunk size")));
                }
                left -= k as u64;
            }
            Ok(out)
        }
    };
}

array_io!(write_u32s, read_u32s, u32);
array_io!(write_f32s, read_f32s, f32);
array_io!(write_f64s, read_f64s, f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn array_round_trip_across_chunk_boundary() {
        let v: Vec<u32> = (0..(CHUNK as u32 * 2 + 5)).collect();
        let mut bytes = Vec::new();
        write_u32s(&mut bytes, &v).unwrap();
        assert_eq!(bytes.len(), v.len() * 4);
        let back = read_u32s(&mut bytes.as_slice(), v.len() as u64, "ids").unwrap();
        assert_eq!(back, v);
    }

    #[test]
    fn huge_count_fails_as_truncated_without_big_allocation() {
        let bytes = [0u8; 16];
        let err = read_f32s(&mut bytes.as_slice(), u64::MAX / 8, "x").unwrap_err();
        assert!(err.to_string().contains("truncated"), "{err}");
    }

    #[test]
    fn little_endian_layout() {
        let mut bytes = Vec::new();
        write_u32s(&mut bytes, &[1]).unwrap();
        assert_eq!(bytes, [1, 0, 0, 0]);
    }
}
