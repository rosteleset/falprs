//! Event data files: descriptors of logged faces and copied events, appended as packed records
//! of a 32-byte event id, a 32-bit descriptor position and the descriptor's floats.

/// Number of floats in a descriptor record.
pub const DESCRIPTOR_SIZE: usize = 512;

const EVENT_ID_SIZE: usize = 32;

/// Size of one packed record in an event data file.
pub const EVENT_DATA_SIZE: usize = EVENT_ID_SIZE + 4 + DESCRIPTOR_SIZE * 4;

pub struct EventData {
    pub event_id: String,
    /// Index of the face in its event.
    #[allow(dead_code)]
    pub position: i32,
    pub descriptor: Vec<f32>,
}

impl EventData {
    /// Reads a record from exactly `EVENT_DATA_SIZE` native-endian bytes.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let (id, rest) = bytes.split_at(EVENT_ID_SIZE);
        let (position, descriptor) = rest.split_at(4);
        Self {
            event_id: String::from_utf8_lossy(id).into_owned(),
            position: i32::from_ne_bytes([position[0], position[1], position[2], position[3]]),
            descriptor: super::caches::descriptor_from_bytes(descriptor),
        }
    }

    #[cfg(test)]
    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(EVENT_DATA_SIZE);
        let mut id = self.event_id.as_bytes().to_vec();
        id.resize(EVENT_ID_SIZE, 0);
        bytes.extend_from_slice(&id);
        bytes.extend_from_slice(&self.position.to_ne_bytes());
        let mut descriptor = self.descriptor.clone();
        descriptor.resize(DESCRIPTOR_SIZE, 0.0);
        bytes.extend_from_slice(&super::caches::descriptor_to_bytes(&descriptor));
        bytes
    }
}

/// Sum of 8 lanes in the order of the C++ service's AVX horizontal adds.
fn reduce_sum(lanes: &[f32; 8]) -> f32 {
    ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]))
}

/// Cosine similarity of two descriptors, accumulated in single precision like the C++ service's
/// AVX version: four 8-lane accumulators per sum over 32-float blocks. Floats beyond the last
/// whole block are ignored, as there.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = [[0f32; 8]; 4];
    let mut sqr_a = [[0f32; 8]; 4];
    let mut sqr_b = [[0f32; 8]; 4];
    for (block_a, block_b) in a.chunks_exact(32).zip(b.chunks_exact(32)) {
        for k in 0..4 {
            for lane in 0..8 {
                let (x, y) = (block_a[k * 8 + lane], block_b[k * 8 + lane]);
                dot[k][lane] += x * y;
                sqr_a[k][lane] += x * x;
                sqr_b[k][lane] += y * y;
            }
        }
    }
    let combine = |sums: &[[f32; 8]; 4]| {
        let mut lanes = [0f32; 8];
        for lane in 0..8 {
            lanes[lane] = (sums[0][lane] + sums[1][lane]) + (sums[2][lane] + sums[3][lane]);
        }
        reduce_sum(&lanes)
    };
    f64::from(combine(&dot)) / f64::from(combine(&sqr_a)).sqrt() / f64::from(combine(&sqr_b)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_round_trip() {
        let data = EventData {
            event_id: "0123456789abcdef0123456789abcdef".to_string(),
            position: 7,
            descriptor: (0..DESCRIPTOR_SIZE).map(|i| i as f32).collect(),
        };
        let bytes = data.to_bytes();
        assert_eq!(bytes.len(), EVENT_DATA_SIZE);
        let read = EventData::from_bytes(&bytes);
        assert_eq!(read.event_id, data.event_id);
        assert_eq!(read.position, 7);
        assert_eq!(read.descriptor, data.descriptor);
    }

    #[test]
    fn cosine() {
        let a: Vec<f32> = (0..DESCRIPTOR_SIZE).map(|i| (i % 7) as f32 - 3.0).collect();
        let negated: Vec<f32> = a.iter().map(|v| -v).collect();
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
        assert!((cosine_similarity(&a, &negated) + 1.0).abs() < 1e-6);
    }
}
