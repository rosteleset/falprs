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
}
