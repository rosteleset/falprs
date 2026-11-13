//! Event data files: descriptors of logged faces and copied events, appended as packed records
//! of a 32-byte event id, a 32-bit descriptor position and the descriptor's floats.

use std::borrow::Cow;

/// Number of floats in a descriptor record.
pub const DESCRIPTOR_SIZE: usize = 512;

const EVENT_ID_SIZE: usize = 32;

/// Size of one packed record in an event data file.
pub const EVENT_DATA_SIZE: usize = EVENT_ID_SIZE + 4 + DESCRIPTOR_SIZE * 4;

/// A packed record of an event data file, read in place.
#[derive(Clone, Copy)]
pub struct EventRecord<'a>(&'a [u8; EVENT_DATA_SIZE]);

impl<'a> EventRecord<'a> {
    /// The records of a file's contents; a trailing partial record is ignored.
    pub fn all(bytes: &'a [u8]) -> impl Iterator<Item = EventRecord<'a>> {
        bytes.as_chunks::<EVENT_DATA_SIZE>().0.iter().map(EventRecord)
    }

    /// The event id, with invalid UTF-8 replaced; borrowed from the record when it is valid.
    pub fn event_id(self) -> Cow<'a, str> {
        String::from_utf8_lossy(&self.0[..EVENT_ID_SIZE])
    }

    /// The raw event id and the face's position in the event: what tells records apart, as an
    /// event's faces have different positions.
    pub fn key(self) -> ([u8; EVENT_ID_SIZE], i32) {
        let (id, rest) = self.0.split_first_chunk::<EVENT_ID_SIZE>().expect("a whole record");
        let position = rest.first_chunk::<4>().expect("a whole record");
        (*id, i32::from_ne_bytes(*position))
    }

    /// Decodes the descriptor's native-endian floats into `descriptor`.
    pub fn read_descriptor(self, descriptor: &mut [f32; DESCRIPTOR_SIZE]) {
        let bytes = self.0[EVENT_ID_SIZE + 4..].as_chunks::<4>().0;
        for (value, bytes) in descriptor.iter_mut().zip(bytes) {
            *value = f32::from_ne_bytes(*bytes);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_are_read_in_place() {
        let descriptor: Vec<f32> = (0..DESCRIPTOR_SIZE).map(|i| i as f32 - 0.5).collect();
        let mut bytes = Vec::new();
        for id in ["0123456789abcdef0123456789abcdef", "fedcba9876543210fedcba9876543210"] {
            bytes.extend_from_slice(id.as_bytes());
            bytes.extend_from_slice(&7i32.to_ne_bytes());
            bytes.extend_from_slice(&super::super::caches::descriptor_to_bytes(&descriptor));
        }
        // A trailing partial record is ignored
        bytes.extend_from_slice(b"partial");
        let records: Vec<EventRecord> = EventRecord::all(&bytes).collect();
        assert_eq!(records.len(), 2);
        assert_eq!(records[1].event_id(), "fedcba9876543210fedcba9876543210");
        assert!(matches!(records[0].event_id(), Cow::Borrowed(_)));
        assert_eq!(records[1].key(), (*b"fedcba9876543210fedcba9876543210", 7));
        let mut read = [0.0; DESCRIPTOR_SIZE];
        records[0].read_descriptor(&mut read);
        assert_eq!(read[..], descriptor[..]);
    }
}
