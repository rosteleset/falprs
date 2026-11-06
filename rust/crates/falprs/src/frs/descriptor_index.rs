//! Flat lists of the face descriptors of each video stream and special group, so that recognition
//! scans them in one pass instead of looking every id up in the descriptor cache. A list holds
//! the cache's own descriptors, not copies, and is rebuilt lazily after either cache it was built
//! from has been updated.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, Weak};

use super::caches::{Descriptor, FaceDescriptorData};

/// Descriptor ids by video stream id or by special group id.
type Links = HashMap<i32, HashSet<i32>>;

/// The descriptors of one video stream or special group, ordered by id.
#[derive(Debug, Default)]
pub struct DescriptorList {
    ids: Vec<i32>,
    descriptors: Vec<Descriptor>,
    /// The size of every descriptor, when they all have the same non-zero size.
    size: Option<usize>,
}

impl DescriptorList {
    fn new(ids: Option<&HashSet<i32>>, faces: &FaceDescriptorData) -> Self {
        let mut ids: Vec<i32> = ids.into_iter().flatten().copied().filter(|id| faces.descriptors.contains_key(id)).collect();
        ids.sort_unstable();
        let descriptors: Vec<Descriptor> = ids.iter().map(|id| faces.descriptors[id].clone()).collect();
        let size = descriptors
            .first()
            .map(|d| d.len())
            .filter(|&size| size > 0 && descriptors.iter().all(|d| d.len() == size));
        Self { ids, descriptors, size }
    }

    /// Raises `best` (similarity, descriptor id) to the first descriptor more similar than it, with
    /// the similarity of `cosine_similarity`.
    pub fn find_best(&self, query: &[f32], best: &mut (f64, i32)) {
        if self.size == Some(query.len()) {
            if let Some((row, similarity)) = falprs::similarity::max_dot(query, self.descriptors.iter().map(|d| &d[..]), best.0) {
                *best = (similarity, self.ids[row]);
            }
            return;
        }
        // Descriptors of different sizes, which cosine_similarity scores -1
        for (&id, known) in self.ids.iter().zip(&self.descriptors) {
            let similarity = super::pipeline::cosine_similarity(query, known);
            if similarity > best.0 {
                *best = (similarity, id);
            }
        }
    }
}

/// Lists by video stream or special group id, valid for the cache snapshots they were built from.
#[derive(Default)]
struct Snapshot {
    links: Weak<Links>,
    faces: Weak<FaceDescriptorData>,
    lists: HashMap<i32, Arc<DescriptorList>>,
}

/// Descriptor lists of either the video streams or the special groups, built on first use.
#[derive(Default)]
pub struct DescriptorLists {
    snapshot: Mutex<Snapshot>,
}

impl DescriptorLists {
    /// The list of `id` in `links`. All lists are dropped once `links` or `faces` is a newer
    /// snapshot than the one they were built from, so deleted descriptors and unused ids go away.
    pub fn get(&self, id: i32, links: &Arc<Links>, faces: &Arc<FaceDescriptorData>) -> Arc<DescriptorList> {
        let mut snapshot = self.snapshot.lock().unwrap_or_else(|e| e.into_inner());
        // A held Weak keeps the allocation, so a new snapshot never reuses the old address
        if snapshot.links.as_ptr() != Arc::as_ptr(links) || snapshot.faces.as_ptr() != Arc::as_ptr(faces) {
            *snapshot = Snapshot {
                links: Arc::downgrade(links),
                faces: Arc::downgrade(faces),
                lists: HashMap::new(),
            };
        }
        snapshot
            .lists
            .entry(id)
            .or_insert_with(|| Arc::new(DescriptorList::new(links.get(&id), faces)))
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn faces(descriptors: &[(i32, Vec<f32>)]) -> Arc<FaceDescriptorData> {
        Arc::new(FaceDescriptorData {
            descriptors: descriptors.iter().map(|(id, d)| (*id, Descriptor::from(d.as_slice()))).collect(),
            spawned_to_parent: HashMap::new(),
        })
    }

    fn unit(values: &[f32]) -> Vec<f32> {
        let mut descriptor: Vec<f32> = values.iter().cycle().take(64).copied().collect();
        super::super::caches::normalize(&mut descriptor);
        descriptor
    }

    /// The scan recognize() did before the lists: every linked id looked up in the cache
    fn reference(query: &[f32], ids: &HashSet<i32>, faces: &FaceDescriptorData, mut best: (f64, i32)) -> (f64, i32) {
        let mut ids: Vec<i32> = ids.iter().copied().collect();
        ids.sort_unstable();
        for id in ids {
            if let Some(known) = faces.descriptors.get(&id) {
                let similarity = super::super::pipeline::cosine_similarity(query, known);
                if similarity > best.0 {
                    best = (similarity, id);
                }
            }
        }
        best
    }

    #[test]
    fn finds_what_the_lookup_scan_found() {
        let query = unit(&[1.0, 2.0, 3.0, -1.0]);
        let faces = faces(&[
            (1, unit(&[3.0, 2.0, 1.0, 0.0])),
            (2, unit(&[1.0, 2.0, 3.0, -0.5])),
            (3, unit(&[-1.0, -2.0, -3.0, 1.0])),
            (4, unit(&[1.0, 2.0, 3.0, -0.5])),
            (5, vec![1.0; 32]),
            (6, Vec::new()),
        ]);
        let cases: [&[i32]; 6] = [&[1, 2, 3, 4], &[3], &[1, 2, 99], &[], &[3, 5, 1], &[6, 3]];
        for ids in cases {
            let ids: HashSet<i32> = ids.iter().copied().collect();
            let list = DescriptorList::new(Some(&ids), &faces);
            for start in [(-2.0, 0), (0.5, 7), (1.5, 7)] {
                let mut best = start;
                list.find_best(&query, &mut best);
                let expected = reference(&query, &ids, &faces, start);
                assert_eq!((best.0.to_bits(), best.1), (expected.0.to_bits(), expected.1), "ids {ids:?}, start {start:?}");
            }
        }
    }

    #[test]
    fn lists_follow_cache_snapshots() {
        let lists = DescriptorLists::default();
        let links = Arc::new(Links::from([(10, HashSet::from([1, 2]))]));
        let first = faces(&[(1, unit(&[1.0])), (2, unit(&[2.0, 1.0]))]);
        assert_eq!(lists.get(10, &links, &first).ids, [1, 2]);
        assert!(Arc::ptr_eq(&lists.get(10, &links, &first), &lists.get(10, &links, &first)));
        assert!(lists.get(11, &links, &first).ids.is_empty());

        let without_2 = faces(&[(1, unit(&[1.0]))]);
        assert_eq!(lists.get(10, &links, &without_2).ids, [1]);
        let relinked = Arc::new(Links::from([(10, HashSet::from([2]))]));
        assert!(lists.get(10, &relinked, &without_2).ids.is_empty());
        assert_eq!(lists.get(10, &relinked, &first).ids, [2]);
    }
}
