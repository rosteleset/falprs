//! Slots for the recognition pipelines of a subsystem: at most a fixed number of them run at
//! once, and video streams ready for their next frame take the free slots in the order they
//! became ready.
//!
//! Without the bound, every ready stream starts its pipeline at once: on hardware that can't keep
//! up all pipelines slow down together, and streams with light pipelines or fast cameras get
//! back sooner and take more frames than the others. With it, the surplus streams wait in a FIFO
//! queue instead, and the running pipelines go at full speed.

use std::num::NonZeroUsize;

use serde_json::{Map, Value};
use tokio::sync::{Semaphore, SemaphorePermit};

use crate::converters;

/// Slots per CPU by default. A pipeline spends most of its time waiting for the camera (about
/// 250 ms) and for inference, and about 17-20 ms of CPU time, Triton's on the same server
/// included, so this many pipelines keep a CPU busy.
pub const DEFAULT_PIPELINES_PER_CPU: usize = 16;

pub struct PipelineSlots {
    /// Tokio's semaphore is fair: permits go out in the order they were asked for.
    semaphore: Semaphore,
    count: usize,
}

impl PipelineSlots {
    /// `pipelines-per-cpu` slots for each CPU available to the process, its cgroup quota counted.
    pub fn from_config(config: &Map<String, Value>) -> Self {
        let cpus = std::thread::available_parallelism().map_or(1, NonZeroUsize::get);
        Self::new(pipelines_per_cpu(config) * cpus)
    }

    pub fn new(count: usize) -> Self {
        let count = count.clamp(1, Semaphore::MAX_PERMITS);
        Self {
            semaphore: Semaphore::new(count),
            count,
        }
    }

    pub fn count(&self) -> usize {
        self.count
    }

    /// Waits for a free slot, behind the pipelines that asked before; the slot is free again
    /// when the permit is dropped.
    pub async fn acquire(&self) -> SemaphorePermit<'_> {
        self.semaphore.acquire().await.expect("the pipeline slots are never closed")
    }
}

/// `pipelines-per-cpu` from the workflow config; values below 1 give the default.
fn pipelines_per_cpu(config: &Map<String, Value>) -> usize {
    let value = converters::to_i64(config.get("pipelines-per-cpu"), DEFAULT_PIPELINES_PER_CPU as i64);
    usize::try_from(value).ok().filter(|&n| n >= 1).unwrap_or(DEFAULT_PIPELINES_PER_CPU)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use serde_json::json;
    use tokio::sync::Mutex;

    use super::*;

    fn config(value: Value) -> Map<String, Value> {
        json!({ "pipelines-per-cpu": value }).as_object().unwrap().clone()
    }

    #[test]
    fn pipelines_per_cpu_from_config() {
        assert_eq!(pipelines_per_cpu(&Map::new()), DEFAULT_PIPELINES_PER_CPU);
        assert_eq!(pipelines_per_cpu(&config(json!(4))), 4);
        assert_eq!(pipelines_per_cpu(&config(json!("8"))), 8);
        assert_eq!(pipelines_per_cpu(&config(json!(0))), DEFAULT_PIPELINES_PER_CPU);
        assert_eq!(pipelines_per_cpu(&config(json!(-3))), DEFAULT_PIPELINES_PER_CPU);
    }

    #[test]
    fn there_is_always_a_slot() {
        assert_eq!(PipelineSlots::new(0).count(), 1);
    }

    #[tokio::test]
    async fn waiting_pipelines_get_slots_in_the_order_they_asked() {
        let slots = Arc::new(PipelineSlots::new(1));
        let order = Arc::new(Mutex::new(Vec::new()));
        let held = slots.acquire().await;

        let mut waiting = Vec::new();
        for stream in 0..5 {
            let (slots, order) = (slots.clone(), order.clone());
            waiting.push(tokio::spawn(async move {
                let _slot = slots.acquire().await;
                order.lock().await.push(stream);
            }));
            // Each stream asks for a slot before the next one does
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        drop(held);
        for task in waiting {
            task.await.unwrap();
        }
        assert_eq!(*order.lock().await, vec![0, 1, 2, 3, 4]);
    }
}
