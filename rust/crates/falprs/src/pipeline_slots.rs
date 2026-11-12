//! Slots for the recognition pipelines of a subsystem: at most a fixed number of them run at
//! once, and video streams ready for their next frame take the free slots in the order they
//! became ready.
//!
//! Without the bound, every ready stream starts its pipeline at once: on hardware that can't keep
//! up all pipelines slow down together, and streams with light pipelines or fast cameras get
//! back sooner and take more frames than the others. With it, the surplus streams wait in a FIFO
//! queue instead, and the running pipelines go at full speed.

use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

use tokio::sync::{Semaphore, SemaphorePermit};

/// Slots per CPU by default. A pipeline spends most of its time waiting for the camera (about
/// 250 ms) and for inference, and about 17-20 ms of CPU time, Triton's on the same server
/// included, so this many pipelines keep a CPU busy.
pub const DEFAULT_PIPELINES_PER_CPU: usize = 16;

/// A taken slot; it is free again when dropped.
pub struct PipelineSlot<'a> {
    _permit: SemaphorePermit<'a>,
    /// How long the pipeline waited for the slot.
    pub waited: Duration,
}

pub struct PipelineSlots {
    /// Tokio's semaphore is fair: permits go out in the order they were asked for.
    semaphore: Semaphore,
    count: usize,
}

impl PipelineSlots {
    /// `pipelines_per_cpu` slots for each CPU available to the process, its cgroup quota counted.
    pub fn per_cpu(pipelines_per_cpu: usize) -> Self {
        let cpus = std::thread::available_parallelism().map_or(1, NonZeroUsize::get);
        Self::new(pipelines_per_cpu.saturating_mul(cpus))
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

    /// Waits for a free slot, behind the pipelines that asked before.
    pub async fn acquire(&self) -> PipelineSlot<'_> {
        let asked = Instant::now();
        let permit = self.semaphore.acquire().await.expect("the pipeline slots are never closed");
        PipelineSlot {
            _permit: permit,
            waited: asked.elapsed(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use tokio::sync::Mutex;

    use super::*;

    #[test]
    fn there_is_always_a_slot() {
        assert_eq!(PipelineSlots::new(0).count(), 1);
    }

    #[tokio::test]
    async fn a_slot_tells_how_long_it_was_waited_for() {
        let slots = Arc::new(PipelineSlots::new(1));
        assert!(slots.acquire().await.waited < Duration::from_millis(50));

        let held = slots.acquire().await;
        let waiting = tokio::spawn({
            let slots = slots.clone();
            async move { slots.acquire().await.waited }
        });
        tokio::time::sleep(Duration::from_millis(100)).await;
        drop(held);
        assert!(waiting.await.unwrap() >= Duration::from_millis(100));
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
