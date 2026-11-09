//! Background tasks of the service: video stream workflows, maintenance and cache updates.
//!
//! They are stopped and awaited before the runtime shuts down. A task left running would keep
//! going on its worker thread after a `block_in_place` section and panic on its next timer, as the
//! runtime's timer driver is already gone.

use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::sync::LazyLock;

use futures::FutureExt;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;

/// A set of tasks that can be cancelled and awaited together.
#[derive(Default)]
pub struct Tasks {
    tracker: TaskTracker,
    cancel: CancellationToken,
}

impl Tasks {
    /// Spawns `task`; it is dropped at its next await point once [`Tasks::shutdown`] is called.
    pub fn spawn<F>(&self, task: F)
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let cancel = self.cancel.clone();
        self.tracker.spawn(async move {
            tokio::select! {
                // A cancelled task does not start or resume, even when it is ready
                biased;
                _ = cancel.cancelled() => {}
                _ = task => {}
            }
        });
    }

    /// Cancels every task and waits until all of them have ended.
    pub async fn shutdown(&self) {
        self.tracker.close();
        self.cancel.cancel();
        self.tracker.wait().await;
    }
}

static BACKGROUND: LazyLock<Tasks> = LazyLock::new(Tasks::default);

/// Spawns a background task of the service.
pub fn spawn<F>(task: F)
where
    F: Future<Output = ()> + Send + 'static,
{
    BACKGROUND.spawn(task);
}

/// Runs `task`, calling `on_panic` with the panic message if it panics: tokio would end the task
/// silently, leaving behind whatever state the task undoes on its normal ways out.
pub async fn catch_panic<F>(task: F, on_panic: impl FnOnce(&str))
where
    F: Future<Output = ()>,
{
    if let Err(payload) = AssertUnwindSafe(task).catch_unwind().await {
        let message = payload
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
            .unwrap_or("unknown panic");
        on_panic(message);
    }
}

/// Stops the background tasks of the service and waits for them.
pub async fn shutdown() {
    BACKGROUND.shutdown().await;
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    use super::*;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn shutdown_waits_for_a_task_inside_block_in_place() {
        let tasks = Tasks::default();
        let left_block_in_place = Arc::new(AtomicBool::new(false));
        let flag = left_block_in_place.clone();
        tasks.spawn(async move {
            // CPU work, like decoding a frame
            tokio::task::block_in_place(|| std::thread::sleep(Duration::from_millis(300)));
            flag.store(true, Ordering::SeqCst);
            // A timer, like a request with a timeout: it must be registered while the runtime lives
            tokio::time::sleep(Duration::from_secs(60)).await;
        });
        tokio::time::sleep(Duration::from_millis(50)).await;

        tokio::time::timeout(Duration::from_secs(5), tasks.shutdown()).await.unwrap();
        assert!(left_block_in_place.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn a_panic_is_reported_instead_of_ending_the_task_silently() {
        let mut reported = None;
        catch_panic(
            async {
                tokio::task::yield_now().await;
                panic!("frame {}", 7);
            },
            |message| reported = Some(message.to_string()),
        )
        .await;
        assert_eq!(reported.as_deref(), Some("frame 7"));

        let mut called = false;
        catch_panic(async {}, |_| called = true).await;
        assert!(!called);
    }

    #[tokio::test]
    async fn tasks_spawned_after_shutdown_do_not_run() {
        let tasks = Tasks::default();
        tasks.shutdown().await;
        let ran = Arc::new(AtomicBool::new(false));
        let flag = ran.clone();
        tasks.spawn(async move { flag.store(true, Ordering::SeqCst) });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!ran.load(Ordering::SeqCst));
    }
}
