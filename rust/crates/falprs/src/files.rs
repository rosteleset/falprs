//! File operations shared by FRS and LPRS, asynchronous so that a slow disk does not stall the
//! runtime's worker threads: single files through `tokio::fs`, directory walks with `std::fs` in one
//! blocking task each, as one `tokio::fs` call per entry would cost a trip to the blocking pool.

use std::fs::Metadata;
use std::io;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::SystemTime;

use anyhow::{Context, Result};

/// Extensions of the image files the screenshot cleanups remove, as in the C++ service.
pub const IMAGE_EXTENSIONS: [&str; 6] = ["png", "jpg", "jpeg", "bmp", "ppm", "tiff"];

/// Writes a file readable and writable by its owner and others (mode 0606), creating its
/// directories, as the C++ service writes screenshots.
pub async fn write_shared(path: &Path, contents: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .with_context(|| format!("can't create {}", parent.display()))?;
    }
    tokio::fs::write(path, contents)
        .await
        .with_context(|| format!("can't write {}", path.display()))?;
    tokio::fs::set_permissions(path, std::fs::Permissions::from_mode(0o606))
        .await
        .with_context(|| format!("can't set the permissions of {}", path.display()))
}

/// Appends `contents` to a file, creating it if needed.
pub async fn append(path: &Path, contents: &[u8]) -> io::Result<()> {
    use tokio::io::AsyncWriteExt;
    let mut file = tokio::fs::OpenOptions::new().create(true).append(true).open(path).await?;
    file.write_all(contents).await?;
    file.flush().await
}

/// Whether `path` exists; an error checking it counts as absent, like `std::filesystem::exists`
/// with an error code.
pub async fn exists(path: &Path) -> bool {
    tokio::fs::try_exists(path).await.unwrap_or(false)
}

/// Regular files under `dir`, recursively, whose extension is one of `extensions`, with their
/// metadata. Directories are walked depth first from the last one found; unreadable entries are
/// skipped. The walk runs on the blocking pool in one go, and stops early once the returned future
/// is dropped, e.g. by a shutdown or a client that went away.
pub async fn files_with_extensions(dir: &Path, extensions: &[&str]) -> Vec<(PathBuf, Metadata)> {
    let (dir, extensions) = (dir.to_path_buf(), owned(extensions));
    run_blocking(move |cancelled| walk(&dir, &extensions, cancelled)).await
}

/// Removes the files under `dir` with one of `extensions` last modified before `before`, and
/// returns the errors of the removals that failed. Runs on the blocking pool like
/// [`files_with_extensions`].
pub async fn remove_old_files(dir: &Path, extensions: &[&str], before: SystemTime) -> Vec<io::Error> {
    let (dir, extensions) = (dir.to_path_buf(), owned(extensions));
    run_blocking(move |cancelled| {
        let mut errors = Vec::new();
        for (path, metadata) in walk(&dir, &extensions, cancelled) {
            if cancelled.load(Ordering::Relaxed) {
                break;
            }
            if metadata.modified().is_ok_and(|modified| modified < before)
                && let Err(e) = std::fs::remove_file(&path)
            {
                errors.push(e);
            }
        }
        errors
    })
    .await
}

fn owned(extensions: &[&str]) -> Vec<String> {
    extensions.iter().map(|e| e.to_string()).collect()
}

/// The walk of [`files_with_extensions`], with `std::fs`; it ends early once `cancelled` is set.
fn walk(dir: &Path, extensions: &[String], cancelled: &AtomicBool) -> Vec<(PathBuf, Metadata)> {
    let mut files = Vec::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(dir) = pending.pop() {
        if cancelled.load(Ordering::Relaxed) {
            break;
        }
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let Ok(file_type) = entry.file_type() else { continue };
            let path = entry.path();
            if file_type.is_dir() {
                pending.push(path);
            } else if file_type.is_file() && path.extension().and_then(|e| e.to_str()).is_some_and(|e| extensions.iter().any(|x| x == e)) {
                let Ok(metadata) = entry.metadata() else { continue };
                files.push((path, metadata));
            }
        }
    }
    files
}

/// Sets its flag when dropped.
struct CancelOnDrop(Arc<AtomicBool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

/// Runs `work` on tokio's blocking pool. `work` gets a flag that is set once the returned future is
/// dropped before it ends, as a blocking task cannot be aborted: checking it lets a long walk stop
/// instead of delaying a shutdown. A panic of `work` resumes in the caller.
async fn run_blocking<T: Send + 'static>(work: impl FnOnce(&AtomicBool) -> T + Send + 'static) -> T {
    let cancelled = Arc::new(AtomicBool::new(false));
    let _cancel_on_drop = CancelOnDrop(cancelled.clone());
    match tokio::task::spawn_blocking(move || work(&cancelled)).await {
        Ok(result) => result,
        Err(e) => std::panic::resume_unwind(e.into_panic()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("falprs_files_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[tokio::test]
    async fn shared_files_get_their_directories_and_mode() {
        let dir = temp_dir("shared");
        let path = dir.join("a/b/c.jpg");
        write_shared(&path, b"jpeg").await.unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"jpeg");
        assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o606);
        append(&dir.join("a/log.dat"), b"12").await.unwrap();
        append(&dir.join("a/log.dat"), b"34").await.unwrap();
        assert_eq!(std::fs::read(dir.join("a/log.dat")).unwrap(), b"1234");
        assert!(exists(&path).await && !exists(&dir.join("missing")).await);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn dropping_the_future_cancels_the_blocking_work() {
        let observed = Arc::new(AtomicBool::new(false));
        let seen = observed.clone();
        let task = tokio::spawn(run_blocking(move |cancelled| {
            // A long walk that only ends when cancelled
            let start = std::time::Instant::now();
            while !cancelled.load(Ordering::Relaxed) && start.elapsed() < Duration::from_secs(5) {
                std::thread::sleep(Duration::from_millis(5));
            }
            seen.store(cancelled.load(Ordering::Relaxed), Ordering::Relaxed);
        }));
        tokio::time::sleep(Duration::from_millis(50)).await;
        task.abort();
        let start = std::time::Instant::now();
        while !observed.load(Ordering::Relaxed) && start.elapsed() < Duration::from_secs(5) {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert!(observed.load(Ordering::Relaxed));
        assert!(walk(Path::new("/"), &owned(&["jpg"]), &AtomicBool::new(true)).is_empty());
    }

    #[tokio::test]
    async fn old_files_with_listed_extensions_are_removed() {
        let dir = temp_dir("old");
        for name in ["x/old.jpg", "x/y/old.dat", "x/keep.txt", "new.jpg"] {
            write_shared(&dir.join(name), b"").await.unwrap();
        }
        let mut found: Vec<PathBuf> = files_with_extensions(&dir, &["jpg", "dat"]).await.into_iter().map(|(p, _)| p).collect();
        found.sort();
        assert_eq!(found, [dir.join("new.jpg"), dir.join("x/old.jpg"), dir.join("x/y/old.dat")]);

        // Everything is older than a moment from now; nothing is older than an hour ago
        assert!(
            remove_old_files(&dir, &["jpg", "dat"], SystemTime::now() - Duration::from_secs(3600))
                .await
                .is_empty()
        );
        assert_eq!(files_with_extensions(&dir, &["jpg", "dat"]).await.len(), 3);
        let later = SystemTime::now() + Duration::from_secs(1);
        assert!(remove_old_files(&dir.join("x"), &["jpg", "dat"], later).await.is_empty());
        assert!(!dir.join("x/old.jpg").exists() && !dir.join("x/y/old.dat").exists());
        assert!(dir.join("x/keep.txt").exists() && dir.join("new.jpg").exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
