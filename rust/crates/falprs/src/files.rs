//! File operations shared by FRS and LPRS, asynchronous through `tokio::fs` so that a slow disk
//! does not stall the runtime's worker threads.

use std::fs::Metadata;
use std::io;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
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
/// skipped.
pub async fn files_with_extensions(dir: &Path, extensions: &[&str]) -> Vec<(PathBuf, Metadata)> {
    let mut files = Vec::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(dir) = pending.pop() {
        let Ok(mut entries) = tokio::fs::read_dir(&dir).await else { continue };
        while let Ok(Some(entry)) = entries.next_entry().await {
            let Ok(file_type) = entry.file_type().await else { continue };
            let path = entry.path();
            if file_type.is_dir() {
                pending.push(path);
            } else if file_type.is_file() && path.extension().and_then(|e| e.to_str()).is_some_and(|e| extensions.contains(&e)) {
                let Ok(metadata) = entry.metadata().await else { continue };
                files.push((path, metadata));
            }
        }
    }
    files
}

/// Removes the files under `dir` with one of `extensions` last modified before `before`, and
/// returns the errors of the removals that failed.
pub async fn remove_old_files(dir: &Path, extensions: &[&str], before: SystemTime) -> Vec<io::Error> {
    let mut errors = Vec::new();
    for (path, metadata) in files_with_extensions(dir, extensions).await {
        if metadata.modified().is_ok_and(|modified| modified < before)
            && let Err(e) = tokio::fs::remove_file(&path).await
        {
            errors.push(e);
        }
    }
    errors
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
