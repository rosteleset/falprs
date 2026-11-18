//! Triton system shared memory for the large inputs, the detector inputs.
//!
//! Each inference server has a pool of regions: files in `/dev/shm` that Triton maps after they are
//! registered with it, so a request only names its region instead of carrying the input. The files are
//! written with positioned writes, without mapping them here. All regions of a pool have the size of the
//! largest input seen so far, and a smaller one grows when it is taken for a larger input. The pool grows
//! while no region is free, up to its limit; an input that finds none free beyond the limit goes over gRPC.
//! A region goes back to the pool as soon as its request ends, whether answered, failed or timed out: if
//! Triton still reads it for a cancelled request, that request's results are not read.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use tonic::transport::Channel;

use super::proto::{
    SystemSharedMemoryRegisterRequest, SystemSharedMemoryRegisterResponse, SystemSharedMemoryStatusRequest, SystemSharedMemoryStatusResponse,
    SystemSharedMemoryUnregisterRequest, SystemSharedMemoryUnregisterResponse,
};
use super::{is_connection_error, unary};

/// The level of a message the pools log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    Info,
    Warning,
}

/// Where the pools log their messages.
pub type Log = Box<dyn Fn(Level, std::fmt::Arguments<'_>) + Send + Sync>;

/// Inputs larger than this go through shared memory: the detector inputs (SCRFD, the YOLO detectors, from
/// about 1.2 MB at 320x320 to 4.9 MB at 640x640), not the face and plate inputs (150 to 600 KB), which come
/// many to a frame and gain little.
pub(super) const THRESHOLD: usize = 1024 * 1024;

/// How often an inference server that could not register shared memory is tried again.
const RETRY_INTERVAL: Duration = Duration::from_secs(3600);

/// How often a pool that reached its limit says so.
const LIMIT_WARNING_INTERVAL: Duration = Duration::from_secs(60);

const SHM_DIR: &str = "/dev/shm";

const REGISTER: &str = "/inference.GRPCInferenceService/SystemSharedMemoryRegister";
const UNREGISTER: &str = "/inference.GRPCInferenceService/SystemSharedMemoryUnregister";
const STATUS: &str = "/inference.GRPCInferenceService/SystemSharedMemoryStatus";

/// The shared memory pools of all inference servers, shared by the subsystems.
pub struct SharedMemory {
    enabled: bool,
    limit: u64,
    /// `falprs-<port>-`: the regions of every run of this service instance.
    instance_prefix: String,
    /// `falprs-<port>-<pid>-`: the regions of this run.
    run_prefix: String,
    next_region: AtomicUsize,
    pools: Mutex<HashMap<String, Arc<Pool>>>,
    log: Log,
}

impl SharedMemory {
    /// The pools, of up to `limit` bytes each, of a service listening on `port`. Removes the region files
    /// its earlier runs left behind; their registrations are removed from each inference server when it is
    /// first used.
    pub fn new(enabled: bool, limit: u64, port: u16, log: Log) -> Arc<Self> {
        let instance_prefix = format!("falprs-{port}-");
        let shared_memory = Self {
            enabled,
            limit,
            run_prefix: format!("{instance_prefix}{}-", std::process::id()),
            instance_prefix,
            next_region: AtomicUsize::new(0),
            pools: Mutex::default(),
            log,
        };
        if enabled {
            shared_memory.remove_files(Path::new(SHM_DIR));
        }
        Arc::new(shared_memory)
    }

    /// No shared memory: every input goes over gRPC.
    pub fn disabled() -> Arc<Self> {
        Arc::new(Self {
            enabled: false,
            limit: 0,
            instance_prefix: String::new(),
            run_prefix: String::new(),
            next_region: AtomicUsize::new(0),
            pools: Mutex::default(),
            log: Box::new(|_, _| {}),
        })
    }

    /// Removes the files in `dir` of this service instance's earlier runs.
    fn remove_files(&self, dir: &Path) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            if entry.file_name().to_str().is_some_and(|name| name.starts_with(&self.instance_prefix))
                && let Err(e) = std::fs::remove_file(entry.path())
            {
                (self.log)(
                    Level::Warning,
                    format_args!("can't remove the old shared memory region {}: {e}", entry.path().display()),
                );
            }
        }
    }

    fn pool(&self, endpoint: &str) -> Arc<Pool> {
        let mut pools = self.pools.lock().unwrap_or_else(|e| e.into_inner());
        pools
            .entry(endpoint.to_string())
            .or_insert_with(|| {
                Arc::new(Pool {
                    endpoint: endpoint.to_string(),
                    limit: self.limit,
                    state: Mutex::new(PoolState {
                        mode: Mode::Untried,
                        free: Vec::new(),
                        names: Vec::new(),
                        reserved_bytes: 0,
                        region_size: 0,
                        channel: None,
                        fallback_warned: false,
                        limit_warned_at: None,
                    }),
                    activation: tokio::sync::Mutex::new(()),
                })
            })
            .clone()
    }

    /// A region of `endpoint`'s pool holding at least `len` bytes, registered through `channel`, or none
    /// when the input is to go over gRPC: shared memory off or unavailable on that server, or the pool at
    /// its limit.
    pub(super) async fn lease(self: &Arc<Self>, endpoint: &str, channel: &Channel, len: usize) -> Option<Lease> {
        if !self.enabled {
            return None;
        }
        let pool = self.pool(endpoint);
        let mode = pool.state().mode;
        if !matches!(mode, Mode::Active) {
            if !mode.activation_due() {
                return None;
            }
            if let Some(lease) = self.activate(&pool, channel, len).await {
                return Some(lease);
            }
            // Activated by another request meanwhile, or not available
            if !matches!(pool.state().mode, Mode::Active) {
                return None;
            }
        }
        let (taken, size) = {
            let mut state = pool.state();
            state.region_size = state.region_size.max(len as u64);
            (state.free.pop(), state.region_size)
        };
        let region = match taken {
            Some(region) if region.size >= size => region,
            Some(region) => {
                if !pool.reserve(size - region.size, &self.log) {
                    pool.state().free.push(region);
                    return None;
                }
                match self.grow(region, size, channel).await {
                    Ok(region) => region,
                    Err((region, e)) => {
                        let _ = unregister(channel, &region.name).await;
                        (self.log)(
                            Level::Warning,
                            format_args!("can't grow shared memory region {} of {}: {e:#}", region.name, pool.endpoint),
                        );
                        pool.discard(region, size);
                        return None;
                    }
                }
            }
            None => {
                if !pool.reserve(size, &self.log) {
                    return None;
                }
                match self.create(size, channel).await {
                    Ok(region) => {
                        pool.state().names.push(region.name.clone());
                        region
                    }
                    Err(e) => {
                        (self.log)(Level::Warning, format_args!("can't add a shared memory region for {}: {e:#}", pool.endpoint));
                        pool.state().reserved_bytes -= size;
                        return None;
                    }
                }
            }
        };
        Some(Lease {
            pool,
            region: Some(region),
            channel: channel.clone(),
        })
    }

    /// Removes the registrations earlier runs left on the server and registers the first region. A server
    /// that refuses is used over gRPC and tried again in an hour; one that can't be reached is tried again
    /// with the next input.
    async fn activate(&self, pool: &Arc<Pool>, channel: &Channel, len: usize) -> Option<Lease> {
        let _activation = pool.activation.lock().await;
        // Another request may have activated the pool meanwhile
        if !pool.state().mode.activation_due() {
            return None;
        }
        let size = len as u64;
        let result = async {
            self.remove_stale_registrations(channel).await?;
            if size > self.limit {
                anyhow::bail!("an input of {size} bytes is larger than inference.shared-memory-limit = {} bytes", self.limit);
            }
            self.create(size, channel).await
        }
        .await;
        let region = match result {
            Ok(region) => region,
            Err(e) => {
                let unreachable = e
                    .downcast_ref::<tonic::Status>()
                    .is_some_and(|status| is_connection_error(status) || status.code() == tonic::Code::Unavailable);
                if !unreachable {
                    pool.fall_back(&e, &self.log);
                }
                return None;
            }
        };
        {
            let mut state = pool.state();
            state.mode = Mode::Active;
            // A later fallback is reported again; the hourly retries of one fallback are not
            state.fallback_warned = false;
            state.names.push(region.name.clone());
            state.reserved_bytes += size;
            state.region_size = state.region_size.max(size);
            state.channel = Some(channel.clone());
        }
        (self.log)(Level::Info, format_args!("shared memory enabled for {}", pool.endpoint));
        Some(Lease {
            pool: pool.clone(),
            region: Some(region),
            channel: channel.clone(),
        })
    }

    /// Sends the inputs of `endpoint` over gRPC from now on, e.g. after the server refused to register a
    /// region again; shared memory is tried again in an hour.
    pub(super) fn fall_back(&self, endpoint: &str, error: &tonic::Status) {
        self.pool(endpoint).fall_back(&anyhow::anyhow!("{error}"), &self.log);
    }

    /// Unregisters the regions of this service instance's earlier runs.
    async fn remove_stale_registrations(&self, channel: &Channel) -> Result<()> {
        let status: SystemSharedMemoryStatusResponse = unary(channel.clone(), STATUS, &SystemSharedMemoryStatusRequest::default()).await?;
        for name in status.regions.into_keys() {
            if name.starts_with(&self.instance_prefix) && !name.starts_with(&self.run_prefix) {
                unregister(channel, &name).await?;
            }
        }
        Ok(())
    }

    /// A new region file of `size` bytes, registered with the server.
    async fn create(&self, size: u64, channel: &Channel) -> Result<Region> {
        let name = format!("{}{}", self.run_prefix, self.next_region.fetch_add(1, Ordering::Relaxed));
        let path = region_path(&name);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            // Frames of the cameras: only the service's user and root (Triton's container) read them
            .mode(0o600)
            .open(&path)
            .with_context(|| format!("can't create {}", path.display()))?;
        let region = Region { name, file, size };
        let registered = async {
            region.file.set_len(size).with_context(|| format!("can't size {}", path.display()))?;
            register(channel, &region.name, size).await?;
            anyhow::Ok(())
        }
        .await;
        match registered {
            Ok(()) => Ok(region),
            Err(e) => {
                let _ = std::fs::remove_file(&path);
                Err(e)
            }
        }
    }

    /// The region enlarged to `size` bytes and registered again at that size.
    async fn grow(&self, region: Region, size: u64, channel: &Channel) -> Result<Region, (Region, anyhow::Error)> {
        let result = async {
            region.file.set_len(size).context("can't resize the file")?;
            unregister(channel, &region.name).await?;
            register(channel, &region.name, size).await?;
            anyhow::Ok(())
        }
        .await;
        match result {
            Ok(()) => Ok(Region { size, ..region }),
            Err(e) => Err((region, e)),
        }
    }

    /// Unregisters the regions of this run and removes their files, as far as the servers answer.
    pub async fn shutdown(&self) {
        let pools: Vec<_> = self.pools.lock().unwrap_or_else(|e| e.into_inner()).values().cloned().collect();
        for pool in pools {
            let (names, channel) = {
                let state = pool.state();
                (state.names.clone(), state.channel.clone())
            };
            if let Some(channel) = channel {
                for name in &names {
                    let _ = tokio::time::timeout(Duration::from_secs(1), unregister(&channel, name)).await;
                }
            }
            for name in &names {
                let _ = std::fs::remove_file(region_path(name));
            }
        }
    }
}

fn region_path(name: &str) -> PathBuf {
    Path::new(SHM_DIR).join(name)
}

async fn register(channel: &Channel, name: &str, size: u64) -> Result<(), tonic::Status> {
    let request = SystemSharedMemoryRegisterRequest {
        name: name.to_string(),
        key: format!("/{name}"),
        offset: 0,
        byte_size: size,
    };
    let _: SystemSharedMemoryRegisterResponse = unary(channel.clone(), REGISTER, &request).await?;
    Ok(())
}

async fn unregister(channel: &Channel, name: &str) -> Result<(), tonic::Status> {
    let request = SystemSharedMemoryUnregisterRequest { name: name.to_string() };
    let _: SystemSharedMemoryUnregisterResponse = unary(channel.clone(), UNREGISTER, &request).await?;
    Ok(())
}

/// Whether Triton failed a request because it does not know a region it names, e.g. after a restart.
pub(super) fn is_missing_region(status: &tonic::Status) -> bool {
    status.message().contains("Unable to find shared memory region")
}

/// Whether Triton refused to register a region because it already has one of that name.
fn is_already_registered(status: &tonic::Status) -> bool {
    status.message().contains("already in manager")
}

struct Pool {
    endpoint: String,
    /// Bytes the regions of the pool may take at most.
    limit: u64,
    state: Mutex<PoolState>,
    /// Held while the pool is activated, so that one request does it.
    activation: tokio::sync::Mutex<()>,
}

impl Pool {
    fn state(&self) -> std::sync::MutexGuard<'_, PoolState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Takes `bytes` more of the limit, or warns, at most once a minute, that the limit is reached.
    fn reserve(&self, bytes: u64, log: &Log) -> bool {
        let mut state = self.state();
        if state.reserved_bytes + bytes <= self.limit {
            state.reserved_bytes += bytes;
            return true;
        }
        let now = Instant::now();
        if state.limit_warned_at.is_none_or(|at| now.duration_since(at) >= LIMIT_WARNING_INTERVAL) {
            state.limit_warned_at = Some(now);
            drop(state);
            log(
                Level::Warning,
                format_args!(
                    "shared memory of {} reached inference.shared-memory-limit = {} bytes, more inputs go over gRPC",
                    self.endpoint, self.limit
                ),
            );
        }
        false
    }

    /// Drops a region that could not be registered at `size` bytes, which the pool had reserved.
    fn discard(&self, region: Region, size: u64) {
        {
            let mut state = self.state();
            state.reserved_bytes -= size;
            state.names.retain(|name| *name != region.name);
        }
        let _ = std::fs::remove_file(region_path(&region.name));
    }

    /// Sends the inputs over gRPC for an hour, warning once until shared memory works again.
    fn fall_back(&self, error: &anyhow::Error, log: &Log) {
        let mut state = self.state();
        state.mode = Mode::Fallback {
            retry_at: Instant::now() + RETRY_INTERVAL,
        };
        if !std::mem::replace(&mut state.fallback_warned, true) {
            drop(state);
            log(
                Level::Warning,
                format_args!(
                    "no shared memory for {}, its inputs go over gRPC; tried again once an hour: {error:#}",
                    self.endpoint
                ),
            );
        }
    }
}

struct PoolState {
    mode: Mode,
    free: Vec<Region>,
    /// All regions of the pool, free or leased.
    names: Vec<String>,
    /// Bytes of all regions of the pool, counted against its limit.
    reserved_bytes: u64,
    /// The size of the largest input so far, which every region of the pool grows to when it is taken.
    region_size: u64,
    /// A connection the regions were registered through, to unregister them at shutdown.
    channel: Option<Channel>,
    fallback_warned: bool,
    limit_warned_at: Option<Instant>,
}

#[derive(Clone, Copy)]
enum Mode {
    Untried,
    Active,
    Fallback { retry_at: Instant },
}

impl Mode {
    /// Whether the pool is to be activated now: never tried, or its hour of fallback is over.
    fn activation_due(self) -> bool {
        match self {
            Mode::Untried => true,
            Mode::Active => false,
            Mode::Fallback { retry_at } => Instant::now() >= retry_at,
        }
    }
}

struct Region {
    name: String,
    file: File,
    size: u64,
}

/// A region taken from a pool for one request; it goes back to the pool when dropped.
pub(super) struct Lease {
    pool: Arc<Pool>,
    region: Option<Region>,
    channel: Channel,
}

impl Lease {
    fn region(&self) -> &Region {
        self.region.as_ref().expect("a lease holds its region until dropped")
    }

    pub(super) fn name(&self) -> &str {
        &self.region().name
    }

    /// Writes the input at the start of the region.
    pub(super) fn write(&self, data: &[u8]) -> std::io::Result<()> {
        self.region().file.write_all_at(data, 0)
    }

    /// Registers the region again, after the server forgot it; a region it still knows is left as it is.
    pub(super) async fn register_again(&self) -> Result<(), tonic::Status> {
        let region = self.region();
        match register(&self.channel, &region.name, region.size).await {
            Err(status) if !is_already_registered(&status) => Err(status),
            _ => Ok(()),
        }
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        if let Some(region) = self.region.take() {
            self.pool.state().free.push(region);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_regions_are_told_from_other_errors() {
        assert!(is_missing_region(&tonic::Status::invalid_argument(
            "Unable to find shared memory region: 'falprs-9051-7-0'"
        )));
        assert!(!is_missing_region(&tonic::Status::invalid_argument("unexpected shape for input 'input.1'")));
    }

    #[test]
    fn registered_regions_are_told_from_other_errors() {
        assert!(is_already_registered(&tonic::Status::already_exists(
            "shared memory region 'falprs-9051-7-0' already in manager"
        )));
        assert!(!is_already_registered(&tonic::Status::internal(
            "Unable to open shared memory region: 'falprs-9051-7-0'"
        )));
    }

    #[test]
    fn old_region_files_of_the_instance_are_removed() {
        let dir = std::env::temp_dir().join(format!("falprs_shm_files_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for name in ["falprs-9051-100-0", "falprs-9051-200-3", "falprs-9061-100-0", "other"] {
            std::fs::write(dir.join(name), b"x").unwrap();
        }
        let shared_memory = SharedMemory::new(false, 1, 9051, Box::new(|_, _| {}));
        shared_memory.remove_files(&dir);
        let mut left: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        left.sort();
        assert_eq!(left, ["falprs-9061-100-0", "other"]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test]
    async fn disabled_shared_memory_leases_nothing() {
        let shared_memory = SharedMemory::disabled();
        let channel = tonic::transport::Endpoint::from_static("http://127.0.0.1:1").connect_lazy();
        assert!(shared_memory.lease("127.0.0.1:1", &channel, 10 * THRESHOLD).await.is_none());
    }
}
