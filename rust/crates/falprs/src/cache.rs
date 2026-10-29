//! In-memory caches reloaded from PostgreSQL on a fixed interval, like userver's `PostgreCache`:
//! either fully on every update (`update-types: only-full`), or incrementally in between full
//! updates (`update-types: full-and-incremental`).

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, RwLock};
use std::time::Instant;

use chrono::{DateTime, Utc};

use anyhow::Result;
use pg_cluster::{Cluster, HostType};

use crate::config::CacheConfig;
use crate::logging::{Level, Logger};
use crate::log;

type Loader<T> = Box<dyn Fn(Cluster, HostType) -> Pin<Box<dyn Future<Output = Result<T>> + Send>> + Send + Sync>;

pub struct PgCache<T> {
    data: RwLock<Arc<T>>,
}

impl<T: Send + Sync + 'static> PgCache<T> {
    /// Loads the cache once, failing if that load fails, then keeps reloading it in the background.
    pub async fn start<F, Fut>(
        name: &'static str,
        cluster: Cluster,
        config: &CacheConfig,
        logger: Arc<Logger>,
        load: F,
    ) -> Result<Arc<Self>>
    where
        F: Fn(Cluster, HostType) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<T>> + Send + 'static,
    {
        let load: Loader<T> = Box::new(move |cluster, host_type| Box::pin(load(cluster, host_type)));
        let initial = load(cluster.clone(), config.host_type)
            .await
            .map_err(|e| e.context(format!("first update of cache {name} failed")))?;
        let cache = Arc::new(Self {
            data: RwLock::new(Arc::new(initial)),
        });

        let weak = Arc::downgrade(&cache);
        let (interval, host_type) = (config.update_interval, config.host_type);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(interval).await;
                let Some(cache) = weak.upgrade() else {
                    return;
                };
                match load(cluster.clone(), host_type).await {
                    Ok(data) => *cache.data.write().unwrap_or_else(|e| e.into_inner()) = Arc::new(data),
                    // Keep serving the previous data
                    Err(e) => log!(logger, Level::Error, "update of cache {name} failed: {e:#}"),
                }
            }
        });
        Ok(cache)
    }

    /// Starts a cache built from rows: `load` returns all rows when given no time, or the rows
    /// updated since the given time, and `apply` merges rows into the data. A full update applies
    /// the rows to empty data; an incremental one applies them to a copy of the current data.
    pub async fn start_incremental<R, F, Fut, A>(
        name: &'static str,
        cluster: Cluster,
        config: &CacheConfig,
        logger: Arc<Logger>,
        load: F,
        apply: A,
    ) -> Result<Arc<Self>>
    where
        T: Default + Clone,
        R: Send + 'static,
        F: Fn(Cluster, HostType, Option<DateTime<Utc>>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Vec<R>>> + Send + 'static,
        A: Fn(&mut T, Vec<R>) + Send + Sync + 'static,
    {
        let host_type = config.host_type;
        let full = move |rows: Vec<R>, apply: &A| {
            let mut data = T::default();
            apply(&mut data, rows);
            data
        };
        let mut last_update = Utc::now();
        let rows = load(cluster.clone(), host_type, None)
            .await
            .map_err(|e| e.context(format!("first update of cache {name} failed")))?;
        let cache = Arc::new(Self {
            data: RwLock::new(Arc::new(full(rows, &apply))),
        });

        let weak = Arc::downgrade(&cache);
        let (interval, full_interval, correction) =
            (config.update_interval, config.full_update_interval, config.update_correction);
        tokio::spawn(async move {
            let mut last_full_update = Instant::now();
            loop {
                tokio::time::sleep(interval).await;
                let Some(cache) = weak.upgrade() else {
                    return;
                };
                let is_full = full_interval.is_none_or(|full_interval| last_full_update.elapsed() >= full_interval);
                let since = (!is_full).then(|| last_update - chrono::TimeDelta::from_std(correction).unwrap_or_default());
                let started = (Utc::now(), Instant::now());
                match load(cluster.clone(), host_type, since).await {
                    Ok(rows) => {
                        let data = if is_full {
                            full(rows, &apply)
                        } else {
                            let mut data = T::clone(&cache.get());
                            apply(&mut data, rows);
                            data
                        };
                        *cache.data.write().unwrap_or_else(|e| e.into_inner()) = Arc::new(data);
                        last_update = started.0;
                        if is_full {
                            last_full_update = started.1;
                        }
                    }
                    // Keep serving the previous data
                    Err(e) => log!(logger, Level::Error, "update of cache {name} failed: {e:#}"),
                }
            }
        });
        Ok(cache)
    }

    pub fn get(&self) -> Arc<T> {
        self.data.read().unwrap_or_else(|e| e.into_inner()).clone()
    }
}
