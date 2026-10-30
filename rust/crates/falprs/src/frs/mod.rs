//! Face recognition subsystem.

mod api;
mod barcodes;
mod caches;
mod config;
mod events;
mod pipeline;
mod workflow;

use std::future::Future;
use std::sync::{Arc, Weak};
use std::time::Duration;

use anyhow::{Context, Result};
use axum::Router;
use axum::extract::DefaultBodyLimit;
use pg_cluster::{Cluster, ClusterSettings};

use crate::cache::PgCache;
use crate::config::FrsConfig;
use crate::logging::Logger;
use crate::triton::Triton;
use caches::{ConfigData, FaceDescriptorData, GroupsData, SGConfigData, SGDescriptorsData, VStreamDescriptorsData, VStreamsConfigData};
use config::LocalConfig;
use workflow::Workflows;

pub struct Frs {
    cluster: Cluster,
    logger: Arc<Logger>,
    local: LocalConfig,
    api_allows_get: bool,
    /// The largest API request body accepted.
    api_max_request_size: usize,
    groups: Arc<PgCache<GroupsData>>,
    config: Arc<PgCache<ConfigData>>,
    vstreams_config: Arc<PgCache<VStreamsConfigData>>,
    face_descriptors: Arc<PgCache<FaceDescriptorData>>,
    vstream_descriptors: Arc<PgCache<VStreamDescriptorsData>>,
    sg_config: Arc<PgCache<SGConfigData>>,
    sg_descriptors: Arc<PgCache<SGDescriptorsData>>,
    triton: Triton,
    http: reqwest::Client,
    workflows: Workflows,
}

impl Frs {
    pub async fn start(config: &FrsConfig, logger: Arc<Logger>) -> Result<Arc<Self>> {
        let mut settings = ClusterSettings::default();
        if let Some(max_pool_size) = config.database.max_pool_size {
            settings.max_pool_size = max_pool_size;
        }
        let cluster = Cluster::connect(&config.database.dsn, settings)
            .await
            .context("can't connect to the FRS database")?;

        let start = |name, cache_config| (name, cluster.clone(), cache_config, logger.clone());
        let (name, c, cfg, l) = start("frs-groups-pg-cache", &config.groups_cache);
        let groups = PgCache::start(name, c, cfg, l, caches::load_groups).await?;
        let (name, c, cfg, l) = start("frs-config-pg-cache", &config.config_cache);
        let common_config = PgCache::start(name, c, cfg, l, caches::load_config).await?;
        let (name, c, cfg, l) = start("frs-vstreams-config-pg-cache", &config.vstreams_config_cache);
        let vstreams_config =
            PgCache::start_incremental(name, c, cfg, l, caches::load_vstreams_config, caches::apply_vstreams_config).await?;
        let (name, c, cfg, l) = start("frs-face-descriptor-pg-cache", &config.face_descriptor_cache);
        let face_descriptors =
            PgCache::start_incremental(name, c, cfg, l, caches::load_face_descriptors, caches::apply_face_descriptors).await?;
        let (name, c, cfg, l) = start("frs-vstream-descriptors-pg-cache", &config.vstream_descriptors_cache);
        let vstream_descriptors = PgCache::start_incremental(
            name,
            c,
            cfg,
            l,
            caches::load_vstream_descriptors,
            caches::apply_vstream_descriptors,
        )
        .await?;
        let (name, c, cfg, l) = start("frs-sg-config-pg-cache", &config.sg_config_cache);
        let sg_config = PgCache::start(name, c, cfg, l, caches::load_sg_config).await?;
        let (name, c, cfg, l) = start("frs-sg-descriptors-pg-cache", &config.sg_descriptors_cache);
        let sg_descriptors =
            PgCache::start_incremental(name, c, cfg, l, caches::load_sg_descriptors, caches::apply_sg_descriptors).await?;

        let frs = Arc::new(Self {
            cluster,
            logger,
            local: LocalConfig::from_config(&config.workflow),
            api_allows_get: config.api_allows_get,
            api_max_request_size: config.api_max_request_size,
            groups,
            config: common_config,
            vstreams_config,
            face_descriptors,
            vstream_descriptors,
            sg_config,
            sg_descriptors,
            triton: Triton::default(),
            http: reqwest::Client::builder().build().context("can't create HTTP client")?,
            workflows: Workflows::default(),
        });
        workflow::load_dnn_stats(&frs);
        frs.start_maintenance();
        Ok(frs)
    }

    /// Runs `task` every `interval`, as long as the subsystem lives; a zero interval disables it.
    fn every<F, Fut>(self: &Arc<Self>, interval: Duration, task: F)
    where
        F: Fn(Arc<Self>) -> Fut + Send + 'static,
        Fut: Future<Output = ()> + Send,
    {
        if interval.is_zero() {
            return;
        }
        let weak: Weak<Self> = Arc::downgrade(self);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(interval).await;
                let Some(frs) = weak.upgrade() else { return };
                task(frs).await;
            }
        });
    }

    fn start_maintenance(self: &Arc<Self>) {
        let local = &self.local;
        self.every(local.clear_old_log_faces, |frs| async move { workflow::old_log_faces_maintenance(&frs).await });
        self.every(local.flag_deleted_maintenance_interval, |frs| async move {
            workflow::flag_deleted_maintenance(&frs).await
        });
        self.every(local.copy_events_maintenance_interval, |frs| async move {
            workflow::copy_events_maintenance(&frs).await
        });
        self.every(local.clear_old_events, |frs| async move { workflow::old_events_maintenance(&frs).await });
        self.every(local.clear_old_log_barcodes, |frs| async move { barcodes::old_log_barcodes_maintenance(&frs).await });
    }

    /// Saves the inference statistics, as the C++ service does when it stops.
    pub async fn shutdown(&self) {
        workflow::save_dnn_stats(self).await;
    }

    pub fn router(self: &Arc<Self>) -> Router {
        api::router(self.clone()).layer(DefaultBodyLimit::max(self.api_max_request_size))
    }
}
