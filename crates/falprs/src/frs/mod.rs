//! Face recognition subsystem.

mod api;
mod barcodes;
mod caches;
mod config;
mod descriptor_index;
mod events;
mod pipeline;
mod workflow;

use std::future::Future;
use std::sync::{Arc, Weak};
use std::time::Duration;

use anyhow::{Context, Result};
use axum::Router;
use pg_cluster::Cluster;

use crate::cache::PgCache;
use crate::config::{FrsConfig, FrsSettings};
use crate::config_schema::StoredConfigWarnings;
use crate::logging::{Level, Logger};
use crate::pipeline_slots::PipelineSlots;
use crate::triton::Triton;
use caches::{ConfigData, FaceDescriptorData, GroupsData, SGConfigData, SGDescriptorsData, VStreamDescriptorsData, VStreamsConfigData};
use descriptor_index::DescriptorLists;
use workflow::Workflows;

pub use config::{COMMON_SCHEMA, VSTREAM_SCHEMA};

pub struct Frs {
    cluster: Cluster,
    logger: Arc<Logger>,
    local: FrsSettings,
    groups: Arc<PgCache<GroupsData>>,
    config: Arc<PgCache<ConfigData>>,
    vstreams_config: Arc<PgCache<VStreamsConfigData>>,
    face_descriptors: Arc<PgCache<FaceDescriptorData>>,
    vstream_descriptors: Arc<PgCache<VStreamDescriptorsData>>,
    sg_config: Arc<PgCache<SGConfigData>>,
    sg_descriptors: Arc<PgCache<SGDescriptorsData>>,
    /// Descriptor lists of the video streams and of the special groups, for recognition.
    vstream_lists: DescriptorLists,
    sg_lists: DescriptorLists,
    triton: Triton,
    http: reqwest::Client,
    /// Bounds the recognition loops' pipelines running at once.
    pipeline_slots: PipelineSlots,
    workflows: Workflows,
}

impl Frs {
    /// Starts the subsystem; `default_logger` gets the messages of its database cluster.
    pub async fn start(config: &FrsConfig, logger: Arc<Logger>, default_logger: Arc<Logger>) -> Result<Arc<Self>> {
        let cluster = crate::database::connect(&config.database, default_logger)
            .await
            .context("can't connect to the FRS database")?;

        let start = |name, cache_config| (name, cluster.clone(), cache_config, logger.clone());
        let (name, c, cfg, l) = start("frs.caches.groups", &config.groups_cache);
        let groups = PgCache::start(name, c, cfg, l, caches::load_groups).await?;
        let (name, c, cfg, l) = start("frs.caches.config", &config.config_cache);
        let warnings = StoredConfigWarnings::new(logger.clone());
        let load_config = {
            let warnings = warnings.clone();
            move |cluster, host_type| caches::load_config(cluster, host_type, warnings.clone())
        };
        let common_config = PgCache::start(name, c, cfg, l, load_config).await?;
        let (name, c, cfg, l) = start("frs.caches.video-streams-config", &config.video_streams_config_cache);
        let load_vstreams_config = move |cluster, host_type, since| caches::load_vstreams_config(cluster, host_type, since, warnings.clone());
        let vstreams_config = PgCache::start_incremental(name, c, cfg, l, load_vstreams_config, caches::apply_vstreams_config).await?;
        let (name, c, cfg, l) = start("frs.caches.face-descriptors", &config.face_descriptors_cache);
        let face_descriptors = PgCache::start_incremental(name, c, cfg, l, caches::load_face_descriptors, caches::apply_face_descriptors).await?;
        let (name, c, cfg, l) = start("frs.caches.video-stream-descriptors", &config.video_stream_descriptors_cache);
        let vstream_descriptors = PgCache::start_incremental(name, c, cfg, l, caches::load_vstream_descriptors, caches::apply_vstream_descriptors).await?;
        let (name, c, cfg, l) = start("frs.caches.sg-config", &config.sg_config_cache);
        let sg_config = PgCache::start(name, c, cfg, l, caches::load_sg_config).await?;
        let (name, c, cfg, l) = start("frs.caches.sg-descriptors", &config.sg_descriptors_cache);
        let sg_descriptors = PgCache::start_incremental(name, c, cfg, l, caches::load_sg_descriptors, caches::apply_sg_descriptors).await?;

        let pipeline_slots = PipelineSlots::per_cpu(config.settings.pipelines_per_cpu);
        crate::log!(logger, Level::Info, "recognition pipeline slots: {}", pipeline_slots.count());
        let frs = Arc::new(Self {
            cluster,
            logger,
            local: config.settings.clone(),
            groups,
            config: common_config,
            vstreams_config,
            face_descriptors,
            vstream_descriptors,
            sg_config,
            sg_descriptors,
            vstream_lists: DescriptorLists::default(),
            sg_lists: DescriptorLists::default(),
            triton: Triton::default(),
            http: reqwest::Client::builder().build().context("can't create HTTP client")?,
            pipeline_slots,
            workflows: Workflows::default(),
        });
        workflow::load_dnn_stats(&frs).await;
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
        crate::tasks::spawn(async move {
            loop {
                tokio::time::sleep(interval).await;
                let Some(frs) = weak.upgrade() else { return };
                task(frs).await;
            }
        });
    }

    fn start_maintenance(self: &Arc<Self>) {
        let local = &self.local;
        self.every(local.log_faces_cleanup_interval, |frs| async move {
            workflow::old_log_faces_maintenance(&frs).await
        });
        self.every(local.flag_deleted_cleanup_interval, |frs| async move {
            workflow::flag_deleted_maintenance(&frs).await
        });
        self.every(local.copy_events_interval, |frs| async move { workflow::copy_events_maintenance(&frs).await });
        self.every(local.events_cleanup_interval, |frs| async move { workflow::old_events_maintenance(&frs).await });
        self.every(local.log_barcodes_cleanup_interval, |frs| async move {
            barcodes::old_log_barcodes_maintenance(&frs).await
        });
    }

    /// Saves the inference statistics, as the C++ service does when it stops.
    pub async fn shutdown(&self) {
        workflow::save_dnn_stats(self).await;
    }

    pub fn router(self: &Arc<Self>) -> Router {
        api::router(self.clone())
    }
}
