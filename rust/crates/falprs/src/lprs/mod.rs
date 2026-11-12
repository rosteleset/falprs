//! Licence plate recognition subsystem.

mod api;
mod caches;
mod footprints;
mod pipeline;
mod vstream_config;
mod workflow;

use std::sync::Arc;

use anyhow::{Context, Result};
use axum::Router;
use pg_cluster::Cluster;

use crate::cache::PgCache;
use crate::config::{LprsConfig, LprsSettings};
use crate::config_schema::StoredConfigWarnings;
use crate::logging::{Level, Logger};
use crate::pipeline_slots::PipelineSlots;
use crate::triton::Triton;
use caches::{GroupsData, VStreamGroupData, VStreamsConfigData};
use workflow::Workflows;

pub use vstream_config::SCHEMA;

pub struct Lprs {
    cluster: Cluster,
    logger: Arc<Logger>,
    groups: Arc<PgCache<GroupsData>>,
    vstreams_config: Arc<PgCache<VStreamsConfigData>>,
    vstream_group: Arc<PgCache<VStreamGroupData>>,
    triton: Triton,
    http: reqwest::Client,
    /// Bounds the workflows' pipelines running at once.
    pipeline_slots: PipelineSlots,
    settings: LprsSettings,
    workflows: Workflows,
}

impl Lprs {
    /// Starts the subsystem; `default_logger` gets the messages of its database cluster.
    pub async fn start(config: &LprsConfig, logger: Arc<Logger>, default_logger: Arc<Logger>) -> Result<Arc<Self>> {
        let cluster = crate::database::connect(&config.database, default_logger)
            .await
            .context("can't connect to the LPRS database")?;

        let groups = PgCache::start("lprs.caches.groups", cluster.clone(), &config.groups_cache, logger.clone(), caches::load_groups).await?;
        let warnings = StoredConfigWarnings::new(logger.clone());
        let vstreams_config = PgCache::start(
            "lprs.caches.video-streams-config",
            cluster.clone(),
            &config.video_streams_config_cache,
            logger.clone(),
            move |cluster, host_type| caches::load_vstreams_config(cluster, host_type, warnings.clone()),
        )
        .await?;
        let vstream_group = PgCache::start(
            "lprs.caches.video-stream-groups",
            cluster.clone(),
            &config.video_stream_groups_cache,
            logger.clone(),
            caches::load_vstream_group,
        )
        .await?;

        let pipeline_slots = PipelineSlots::per_cpu(config.settings.pipelines_per_cpu);
        crate::log!(logger, Level::Info, "recognition pipeline slots: {}", pipeline_slots.count());
        let lprs = Arc::new(Self {
            cluster,
            logger,
            groups,
            vstreams_config,
            vstream_group,
            triton: Triton::default(),
            http: reqwest::Client::builder().build().context("can't create HTTP client")?,
            pipeline_slots,
            settings: config.settings.clone(),
            workflows: Workflows::default(),
        });
        lprs.start_maintenance();
        Ok(lprs)
    }

    /// Periodic ban list, events log and removed video streams cleanup; an interval of zero disables a task.
    fn start_maintenance(self: &Arc<Self>) {
        let weak = Arc::downgrade(self);
        let interval = self.settings.ban_cleanup_interval;
        if !interval.is_zero() {
            let weak = weak.clone();
            crate::tasks::spawn(async move {
                loop {
                    tokio::time::sleep(interval).await;
                    let Some(lprs) = weak.upgrade() else { return };
                    workflow::ban_maintenance(&lprs);
                }
            });
        }
        let interval = self.settings.events_log_cleanup_interval;
        if !interval.is_zero() {
            let weak = weak.clone();
            crate::tasks::spawn(async move {
                loop {
                    tokio::time::sleep(interval).await;
                    let Some(lprs) = weak.upgrade() else { return };
                    workflow::events_log_maintenance(&lprs).await;
                }
            });
        }
        let interval = self.settings.flag_deleted_cleanup_interval;
        if !interval.is_zero() {
            crate::tasks::spawn(async move {
                loop {
                    tokio::time::sleep(interval).await;
                    let Some(lprs) = weak.upgrade() else { return };
                    workflow::flag_deleted_maintenance(&lprs).await;
                }
            });
        }
    }

    pub fn router(self: &Arc<Self>) -> Router {
        api::router(self.clone())
    }
}
