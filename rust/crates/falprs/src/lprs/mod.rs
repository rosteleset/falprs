//! Licence plate recognition subsystem.

mod api;
mod caches;
mod geometry;
mod obb;
mod pipeline;
mod vstream_config;
mod workflow;

use std::sync::Arc;

use anyhow::{Context, Result};
use axum::Router;
use axum::extract::DefaultBodyLimit;
use pg_cluster::{Cluster, ClusterSettings};

use crate::cache::PgCache;
use crate::config::LprsConfig;
use crate::converters;
use crate::logging::Logger;
use crate::triton::Triton;
use caches::{GroupsData, VStreamGroupData, VStreamsConfigData};
use workflow::{WorkflowSettings, Workflows};

pub struct Lprs {
    cluster: Cluster,
    logger: Arc<Logger>,
    /// Stream group used for requests without an `Authorization` header; 0 or less forbids them.
    allow_group_id_without_auth: i32,
    /// The largest API request body accepted.
    api_max_request_size: usize,
    groups: Arc<PgCache<GroupsData>>,
    vstreams_config: Arc<PgCache<VStreamsConfigData>>,
    vstream_group: Arc<PgCache<VStreamGroupData>>,
    triton: Triton,
    http: reqwest::Client,
    settings: WorkflowSettings,
    workflows: Workflows,
}

impl Lprs {
    pub async fn start(config: &LprsConfig, logger: Arc<Logger>) -> Result<Arc<Self>> {
        let mut settings = ClusterSettings::default();
        if let Some(max_pool_size) = config.database.max_pool_size {
            settings.max_pool_size = max_pool_size;
        }
        let cluster = Cluster::connect(&config.database.dsn, settings)
            .await
            .context("can't connect to the LPRS database")?;

        let groups = PgCache::start("lprs-groups-pg-cache", cluster.clone(), &config.groups_cache, logger.clone(), caches::load_groups).await?;
        let vstreams_config = PgCache::start(
            "lprs-vstreams-config-pg-cache",
            cluster.clone(),
            &config.vstreams_config_cache,
            logger.clone(),
            caches::load_vstreams_config,
        )
        .await?;
        let vstream_group = PgCache::start(
            "lprs-vstream-group-pg-cache",
            cluster.clone(),
            &config.vstream_group_cache,
            logger.clone(),
            caches::load_vstream_group,
        )
        .await?;

        let allow_group_id_without_auth =
            converters::to_i64(config.workflow.get("allow-group-id-without-auth"), 0) as i32;
        let lprs = Arc::new(Self {
            cluster,
            logger,
            allow_group_id_without_auth,
            api_max_request_size: config.api_max_request_size,
            groups,
            vstreams_config,
            vstream_group,
            triton: Triton::default(),
            http: reqwest::Client::builder().build().context("can't create HTTP client")?,
            settings: WorkflowSettings::from_config(&config.workflow),
            workflows: Workflows::default(),
        });
        lprs.start_maintenance();
        Ok(lprs)
    }

    /// Periodic ban list and events log cleanup; an interval of zero disables a task.
    fn start_maintenance(self: &Arc<Self>) {
        let weak = Arc::downgrade(self);
        let interval = self.settings.ban_maintenance_interval;
        if !interval.is_zero() {
            let weak = weak.clone();
            tokio::spawn(async move {
                loop {
                    tokio::time::sleep(interval).await;
                    let Some(lprs) = weak.upgrade() else { return };
                    workflow::ban_maintenance(&lprs);
                }
            });
        }
        let interval = self.settings.events_log_maintenance_interval;
        if !interval.is_zero() {
            tokio::spawn(async move {
                loop {
                    tokio::time::sleep(interval).await;
                    let Some(lprs) = weak.upgrade() else { return };
                    workflow::events_log_maintenance(&lprs).await;
                }
            });
        }
    }

    pub fn router(self: &Arc<Self>) -> Router {
        api::router(self.clone()).layer(DefaultBodyLimit::max(self.api_max_request_size))
    }
}
