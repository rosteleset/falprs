//! Connecting to the PostgreSQL cluster of a subsystem.

use std::sync::Arc;

use pg_cluster::{Cluster, ClusterSettings, LogLevel, LogSink};

use crate::config::DatabaseConfig;
use crate::logging::{Level, Location, Logger};

/// Connects to the cluster of `config`. Messages of the cluster itself, such as timeouts and host
/// role changes, go to `logger`, as userver's PostgreSQL driver writes them to the default logger.
pub async fn connect(config: &DatabaseConfig, logger: Arc<Logger>) -> pg_cluster::Result<Cluster> {
    let mut settings = ClusterSettings {
        command_control: config.command_control,
        discovery_timeout: config.discovery_timeout,
        log: LogSink::new(move |record| {
            let level = match record.level {
                LogLevel::Info => Level::Info,
                LogLevel::Warning => Level::Warning,
            };
            let location = Location {
                nested_fn: record.nested_fn,
                file: record.file,
                line: record.line,
            };
            logger.log(level, location, record.text);
        }),
        ..ClusterSettings::default()
    };
    if let Some(max_pool_size) = config.max_pool_size {
        settings.max_pool_size = max_pool_size;
    }
    Cluster::connect(&config.dsn, settings).await
}
