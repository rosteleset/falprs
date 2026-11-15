//! Connecting to the PostgreSQL cluster of a subsystem, and the deletes of the log cleanups.

use std::future::Future;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use pg_cluster::tokio_postgres::types::ToSql;
use pg_cluster::{Cluster, ClusterSettings, HostType, LogLevel, LogSink};

use crate::config::DatabaseConfig;
use crate::logging::{Level, Location, Logger};

/// Connects to the cluster of `config`. Messages of the cluster itself, such as timeouts and host
/// role changes, go to `logger`, as userver's PostgreSQL driver writes them to the default logger.
pub async fn connect(config: &DatabaseConfig, logger: Arc<Logger>) -> pg_cluster::Result<Cluster> {
    let settings = ClusterSettings {
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
        max_pool_size: config.max_pool_size,
        ..ClusterSettings::default()
    };
    Cluster::connect(&config.dsn, settings).await
}

/// Rows deleted by one statement of [`delete_outdated`]: small enough to stay well under the
/// statement timeout on a slow disk and to spread the WAL of a large cleanup, large enough that a
/// cleanup of an hour of events takes a few hundred statements.
pub const DELETE_BATCH: i64 = 5_000;

/// A log table cleaned by [`delete_outdated`]: its name and its `serial` primary key.
pub struct LogTable {
    pub name: &'static str,
    pub key: &'static str,
}

pub const LOG_FACES: LogTable = LogTable {
    name: "log_faces",
    key: "id_log",
};
pub const LOG_BARCODES: LogTable = LogTable {
    name: "log_barcodes",
    key: "id_log",
};
pub const EVENTS_LOG: LogTable = LogTable {
    name: "events_log",
    key: "id_event",
};

/// What [`delete_outdated`] did: the rows it deleted, and the error that stopped it, if any.
pub struct Deleted {
    pub rows: u64,
    pub error: Option<pg_cluster::Error>,
}

/// Deletes the rows of `table` whose `log_date` is before `before`, [`DELETE_BATCH`] rows per
/// statement, until a statement deletes fewer. Each statement takes the oldest outdated rows by the
/// primary key: it is assigned in insert order, close to `log_date` order, so they lead the key and
/// no index on `log_date` is needed. A row whose `log_date` is out of step with its key is left to a
/// later batch or cleanup. Each statement commits on its own, so a failed one keeps what the
/// previous ones deleted, and a shutdown stops the loop between statements.
///
/// The keys are passed as an array (`= any(array(...))`), not as `in (select ...)`: PostgreSQL
/// plans the latter as a hash semi join over a sequential scan of the whole table for every batch,
/// the former as primary key lookups. On a table of 1 M rows (PostgreSQL 16) a batch takes ~7 ms
/// this way, flat over the batches of a cleanup; only the last one walks the rest of the key to find
/// no more outdated rows, once per cleanup, as the single delete did.
pub async fn delete_outdated(cluster: &Cluster, table: &LogTable, before: DateTime<Utc>) -> Deleted {
    let LogTable { name, key } = table;
    let sql = format!("delete from {name} where {key} = any(array(select {key} from {name} where log_date < $1 order by {key} limit $2))");
    let params: [&(dyn ToSql + Sync); 2] = [&before, &DELETE_BATCH];
    in_batches(DELETE_BATCH, || cluster.execute(HostType::Master, &sql, &params)).await
}

/// Runs `batch` until it deletes fewer than `limit` rows or fails.
async fn in_batches<F>(limit: i64, mut batch: impl FnMut() -> F) -> Deleted
where
    F: Future<Output = pg_cluster::Result<u64>>,
{
    let limit = u64::try_from(limit).unwrap_or(0);
    let mut rows = 0;
    loop {
        match batch().await {
            Ok(deleted) => {
                rows += deleted;
                if deleted < limit {
                    return Deleted { rows, error: None };
                }
            }
            Err(e) => return Deleted { rows, error: Some(e) },
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    async fn run(results: Vec<pg_cluster::Result<u64>>) -> (Deleted, usize) {
        let mut results = results.into_iter();
        let mut calls = 0;
        let deleted = in_batches(10, || {
            calls += 1;
            std::future::ready(results.next().expect("no batch after the last one"))
        })
        .await;
        (deleted, calls)
    }

    #[tokio::test]
    async fn batches_go_on_while_full() {
        let (deleted, calls) = run(vec![Ok(10), Ok(10), Ok(3)]).await;
        assert_eq!((deleted.rows, calls), (23, 3));
        assert!(deleted.error.is_none());
        // An exact multiple of the limit takes one more, empty batch to see the end
        let (deleted, calls) = run(vec![Ok(10), Ok(0)]).await;
        assert_eq!((deleted.rows, calls), (10, 2));
    }

    #[tokio::test]
    async fn a_failed_batch_stops_and_keeps_the_count() {
        let timeout = pg_cluster::Error::NetworkTimeout(Duration::from_millis(500));
        let (deleted, calls) = run(vec![Ok(10), Err(timeout)]).await;
        assert_eq!((deleted.rows, calls), (10, 2));
        assert!(matches!(deleted.error, Some(pg_cluster::Error::NetworkTimeout(_))));
    }
}
