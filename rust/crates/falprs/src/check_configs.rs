//! `falprs check-configs [--fix]`: checks the configs stored in the databases against their
//! schemas and, with `--fix`, cleans them: values the old lenient readers took are normalized,
//! keys stored in the wrong FRS group table are moved to the right one, and invalid values and
//! unknown keys are removed. The original configs of the changed rows are saved to a backup file
//! first.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use chrono::Local;
use pg_cluster::{HostType, Transaction};
use serde_json::{Map, Value, json};

use crate::config::{Config, DatabaseConfig};
use crate::config_schema::{Cleanup, Schema};
use crate::logging::Logger;
use crate::{frs, lprs};

/// A stored config: a row of one of the config tables.
#[derive(Debug, Clone)]
pub struct StoredRow {
    pub table: &'static str,
    /// `id_vstream` of a video stream config, `id_group` of a stream group's config.
    pub id: i32,
    pub id_group: i32,
    /// The row for messages, e.g. "stream group 1, video stream 7".
    pub label: String,
    pub config: Value,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    Normalized(Value),
    Moved(&'static str),
    Removed(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Change {
    pub table: &'static str,
    pub label: String,
    pub key: String,
    pub old: Value,
    pub action: Action,
}

/// A row whose config changes; `original` is None for a row the cleanup inserts.
#[derive(Debug, Clone, PartialEq)]
pub struct RowUpdate {
    pub table: &'static str,
    pub id: i32,
    pub original: Option<Value>,
    pub config: Value,
}

#[derive(Debug, Default, PartialEq)]
pub struct Plan {
    pub changes: Vec<Change>,
    pub updates: Vec<RowUpdate>,
}

/// The tables of a subsystem's configs and the schema of each.
struct Tables {
    /// Video stream config table, keyed by `id_vstream`.
    streams: &'static str,
    /// Stream group tables, keyed by `id_group`, with their schemas.
    groups: Vec<(&'static str, &'static Schema)>,
    stream_schema: &'static Schema,
}

fn lprs_tables() -> Tables {
    Tables {
        streams: "vstreams",
        groups: vec![("default_vstream_config", &lprs::SCHEMA)],
        stream_schema: &lprs::SCHEMA,
    }
}

fn frs_tables() -> Tables {
    Tables {
        streams: "video_streams",
        groups: vec![("default_vstream_config", &frs::VSTREAM_SCHEMA), ("common_config", &frs::COMMON_SCHEMA)],
        stream_schema: &frs::VSTREAM_SCHEMA,
    }
}

impl Tables {
    fn schema(&self, table: &str) -> &'static Schema {
        if table == self.streams {
            return self.stream_schema;
        }
        self.groups
            .iter()
            .find(|(name, _)| *name == table)
            .map(|(_, schema)| *schema)
            .expect("a config table")
    }

    /// The other stream group table, whose keys may have been stored in `table`.
    fn other_group_table(&self, table: &str) -> Option<(&'static str, &'static Schema)> {
        if table == self.streams {
            return None;
        }
        self.groups.iter().find(|(name, _)| *name != table).copied()
    }
}

/// Whether the cleanup keeps the value of `key` in `config`.
fn keeps(schema: &Schema, config: &Value, key: &str) -> bool {
    config.get(key).is_some_and(|value| !matches!(schema.cleanup(key, value), Cleanup::Remove(_)))
}

/// Plans the cleanup of a subsystem's stored configs.
fn plan(tables: &Tables, rows: &[StoredRow]) -> Plan {
    let mut plan = Plan::default();
    // The new configs by (table, id), starting from the stored ones
    let mut configs: BTreeMap<(&'static str, i32), Map<String, Value>> = BTreeMap::new();
    let mut originals: HashMap<(&'static str, i32), Value> = HashMap::new();
    for row in rows {
        if let Value::Object(config) = &row.config {
            configs.insert((row.table, row.id), config.clone());
            originals.insert((row.table, row.id), row.config.clone());
        }
    }
    for row in rows {
        let Value::Object(config) = &row.config else {
            continue;
        };
        let schema = tables.schema(row.table);
        for (key, value) in config {
            let mut change = |action: Action| {
                plan.changes.push(Change {
                    table: row.table,
                    label: row.label.clone(),
                    key: key.clone(),
                    old: value.clone(),
                    action,
                })
            };
            let own = configs.get_mut(&(row.table, row.id)).expect("the row's config");
            if schema.spec(key).is_some() {
                match schema.cleanup(key, value) {
                    Cleanup::Keep => {}
                    Cleanup::Normalize(canonical) => {
                        own.insert(key.clone(), canonical.clone());
                        change(Action::Normalized(canonical));
                    }
                    Cleanup::Remove(why) => {
                        own.remove(key);
                        change(Action::Removed(why));
                    }
                }
                continue;
            }
            own.remove(key);
            // A key of the other stream group table moves there, unless that table sets it
            let other = tables.other_group_table(row.table).filter(|(_, other)| other.spec(key).is_some());
            let Some((other_table, other_schema)) = other else {
                let Cleanup::Remove(why) = schema.cleanup(key, value) else {
                    unreachable!("an unknown key is removed")
                };
                change(Action::Removed(why));
                continue;
            };
            let target = (other_table, row.id_group);
            let canonical = match other_schema.cleanup(key, value) {
                Cleanup::Keep => value.clone(),
                Cleanup::Normalize(canonical) => canonical,
                Cleanup::Remove(why) => {
                    change(Action::Removed(format!("{why} (a key of the {other_table})")));
                    continue;
                }
            };
            if originals.get(&target).is_some_and(|stored| keeps(other_schema, stored, key)) {
                change(Action::Removed(format!("set in the {other_table} already")));
                continue;
            }
            configs.entry(target).or_default().insert(key.clone(), canonical);
            change(Action::Moved(other_table));
        }
    }
    for ((table, id), config) in configs {
        let config = Value::Object(config);
        let original = originals.get(&(table, id)).cloned();
        if original.as_ref() != Some(&config) {
            plan.updates.push(RowUpdate { table, id, original, config });
        }
    }
    plan
}

/// Reads a subsystem's stored configs in `transaction`, locking the rows.
async fn read_rows(transaction: &Transaction<'_>, tables: &Tables, ext_column: &str) -> Result<Vec<StoredRow>> {
    let mut rows = Vec::new();
    let query = format!(
        "select id_vstream, coalesce(id_group, 0), {ext_column}, config from {} order by id_vstream for update",
        tables.streams
    );
    for row in transaction.query(&query, &[]).await? {
        let (id_group, ext): (i32, String) = (row.get(1), row.get(2));
        rows.push(StoredRow {
            table: tables.streams,
            id: row.get(0),
            id_group,
            label: format!("stream group {id_group}, video stream {ext}"),
            config: row.get::<_, Option<Value>>(3).unwrap_or(Value::Null),
        });
    }
    for (table, _) in &tables.groups {
        let query = format!("select id_group, config from {table} order by id_group for update");
        for row in transaction.query(&query, &[]).await? {
            let id_group: i32 = row.get(0);
            rows.push(StoredRow {
                table,
                id: id_group,
                id_group,
                label: format!("stream group {id_group}"),
                config: row.get::<_, Option<Value>>(1).unwrap_or(Value::Null),
            });
        }
    }
    Ok(rows)
}

async fn apply(transaction: &Transaction<'_>, tables: &Tables, updates: &[RowUpdate]) -> Result<()> {
    for update in updates {
        if update.table == tables.streams {
            // The running service picks the change up with its next cache update
            let query = format!("update {} set config = $1, last_updated = now() where id_vstream = $2", update.table);
            transaction.execute(&query, &[&update.config, &update.id]).await?;
        } else if update.original.is_some() {
            let query = format!("update {} set config = $1 where id_group = $2", update.table);
            transaction.execute(&query, &[&update.config, &update.id]).await?;
        } else {
            let query = format!("insert into {} (id_group, config) values ($2, $1)", update.table);
            transaction.execute(&query, &[&update.config, &update.id]).await?;
        }
    }
    Ok(())
}

fn print_changes(subsystem: &str, changes: &[Change]) {
    for change in changes {
        let action = match &change.action {
            Action::Normalized(value) => format!("normalized to {value}"),
            Action::Moved(table) => format!("moved to the {table}"),
            Action::Removed(why) => format!("removed: {why}"),
        };
        println!("{subsystem} {}, {}: `{}` = {}: {action}", change.table, change.label, change.key, change.old);
    }
}

fn backup_path(config_path: &Path) -> PathBuf {
    let dir = config_path.parent().filter(|dir| !dir.as_os_str().is_empty()).unwrap_or(Path::new("."));
    dir.join(format!("config-cleanup-{}.json", Local::now().format("%Y%m%d-%H%M%S")))
}

fn backup(updates: &[(&str, &Tables, &[RowUpdate])]) -> Value {
    let rows: Vec<Value> = updates
        .iter()
        .flat_map(|(subsystem, tables, updates)| {
            updates.iter().map(move |update| {
                let id_name = if update.table == tables.streams { "id_vstream" } else { "id_group" };
                json!({"subsystem": subsystem, "table": update.table, id_name: update.id, "config": update.original})
            })
        })
        .collect();
    json!({"time": Local::now().to_rfc3339(), "rows": rows})
}

/// Runs the subcommand; returns the process exit code.
pub async fn run(config_path: &Path, fix: bool) -> Result<i32> {
    let config = Config::load(config_path)?;
    let logger = Arc::new(Logger::stderr()?);
    let mut subsystems: Vec<(&str, &DatabaseConfig, Tables, &str)> = Vec::new();
    if let Some(lprs) = &config.lprs {
        subsystems.push(("lprs", &lprs.database, lprs_tables(), "ext_id"));
    }
    if let Some(frs) = &config.frs {
        subsystems.push(("frs", &frs.database, frs_tables(), "vstream_ext"));
    }
    let mut clients = Vec::new();
    for (name, database, ..) in &subsystems {
        let cluster = crate::database::connect(database, logger.clone())
            .await
            .with_context(|| format!("can't connect to the {} database", name.to_uppercase()))?;
        clients.push(cluster.client(HostType::Master).await?);
    }
    let mut transactions = Vec::new();
    for client in &mut clients {
        transactions.push(client.transaction().await?);
    }
    let mut plans = Vec::new();
    for ((name, _, tables, ext_column), transaction) in subsystems.iter().zip(&transactions) {
        let rows = read_rows(transaction, tables, ext_column).await?;
        let plan = plan(tables, &rows);
        print_changes(name, &plan.changes);
        plans.push(plan);
    }
    let changes: usize = plans.iter().map(|plan| plan.changes.len()).sum();
    if changes == 0 {
        println!("The stored configs are clean");
        return Ok(0);
    }
    if !fix {
        println!("{changes} change(s) needed; run with --fix to apply them");
        return Ok(1);
    }
    let path = backup_path(config_path);
    let updates: Vec<_> = subsystems
        .iter()
        .zip(&plans)
        .map(|((name, _, tables, _), plan)| (*name, tables, plan.updates.as_slice()))
        .collect();
    let text = serde_json::to_string_pretty(&backup(&updates))?;
    std::fs::write(&path, text).with_context(|| format!("can't write the backup {}", path.display()))?;
    println!("The original configs of the changed rows are saved to {}", path.display());
    for (((_, _, tables, _), plan), transaction) in subsystems.iter().zip(&plans).zip(&transactions) {
        apply(transaction, tables, &plan.updates).await?;
    }
    for transaction in transactions {
        transaction.commit().await?;
    }
    println!("{changes} change(s) applied");
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(table: &'static str, id: i32, id_group: i32, config: Value) -> StoredRow {
        StoredRow {
            table,
            id,
            id_group,
            label: format!("row {id}"),
            config,
        }
    }

    fn update<'a>(plan: &'a Plan, table: &str, id: i32) -> Option<&'a RowUpdate> {
        plan.updates.iter().find(|update| update.table == table && update.id == id)
    }

    #[test]
    fn lenient_values_are_normalized_and_ignored_ones_removed() {
        let rows = [row(
            "vstreams",
            7,
            1,
            json!({"flag-save-failed": "yes", "plate-confidence": "0.6", "ban-duration": 30, "char-score": 0.4,
                   "ovd-net-inference-server": "127.0.0.1:8001", "logs-level": null, "vehicle-confidence": 1.5}),
        )];
        let plan = plan(&lprs_tables(), &rows);
        let config = &update(&plan, "vstreams", 7).unwrap().config;
        assert_eq!(config, &json!({"flag-save-failed": true, "plate-confidence": 0.6, "char-score": 0.4}));
        assert_eq!(plan.changes.len(), 6);
        assert!(plan.changes.contains(&Change {
            table: "vstreams",
            label: "row 7".into(),
            key: "flag-save-failed".into(),
            old: json!("yes"),
            action: Action::Normalized(json!(true)),
        }));
    }

    #[test]
    fn frs_keys_move_to_their_group_table() {
        let rows = [
            row(
                "common_config",
                1,
                1,
                json!({"inference-timeout": "1s", "logs-level": "trace", "callback-timeout": "2s",
                                               "comments-url-image-error": "Failed to receive image."}),
            ),
            row(
                "default_vstream_config",
                1,
                1,
                json!({"logs-level": "info", "sg-max-descriptor-count": "500", "open-door-duration": "10s"}),
            ),
            row("video_streams", 5, 1, json!({"blur": 300, "comments-no-faces": "none"})),
            // A stream group without a default video stream config row
            row("common_config", 2, 2, json!({"margin": 10})),
        ];
        let plan = plan(&frs_tables(), &rows);
        assert_eq!(
            update(&plan, "common_config", 1).unwrap().config,
            json!({"callback-timeout": "2s", "sg-max-descriptor-count": 500})
        );
        // logs-level is set in the default video stream config already
        assert_eq!(
            update(&plan, "default_vstream_config", 1).unwrap().config,
            json!({"logs-level": "info", "inference-timeout": "1s"})
        );
        // A common config key is not a video stream config key
        assert_eq!(update(&plan, "video_streams", 5).unwrap().config, json!({"blur": 300}));
        let inserted = update(&plan, "default_vstream_config", 2).unwrap();
        assert_eq!((inserted.original.clone(), inserted.config.clone()), (None, json!({"margin": 10})));
        assert_eq!(update(&plan, "common_config", 2).unwrap().config, json!({}));
        let moved: Vec<_> = plan
            .changes
            .iter()
            .filter(|c| matches!(c.action, Action::Moved(_)))
            .map(|c| c.key.as_str())
            .collect();
        assert_eq!(moved, vec!["inference-timeout", "sg-max-descriptor-count", "margin"]);
    }

    #[test]
    fn clean_configs_need_nothing_and_a_second_run_finds_nothing() {
        let rows = vec![
            row("common_config", 1, 1, json!({"inference-timeout": "1s", "flag-copy-event-data": "true"})),
            row("default_vstream_config", 1, 1, json!({"blur": "300"})),
            row("video_streams", 3, 1, Value::Null),
        ];
        let tables = frs_tables();
        let first = plan(&tables, &rows);
        assert!(!first.changes.is_empty());
        let cleaned: Vec<_> = rows
            .iter()
            .map(|r| {
                let config = update(&first, r.table, r.id).map_or(r.config.clone(), |u| u.config.clone());
                StoredRow { config, ..r.clone() }
            })
            .collect();
        assert_eq!(plan(&tables, &cleaned), Plan::default());
    }
}
