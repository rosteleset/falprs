//! Reading the existing userver `config.yaml`.
//!
//! Only the keys the service needs are deserialized; userver-specific sections are ignored.
//! userver rejects unknown keys anywhere in its static config, so settings only this service
//! understands live under `dynamic-config.defaults`, which userver does not validate.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use pg_cluster::HostType;
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::converters::parse_duration;

/// Dynamic config key mapping cache component names to the host type their fills read from.
const CACHE_HOST_TYPES_KEY: &str = "FALPRS_PG_CACHE_HOST_TYPES";

/// userver's default `max_request_size` of a handler.
const DEFAULT_MAX_REQUEST_SIZE: usize = 1024 * 1024;

/// Settings the service takes from `config.yaml`.
#[derive(Debug, Clone)]
pub struct Config {
    pub port: u16,
    pub monitor_port: Option<u16>,
    pub loggers: HashMap<String, LoggerConfig>,
    pub static_dir: Option<PathBuf>,
    pub lprs: Option<LprsConfig>,
    pub frs: Option<FrsConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LoggerConfig {
    pub file_path: String,
    #[serde(default = "default_log_level")]
    pub level: String,
}

fn default_log_level() -> String {
    "info".to_string()
}

#[derive(Debug, Clone)]
pub struct DatabaseConfig {
    pub dsn: String,
    pub max_pool_size: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub update_interval: Duration,
    /// Set for `update-types: full-and-incremental`: updates in between full ones are incremental.
    pub full_update_interval: Option<Duration>,
    /// How far back from the previous update's start an incremental update looks.
    pub update_correction: Duration,
    pub host_type: HostType,
}

#[derive(Debug, Clone)]
pub struct LprsConfig {
    pub database: DatabaseConfig,
    /// The largest API request body accepted, as userver's `max_request_size`.
    pub api_max_request_size: usize,
    pub groups_cache: CacheConfig,
    pub vstreams_config_cache: CacheConfig,
    pub vstream_group_cache: CacheConfig,
    /// The `config` section of `lprs-workflow`.
    pub workflow: Map<String, Value>,
}

#[derive(Debug, Clone)]
pub struct FrsConfig {
    pub database: DatabaseConfig,
    /// Whether the API also answers GET requests (only `motionDetection` handles them).
    pub api_allows_get: bool,
    /// The largest API request body accepted, as userver's `max_request_size`.
    pub api_max_request_size: usize,
    pub groups_cache: CacheConfig,
    pub config_cache: CacheConfig,
    pub vstreams_config_cache: CacheConfig,
    pub face_descriptor_cache: CacheConfig,
    pub vstream_descriptors_cache: CacheConfig,
    pub sg_config_cache: CacheConfig,
    pub sg_descriptors_cache: CacheConfig,
    /// The `config` section of `frs-workflow`.
    pub workflow: Map<String, Value>,
}

#[derive(Deserialize)]
struct Root {
    components_manager: ComponentsManager,
}

#[derive(Deserialize)]
struct ComponentsManager {
    components: Components,
}

#[derive(Deserialize)]
struct Components {
    server: Server,
    logging: Logging,
    #[serde(rename = "fs-cache-main")]
    fs_cache_main: Option<FsCache>,
    #[serde(rename = "dynamic-config")]
    dynamic_config: Option<DynamicConfig>,
    #[serde(rename = "lprs-api-http")]
    lprs_api: Option<Handler>,
    #[serde(rename = "lprs-postgresql-database")]
    lprs_database: Option<Database>,
    #[serde(rename = "lprs-groups-pg-cache")]
    lprs_groups_cache: Option<Cache>,
    #[serde(rename = "lprs-vstreams-config-pg-cache")]
    lprs_vstreams_config_cache: Option<Cache>,
    #[serde(rename = "lprs-vstream-group-pg-cache")]
    lprs_vstream_group_cache: Option<Cache>,
    #[serde(rename = "lprs-workflow")]
    lprs_workflow: Option<Workflow>,
    #[serde(rename = "frs-api-http")]
    frs_api: Option<Handler>,
    #[serde(rename = "frs-postgresql-database")]
    frs_database: Option<Database>,
    #[serde(rename = "frs-groups-pg-cache")]
    frs_groups_cache: Option<Cache>,
    #[serde(rename = "frs-config-pg-cache")]
    frs_config_cache: Option<Cache>,
    #[serde(rename = "frs-vstreams-config-pg-cache")]
    frs_vstreams_config_cache: Option<Cache>,
    #[serde(rename = "frs-face-descriptor-pg-cache")]
    frs_face_descriptor_cache: Option<Cache>,
    #[serde(rename = "frs-vstream-descriptors-pg-cache")]
    frs_vstream_descriptors_cache: Option<Cache>,
    #[serde(rename = "frs-sg-config-pg-cache")]
    frs_sg_config_cache: Option<Cache>,
    #[serde(rename = "frs-sg-descriptors-pg-cache")]
    frs_sg_descriptors_cache: Option<Cache>,
    #[serde(rename = "frs-workflow")]
    frs_workflow: Option<Workflow>,
}

#[derive(Deserialize)]
struct Handler {
    method: Option<String>,
    max_request_size: Option<usize>,
}

#[derive(Deserialize)]
struct Server {
    listener: Listener,
    #[serde(rename = "listener-monitor")]
    listener_monitor: Option<Listener>,
    #[serde(rename = "handler-defaults")]
    handler_defaults: Option<HandlerDefaults>,
}

#[derive(Deserialize)]
struct HandlerDefaults {
    max_request_size: Option<usize>,
}

#[derive(Deserialize)]
struct Listener {
    port: u16,
}

#[derive(Deserialize)]
struct Logging {
    loggers: HashMap<String, LoggerConfig>,
}

#[derive(Deserialize)]
struct FsCache {
    dir: PathBuf,
}

#[derive(Deserialize, Default)]
struct DynamicConfig {
    #[serde(default)]
    defaults: Map<String, Value>,
}

#[derive(Deserialize)]
struct Database {
    dbconnection: String,
}

#[derive(Deserialize)]
struct Cache {
    #[serde(rename = "update-interval")]
    update_interval: String,
    #[serde(rename = "full-update-interval")]
    full_update_interval: Option<String>,
    #[serde(rename = "update-correction")]
    update_correction: Option<String>,
    #[serde(rename = "update-types")]
    update_types: Option<String>,
}

#[derive(Deserialize)]
struct Workflow {
    #[serde(default)]
    config: Map<String, Value>,
}

fn parse_host_type(text: &str) -> Result<HostType> {
    match text {
        "master" => Ok(HostType::Master),
        "replica" => Ok(HostType::Replica),
        "replica-or-master" => Ok(HostType::ReplicaOrMaster),
        _ => bail!("unknown host type {text:?}, expected master, replica or replica-or-master"),
    }
}

fn cache_config(name: &str, cache: Option<Cache>, defaults: &Map<String, Value>) -> Result<CacheConfig> {
    let cache = cache.ok_or_else(|| anyhow!("component {name} is required"))?;
    let update_interval =
        parse_duration(&cache.update_interval).ok_or_else(|| anyhow!("{name}: invalid update-interval"))?;
    let full_update_interval = match cache.update_types.as_deref() {
        None | Some("only-full") => None,
        Some("full-and-incremental") => {
            let text = cache
                .full_update_interval
                .ok_or_else(|| anyhow!("{name}: full-update-interval is required for incremental updates"))?;
            Some(parse_duration(&text).ok_or_else(|| anyhow!("{name}: invalid full-update-interval"))?)
        }
        Some(other) => bail!("{name}: unsupported update-types {other:?}"),
    };
    let update_correction = match cache.update_correction {
        None => Duration::ZERO,
        Some(text) => parse_duration(&text).ok_or_else(|| anyhow!("{name}: invalid update-correction"))?,
    };
    let host_type = match defaults.get(CACHE_HOST_TYPES_KEY).and_then(|types| types.get(name)) {
        None => HostType::Master,
        Some(Value::String(text)) => parse_host_type(text).with_context(|| format!("{CACHE_HOST_TYPES_KEY}.{name}"))?,
        Some(_) => bail!("{CACHE_HOST_TYPES_KEY}.{name} must be a string"),
    };
    Ok(CacheConfig {
        update_interval,
        full_update_interval,
        update_correction,
        host_type,
    })
}

fn max_pool_size(defaults: &Map<String, Value>, database: &str) -> Option<usize> {
    defaults
        .get("POSTGRES_CONNECTION_POOL_SETTINGS")?
        .get(database)?
        .get("max_pool_size")?
        .as_u64()
        .map(|size| size as usize)
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("can't read config {}", path.display()))?;
        Self::parse(&text).with_context(|| format!("invalid config {}", path.display()))
    }

    pub fn parse(text: &str) -> Result<Self> {
        let root: Root = serde_saphyr::from_str(text)?;
        let components = root.components_manager.components;
        if !components.logging.loggers.contains_key("default") {
            bail!("logging.loggers.default is required");
        }
        let defaults = components.dynamic_config.unwrap_or_default().defaults;
        // A handler's own limit, else the server's handler defaults, else userver's default
        let default_max_request_size = components
            .server
            .handler_defaults
            .as_ref()
            .and_then(|defaults| defaults.max_request_size)
            .unwrap_or(DEFAULT_MAX_REQUEST_SIZE);
        let max_request_size =
            |handler: Option<&Handler>| handler.and_then(|h| h.max_request_size).unwrap_or(default_max_request_size);
        let lprs = match components.lprs_database {
            None => None,
            Some(database) => Some(LprsConfig {
                database: DatabaseConfig {
                    dsn: database.dbconnection,
                    max_pool_size: max_pool_size(&defaults, "lprs-postgresql-database"),
                },
                api_max_request_size: max_request_size(components.lprs_api.as_ref()),
                groups_cache: cache_config("lprs-groups-pg-cache", components.lprs_groups_cache, &defaults)?,
                vstreams_config_cache: cache_config(
                    "lprs-vstreams-config-pg-cache",
                    components.lprs_vstreams_config_cache,
                    &defaults,
                )?,
                vstream_group_cache: cache_config(
                    "lprs-vstream-group-pg-cache",
                    components.lprs_vstream_group_cache,
                    &defaults,
                )?,
                workflow: components.lprs_workflow.map(|w| w.config).unwrap_or_default(),
            }),
        };
        let frs = match components.frs_database {
            None => None,
            Some(database) => Some(FrsConfig {
                database: DatabaseConfig {
                    dsn: database.dbconnection,
                    max_pool_size: max_pool_size(&defaults, "frs-postgresql-database"),
                },
                api_max_request_size: max_request_size(components.frs_api.as_ref()),
                api_allows_get: components
                    .frs_api
                    .as_ref()
                    .and_then(|handler| handler.method.as_deref())
                    .is_some_and(|methods| methods.split(',').any(|method| method.trim() == "GET")),
                groups_cache: cache_config("frs-groups-pg-cache", components.frs_groups_cache, &defaults)?,
                config_cache: cache_config("frs-config-pg-cache", components.frs_config_cache, &defaults)?,
                vstreams_config_cache: cache_config(
                    "frs-vstreams-config-pg-cache",
                    components.frs_vstreams_config_cache,
                    &defaults,
                )?,
                face_descriptor_cache: cache_config(
                    "frs-face-descriptor-pg-cache",
                    components.frs_face_descriptor_cache,
                    &defaults,
                )?,
                vstream_descriptors_cache: cache_config(
                    "frs-vstream-descriptors-pg-cache",
                    components.frs_vstream_descriptors_cache,
                    &defaults,
                )?,
                sg_config_cache: cache_config("frs-sg-config-pg-cache", components.frs_sg_config_cache, &defaults)?,
                sg_descriptors_cache: cache_config(
                    "frs-sg-descriptors-pg-cache",
                    components.frs_sg_descriptors_cache,
                    &defaults,
                )?,
                workflow: components.frs_workflow.map(|w| w.config).unwrap_or_default(),
            }),
        };
        Ok(Self {
            port: components.server.listener.port,
            monitor_port: components.server.listener_monitor.map(|listener| listener.port),
            loggers: components.logging.loggers,
            static_dir: components.fs_cache_main.map(|cache| cache.dir),
            lprs,
            frs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config_text() -> String {
        std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../tests/test_config.yaml")).unwrap()
    }

    #[test]
    fn parses_test_config() {
        let config = Config::parse(&test_config_text()).unwrap();
        assert_eq!(config.port, 9071);
        assert_eq!(config.monitor_port, Some(9072));
        assert_eq!(config.loggers["default"].file_path, "/tmp/test_falprs/logs/log.txt");
        assert_eq!(config.loggers["default"].level, "error");
        assert_eq!(config.loggers["lprs"].level, "debug");
        assert_eq!(config.static_dir.as_deref(), Some(Path::new("/tmp/test_falprs/static/")));

        let lprs = config.lprs.unwrap();
        assert_eq!(lprs.database.dsn, "postgresql://test_falprs:123@localhost:5432/test_lprs");
        assert_eq!(lprs.database.max_pool_size, Some(15));
        assert_eq!(lprs.groups_cache.update_interval, Duration::from_secs(5));
        assert_eq!(lprs.groups_cache.host_type, HostType::Master);
        assert_eq!(lprs.workflow["allow-group-id-without-auth"], 1);
        assert_eq!(lprs.groups_cache.full_update_interval, None);
        assert_eq!(lprs.api_max_request_size, 1024 * 1024);

        let frs = config.frs.unwrap();
        assert_eq!(frs.database.dsn, "postgresql://test_falprs:123@localhost:5432/test_frs");
        assert!(!frs.api_allows_get);
        assert_eq!(frs.face_descriptor_cache.update_interval, Duration::from_secs(5));
        assert_eq!(frs.face_descriptor_cache.full_update_interval, Some(Duration::from_secs(600)));
        assert_eq!(frs.face_descriptor_cache.update_correction, Duration::from_secs(1));
        assert_eq!(frs.sg_config_cache.full_update_interval, None);
        assert_eq!(frs.workflow["events-path"], "/tmp/test_falprs/static/frs/events/");
    }

    #[test]
    fn parses_example_config() {
        let config = Config::load(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../config.yaml.example"))).unwrap();
        assert!(config.loggers.contains_key("lprs"));
        assert!(config.loggers.contains_key("frs"));
        assert_eq!(config.lprs.unwrap().api_max_request_size, 5 * 1024 * 1024);
        assert_eq!(config.frs.unwrap().api_max_request_size, 5 * 1024 * 1024);
    }

    #[test]
    fn max_request_size_from_handler_defaults() {
        let text = test_config_text().replacen(
            "        server:\n",
            "        server:\n            handler-defaults:\n                max_request_size: 3000000\n",
            1,
        );
        let config = Config::parse(&text).unwrap();
        assert_eq!(config.lprs.unwrap().api_max_request_size, 3_000_000);
        assert_eq!(config.frs.unwrap().api_max_request_size, 3_000_000);
    }

    #[test]
    fn cache_host_types_from_dynamic_config() {
        let text = test_config_text().replacen(
            "            defaults:\n",
            "            defaults:\n                FALPRS_PG_CACHE_HOST_TYPES:\n                    lprs-groups-pg-cache: replica-or-master\n",
            1,
        );
        let lprs = Config::parse(&text).unwrap().lprs.unwrap();
        assert_eq!(lprs.groups_cache.host_type, HostType::ReplicaOrMaster);
        assert_eq!(lprs.vstreams_config_cache.host_type, HostType::Master);

        let invalid = test_config_text().replacen(
            "            defaults:\n",
            "            defaults:\n                FALPRS_PG_CACHE_HOST_TYPES:\n                    lprs-groups-pg-cache: slave\n",
            1,
        );
        assert!(Config::parse(&invalid).is_err());
    }
}
