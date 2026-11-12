//! Reading the service config, `config.toml`.
//!
//! Every key has a built-in default except the database DSNs, and unknown keys are rejected, so
//! a typo stops the service instead of being silently ignored. A subsystem is enabled by its
//! section (`[lprs]`, `[frs]`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use pg_cluster::{CommandControl, HostType};
use serde::{Deserialize, Deserializer};

use crate::converters::parse_duration;
use crate::logging::Level;
use crate::pipeline_slots::DEFAULT_PIPELINES_PER_CPU;

/// Settings the service takes from its config.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub port: u16,
    pub monitor_port: u16,
    pub loggers: HashMap<String, LoggerConfig>,
    pub static_dir: PathBuf,
    /// The largest API request body accepted, in bytes.
    pub max_request_size: usize,
    pub lprs: Option<LprsConfig>,
    pub frs: Option<FrsConfig>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LoggerConfig {
    pub file_path: String,
    pub level: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DatabaseConfig {
    pub dsn: String,
    pub max_pool_size: usize,
    /// Timeouts of requests other than cache updates.
    pub command_control: CommandControl,
    /// Time limit for checking the role of one host.
    pub discovery_timeout: Duration,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheConfig {
    pub update_interval: Duration,
    /// Set when the updates in between full ones are incremental.
    pub full_update_interval: Option<Duration>,
    /// How far back from the previous update's start an incremental update looks.
    pub update_correction: Duration,
    pub host_type: HostType,
    /// Network timeout of the requests of a full update; their statement timeout is off.
    pub full_update_timeout: Duration,
    /// Network timeout of the requests of an incremental update; their statement timeout is off.
    pub incremental_update_timeout: Duration,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LprsConfig {
    pub database: DatabaseConfig,
    pub groups_cache: CacheConfig,
    pub video_streams_config_cache: CacheConfig,
    pub video_stream_groups_cache: CacheConfig,
    pub settings: LprsSettings,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FrsConfig {
    pub database: DatabaseConfig,
    pub groups_cache: CacheConfig,
    pub config_cache: CacheConfig,
    pub video_streams_config_cache: CacheConfig,
    pub face_descriptors_cache: CacheConfig,
    pub video_stream_descriptors_cache: CacheConfig,
    pub sg_config_cache: CacheConfig,
    pub sg_descriptors_cache: CacheConfig,
    pub settings: FrsSettings,
}

/// The `[lprs]` settings. Intervals of zero disable their task.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct LprsSettings {
    /// Stream group used for requests without an `Authorization` header; 0 or less forbids them.
    pub no_auth_group_id: i32,
    #[serde(deserialize_with = "pipelines_per_cpu")]
    pub pipelines_per_cpu: usize,
    pub screenshots_path: String,
    pub screenshots_url_prefix: String,
    pub failed_path: String,
    #[serde(deserialize_with = "duration")]
    pub failed_ttl: Duration,
    #[serde(deserialize_with = "duration")]
    pub ban_cleanup_interval: Duration,
    #[serde(deserialize_with = "duration")]
    pub events_log_ttl: Duration,
    /// Also removes the failed files older than `failed_ttl`.
    #[serde(deserialize_with = "duration")]
    pub events_log_cleanup_interval: Duration,
    #[serde(deserialize_with = "duration")]
    pub flag_deleted_ttl: Duration,
    #[serde(deserialize_with = "duration")]
    pub flag_deleted_cleanup_interval: Duration,
    database: Option<DatabaseFile>,
    caches: LprsCaches,
}

impl Default for LprsSettings {
    fn default() -> Self {
        Self {
            no_auth_group_id: 1,
            pipelines_per_cpu: DEFAULT_PIPELINES_PER_CPU,
            screenshots_path: "/opt/falprs/static/lprs/".to_string(),
            screenshots_url_prefix: "http://localhost:9051/lprs/".to_string(),
            failed_path: "/opt/falprs/static/lprs/failed/".to_string(),
            failed_ttl: days(60),
            ban_cleanup_interval: secs(5),
            events_log_ttl: hours(4),
            events_log_cleanup_interval: hours(2),
            flag_deleted_ttl: secs(300),
            flag_deleted_cleanup_interval: secs(10),
            database: None,
            caches: LprsCaches::default(),
        }
    }
}

/// The `[frs]` settings. Intervals of zero disable their task.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct FrsSettings {
    /// Stream group used for requests without a token; 0 or less forbids them.
    pub no_auth_group_id: i32,
    #[serde(deserialize_with = "pipelines_per_cpu")]
    pub pipelines_per_cpu: usize,
    pub screenshots_path: String,
    pub screenshots_url_prefix: String,
    pub events_path: String,
    pub barcodes_path: String,
    pub barcodes_url_prefix: String,
    #[serde(deserialize_with = "duration")]
    pub log_faces_ttl: Duration,
    #[serde(deserialize_with = "duration")]
    pub log_faces_cleanup_interval: Duration,
    #[serde(deserialize_with = "duration")]
    pub log_barcodes_ttl: Duration,
    #[serde(deserialize_with = "duration")]
    pub log_barcodes_cleanup_interval: Duration,
    #[serde(deserialize_with = "duration")]
    pub events_ttl: Duration,
    #[serde(deserialize_with = "duration")]
    pub events_cleanup_interval: Duration,
    #[serde(deserialize_with = "duration")]
    pub flag_deleted_ttl: Duration,
    #[serde(deserialize_with = "duration")]
    pub flag_deleted_cleanup_interval: Duration,
    #[serde(deserialize_with = "duration")]
    pub copy_events_interval: Duration,
    database: Option<DatabaseFile>,
    caches: FrsCaches,
}

impl Default for FrsSettings {
    fn default() -> Self {
        Self {
            no_auth_group_id: 1,
            pipelines_per_cpu: DEFAULT_PIPELINES_PER_CPU,
            screenshots_path: "/opt/falprs/static/frs/screenshots/".to_string(),
            screenshots_url_prefix: "http://localhost:9051/frs/screenshots/".to_string(),
            events_path: "/opt/falprs/static/frs/events/".to_string(),
            barcodes_path: "/opt/falprs/static/frs/barcodes/".to_string(),
            barcodes_url_prefix: "http://localhost:9051/frs/barcodes/".to_string(),
            log_faces_ttl: hours(4),
            log_faces_cleanup_interval: hours(1),
            log_barcodes_ttl: hours(4),
            log_barcodes_cleanup_interval: hours(1),
            events_ttl: days(30),
            events_cleanup_interval: days(1),
            flag_deleted_ttl: secs(300),
            flag_deleted_cleanup_interval: secs(10),
            copy_events_interval: secs(30),
            database: None,
            caches: FrsCaches::default(),
        }
    }
}

/// userver's default `full-update-op-timeout` of a PostgreSQL cache.
const DEFAULT_FULL_UPDATE_TIMEOUT: Duration = Duration::from_secs(60);

/// userver's default `incremental-update-op-timeout` of a PostgreSQL cache.
const DEFAULT_INCREMENTAL_UPDATE_TIMEOUT: Duration = Duration::from_secs(1);

const fn secs(s: u64) -> Duration {
    Duration::from_secs(s)
}

const fn hours(h: u64) -> Duration {
    secs(h * 3600)
}

const fn days(d: u64) -> Duration {
    hours(d * 24)
}

fn duration<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Duration, D::Error> {
    let text = String::deserialize(deserializer)?;
    parse_duration(&text)
        .ok_or_else(|| serde::de::Error::custom(format!("invalid duration {text:?}, expected e.g. \"500ms\", \"5s\", \"1m\", \"2h\", \"30d\"")))
}

fn optional_duration<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<Duration>, D::Error> {
    duration(deserializer).map(Some)
}

fn pipelines_per_cpu<'de, D: Deserializer<'de>>(deserializer: D) -> Result<usize, D::Error> {
    match usize::deserialize(deserializer)? {
        0 => Err(serde::de::Error::custom("must be at least 1")),
        value => Ok(value),
    }
}

fn host_type<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<HostType>, D::Error> {
    match String::deserialize(deserializer)?.as_str() {
        "master" => Ok(Some(HostType::Master)),
        "replica" => Ok(Some(HostType::Replica)),
        "replica-or-master" => Ok(Some(HostType::ReplicaOrMaster)),
        other => Err(serde::de::Error::custom(format!(
            "unknown host type {other:?}, expected master, replica or replica-or-master"
        ))),
    }
}

#[derive(Deserialize, Default)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
struct File {
    server: ServerFile,
    logging: LoggingFile,
    lprs: Option<LprsSettings>,
    frs: Option<FrsSettings>,
}

#[derive(Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
struct ServerFile {
    port: u16,
    monitor_port: u16,
    static_dir: PathBuf,
    max_request_size: usize,
}

impl Default for ServerFile {
    fn default() -> Self {
        Self {
            port: 9051,
            monitor_port: 9052,
            static_dir: PathBuf::from("/opt/falprs/static/"),
            max_request_size: 5 * 1024 * 1024,
        }
    }
}

#[derive(Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct LoggingFile {
    default: LoggerFile,
    lprs: LoggerFile,
    frs: LoggerFile,
}

#[derive(Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct LoggerFile {
    file: Option<String>,
    level: Option<String>,
}

impl LoggerFile {
    fn resolve(self, name: &str, file: &str, level: &str) -> Result<LoggerConfig> {
        let level = self.level.unwrap_or_else(|| level.to_string());
        level.parse::<Level>().with_context(|| format!("logging.{name}.level"))?;
        Ok(LoggerConfig {
            file_path: self.file.unwrap_or_else(|| file.to_string()),
            level,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
struct DatabaseFile {
    dsn: Option<String>,
    max_pool_size: usize,
    #[serde(deserialize_with = "duration")]
    network_timeout: Duration,
    /// Zero turns the limit off.
    #[serde(deserialize_with = "duration")]
    statement_timeout: Duration,
    #[serde(deserialize_with = "duration")]
    discovery_timeout: Duration,
}

impl Default for DatabaseFile {
    fn default() -> Self {
        Self {
            dsn: None,
            max_pool_size: 16,
            network_timeout: secs(5),
            statement_timeout: secs(5),
            discovery_timeout: secs(2),
        }
    }
}

impl DatabaseFile {
    fn resolve(database: Option<Self>, subsystem: &str) -> Result<DatabaseConfig> {
        let database = database.unwrap_or_default();
        let dsn = database
            .dsn
            .filter(|dsn| !dsn.is_empty())
            .ok_or_else(|| anyhow!("{subsystem}.database.dsn is required"))?;
        if database.max_pool_size == 0 {
            bail!("{subsystem}.database.max-pool-size must be at least 1");
        }
        if database.network_timeout.is_zero() || database.discovery_timeout.is_zero() {
            bail!("{subsystem}.database: network-timeout and discovery-timeout must be positive");
        }
        Ok(DatabaseConfig {
            dsn,
            max_pool_size: database.max_pool_size,
            command_control: CommandControl {
                network_timeout: database.network_timeout,
                statement_timeout: (!database.statement_timeout.is_zero()).then_some(database.statement_timeout),
            },
            discovery_timeout: database.discovery_timeout,
        })
    }
}

/// A cache updated by full updates only.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
struct FullCacheFile {
    #[serde(deserialize_with = "optional_duration")]
    update_interval: Option<Duration>,
    #[serde(deserialize_with = "host_type")]
    host_type: Option<HostType>,
    #[serde(deserialize_with = "optional_duration")]
    full_update_timeout: Option<Duration>,
}

/// A cache that can update incrementally in between full updates.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
struct IncrementalCacheFile {
    #[serde(deserialize_with = "optional_duration")]
    update_interval: Option<Duration>,
    /// Zero makes every update a full one.
    #[serde(deserialize_with = "optional_duration")]
    full_update_interval: Option<Duration>,
    #[serde(deserialize_with = "optional_duration")]
    update_correction: Option<Duration>,
    #[serde(deserialize_with = "host_type")]
    host_type: Option<HostType>,
    #[serde(deserialize_with = "optional_duration")]
    full_update_timeout: Option<Duration>,
    #[serde(deserialize_with = "optional_duration")]
    incremental_update_timeout: Option<Duration>,
}

fn positive(interval: Duration, name: &str) -> Result<Duration> {
    if interval.is_zero() {
        bail!("{name}.update-interval must be positive");
    }
    Ok(interval)
}

impl FullCacheFile {
    fn resolve(self, name: &str, update_interval: Duration) -> Result<CacheConfig> {
        Ok(CacheConfig {
            update_interval: positive(self.update_interval.unwrap_or(update_interval), name)?,
            full_update_interval: None,
            update_correction: Duration::ZERO,
            host_type: self.host_type.unwrap_or(HostType::Master),
            full_update_timeout: self.full_update_timeout.unwrap_or(DEFAULT_FULL_UPDATE_TIMEOUT),
            incremental_update_timeout: DEFAULT_INCREMENTAL_UPDATE_TIMEOUT,
        })
    }
}

impl IncrementalCacheFile {
    fn resolve(self, name: &str, update_interval: Duration, full_update_interval: Duration) -> Result<CacheConfig> {
        let full_update_interval = self.full_update_interval.unwrap_or(full_update_interval);
        Ok(CacheConfig {
            update_interval: positive(self.update_interval.unwrap_or(update_interval), name)?,
            full_update_interval: (!full_update_interval.is_zero()).then_some(full_update_interval),
            update_correction: self.update_correction.unwrap_or(secs(1)),
            host_type: self.host_type.unwrap_or(HostType::Master),
            full_update_timeout: self.full_update_timeout.unwrap_or(DEFAULT_FULL_UPDATE_TIMEOUT),
            incremental_update_timeout: self.incremental_update_timeout.unwrap_or(DEFAULT_INCREMENTAL_UPDATE_TIMEOUT),
        })
    }
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
struct LprsCaches {
    groups: FullCacheFile,
    video_streams_config: FullCacheFile,
    video_stream_groups: FullCacheFile,
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
struct FrsCaches {
    groups: FullCacheFile,
    config: FullCacheFile,
    video_streams_config: IncrementalCacheFile,
    face_descriptors: IncrementalCacheFile,
    video_stream_descriptors: IncrementalCacheFile,
    sg_config: FullCacheFile,
    sg_descriptors: IncrementalCacheFile,
}

/// Directories and URL prefixes are used with a trailing slash.
fn with_trailing_slash(path: &mut String) {
    if !path.is_empty() && !path.ends_with('/') {
        path.push('/');
    }
}

impl LprsConfig {
    fn resolve(mut settings: LprsSettings) -> Result<Self> {
        let minute = secs(60);
        let caches = std::mem::take(&mut settings.caches);
        let database = DatabaseFile::resolve(settings.database.take(), "lprs")?;
        for path in [&mut settings.screenshots_path, &mut settings.screenshots_url_prefix, &mut settings.failed_path] {
            with_trailing_slash(path);
        }
        Ok(Self {
            database,
            groups_cache: caches.groups.resolve("lprs.caches.groups", minute)?,
            video_streams_config_cache: caches.video_streams_config.resolve("lprs.caches.video-streams-config", minute)?,
            video_stream_groups_cache: caches.video_stream_groups.resolve("lprs.caches.video-stream-groups", minute)?,
            settings,
        })
    }
}

impl FrsConfig {
    fn resolve(mut settings: FrsSettings) -> Result<Self> {
        let (ten_seconds, minute) = (secs(10), secs(60));
        let caches = std::mem::take(&mut settings.caches);
        let database = DatabaseFile::resolve(settings.database.take(), "frs")?;
        for path in [
            &mut settings.screenshots_path,
            &mut settings.screenshots_url_prefix,
            &mut settings.events_path,
            &mut settings.barcodes_path,
            &mut settings.barcodes_url_prefix,
        ] {
            with_trailing_slash(path);
        }
        Ok(Self {
            database,
            groups_cache: caches.groups.resolve("frs.caches.groups", minute)?,
            config_cache: caches.config.resolve("frs.caches.config", minute)?,
            video_streams_config_cache: caches.video_streams_config.resolve("frs.caches.video-streams-config", ten_seconds, secs(300))?,
            face_descriptors_cache: caches.face_descriptors.resolve("frs.caches.face-descriptors", ten_seconds, secs(600))?,
            video_stream_descriptors_cache: caches
                .video_stream_descriptors
                .resolve("frs.caches.video-stream-descriptors", ten_seconds, secs(600))?,
            sg_config_cache: caches.sg_config.resolve("frs.caches.sg-config", minute)?,
            sg_descriptors_cache: caches.sg_descriptors.resolve("frs.caches.sg-descriptors", ten_seconds, secs(600))?,
            settings,
        })
    }
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        if path.extension().is_some_and(|extension| extension == "yaml" || extension == "yml") {
            bail!(
                "{}: the YAML config is no longer supported, convert it with scripts/convert_config.py",
                path.display()
            );
        }
        let text = std::fs::read_to_string(path).with_context(|| format!("can't read config {}", path.display()))?;
        Self::parse(&text).with_context(|| format!("invalid config {}", path.display()))
    }

    pub fn parse(text: &str) -> Result<Self> {
        let file: File = toml::from_str(text)?;
        if file.server.max_request_size == 0 {
            bail!("server.max-request-size must be positive");
        }
        let logging = file.logging;
        let loggers = HashMap::from([
            ("default".to_string(), logging.default.resolve("default", "/var/log/falprs/log.txt", "error")?),
            ("lprs".to_string(), logging.lprs.resolve("lprs", "/var/log/falprs/lprs-log.txt", "info")?),
            ("frs".to_string(), logging.frs.resolve("frs", "/var/log/falprs/frs-log.txt", "info")?),
        ]);
        Ok(Self {
            port: file.server.port,
            monitor_port: file.server.monitor_port,
            loggers,
            static_dir: file.server.static_dir,
            max_request_size: file.server.max_request_size,
            lprs: file.lprs.map(LprsConfig::resolve).transpose()?,
            frs: file.frs.map(FrsConfig::resolve).transpose()?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL: &str = "[lprs.database]\ndsn = \"postgresql://u:p@h/lprs\"\n[frs.database]\ndsn = \"postgresql://u:p@h/frs\"\n";

    /// The minimal config with the DSNs of the example config.
    const MINIMAL_EXAMPLE: &str = "[lprs.database]\ndsn = \"postgresql://falprs:123@localhost:5432/lprs\"\n\
                                   [frs.database]\ndsn = \"postgresql://falprs:123@localhost:5432/frs\"\n";

    fn repo_file(path: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..").join(path)
    }

    fn error_of(text: &str) -> String {
        format!("{:#}", Config::parse(text).unwrap_err())
    }

    #[test]
    fn parses_test_config() {
        let config = Config::load(&repo_file("tests/test_config.toml")).unwrap();
        assert_eq!(config.port, 9071);
        assert_eq!(config.monitor_port, 9072);
        assert_eq!(config.loggers["default"].file_path, "/tmp/test_falprs/logs/log.txt");
        assert_eq!(config.loggers["default"].level, "error");
        assert_eq!(config.loggers["lprs"].level, "debug");
        assert_eq!(config.static_dir, Path::new("/tmp/test_falprs/static/"));
        assert_eq!(config.max_request_size, 1024 * 1024);

        let lprs = config.lprs.unwrap();
        assert_eq!(lprs.database.dsn, "postgresql://test_falprs:123@localhost:5432/test_lprs");
        assert_eq!(lprs.database.max_pool_size, 15);
        assert_eq!(lprs.groups_cache.update_interval, secs(5));
        assert_eq!(lprs.groups_cache.host_type, HostType::Master);
        assert_eq!(lprs.groups_cache.full_update_interval, None);
        assert_eq!(lprs.settings.no_auth_group_id, 1);
        assert_eq!(lprs.settings.failed_ttl, days(2));

        let frs = config.frs.unwrap();
        assert_eq!(frs.database.dsn, "postgresql://test_falprs:123@localhost:5432/test_frs");
        assert_eq!(frs.face_descriptors_cache.update_interval, secs(5));
        assert_eq!(frs.face_descriptors_cache.full_update_interval, Some(secs(600)));
        assert_eq!(frs.face_descriptors_cache.update_correction, secs(1));
        assert_eq!(frs.video_streams_config_cache.full_update_interval, Some(secs(30)));
        assert_eq!(frs.sg_config_cache.full_update_interval, None);
        assert_eq!(frs.settings.events_path, "/tmp/test_falprs/static/frs/events/");
        assert_eq!(frs.settings.events_ttl, hours(2));
    }

    #[test]
    fn example_config_states_the_defaults() {
        assert_eq!(
            Config::load(&repo_file("config.toml.example")).unwrap(),
            Config::parse(MINIMAL_EXAMPLE).unwrap()
        );
    }

    /// The converter's output of the old configs, as scripts/tests/test_convert_config.py checks it.
    #[test]
    fn converted_configs_mean_the_same_as_the_new_ones() {
        let converted = Config::load(&repo_file("scripts/tests/fixtures/test_config.converted.toml")).unwrap();
        assert_eq!(converted, Config::load(&repo_file("tests/test_config.toml")).unwrap());

        // The old example set a pool size of 15 that was ignored before; it is carried over now
        let mut converted = Config::load(&repo_file("scripts/tests/fixtures/config.converted.toml")).unwrap();
        for database in [&mut converted.lprs.as_mut().unwrap().database, &mut converted.frs.as_mut().unwrap().database] {
            assert_eq!(database.max_pool_size, 15);
            database.max_pool_size = 16;
        }
        assert_eq!(converted, Config::load(&repo_file("config.toml.example")).unwrap());
    }

    #[test]
    fn minimal_config_has_the_documented_defaults() {
        let config = Config::parse(MINIMAL).unwrap();
        assert_eq!((config.port, config.monitor_port), (9051, 9052));
        assert_eq!(config.static_dir, Path::new("/opt/falprs/static/"));
        assert_eq!(config.max_request_size, 5_242_880);
        assert_eq!(config.loggers["default"].file_path, "/var/log/falprs/log.txt");
        assert_eq!(config.loggers["default"].level, "error");
        assert_eq!(config.loggers["frs"].file_path, "/var/log/falprs/frs-log.txt");
        assert_eq!(config.loggers["frs"].level, "info");

        let lprs = config.lprs.unwrap();
        assert_eq!(lprs.database.max_pool_size, 16);
        assert_eq!(
            lprs.database.command_control,
            CommandControl {
                network_timeout: secs(5),
                statement_timeout: Some(secs(5)),
            }
        );
        assert_eq!(lprs.database.discovery_timeout, secs(2));
        assert_eq!(lprs.settings.no_auth_group_id, 1);
        assert_eq!(lprs.settings.pipelines_per_cpu, DEFAULT_PIPELINES_PER_CPU);
        assert_eq!(lprs.settings.ban_cleanup_interval, secs(5));
        assert_eq!(lprs.groups_cache.update_interval, secs(60));
        assert_eq!(lprs.groups_cache.full_update_timeout, secs(60));

        let frs = config.frs.unwrap();
        assert_eq!(frs.settings.no_auth_group_id, 1);
        assert_eq!(frs.settings.copy_events_interval, secs(30));
        assert_eq!(frs.video_streams_config_cache.update_interval, secs(10));
        assert_eq!(frs.video_streams_config_cache.full_update_interval, Some(secs(300)));
        assert_eq!(frs.face_descriptors_cache.full_update_interval, Some(secs(600)));
        assert_eq!(frs.face_descriptors_cache.update_correction, secs(1));
        assert_eq!(frs.face_descriptors_cache.incremental_update_timeout, secs(1));
        assert_eq!(frs.sg_config_cache.update_interval, secs(60));
    }

    #[test]
    fn subsystems_are_enabled_by_their_section() {
        let config = Config::parse("[frs.database]\ndsn = \"x\"\n").unwrap();
        assert!(config.lprs.is_none());
        assert!(config.frs.is_some());
        let config = Config::parse("").unwrap();
        assert!(config.lprs.is_none() && config.frs.is_none());
    }

    #[test]
    fn rejects_unknown_keys_naming_them() {
        for (text, key) in [
            ("[server]\nprot = 1\n", "prot"),
            ("[logging.default]\nfile_path = \"x\"\n", "file_path"),
            ("[logging.other]\nfile = \"x\"\n", "other"),
            (
                "[lprs]\nban-maintenance-interval = \"5s\"\n[lprs.database]\ndsn = \"x\"\n",
                "ban-maintenance-interval",
            ),
            ("[frs]\nclear-old-events = \"1d\"\n[frs.database]\ndsn = \"x\"\n", "clear-old-events"),
            ("[frs.database]\ndsn = \"x\"\nmin-pool-size = 1\n", "min-pool-size"),
            ("[frs.database]\ndsn = \"x\"\n[frs.caches.faces]\n", "faces"),
            // A cache updated by full updates only has no incremental settings
            (
                "[frs.database]\ndsn = \"x\"\n[frs.caches.groups]\nfull-update-interval = \"5m\"\n",
                "full-update-interval",
            ),
            ("[dynamic-config]\n", "dynamic-config"),
        ] {
            let error = error_of(text);
            assert!(error.contains(key), "{text}: {error}");
        }
    }

    #[test]
    fn rejects_a_missing_dsn() {
        assert!(error_of("[lprs]\nno-auth-group-id = 0\n").contains("lprs.database.dsn is required"));
        assert!(error_of("[frs.database]\nmax-pool-size = 4\n").contains("frs.database.dsn is required"));
    }

    #[test]
    fn rejects_invalid_values() {
        for text in [
            "[lprs.database]\ndsn = \"x\"\nnetwork-timeout = 5000\n",
            "[lprs.database]\ndsn = \"x\"\nnetwork-timeout = \"0s\"\n",
            "[lprs.database]\ndsn = \"x\"\ndiscovery-timeout = \"-1s\"\n",
            "[lprs.database]\ndsn = \"x\"\nmax-pool-size = 0\n",
            "[lprs.database]\ndsn = \"x\"\n[lprs.caches.groups]\nhost-type = \"slave\"\n",
            "[lprs.database]\ndsn = \"x\"\n[lprs.caches.groups]\nupdate-interval = \"0s\"\n",
            "[lprs]\npipelines-per-cpu = 0\n[lprs.database]\ndsn = \"x\"\n",
            "[frs]\nevents-ttl = \"30 days\"\n[frs.database]\ndsn = \"x\"\n",
            "[logging.lprs]\nlevel = \"verbose\"\n",
            "[server]\nmax-request-size = 0\n",
        ] {
            assert!(Config::parse(text).is_err(), "{text}");
        }
    }

    #[test]
    fn cache_settings() {
        let text = format!(
            "{MINIMAL}[lprs.caches.groups]\nhost-type = \"replica-or-master\"\nfull-update-timeout = \"3m\"\n\
             [frs.caches.face-descriptors]\nfull-update-interval = \"0s\"\nincremental-update-timeout = \"2s\"\n"
        );
        let config = Config::parse(&text).unwrap();
        let lprs = config.lprs.unwrap();
        assert_eq!(lprs.groups_cache.host_type, HostType::ReplicaOrMaster);
        assert_eq!(lprs.groups_cache.full_update_timeout, secs(180));
        assert_eq!(lprs.video_streams_config_cache.host_type, HostType::Master);
        let frs = config.frs.unwrap();
        // A zero full update interval makes every update a full one
        assert_eq!(frs.face_descriptors_cache.full_update_interval, None);
        assert_eq!(frs.face_descriptors_cache.incremental_update_timeout, secs(2));
    }

    #[test]
    fn database_settings_are_per_subsystem() {
        let text = "[lprs.database]\ndsn = \"a\"\nnetwork-timeout = \"700ms\"\nstatement-timeout = \"0s\"\nmax-pool-size = 4\n\
                    [frs.database]\ndsn = \"b\"\ndiscovery-timeout = \"700ms\"\n";
        let config = Config::parse(text).unwrap();
        let (lprs, frs) = (config.lprs.unwrap(), config.frs.unwrap());
        assert_eq!(
            lprs.database.command_control,
            CommandControl {
                network_timeout: Duration::from_millis(700),
                statement_timeout: None,
            }
        );
        assert_eq!(lprs.database.max_pool_size, 4);
        assert_eq!(lprs.database.discovery_timeout, secs(2));
        assert_eq!(frs.database.command_control.network_timeout, secs(5));
        assert_eq!(frs.database.max_pool_size, 16);
        assert_eq!(frs.database.discovery_timeout, Duration::from_millis(700));
    }

    #[test]
    fn paths_get_a_trailing_slash() {
        let text = "[lprs]\nfailed-path = \"/data/failed\"\n[lprs.database]\ndsn = \"x\"\n\
                    [frs]\nbarcodes-url-prefix = \"http://host/barcodes\"\nevents-path = \"\"\n[frs.database]\ndsn = \"x\"\n";
        let config = Config::parse(text).unwrap();
        assert_eq!(config.lprs.unwrap().settings.failed_path, "/data/failed/");
        let frs = config.frs.unwrap().settings;
        assert_eq!(frs.barcodes_url_prefix, "http://host/barcodes/");
        assert_eq!(frs.events_path, "");
    }

    #[test]
    fn yaml_config_gives_the_conversion_hint() {
        let error = format!("{:#}", Config::load(Path::new("/opt/falprs/config.yaml")).unwrap_err());
        assert!(error.contains("scripts/convert_config.py"), "{error}");
    }
}
