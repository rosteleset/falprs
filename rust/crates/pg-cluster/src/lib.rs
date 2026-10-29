//! PostgreSQL cluster access with explicit routing by host type, in the spirit of userver's
//! `storages::postgres::Cluster`.
//!
//! Given a multi-host connection string, the cluster keeps a connection pool per host, periodically
//! discovers which host is the master and which are replicas, follows a failover without a restart,
//! and routes every request to the host type the caller names. See
//! `docs/adr/0001-api-reads-from-master-only.md` for which requests may go to replicas.

use std::fmt;
use std::str::FromStr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock, Weak};
use std::time::Duration;

use deadpool_postgres::{Manager, ManagerConfig, Pool, PoolError, RecyclingMethod, Runtime};
use tokio::sync::Notify;
use tokio_postgres::config::Host as ConfigHost;
use tokio_postgres::types::ToSql;
use tokio_postgres::{Config, NoTls, Row};

pub use deadpool_postgres::Object as Client;
pub use tokio_postgres;

/// Where a request is routed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostType {
    /// The master only.
    Master,
    /// Replicas only.
    Replica,
    /// A replica if one is available, the master otherwise.
    ReplicaOrMaster,
}

/// The role of a host as last discovered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Unknown,
    Master,
    Replica,
    Unavailable,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid connection string: {0}")]
    Config(String),
    #[error("no available host for {0:?}")]
    NoHost(HostType),
    #[error(transparent)]
    Pool(#[from] PoolError),
    #[error("{}", error_message(.0))]
    Db(#[from] tokio_postgres::Error),
}

/// The server's message for errors reported by the server (tokio-postgres displays them only as
/// "db error"), the error itself otherwise.
pub fn error_message(error: &tokio_postgres::Error) -> String {
    match error.as_db_error() {
        Some(db_error) => db_error.to_string(),
        None => error.to_string(),
    }
}

impl Error {
    /// Whether the server rejected the statement for violating a unique constraint.
    pub fn is_unique_violation(&self) -> bool {
        matches!(self, Error::Db(e) if e.code() == Some(&tokio_postgres::error::SqlState::UNIQUE_VIOLATION))
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone)]
pub struct ClusterSettings {
    /// Maximum number of connections per host.
    pub max_pool_size: usize,
    /// How often host roles are rediscovered.
    pub discovery_interval: Duration,
    /// Timeout for establishing a connection and for waiting for a free one.
    pub connect_timeout: Duration,
}

impl Default for ClusterSettings {
    fn default() -> Self {
        Self {
            max_pool_size: 16,
            discovery_interval: Duration::from_secs(1),
            connect_timeout: Duration::from_secs(2),
        }
    }
}

struct Host {
    name: String,
    pool: Pool,
    role: RwLock<Role>,
}

impl Host {
    fn role(&self) -> Role {
        *self.role.read().unwrap_or_else(|e| e.into_inner())
    }

    fn set_role(&self, role: Role) {
        *self.role.write().unwrap_or_else(|e| e.into_inner()) = role;
    }

    async fn discover_role(&self) -> Role {
        let role = match self.pool.get().await {
            Ok(client) => match client.query_one("SELECT pg_is_in_recovery()", &[]).await {
                Ok(row) if row.get::<_, bool>(0) => Role::Replica,
                Ok(_) => Role::Master,
                Err(_) => Role::Unavailable,
            },
            Err(_) => Role::Unavailable,
        };
        self.set_role(role);
        role
    }
}

struct Inner {
    hosts: Vec<Arc<Host>>,
    next_replica: AtomicUsize,
    rediscover: Notify,
}

impl Inner {
    async fn discover(&self) {
        let mut tasks = tokio::task::JoinSet::new();
        for host in &self.hosts {
            let host = host.clone();
            tasks.spawn(async move { host.discover_role().await });
        }
        while tasks.join_next().await.is_some() {}
    }

    /// Hosts to try for `host_type`, best first.
    fn candidates(&self, host_type: HostType) -> Vec<Arc<Host>> {
        let with_role = |role| self.hosts.iter().filter(move |h| h.role() == role).cloned();
        let mut replicas: Vec<_> = with_role(Role::Replica).collect();
        if !replicas.is_empty() {
            // Spread load over replicas
            let shift = self.next_replica.fetch_add(1, Ordering::Relaxed) % replicas.len();
            replicas.rotate_left(shift);
        }
        match host_type {
            HostType::Master => with_role(Role::Master).collect(),
            HostType::Replica => replicas,
            HostType::ReplicaOrMaster => replicas.into_iter().chain(with_role(Role::Master)).collect(),
        }
    }

    async fn client(&self, host_type: HostType) -> Result<Client> {
        let mut last_error = None;
        for attempt in 0..2 {
            if attempt > 0 {
                // Nothing usable is known: the topology may have changed since the last discovery
                self.discover().await;
            }
            for host in self.candidates(host_type) {
                match host.pool.get().await {
                    Ok(client) => return Ok(client),
                    Err(e) => {
                        host.set_role(Role::Unavailable);
                        self.rediscover.notify_one();
                        last_error = Some(e);
                    }
                }
            }
        }
        Err(last_error.map_or(Error::NoHost(host_type), Error::Pool))
    }
}

async fn run_discovery(cluster: Weak<Inner>, interval: Duration) {
    loop {
        let Some(inner) = cluster.upgrade() else {
            return;
        };
        tokio::select! {
            _ = tokio::time::sleep(interval) => {}
            _ = inner.rediscover.notified() => {}
        }
        inner.discover().await;
    }
}

/// A PostgreSQL cluster of one master and any number of replicas.
#[derive(Clone)]
pub struct Cluster {
    inner: Arc<Inner>,
}

impl fmt::Debug for Cluster {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Cluster").field("topology", &self.topology()).finish()
    }
}

impl Cluster {
    /// Connects to every host of `dsn` (URL or key-value form, several hosts separated by commas)
    /// and discovers their roles before returning.
    pub async fn connect(dsn: &str, settings: ClusterSettings) -> Result<Self> {
        let config = Config::from_str(dsn).map_err(|e| Error::Config(e.to_string()))?;
        let hosts = split_hosts(&config)?
            .into_iter()
            .map(|(name, host_config)| make_pool(host_config, &settings).map(|pool| Host { name, pool, role: RwLock::new(Role::Unknown) }))
            .map(|host| host.map(Arc::new))
            .collect::<Result<Vec<_>>>()?;
        let inner = Arc::new(Inner {
            hosts,
            next_replica: AtomicUsize::new(0),
            rediscover: Notify::new(),
        });
        inner.discover().await;
        tokio::spawn(run_discovery(Arc::downgrade(&inner), settings.discovery_interval));
        Ok(Self { inner })
    }

    /// A pooled connection to a host of `host_type`, e.g. to run a transaction on.
    pub async fn client(&self, host_type: HostType) -> Result<Client> {
        self.inner.client(host_type).await
    }

    pub async fn query(&self, host_type: HostType, sql: &str, params: &[&(dyn ToSql + Sync)]) -> Result<Vec<Row>> {
        let client = self.client(host_type).await?;
        let statement = client.prepare_cached(sql).await?;
        Ok(client.query(&statement, params).await?)
    }

    /// Runs a query expected to return at most one row.
    pub async fn query_opt(
        &self,
        host_type: HostType,
        sql: &str,
        params: &[&(dyn ToSql + Sync)],
    ) -> Result<Option<Row>> {
        let client = self.client(host_type).await?;
        let statement = client.prepare_cached(sql).await?;
        Ok(client.query_opt(&statement, params).await?)
    }

    /// Runs a query expected to return exactly one row.
    pub async fn query_one(&self, host_type: HostType, sql: &str, params: &[&(dyn ToSql + Sync)]) -> Result<Row> {
        let client = self.client(host_type).await?;
        let statement = client.prepare_cached(sql).await?;
        Ok(client.query_one(&statement, params).await?)
    }

    pub async fn execute(&self, host_type: HostType, sql: &str, params: &[&(dyn ToSql + Sync)]) -> Result<u64> {
        let client = self.client(host_type).await?;
        let statement = client.prepare_cached(sql).await?;
        Ok(client.execute(&statement, params).await?)
    }

    /// Hosts in connection string order, with their last discovered roles.
    pub fn topology(&self) -> Vec<(String, Role)> {
        self.inner.hosts.iter().map(|h| (h.name.clone(), h.role())).collect()
    }
}

/// One single-host config per host of `config`, keeping the connection parameters.
fn split_hosts(config: &Config) -> Result<Vec<(String, Config)>> {
    let hosts = config.get_hosts();
    let ports = config.get_ports();
    if hosts.is_empty() {
        return Err(Error::Config("no hosts".to_string()));
    }
    if ports.len() > 1 && ports.len() != hosts.len() {
        return Err(Error::Config("the number of ports must be 1 or match the number of hosts".to_string()));
    }
    hosts
        .iter()
        .enumerate()
        .map(|(i, host)| {
            let port = ports.get(i).or(ports.first()).copied().unwrap_or(5432);
            let mut single = Config::new();
            let name = match host {
                ConfigHost::Tcp(name) => {
                    single.host(name);
                    format!("{name}:{port}")
                }
                ConfigHost::Unix(path) => {
                    single.host_path(path);
                    format!("{}:{port}", path.display())
                }
            };
            single.port(port);
            if let Some(user) = config.get_user() {
                single.user(user);
            }
            if let Some(password) = config.get_password() {
                single.password(password);
            }
            if let Some(dbname) = config.get_dbname() {
                single.dbname(dbname);
            }
            if let Some(options) = config.get_options() {
                single.options(options);
            }
            if let Some(application_name) = config.get_application_name() {
                single.application_name(application_name);
            }
            Ok((name, single))
        })
        .collect()
}

fn make_pool(config: Config, settings: &ClusterSettings) -> Result<Pool> {
    let mut config = config;
    config.connect_timeout(settings.connect_timeout);
    let manager = Manager::from_config(config, NoTls, ManagerConfig {
        recycling_method: RecyclingMethod::Fast,
    });
    Pool::builder(manager)
        .max_size(settings.max_pool_size)
        .runtime(Runtime::Tokio1)
        .create_timeout(Some(settings.connect_timeout))
        .wait_timeout(Some(settings.connect_timeout))
        .build()
        .map_err(|e| Error::Config(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(dsn: &str) -> Vec<String> {
        split_hosts(&Config::from_str(dsn).unwrap()).unwrap().into_iter().map(|(name, _)| name).collect()
    }

    #[test]
    fn splits_url_hosts() {
        assert_eq!(names("postgresql://u:p@h1:5433,h2:5434/db"), ["h1:5433", "h2:5434"]);
        assert_eq!(names("postgresql://u:p@h1,h2/db"), ["h1:5432", "h2:5432"]);
        assert_eq!(names("postgresql://u:p@localhost:5432/db"), ["localhost:5432"]);
    }

    #[test]
    fn keeps_connection_parameters() {
        let hosts = split_hosts(&Config::from_str("postgresql://falprs:123@h1,h2:5433/lprs").unwrap()).unwrap();
        for (_, config) in hosts {
            assert_eq!(config.get_user(), Some("falprs"));
            assert_eq!(config.get_password(), Some(&b"123"[..]));
            assert_eq!(config.get_dbname(), Some("lprs"));
        }
    }
}
