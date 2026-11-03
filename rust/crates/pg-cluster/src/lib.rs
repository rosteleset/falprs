//! PostgreSQL cluster access with explicit routing by host type, in the spirit of userver's
//! `storages::postgres::Cluster`.
//!
//! Given a multi-host connection string, the cluster keeps a connection pool per host, periodically
//! discovers which host is the master and which are replicas, follows a failover without a restart,
//! and routes every request to the host type the caller names. See
//! `docs/adr/0001-api-reads-from-master-only.md` for which requests may go to replicas.
//!
//! Every request is bounded as userver's `CommandControl` bounds it: by a network timeout, how long
//! the client waits for each response, and by a statement timeout, how long the server may execute
//! a statement. A connection that timed out on the network or failed is closed at once, and its host
//! is marked unavailable until the next discovery finds it again.

use std::fmt;
use std::future::Future;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock, Weak};
use std::time::Duration;

use deadpool_postgres::{Manager, ManagerConfig, Object, Pool, PoolError, RecyclingMethod, Runtime};
use tokio::sync::Notify;
use tokio_postgres::config::Host as ConfigHost;
use tokio_postgres::error::SqlState;
use tokio_postgres::types::ToSql;
use tokio_postgres::{CancelToken, Config, NoTls, Row, ToStatement};

pub use tokio_postgres;

/// Logs to a [`LogSink`] with the location of the call, see [`LogRecord`].
macro_rules! log {
    ($sink:expr, $level:expr, $($arg:tt)+) => {
        if let Some(sink) = &$sink.0 {
            sink(&LogRecord {
                level: $level,
                nested_fn: {
                    fn f() {}
                    std::any::type_name_of_val(&f)
                },
                file: file!(),
                line: line!(),
                text: format_args!($($arg)+),
            })
        }
    };
}

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

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Role::Unknown => "unknown",
            Role::Master => "master",
            Role::Replica => "replica",
            Role::Unavailable => "unavailable",
        })
    }
}

/// Timeouts of a request, as userver's `storages::postgres::CommandControl`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommandControl {
    /// How long the client waits for each response of the server.
    pub network_timeout: Duration,
    /// How long the server may execute a statement; `None` turns the limit off.
    pub statement_timeout: Option<Duration>,
}

impl Default for CommandControl {
    /// userver's built-in `components::Postgres::kDefaultCommandControl`.
    fn default() -> Self {
        Self {
            network_timeout: Duration::from_millis(500),
            statement_timeout: Some(Duration::from_millis(250)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Info,
    Warning,
}

/// A message of the cluster for the service's log.
pub struct LogRecord<'a> {
    pub level: LogLevel,
    /// The type name of a function nested in the logging one, as `std::any::type_name_of_val` gives it.
    pub nested_fn: &'static str,
    pub file: &'static str,
    pub line: u32,
    pub text: fmt::Arguments<'a>,
}

type LogFn = dyn Fn(&LogRecord<'_>) + Send + Sync;

/// Where the cluster writes its messages; the default one drops them.
#[derive(Clone, Default)]
pub struct LogSink(Option<Arc<LogFn>>);

impl LogSink {
    pub fn new(log: impl Fn(&LogRecord<'_>) + Send + Sync + 'static) -> Self {
        Self(Some(Arc::new(log)))
    }
}

impl fmt::Debug for LogSink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(if self.0.is_some() { "LogSink" } else { "LogSink(none)" })
    }
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
    #[error("network timeout: no response within {} ms", .0.as_millis())]
    NetworkTimeout(Duration),
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
        matches!(self, Error::Db(e) if e.code() == Some(&SqlState::UNIQUE_VIOLATION))
    }

    /// Whether the server cancelled the statement, e.g. because it ran past the statement timeout.
    pub fn is_query_cancelled(&self) -> bool {
        matches!(self, Error::Db(e) if e.code() == Some(&SqlState::QUERY_CANCELED))
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone)]
pub struct ClusterSettings {
    /// Maximum number of connections per host.
    pub max_pool_size: usize,
    /// How often host roles are rediscovered.
    pub discovery_interval: Duration,
    /// Time limit for checking the role of one host: getting a connection to it and asking it.
    /// It also bounds a request to cancel a statement the client stopped waiting for.
    pub discovery_timeout: Duration,
    /// Timeout for establishing a connection, unless the connection string sets one, and for
    /// waiting for a free one.
    pub connect_timeout: Duration,
    /// Timeouts of requests, unless a view of the cluster overrides them.
    pub command_control: CommandControl,
    pub log: LogSink,
}

impl Default for ClusterSettings {
    fn default() -> Self {
        Self {
            max_pool_size: 16,
            discovery_interval: Duration::from_secs(1),
            discovery_timeout: Duration::from_secs(2),
            connect_timeout: Duration::from_secs(2),
            command_control: CommandControl::default(),
            log: LogSink::default(),
        }
    }
}

struct Host {
    name: String,
    /// The host with its database, `host:port/dbname`, as log messages name it: services may use
    /// several databases on the same hosts.
    label: String,
    pool: Pool,
    role: RwLock<Role>,
}

impl Host {
    fn role(&self) -> Role {
        *self.role.read().unwrap_or_else(|e| e.into_inner())
    }
}

struct Inner {
    hosts: Vec<Arc<Host>>,
    next_replica: AtomicUsize,
    rediscover: Notify,
    discovery_timeout: Duration,
    /// The statement timeout every connection starts its session with.
    session_statement_timeout: Option<Duration>,
    log: LogSink,
    /// Whether more than one host reported being the master on the last discovery.
    several_masters: AtomicBool,
}

impl Inner {
    fn set_role(&self, host: &Host, role: Role) {
        let previous = std::mem::replace(&mut *host.role.write().unwrap_or_else(|e| e.into_inner()), role);
        if previous != role {
            log!(self.log, LogLevel::Info, "PostgreSQL host {}: {previous} -> {role}", host.label);
        }
    }

    /// Marks a host a request failed on as unavailable and asks for a discovery right away, unless
    /// the host is already known to be unavailable: that keeps discovery to its interval while a
    /// host stays down.
    fn mark_unavailable(&self, host: &Host) {
        if host.role() != Role::Unavailable {
            self.set_role(host, Role::Unavailable);
            self.rediscover.notify_one();
        }
    }

    /// Checks the roles of all hosts in parallel; each check is bounded by the discovery timeout,
    /// so a host that does not answer delays neither the others nor the next discovery.
    async fn discover(self: &Arc<Self>) {
        let mut tasks = tokio::task::JoinSet::new();
        for host in &self.hosts {
            let (inner, host) = (self.clone(), host.clone());
            tasks.spawn(async move { inner.discover_role(&host).await });
        }
        while tasks.join_next().await.is_some() {}
        self.check_masters();
    }

    async fn discover_role(self: &Arc<Self>, host: &Arc<Host>) {
        let command_control = CommandControl {
            network_timeout: self.discovery_timeout,
            statement_timeout: self.session_statement_timeout,
        };
        let check = async {
            let mut client = self.connect_to(host, command_control).await.ok()?;
            // Hosts are reported by their roles, not by each failed check
            client.ctx.log_timeouts = false;
            let row = client.query_one_prepared("SELECT pg_is_in_recovery()", &[]).await.ok()?;
            row.try_get::<_, bool>(0).ok()
        };
        let role = match tokio::time::timeout(self.discovery_timeout, check).await {
            Ok(Some(true)) => Role::Replica,
            Ok(Some(false)) => Role::Master,
            _ => Role::Unavailable,
        };
        self.set_role(host, role);
    }

    /// Warns once when several hosts start reporting being the master, e.g. when the old master
    /// comes back after a failover without being demoted: requests go to the first one.
    fn check_masters(&self) {
        let masters: Vec<&str> = self.hosts.iter().filter(|h| h.role() == Role::Master).map(|h| h.label.as_str()).collect();
        if masters.len() > 1 {
            if !self.several_masters.swap(true, Ordering::Relaxed) {
                log!(
                    self.log,
                    LogLevel::Warning,
                    "Several PostgreSQL hosts report being the master: {}; requests go to {}",
                    masters.join(", "),
                    masters[0]
                );
            }
        } else {
            self.several_masters.store(false, Ordering::Relaxed);
        }
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

    async fn client(self: &Arc<Self>, host_type: HostType, command_control: CommandControl) -> Result<Client> {
        let mut last_error = None;
        for attempt in 0..2 {
            if attempt > 0 {
                // Nothing usable is known: the topology may have changed since the last discovery
                self.discover().await;
            }
            for host in self.candidates(host_type) {
                match self.connect_to(&host, command_control).await {
                    Ok(client) => return Ok(client),
                    Err(e) => {
                        self.mark_unavailable(&host);
                        last_error = Some(e);
                    }
                }
            }
        }
        Err(last_error.unwrap_or(Error::NoHost(host_type)))
    }

    async fn connect_to(self: &Arc<Self>, host: &Arc<Host>, command_control: CommandControl) -> Result<Client> {
        let object = host.pool.get().await?;
        Ok(Client {
            object: Some(object),
            ctx: Context {
                inner: self.clone(),
                host: host.clone(),
                command_control,
                busy: AtomicBool::new(false),
                broken: AtomicBool::new(false),
                log_timeouts: true,
            },
        })
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

/// Asks the server in the background to cancel the statement running on a connection, giving up
/// after `timeout`: the cancel request opens a connection of its own to a host that may not answer.
fn spawn_cancel(token: CancelToken, timeout: Duration) {
    if let Ok(runtime) = tokio::runtime::Handle::try_current() {
        runtime.spawn(async move {
            let _ = tokio::time::timeout(timeout, token.cancel_query(NoTls)).await;
        });
    }
}

/// Whether the server reported that the connection is lost or the server is going away.
fn is_connection_failure(code: &SqlState) -> bool {
    code.code().starts_with("08") || [SqlState::ADMIN_SHUTDOWN, SqlState::CRASH_SHUTDOWN, SqlState::CANNOT_CONNECT_NOW].contains(code)
}

/// A statement timeout in milliseconds as PostgreSQL takes it, 0 turning it off.
fn statement_timeout_ms(timeout: Option<Duration>) -> u128 {
    timeout.map_or(0, |timeout| timeout.as_millis().max(1))
}

fn describe_timeout(timeout: Option<Duration>) -> String {
    match timeout {
        Some(timeout) => format!("{}ms", timeout.as_millis()),
        None => "off".to_string(),
    }
}

/// What a client knows about its connection besides the connection itself.
struct Context {
    inner: Arc<Inner>,
    host: Arc<Host>,
    command_control: CommandControl,
    /// Set while a request waits for the server: a client dropped then leaves the connection in
    /// an unknown state.
    busy: AtomicBool,
    /// Set once the connection is known to be unusable.
    broken: AtomicBool,
    log_timeouts: bool,
}

impl Context {
    /// Runs one request to the server within the network timeout. `cancel_token` is only called
    /// once the request is given up, when nothing borrows the connection any more.
    async fn run<T>(
        &self,
        request: impl Future<Output = std::result::Result<T, tokio_postgres::Error>>,
        cancel_token: impl FnOnce() -> CancelToken,
    ) -> Result<T> {
        self.busy.store(true, Ordering::Relaxed);
        let result = tokio::time::timeout(self.command_control.network_timeout, request).await;
        self.busy.store(false, Ordering::Relaxed);
        match result {
            Ok(Ok(value)) => Ok(value),
            Ok(Err(e)) => {
                self.inspect(&e);
                Err(Error::Db(e))
            }
            Err(_) => {
                let timeout = self.command_control.network_timeout;
                if self.log_timeouts {
                    log!(
                        self.inner.log,
                        LogLevel::Warning,
                        "Network timeout error: no response from PostgreSQL host {}. Network timeout was {}ms",
                        self.host.label,
                        timeout.as_millis()
                    );
                }
                self.break_connection();
                spawn_cancel(cancel_token(), self.inner.discovery_timeout);
                Err(Error::NetworkTimeout(timeout))
            }
        }
    }

    fn inspect(&self, error: &tokio_postgres::Error) {
        let code = error.code();
        if error.is_closed() || code.is_some_and(is_connection_failure) {
            self.break_connection();
        } else if code == Some(&SqlState::QUERY_CANCELED) {
            if self.log_timeouts {
                log!(
                    self.inner.log,
                    LogLevel::Warning,
                    "Statement was cancelled on PostgreSQL host {}: {}. Statement timeout was {}",
                    self.host.label,
                    error_message(error),
                    describe_timeout(self.command_control.statement_timeout)
                );
            }
        } else if code == Some(&SqlState::READ_ONLY_SQL_TRANSACTION) {
            // The master may have been demoted
            self.inner.rediscover.notify_one();
        }
    }

    fn break_connection(&self) {
        self.broken.store(true, Ordering::Relaxed);
        self.inner.mark_unavailable(&self.host);
    }

    /// Whether requests need a statement timeout other than the one their session starts with.
    fn overrides_statement_timeout(&self) -> bool {
        self.command_control.statement_timeout != self.inner.session_statement_timeout
    }
}

/// A pooled connection to a host, e.g. to run a transaction on. A connection that failed or was
/// given up on mid-request is closed when the client is dropped instead of going back to the pool.
pub struct Client {
    /// Always set until the client is dropped.
    object: Option<Object>,
    ctx: Context,
}

impl Client {
    fn object(&self) -> &Object {
        self.object.as_ref().expect("the connection is only taken on drop")
    }

    /// Starts a transaction, with the client's statement timeout if it differs from the session's.
    pub async fn transaction(&mut self) -> Result<Transaction<'_>> {
        let object = self.object.as_mut().expect("the connection is only taken on drop");
        let ctx = &self.ctx;
        let token = object.cancel_token();
        let transaction = ctx.run(object.transaction(), || token).await?;
        let transaction = Transaction { transaction, ctx };
        if ctx.overrides_statement_timeout() {
            let sql = format!("SET LOCAL statement_timeout = {}", statement_timeout_ms(ctx.command_control.statement_timeout));
            transaction
                .ctx
                .run(transaction.transaction.batch_execute(&sql), || transaction.transaction.cancel_token())
                .await?;
        }
        Ok(transaction)
    }

    async fn query_prepared(&self, sql: &str, params: &[&(dyn ToSql + Sync)]) -> Result<Vec<Row>> {
        let client = self.object();
        let request = async { client.query(&client.prepare_cached(sql).await?, params).await };
        self.ctx.run(request, || client.cancel_token()).await
    }

    async fn query_opt_prepared(&self, sql: &str, params: &[&(dyn ToSql + Sync)]) -> Result<Option<Row>> {
        let client = self.object();
        let request = async { client.query_opt(&client.prepare_cached(sql).await?, params).await };
        self.ctx.run(request, || client.cancel_token()).await
    }

    async fn query_one_prepared(&self, sql: &str, params: &[&(dyn ToSql + Sync)]) -> Result<Row> {
        let client = self.object();
        let request = async { client.query_one(&client.prepare_cached(sql).await?, params).await };
        self.ctx.run(request, || client.cancel_token()).await
    }

    async fn execute_prepared(&self, sql: &str, params: &[&(dyn ToSql + Sync)]) -> Result<u64> {
        let client = self.object();
        let request = async { client.execute(&client.prepare_cached(sql).await?, params).await };
        self.ctx.run(request, || client.cancel_token()).await
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        let Some(object) = self.object.take() else {
            return;
        };
        let busy = self.ctx.busy.load(Ordering::Relaxed);
        if busy {
            // Dropped mid-request, e.g. with a cancelled handler: the statement may still run
            spawn_cancel(object.cancel_token(), self.ctx.inner.discovery_timeout);
        }
        if busy || self.ctx.broken.load(Ordering::Relaxed) {
            // Out of the pool, the connection is dropped, which aborts its task and closes the
            // socket at once instead of waiting for the server to answer
            drop(Object::take(object));
        }
    }
}

/// A transaction on a [`Client`]; each request is bounded by the client's network timeout.
/// Dropping it without committing rolls it back.
pub struct Transaction<'a> {
    transaction: deadpool_postgres::Transaction<'a>,
    ctx: &'a Context,
}

impl Transaction<'_> {
    pub async fn query<T>(&self, statement: &T, params: &[&(dyn ToSql + Sync)]) -> Result<Vec<Row>>
    where
        T: ?Sized + ToStatement + Sync,
    {
        let transaction = &self.transaction;
        self.ctx.run(transaction.query(statement, params), || transaction.cancel_token()).await
    }

    pub async fn query_opt<T>(&self, statement: &T, params: &[&(dyn ToSql + Sync)]) -> Result<Option<Row>>
    where
        T: ?Sized + ToStatement + Sync,
    {
        let transaction = &self.transaction;
        self.ctx.run(transaction.query_opt(statement, params), || transaction.cancel_token()).await
    }

    pub async fn query_one<T>(&self, statement: &T, params: &[&(dyn ToSql + Sync)]) -> Result<Row>
    where
        T: ?Sized + ToStatement + Sync,
    {
        let transaction = &self.transaction;
        self.ctx.run(transaction.query_one(statement, params), || transaction.cancel_token()).await
    }

    pub async fn execute<T>(&self, statement: &T, params: &[&(dyn ToSql + Sync)]) -> Result<u64>
    where
        T: ?Sized + ToStatement + Sync,
    {
        let transaction = &self.transaction;
        self.ctx.run(transaction.execute(statement, params), || transaction.cancel_token()).await
    }

    pub async fn commit(self) -> Result<()> {
        let token = self.transaction.cancel_token();
        self.ctx.run(self.transaction.commit(), || token).await
    }

    pub async fn rollback(self) -> Result<()> {
        let token = self.transaction.cancel_token();
        self.ctx.run(self.transaction.rollback(), || token).await
    }

    async fn query_prepared(&self, sql: &str, params: &[&(dyn ToSql + Sync)]) -> Result<Vec<Row>> {
        let transaction = &self.transaction;
        let request = async { transaction.query(&transaction.prepare_cached(sql).await?, params).await };
        self.ctx.run(request, || transaction.cancel_token()).await
    }

    async fn query_opt_prepared(&self, sql: &str, params: &[&(dyn ToSql + Sync)]) -> Result<Option<Row>> {
        let transaction = &self.transaction;
        let request = async { transaction.query_opt(&transaction.prepare_cached(sql).await?, params).await };
        self.ctx.run(request, || transaction.cancel_token()).await
    }

    async fn query_one_prepared(&self, sql: &str, params: &[&(dyn ToSql + Sync)]) -> Result<Row> {
        let transaction = &self.transaction;
        let request = async { transaction.query_one(&transaction.prepare_cached(sql).await?, params).await };
        self.ctx.run(request, || transaction.cancel_token()).await
    }

    async fn execute_prepared(&self, sql: &str, params: &[&(dyn ToSql + Sync)]) -> Result<u64> {
        let transaction = &self.transaction;
        let request = async { transaction.execute(&transaction.prepare_cached(sql).await?, params).await };
        self.ctx.run(request, || transaction.cancel_token()).await
    }
}

/// Runs one prepared request on a client of `host_type`; when the cluster's statement timeout
/// differs from the session's, in a transaction that sets it.
macro_rules! single_request {
    ($cluster:expr, $host_type:expr, $method:ident($($arg:expr),*)) => {{
        let mut client = $cluster.client($host_type).await?;
        if client.ctx.overrides_statement_timeout() {
            let transaction = client.transaction().await?;
            let result = transaction.$method($($arg),*).await?;
            transaction.commit().await?;
            Ok(result)
        } else {
            client.$method($($arg),*).await
        }
    }};
}

/// A PostgreSQL cluster of one master and any number of replicas.
#[derive(Clone)]
pub struct Cluster {
    inner: Arc<Inner>,
    command_control: CommandControl,
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
        // Without a database name PostgreSQL connects to the user's database
        let dbname = config.get_dbname().or(config.get_user()).unwrap_or_default().to_string();
        let hosts = split_hosts(&config)?
            .into_iter()
            .map(|(name, host_config)| {
                make_pool(host_config, &settings).map(|pool| Host {
                    label: format!("{name}/{dbname}"),
                    name,
                    pool,
                    role: RwLock::new(Role::Unknown),
                })
            })
            .map(|host| host.map(Arc::new))
            .collect::<Result<Vec<_>>>()?;
        let inner = Arc::new(Inner {
            hosts,
            next_replica: AtomicUsize::new(0),
            rediscover: Notify::new(),
            discovery_timeout: settings.discovery_timeout,
            session_statement_timeout: settings.command_control.statement_timeout,
            log: settings.log.clone(),
            several_masters: AtomicBool::new(false),
        });
        inner.discover().await;
        tokio::spawn(run_discovery(Arc::downgrade(&inner), settings.discovery_interval));
        Ok(Self {
            inner,
            command_control: settings.command_control,
        })
    }

    /// The same cluster with requests bounded by other timeouts, e.g. for long cache fills.
    pub fn with_command_control(&self, command_control: CommandControl) -> Cluster {
        Self {
            inner: self.inner.clone(),
            command_control,
        }
    }

    pub fn command_control(&self) -> CommandControl {
        self.command_control
    }

    /// A pooled connection to a host of `host_type`, e.g. to run a transaction on.
    pub async fn client(&self, host_type: HostType) -> Result<Client> {
        self.inner.client(host_type, self.command_control).await
    }

    pub async fn query(&self, host_type: HostType, sql: &str, params: &[&(dyn ToSql + Sync)]) -> Result<Vec<Row>> {
        single_request!(self, host_type, query_prepared(sql, params))
    }

    /// Runs a query expected to return at most one row.
    pub async fn query_opt(&self, host_type: HostType, sql: &str, params: &[&(dyn ToSql + Sync)]) -> Result<Option<Row>> {
        single_request!(self, host_type, query_opt_prepared(sql, params))
    }

    /// Runs a query expected to return exactly one row.
    pub async fn query_one(&self, host_type: HostType, sql: &str, params: &[&(dyn ToSql + Sync)]) -> Result<Row> {
        single_request!(self, host_type, query_one_prepared(sql, params))
    }

    pub async fn execute(&self, host_type: HostType, sql: &str, params: &[&(dyn ToSql + Sync)]) -> Result<u64> {
        single_request!(self, host_type, execute_prepared(sql, params))
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
            // TCP settings, as libpq applies them for the C++ service
            if let Some(connect_timeout) = config.get_connect_timeout() {
                single.connect_timeout(*connect_timeout);
            }
            if let Some(tcp_user_timeout) = config.get_tcp_user_timeout() {
                // libpq reads the value in milliseconds, tokio-postgres in seconds
                single.tcp_user_timeout(Duration::from_millis(tcp_user_timeout.as_secs()));
            }
            single.keepalives(config.get_keepalives());
            single.keepalives_idle(config.get_keepalives_idle());
            if let Some(interval) = config.get_keepalives_interval() {
                single.keepalives_interval(interval);
            }
            if let Some(retries) = config.get_keepalives_retries() {
                single.keepalives_retries(retries);
            }
            Ok((name, single))
        })
        .collect()
}

fn make_pool(mut config: Config, settings: &ClusterSettings) -> Result<Pool> {
    let connect_timeout = config.get_connect_timeout().copied().unwrap_or(settings.connect_timeout);
    config.connect_timeout(connect_timeout);
    // Appended last, so it takes precedence over a statement_timeout in the connection string,
    // as userver's command control does
    let statement_timeout = format!("-c statement_timeout={}", statement_timeout_ms(settings.command_control.statement_timeout));
    let options = match config.get_options() {
        Some(options) => format!("{options} {statement_timeout}"),
        None => statement_timeout,
    };
    config.options(&options);
    let manager = Manager::from_config(
        config,
        NoTls,
        ManagerConfig {
            recycling_method: RecyclingMethod::Fast,
        },
    );
    Pool::builder(manager)
        .max_size(settings.max_pool_size)
        .runtime(Runtime::Tokio1)
        .create_timeout(Some(connect_timeout))
        .wait_timeout(Some(connect_timeout))
        .build()
        .map_err(|e| Error::Config(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(dsn: &str) -> Vec<String> {
        split_hosts(&Config::from_str(dsn).unwrap())
            .unwrap()
            .into_iter()
            .map(|(name, _)| name)
            .collect()
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

    #[test]
    fn keeps_tcp_parameters() {
        let dsn = "postgresql://u:p@h1,h2/db?connect_timeout=7&keepalives_idle=30&keepalives_interval=5&keepalives_retries=3&tcp_user_timeout=10000";
        for (_, config) in split_hosts(&Config::from_str(dsn).unwrap()).unwrap() {
            assert_eq!(config.get_connect_timeout(), Some(&Duration::from_secs(7)));
            assert!(config.get_keepalives());
            assert_eq!(config.get_keepalives_idle(), Duration::from_secs(30));
            assert_eq!(config.get_keepalives_interval(), Some(Duration::from_secs(5)));
            assert_eq!(config.get_keepalives_retries(), Some(3));
            assert_eq!(config.get_tcp_user_timeout(), Some(&Duration::from_secs(10)));
        }
        let (_, config) = split_hosts(&Config::from_str("postgresql://u:p@h1/db?keepalives=0").unwrap())
            .unwrap()
            .remove(0);
        assert!(!config.get_keepalives());
        assert_eq!(config.get_connect_timeout(), None);
    }

    #[test]
    fn statement_timeouts_in_milliseconds() {
        assert_eq!(statement_timeout_ms(None), 0);
        assert_eq!(statement_timeout_ms(Some(Duration::from_millis(250))), 250);
        // Never 0, which would turn the limit off
        assert_eq!(statement_timeout_ms(Some(Duration::from_micros(10))), 1);
    }
}
