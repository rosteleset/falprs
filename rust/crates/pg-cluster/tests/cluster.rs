//! Tests against real PostgreSQL in Docker: a primary and a streaming replica.
//! The user running them must be able to use Docker.

use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use pg_cluster::{Cluster, ClusterSettings, CommandControl, Error, HostType, LogSink, Role};

const IMAGE: &str = "postgres:16";
const REPLICA_DATA: &str = "/tmp/replica";
const READY_TIMEOUT: Duration = Duration::from_secs(60);

fn docker(args: &[&str]) -> String {
    let output = Command::new("docker").args(args).output().expect("docker is not available");
    assert!(
        output.status.success(),
        "docker {} failed: {}",
        args.join(" "),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).trim().to_string()
}

fn docker_ok(args: &[&str]) -> bool {
    Command::new("docker").args(args).output().is_ok_and(|output| output.status.success())
}

fn wait_until(what: &str, mut condition: impl FnMut() -> bool) {
    let deadline = Instant::now() + READY_TIMEOUT;
    while !condition() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        thread::sleep(Duration::from_millis(200));
    }
}

fn unique_name(role: &str) -> String {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    format!("falprs-pg-test-{}-{}-{role}", std::process::id(), COUNTER.fetch_add(1, Ordering::Relaxed))
}

/// A container removed when dropped, also when a test fails.
struct Container {
    name: String,
    /// Host address of the published PostgreSQL port; Docker forgets it once the container stops.
    address: String,
}

impl Container {
    fn run(name: String, args: &[&str]) -> Container {
        docker(&[&["run", "-d", "--name", &name, "-p", "127.0.0.1::5432"], args].concat());
        let published = docker(&["port", &name, "5432/tcp"]);
        let port = published.lines().next().unwrap_or_default().rsplit(':').next().unwrap().to_string();
        Container {
            name,
            address: format!("127.0.0.1:{port}"),
        }
    }

    fn is_ready(&self) -> bool {
        // The entrypoint's init server listens on the Unix socket only, so TCP readiness means the final server is up
        docker_ok(&["exec", &self.name, "pg_isready", "-h", "127.0.0.1", "-U", "postgres"])
    }

    fn stop(&self) {
        docker(&["stop", "-t", "1", &self.name]);
    }

    /// Freezes the server: the kernel still accepts and acknowledges TCP, but nothing answers.
    fn pause(&self) {
        docker(&["pause", &self.name]);
    }

    fn unpause(&self) {
        docker(&["unpause", &self.name]);
    }

    fn port(&self) -> u16 {
        self.address.rsplit(':').next().unwrap().parse().unwrap()
    }
}

impl Drop for Container {
    fn drop(&mut self) {
        let _ = Command::new("docker").args(["rm", "-f", "-v", &self.name]).output();
    }
}

struct Network {
    name: String,
}

impl Drop for Network {
    fn drop(&mut self) {
        let _ = Command::new("docker").args(["network", "rm", &self.name]).output();
    }
}

/// A primary and, optionally, a streaming replica. Field order makes containers go before the network.
struct TestCluster {
    primary: Container,
    replica: Option<Container>,
    _network: Network,
}

impl TestCluster {
    fn start(with_replica: bool) -> TestCluster {
        let network = Network { name: unique_name("net") };
        docker(&["network", "create", &network.name]);

        let primary = Container::run(unique_name("primary"), &["--network", &network.name, "-e", "POSTGRES_PASSWORD=test", IMAGE]);
        wait_until("primary", || primary.is_ready());
        docker(&[
            "exec",
            &primary.name,
            "bash",
            "-c",
            "echo 'host replication all all trust' >> \"$PGDATA/pg_hba.conf\" && psql -U postgres -c 'SELECT pg_reload_conf()'",
        ]);

        let replica = with_replica.then(|| {
            let script = format!(
                "until pg_basebackup -h {} -U postgres -D {REPLICA_DATA} -R -X stream; do rm -rf {REPLICA_DATA}; sleep 1; done; \
                 chmod 700 {REPLICA_DATA}; exec postgres -D {REPLICA_DATA}",
                primary.name
            );
            let replica = Container::run(
                unique_name("replica"),
                &["--network", &network.name, "--user", "postgres", "--entrypoint", "bash", IMAGE, "-c", &script],
            );
            wait_until("replica", || replica.is_ready());
            replica
        });

        TestCluster {
            primary,
            replica,
            _network: network,
        }
    }

    fn replica(&self) -> &Container {
        self.replica.as_ref().expect("cluster has no replica")
    }

    fn dsn(&self) -> String {
        let hosts: Vec<String> = std::iter::once(&self.primary).chain(self.replica.as_ref()).map(|c| c.address.clone()).collect();
        format!("postgresql://postgres:test@{}/postgres", hosts.join(","))
    }
}

fn settings() -> ClusterSettings {
    ClusterSettings {
        discovery_interval: Duration::from_millis(500),
        discovery_timeout: Duration::from_secs(1),
        connect_timeout: Duration::from_secs(1),
        ..ClusterSettings::default()
    }
}

/// Timeouts long enough for the slow statements of a test.
fn long_command_control() -> CommandControl {
    CommandControl {
        network_timeout: Duration::from_secs(60),
        statement_timeout: None,
    }
}

/// Collects the cluster's messages.
fn log_lines() -> (LogSink, Arc<Mutex<Vec<String>>>) {
    let lines = Arc::new(Mutex::new(Vec::new()));
    let sink = {
        let lines = lines.clone();
        LogSink::new(move |record| lines.lock().unwrap().push(record.text.to_string()))
    };
    (sink, lines)
}

/// Established TCP connections of this machine to `port` on 127.0.0.1, i.e. the client sides of
/// connections to a container.
fn connections_to(port: u16) -> usize {
    let peer = format!("0100007F:{port:04X}");
    std::fs::read_to_string("/proc/net/tcp")
        .unwrap()
        .lines()
        .skip(1)
        .filter(|line| {
            let fields: Vec<&str> = line.split_whitespace().collect();
            fields[2] == peer && fields[3] == "01"
        })
        .count()
}

/// Whether the host serving `host_type` is in recovery, i.e. is a replica.
async fn in_recovery(cluster: &Cluster, host_type: HostType) -> pg_cluster::Result<bool> {
    let rows = cluster.query(host_type, "SELECT pg_is_in_recovery()", &[]).await?;
    Ok(rows[0].get(0))
}

async fn eventually<T>(what: &str, mut check: impl AsyncFnMut() -> Option<T>) -> T {
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        if let Some(value) = check().await {
            return value;
        }
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn discovers_roles() {
    let test = TestCluster::start(true);
    let cluster = Cluster::connect(&test.dsn(), settings()).await.unwrap();

    assert_eq!(
        cluster.topology(),
        vec![(test.primary.address.clone(), Role::Master), (test.replica().address.clone(), Role::Replica),]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn routes_by_host_type() {
    let test = TestCluster::start(true);
    let cluster = Cluster::connect(&test.dsn(), settings()).await.unwrap();

    assert!(!in_recovery(&cluster, HostType::Master).await.unwrap());
    assert!(in_recovery(&cluster, HostType::Replica).await.unwrap());
    assert!(in_recovery(&cluster, HostType::ReplicaOrMaster).await.unwrap());

    // Writes go to the master and reach the replica
    cluster.execute(HostType::Master, "CREATE TABLE t (v integer)", &[]).await.unwrap();
    cluster.execute(HostType::Master, "INSERT INTO t VALUES ($1)", &[&42i32]).await.unwrap();
    let value: i32 = eventually("replication", async || {
        let rows = cluster.query(HostType::Replica, "SELECT v FROM t", &[]).await.ok()?;
        rows.first().map(|row| row.get(0))
    })
    .await;
    assert_eq!(value, 42);
}

#[tokio::test(flavor = "multi_thread")]
async fn transaction_on_master() {
    let test = TestCluster::start(false);
    let cluster = Cluster::connect(&test.dsn(), settings()).await.unwrap();

    let mut client = cluster.client(HostType::Master).await.unwrap();
    let transaction = client.transaction().await.unwrap();
    transaction.execute("CREATE TABLE t (v integer)", &[]).await.unwrap();
    transaction.execute("INSERT INTO t VALUES (1)", &[]).await.unwrap();
    transaction.rollback().await.unwrap();

    let rows = cluster.query(HostType::Master, "SELECT to_regclass('t') IS NULL", &[]).await.unwrap();
    assert!(rows[0].get::<_, bool>(0), "rolled back table must not exist");
}

#[tokio::test(flavor = "multi_thread")]
async fn replica_or_master_falls_back_to_master() {
    let test = TestCluster::start(true);
    let cluster = Cluster::connect(&test.dsn(), settings()).await.unwrap();

    test.replica().stop();
    eventually("fallback to the master", async || {
        (!in_recovery(&cluster, HostType::ReplicaOrMaster).await.ok()?).then_some(())
    })
    .await;
    assert!(in_recovery(&cluster, HostType::Replica).await.is_err());
    assert!(!in_recovery(&cluster, HostType::Master).await.unwrap());
}

#[tokio::test(flavor = "multi_thread")]
async fn follows_failover() {
    let test = TestCluster::start(true);
    let cluster = Cluster::connect(&test.dsn(), settings()).await.unwrap();
    assert!(!in_recovery(&cluster, HostType::Master).await.unwrap());

    test.primary.stop();
    docker(&["exec", &test.replica().name, "pg_ctl", "promote", "-D", REPLICA_DATA]);

    eventually("the promoted replica to serve the master", async || {
        (!in_recovery(&cluster, HostType::Master).await.ok()?).then_some(())
    })
    .await;
    cluster.execute(HostType::Master, "CREATE TABLE after_failover (v integer)", &[]).await.unwrap();
    assert_eq!(
        cluster.topology(),
        vec![
            (test.primary.address.clone(), Role::Unavailable),
            (test.replica().address.clone(), Role::Master),
        ]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn single_host() {
    let test = TestCluster::start(false);
    let cluster = Cluster::connect(&test.dsn(), settings()).await.unwrap();

    assert!(!in_recovery(&cluster, HostType::Master).await.unwrap());
    assert!(!in_recovery(&cluster, HostType::ReplicaOrMaster).await.unwrap());
    assert!(matches!(in_recovery(&cluster, HostType::Replica).await, Err(Error::NoHost(HostType::Replica))));
}

#[tokio::test(flavor = "multi_thread")]
async fn follows_failover_from_a_hung_master() {
    let test = TestCluster::start(true);
    let (log, lines) = log_lines();
    let cluster = Cluster::connect(&test.dsn(), ClusterSettings { log, ..settings() }).await.unwrap();
    assert!(!in_recovery(&cluster, HostType::Master).await.unwrap());

    // A master that stops answering without closing connections, as behind a network partition
    test.primary.pause();
    docker(&["exec", &test.replica().name, "pg_ctl", "promote", "-D", REPLICA_DATA]);

    eventually("the promoted replica to serve the master", async || {
        (!in_recovery(&cluster, HostType::Master).await.ok()?).then_some(())
    })
    .await;
    cluster.execute(HostType::Master, "CREATE TABLE after_failover (v integer)", &[]).await.unwrap();
    assert_eq!(cluster.topology()[0].1, Role::Unavailable);
    let promoted = format!("PostgreSQL host {}/postgres: replica -> master", test.replica().address);
    assert!(lines.lock().unwrap().contains(&promoted), "{:?}", lines.lock().unwrap());
    test.primary.unpause();
}

#[tokio::test(flavor = "multi_thread")]
async fn statement_timeout_cancels_the_statement_only() {
    let test = TestCluster::start(false);
    let (log, lines) = log_lines();
    let command_control = CommandControl {
        network_timeout: Duration::from_secs(5),
        statement_timeout: Some(Duration::from_millis(300)),
    };
    let cluster = Cluster::connect(
        &test.dsn(),
        ClusterSettings {
            command_control,
            log,
            ..settings()
        },
    )
    .await
    .unwrap();

    let error = cluster.query(HostType::Master, "SELECT pg_sleep(2)", &[]).await.unwrap_err();
    assert!(error.is_query_cancelled(), "{error}");
    assert!(lines.lock().unwrap().iter().any(
        |line| line.starts_with(&format!("Statement was cancelled on PostgreSQL host {}/postgres: ", test.primary.address))
            && line.ends_with("Statement timeout was 300ms")
    ));
    // The connection and the host stay usable
    assert_eq!(cluster.topology()[0].1, Role::Master);
    assert!(!in_recovery(&cluster, HostType::Master).await.unwrap());

    // A view without a statement timeout, as cache updates use, sets it off for its requests only
    let unlimited = cluster.with_command_control(long_command_control());
    unlimited.query(HostType::Master, "SELECT pg_sleep(1)", &[]).await.unwrap();
    let mut client = unlimited.client(HostType::Master).await.unwrap();
    let transaction = client.transaction().await.unwrap();
    let row = transaction.query_one("SHOW statement_timeout", &[]).await.unwrap();
    assert_eq!(row.get::<_, String>(0), "0");
    drop(transaction);
    drop(client);
    let rows = cluster.query(HostType::Master, "SHOW statement_timeout", &[]).await.unwrap();
    assert_eq!(rows[0].get::<_, String>(0), "300ms");
}

#[tokio::test(flavor = "multi_thread")]
async fn network_timeout_closes_connections_to_a_hung_host() {
    let test = TestCluster::start(false);
    let (log, lines) = log_lines();
    let command_control = CommandControl {
        network_timeout: Duration::from_millis(1000),
        statement_timeout: Some(Duration::from_secs(5)),
    };
    let settings = ClusterSettings {
        max_pool_size: 10,
        command_control,
        log,
        ..settings()
    };
    let cluster = Cluster::connect(&test.dsn(), settings).await.unwrap();

    // Fill the pool with connections
    let sleeps = (0..10).map(|_| cluster.query(HostType::Master, "SELECT pg_sleep(0.3)", &[]));
    for result in futures::future::join_all(sleeps).await {
        result.unwrap();
    }
    let port = test.primary.port();
    assert!(connections_to(port) >= 10);

    test.primary.pause();
    let started = Instant::now();
    let requests = (0..10).map(|_| cluster.query(HostType::Master, "SELECT 1", &[]));
    for result in futures::future::join_all(requests).await {
        assert!(matches!(result, Err(Error::NetworkTimeout(_))), "{result:?}");
    }
    assert!(started.elapsed() < Duration::from_secs(5));
    assert_eq!(cluster.topology()[0].1, Role::Unavailable);
    assert!(lines.lock().unwrap().iter().any(|line| line.starts_with(&format!(
        "Network timeout error: no response from PostgreSQL host {}/postgres.",
        test.primary.address
    )) && line.ends_with("Network timeout was 1000ms")));
    // The connections that timed out are closed, not left waiting for the server; discovery may
    // hold one more while it tries the host
    // Sockets close as soon as the runtime drops the aborted connection tasks
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(connections_to(port) <= 1, "{} connections left", connections_to(port));
    // Requests fail fast while the host stays down
    let started = Instant::now();
    assert!(cluster.query(HostType::Master, "SELECT 1", &[]).await.is_err());
    assert!(started.elapsed() < Duration::from_secs(3));

    test.primary.unpause();
    eventually("the host to come back", async || {
        (!in_recovery(&cluster, HostType::Master).await.ok()?).then_some(())
    })
    .await;
}

#[tokio::test(flavor = "multi_thread")]
async fn lost_connection_fails_the_request_at_once() {
    let test = TestCluster::start(false);
    let cluster = Cluster::connect(&test.dsn(), settings()).await.unwrap();
    let unlimited = cluster.with_command_control(long_command_control());

    let started = Instant::now();
    let request = tokio::spawn(async move { unlimited.query(HostType::Master, "SELECT pg_sleep(30)", &[]).await });
    tokio::time::sleep(Duration::from_secs(1)).await;
    docker(&["kill", &test.primary.name]);

    let result = request.await.unwrap();
    assert!(matches!(result, Err(Error::Db(_))), "{result:?}");
    assert!(started.elapsed() < Duration::from_secs(10));
    assert_eq!(cluster.topology()[0].1, Role::Unavailable);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_busy_pool_is_not_a_host_failure() {
    let test = TestCluster::start(false);
    let settings = ClusterSettings {
        max_pool_size: 1,
        connect_timeout: Duration::from_millis(300),
        ..settings()
    };
    let cluster = Cluster::connect(&test.dsn(), settings).await.unwrap();
    let held = cluster.client(HostType::Master).await.unwrap();

    // The only pooled connection is taken: requests wait for it and fail, the master stays
    let busy = in_recovery(&cluster, HostType::Master).await;
    assert!(matches!(busy, Err(Error::PoolExhausted { max_size: 1, .. })), "{busy:?}");
    assert_eq!(cluster.topology()[0].1, Role::Master);
    // Discoveries keep running on their own connection and still find the master
    tokio::time::sleep(Duration::from_millis(1500)).await;
    assert_eq!(cluster.topology()[0].1, Role::Master);
    assert!(matches!(in_recovery(&cluster, HostType::Master).await, Err(Error::PoolExhausted { .. })));

    drop(held);
    assert!(!in_recovery(&cluster, HostType::Master).await.unwrap());
}
