//! Tests against real PostgreSQL in Docker: a primary and a streaming replica.
//! The user running them must be able to use Docker.

use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use pg_cluster::{Cluster, ClusterSettings, Error, HostType, Role};

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
    Command::new("docker")
        .args(args)
        .output()
        .is_ok_and(|output| output.status.success())
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
    format!(
        "falprs-pg-test-{}-{}-{role}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    )
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
        Container { name, address: format!("127.0.0.1:{port}") }
    }

    fn is_ready(&self) -> bool {
        // The entrypoint's init server listens on the Unix socket only, so TCP readiness means the final server is up
        docker_ok(&["exec", &self.name, "pg_isready", "-h", "127.0.0.1", "-U", "postgres"])
    }

    fn stop(&self) {
        docker(&["stop", "-t", "1", &self.name]);
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

        let primary = Container::run(unique_name("primary"), &[
            "--network", &network.name, "-e", "POSTGRES_PASSWORD=test", IMAGE,
        ]);
        wait_until("primary", || primary.is_ready());
        docker(&[
            "exec", &primary.name, "bash", "-c",
            "echo 'host replication all all trust' >> \"$PGDATA/pg_hba.conf\" && psql -U postgres -c 'SELECT pg_reload_conf()'",
        ]);

        let replica = with_replica.then(|| {
            let script = format!(
                "until pg_basebackup -h {} -U postgres -D {REPLICA_DATA} -R -X stream; do rm -rf {REPLICA_DATA}; sleep 1; done; \
                 chmod 700 {REPLICA_DATA}; exec postgres -D {REPLICA_DATA}",
                primary.name
            );
            let replica = Container::run(unique_name("replica"), &[
                "--network", &network.name, "--user", "postgres", "--entrypoint", "bash", IMAGE, "-c", &script,
            ]);
            wait_until("replica", || replica.is_ready());
            replica
        });

        TestCluster { primary, replica, _network: network }
    }

    fn replica(&self) -> &Container {
        self.replica.as_ref().expect("cluster has no replica")
    }

    fn dsn(&self) -> String {
        let hosts: Vec<String> = std::iter::once(&self.primary)
            .chain(self.replica.as_ref())
            .map(|c| c.address.clone())
            .collect();
        format!("postgresql://postgres:test@{}/postgres", hosts.join(","))
    }
}

fn settings() -> ClusterSettings {
    ClusterSettings {
        discovery_interval: Duration::from_millis(500),
        connect_timeout: Duration::from_secs(1),
        ..ClusterSettings::default()
    }
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
        vec![
            (test.primary.address.clone(), Role::Master),
            (test.replica().address.clone(), Role::Replica),
        ]
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

    let rows = cluster
        .query(HostType::Master, "SELECT to_regclass('t') IS NULL", &[])
        .await
        .unwrap();
    assert!(rows[0].get::<_, bool>(0), "rolled back table must not exist");
}

#[tokio::test(flavor = "multi_thread")]
async fn replica_or_master_falls_back_to_master() {
    let test = TestCluster::start(true);
    let cluster = Cluster::connect(&test.dsn(), settings()).await.unwrap();

    test.replica().stop();
    eventually("fallback to the master", async || {
        (in_recovery(&cluster, HostType::ReplicaOrMaster).await.ok()? == false).then_some(())
    })
    .await;
    assert!(matches!(in_recovery(&cluster, HostType::Replica).await, Err(_)));
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
        (in_recovery(&cluster, HostType::Master).await.ok()? == false).then_some(())
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
    assert!(matches!(
        in_recovery(&cluster, HostType::Replica).await,
        Err(Error::NoHost(HostType::Replica))
    ));
}
