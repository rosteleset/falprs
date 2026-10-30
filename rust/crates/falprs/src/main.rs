mod cache;
mod config;
mod converters;
mod fetch;
mod frs;
mod http_api;
mod logging;
mod lprs;
mod osd;
mod service;

use std::net::{Ipv6Addr, SocketAddr};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use axum::Router;
use clap::Parser;
use tokio::net::TcpListener;
use tokio::signal::unix::{SignalKind, signal};
use tokio::sync::watch;

use falprs::{imaging, triton};

use crate::config::Config;
use crate::logging::{Level, Loggers};

#[derive(Parser)]
#[command(about = "FALPRS: licence plate and face recognition service", disable_version_flag = true)]
struct Args {
    /// Print the version and exit; only the number is printed, like the C++ service, as the
    /// update script compares it
    #[arg(short, long)]
    version: bool,
    /// Path to the service config (the userver config.yaml)
    #[arg(short, long, required_unless_present = "version")]
    config: Option<PathBuf>,
}

async fn serve(port: u16, router: Router, mut shutdown: watch::Receiver<bool>) -> Result<()> {
    // Dual-stack listener, like userver's default
    let address = SocketAddr::from((Ipv6Addr::UNSPECIFIED, port));
    let listener = TcpListener::bind(address)
        .await
        .with_context(|| format!("can't listen on port {port}"))?;
    axum::serve(listener, router.into_make_service_with_connect_info::<SocketAddr>())
        .with_graceful_shutdown(async move {
            let _ = shutdown.wait_for(|stop| *stop).await;
        })
        .await?;
    Ok(())
}

/// Ignores SIGUSR1, which would otherwise terminate the process: userver reopens its logs on it,
/// here logs are reopened through the on-log-rotate handler instead.
fn ignore_sigusr1() -> Result<()> {
    let mut user1 = signal(SignalKind::user_defined1())?;
    tokio::spawn(async move { while user1.recv().await.is_some() {} });
    Ok(())
}

async fn wait_for_termination() -> Result<()> {
    let mut terminate = signal(SignalKind::terminate())?;
    let mut interrupt = signal(SignalKind::interrupt())?;
    tokio::select! {
        _ = terminate.recv() => {}
        _ = interrupt.recv() => {}
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let Some(config_path) = args.config.filter(|_| !args.version) else {
        println!("{}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    };
    // TLS for outgoing HTTPS uses ring, the only dependency with C/asm code
    rustls::crypto::ring::default_provider()
        .install_default()
        .map_err(|_| anyhow::anyhow!("can't install the TLS crypto provider"))?;
    ignore_sigusr1()?;
    let config = Config::load(&config_path)?;
    let loggers = Arc::new(Loggers::new(&config.loggers)?);
    let logger = loggers.default_logger();
    log!(logger, Level::Info, "starting service");

    let mut main_router = service::main_router(config.static_dir.clone());
    let mut frs_subsystem = None;
    if let Some(lprs_config) = &config.lprs {
        let lprs_logger = loggers.get("lprs").unwrap_or_else(|| logger.clone());
        let lprs = lprs::Lprs::start(lprs_config, lprs_logger).await?;
        main_router = main_router.merge(lprs.router());
    }
    if let Some(frs_config) = &config.frs {
        let frs_logger = loggers.get("frs").unwrap_or_else(|| logger.clone());
        let frs = frs::Frs::start(frs_config, frs_logger).await?;
        main_router = main_router.merge(frs.router());
        frs_subsystem = Some(frs);
    }

    let (stop, shutdown) = watch::channel(false);
    let mut servers = tokio::task::JoinSet::new();
    servers.spawn(serve(config.port, main_router, shutdown.clone()));
    if let Some(port) = config.monitor_port {
        servers.spawn(serve(port, service::monitor_router(loggers.clone()), shutdown));
    }

    tokio::select! {
        result = wait_for_termination() => result?,
        Some(result) = servers.join_next() => {
            // A listener stopped on its own: report why and shut down the rest
            result??;
        }
    }
    log!(logger, Level::Info, "stopping service");
    let _ = stop.send(true);
    while let Some(result) = servers.join_next().await {
        result??;
    }
    if let Some(frs) = frs_subsystem {
        frs.shutdown().await;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn sigusr1_does_not_terminate_the_process() {
        ignore_sigusr1().unwrap();
        let status = std::process::Command::new("kill")
            .args(["-USR1", &std::process::id().to_string()])
            .status()
            .unwrap();
        assert!(status.success());
        // Without the handler the default action would have killed the test process by now
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }
}
