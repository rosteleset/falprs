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
#[command(version, about = "FALPRS: licence plate and face recognition service")]
struct Args {
    /// Path to the service config (the userver config.yaml)
    #[arg(short, long)]
    config: PathBuf,
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
    // TLS for outgoing HTTPS uses ring, the only dependency with C/asm code
    rustls::crypto::ring::default_provider()
        .install_default()
        .map_err(|_| anyhow::anyhow!("can't install the TLS crypto provider"))?;
    let config = Config::load(&args.config)?;
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
