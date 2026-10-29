//! Service endpoints shared by LPRS and FRS: ping, static files, log level and log rotation.

use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::extract::{Path as UrlPath, Query, State};
use axum::http::{Method, StatusCode, Uri, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use serde::Deserialize;
use serde_json::json;

use crate::logging::{Level, Logger, Loggers};

/// Routes of the main listener.
pub fn main_router(static_dir: Option<PathBuf>) -> Router {
    Router::new()
        .route("/ping", get(ping))
        .fallback(serve_static)
        .with_state(static_dir.map(Arc::new))
}

/// Routes of the monitor listener.
pub fn monitor_router(loggers: Arc<Loggers>) -> Router {
    let log_level = get(get_log_level).put(put_log_level);
    Router::new()
        .route("/service/log-level", log_level.clone())
        .route("/service/log-level/", log_level.clone())
        .route("/service/log-level/{level}", log_level.clone())
        .route("/service/log-level/{level}/", log_level)
        .route("/service/on-log-rotate", post(on_log_rotate))
        .route("/service/on-log-rotate/", post(on_log_rotate))
        .with_state(loggers)
}

async fn ping() -> StatusCode {
    StatusCode::OK
}

fn not_found() -> Response {
    (StatusCode::NOT_FOUND, "File not found").into_response()
}

/// Maps a URL path to a file inside `root`, refusing anything that could escape it.
fn resolve_static_path(root: &Path, url_path: &str) -> Option<PathBuf> {
    let relative = Path::new(url_path.trim_start_matches('/'));
    if relative.as_os_str().is_empty() || !relative.components().all(|c| matches!(c, Component::Normal(_))) {
        return None;
    }
    Some(root.join(relative))
}

async fn serve_static(State(root): State<Option<Arc<PathBuf>>>, method: Method, uri: Uri) -> Response {
    if method != Method::GET {
        return not_found();
    }
    let Some(path) = root.and_then(|root| resolve_static_path(&root, uri.path())) else {
        return not_found();
    };
    match tokio::fs::read(&path).await {
        Ok(content) => {
            let mime = mime_guess::from_path(&path).first_or_octet_stream();
            ([(header::CONTENT_TYPE, mime.as_ref())], Body::from(content)).into_response()
        }
        Err(_) => not_found(),
    }
}

#[derive(Deserialize)]
struct LoggerQuery {
    logger: Option<String>,
}

fn find_logger(loggers: &Loggers, query: &LoggerQuery) -> Option<Arc<Logger>> {
    match query.logger.as_deref() {
        None | Some("") => Some(loggers.default_logger()),
        Some(name) => loggers.get(name),
    }
}

fn levels_response(init: Level, current: Level) -> Response {
    let body = json!({"init-log-level": init.as_str(), "current-log-level": current.as_str()});
    format!("{body}\n").into_response()
}

async fn get_log_level(
    State(loggers): State<Arc<Loggers>>,
    level: Option<UrlPath<String>>,
    Query(query): Query<LoggerQuery>,
) -> Response {
    if let Some(UrlPath(level)) = level {
        let message = format!("unexpected parameter for log level getting: {level}");
        return (StatusCode::BAD_REQUEST, message).into_response();
    }
    let Some(logger) = find_logger(&loggers, &query) else {
        return StatusCode::INTERNAL_SERVER_ERROR.into_response();
    };
    levels_response(logger.init_level(), logger.level())
}

async fn put_log_level(
    State(loggers): State<Arc<Loggers>>,
    level: Option<UrlPath<String>>,
    Query(query): Query<LoggerQuery>,
) -> Response {
    let Some(logger) = find_logger(&loggers, &query) else {
        return StatusCode::INTERNAL_SERVER_ERROR.into_response();
    };
    let level = match level.as_ref().map(|UrlPath(level)| level.as_str()) {
        Some("reset") => logger.init_level(),
        Some(level) => match level.parse() {
            Ok(level) => level,
            Err(_) => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
        },
        None => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    };
    logger.set_level(level);
    levels_response(logger.init_level(), level)
}

async fn on_log_rotate(State(loggers): State<Arc<Loggers>>) -> StatusCode {
    match tokio::task::spawn_blocking(move || loggers.reopen_all()).await {
        Ok(()) => StatusCode::OK,
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_path_stays_inside_root() {
        let root = Path::new("/srv/static");
        assert_eq!(resolve_static_path(root, "/a/b.jpg"), Some(PathBuf::from("/srv/static/a/b.jpg")));
        assert_eq!(resolve_static_path(root, "/../etc/passwd"), None);
        assert_eq!(resolve_static_path(root, "/a/../../etc/passwd"), None);
        assert_eq!(resolve_static_path(root, "/"), None);
    }
}
