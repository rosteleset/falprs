//! Building blocks shared by the LPRS and FRS HTTP APIs: JSON responses shaped like userver's,
//! error answers and request member validation.

use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};

/// An error answered as `{"code": "<status>", "message": "<message>"}`.
#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    pub fn unauthorized() -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            message: "Unauthorized".to_string(),
        }
    }

    pub fn server_error() -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: "Internal server error".to_string(),
        }
    }
}

/// Any database failure is reported to the client as an internal server error.
impl From<pg_cluster::Error> for ApiError {
    fn from(_: pg_cluster::Error) -> Self {
        Self::server_error()
    }
}

impl From<pg_cluster::tokio_postgres::Error> for ApiError {
    fn from(_: pg_cluster::tokio_postgres::Error) -> Self {
        Self::server_error()
    }
}

pub fn json_response(status: StatusCode, body: Value) -> Response {
    let content_type = HeaderValue::from_static("application/json; charset=utf-8");
    (status, [(header::CONTENT_TYPE, content_type)], body.to_string()).into_response()
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        json_response(self.status, json!({"code": self.status.as_u16().to_string(), "message": self.message}))
    }
}

pub fn no_content() -> Response {
    let content_type = HeaderValue::from_static("application/json; charset=utf-8");
    (StatusCode::NO_CONTENT, [(header::CONTENT_TYPE, content_type)]).into_response()
}

/// Wraps method data the way the C++ service does: empty data means "204 No Content".
pub fn data_response(data: Value) -> Response {
    let is_empty = match &data {
        Value::Null => true,
        Value::Array(items) => items.is_empty(),
        Value::Object(members) => members.is_empty(),
        _ => false,
    };
    if is_empty {
        return no_content();
    }
    json_response(StatusCode::OK, json!({"code": "200", "message": "Ok", "data": data}))
}

pub type ApiResult = Result<Response, ApiError>;

/// The member must be present and be a non-empty scalar.
pub fn require_member<'a>(body: &'a Value, member: &str) -> Result<&'a Value, ApiError> {
    let value = body
        .get(member)
        .ok_or_else(|| ApiError::bad_request(format!("Required member `{member}` not found.")))?;
    let problem = match value {
        Value::Null => Some("must not be null"),
        Value::Array(_) => Some("must not be an array"),
        Value::Object(_) => Some("must not be an object"),
        Value::String(text) if text.is_empty() => Some("must not be empty"),
        _ => None,
    };
    match problem {
        Some(problem) => Err(ApiError::bad_request(format!("Member `{member}` {problem}."))),
        None => Ok(value),
    }
}
