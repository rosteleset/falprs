//! Building blocks shared by the LPRS and FRS HTTP APIs: JSON responses shaped like userver's,
//! error answers and request member validation.

use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde_json::{Map, Value, json};

use crate::config_schema::{Invalid, Schema, Validated};
use crate::log;
use crate::logging::{Level, Logger};

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

/// The first `max_chars` characters of `value` as compact JSON, serialising no more of it than
/// needed: a request body may hold a Base64 image of megabytes.
pub fn clipped_json(value: &Value, max_chars: usize) -> String {
    /// Collects bytes until it has enough for `max_chars` characters of any width, then stops the
    /// serialisation with an error.
    struct Clipped {
        bytes: Vec<u8>,
        limit: usize,
    }
    impl std::io::Write for Clipped {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            let room = self.limit - self.bytes.len();
            self.bytes.extend_from_slice(&buf[..buf.len().min(room)]);
            if self.bytes.len() == self.limit {
                return Err(std::io::Error::other("clipped"));
            }
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut clipped = Clipped {
        bytes: Vec::new(),
        limit: max_chars * 4,
    };
    // An error means only that the output was clipped
    let _ = serde_json::to_writer(&mut clipped, value);
    // A character cut at the end lies beyond the first `max_chars`
    String::from_utf8_lossy(&clipped.bytes).chars().take(max_chars).collect()
}

/// Validates a config given through the API: an invalid value is answered with 400 naming the
/// member (`prefix` and the key, e.g. `config.blur`); unknown keys are dropped with a warning.
pub fn validate_config(schema: &Schema, config: &Map<String, Value>, prefix: &str, logger: &Logger) -> Result<Validated, ApiError> {
    let validated = schema
        .validate(config)
        .map_err(|(key, Invalid(expected))| ApiError::bad_request(format!("Member `{prefix}{key}` must be {expected}.")))?;
    if !validated.unknown.is_empty() {
        log!(
            logger,
            Level::Warning,
            "unknown keys of the {} are dropped: {}",
            schema.name,
            validated.unknown.join(", ")
        );
    }
    Ok(validated)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clipped_json_is_the_start_of_the_whole() {
        let values = [
            json!({"url": "data:image/jpeg;base64,".to_string() + &"A".repeat(100_000), "streamId": 7}),
            json!({"name": "\u{0436}\u{1F600}".repeat(500)}),
            json!({"a": 1}),
            Value::Null,
            json!(""),
        ];
        for value in &values {
            let whole: String = value.to_string().chars().take(300).collect();
            assert_eq!(clipped_json(value, 300), whole);
        }
    }
}
