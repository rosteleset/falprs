//! Lenient conversions of JSON config values, matching the C++ service: a missing, null or
//! unconvertible value yields the default, and numbers may also be given as strings.

use std::time::Duration;

use chrono::{DateTime, Timelike, Utc};
use serde_json::Value;

/// Formats a time like userver's JSON serialization of time points: RFC 3339 in UTC with only the
/// significant digits of the fractional seconds (`%Y-%m-%dT%H:%M:%E*S%Ez`).
pub fn format_time(time: DateTime<Utc>) -> String {
    let seconds = time.format("%Y-%m-%dT%H:%M:%S");
    match time.nanosecond() % 1_000_000_000 {
        0 => format!("{seconds}+00:00"),
        nanos => {
            let fraction = format!("{nanos:09}");
            format!("{seconds}.{}+00:00", fraction.trim_end_matches('0'))
        }
    }
}

/// Parses userver durations: a non-negative integer with an optional `ms`, `s`, `m`, `h` or `d`
/// suffix; no suffix means seconds.
pub fn parse_duration(text: &str) -> Option<Duration> {
    let digits = text.find(|c: char| !c.is_ascii_digit()).unwrap_or(text.len());
    let value: u64 = text[..digits].parse().ok()?;
    let seconds = |multiplier: u64| value.checked_mul(multiplier).map(Duration::from_secs);
    match &text[digits..] {
        "" | "s" => Some(Duration::from_secs(value)),
        "ms" => Some(Duration::from_millis(value)),
        "m" => seconds(60),
        "h" => seconds(60 * 60),
        "d" => seconds(24 * 60 * 60),
        _ => None,
    }
}

pub fn to_duration(value: Option<&Value>, default: Duration) -> Duration {
    match value {
        Some(Value::String(text)) => parse_duration(text).unwrap_or(default),
        _ => default,
    }
}

pub fn to_string(value: Option<&Value>, default: &str) -> String {
    match value {
        None | Some(Value::Null) => default.to_string(),
        Some(Value::String(text)) => text.clone(),
        Some(Value::Number(number)) if number.is_f64() => format!("{:.6}", number.as_f64().unwrap_or_default()),
        Some(Value::Number(number)) => number.to_string(),
        Some(_) => String::new(),
    }
}

pub fn to_f32(value: Option<&Value>, default: f32) -> f32 {
    match value {
        Some(Value::Number(number)) => number.as_f64().map_or(default, |n| n as f32),
        Some(Value::String(text)) => text.trim().parse().unwrap_or(default),
        _ => default,
    }
}

/// Booleans also come as numbers (non-zero is true) or as `true`/`false`, `yes`/`no`, `1`/`0` strings.
pub fn to_bool(value: Option<&Value>, default: bool) -> bool {
    match value {
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(number)) => number.as_f64().map_or(default, |n| n != 0.0),
        Some(Value::String(text)) => match text.to_ascii_lowercase().as_str() {
            "true" | "t" | "yes" | "y" | "1" => true,
            "false" | "f" | "no" | "n" | "0" => false,
            _ => default,
        },
        _ => default,
    }
}

pub fn to_i64(value: Option<&Value>, default: i64) -> i64 {
    match value {
        Some(Value::Number(number)) => number
            .as_i64()
            .or_else(|| number.as_f64().map(|n| n as i64))
            .unwrap_or(default),
        Some(Value::String(text)) => text.trim().parse().unwrap_or(default),
        _ => default,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn durations() {
        assert_eq!(parse_duration("5"), Some(Duration::from_secs(5)));
        assert_eq!(parse_duration("5s"), Some(Duration::from_secs(5)));
        assert_eq!(parse_duration("300ms"), Some(Duration::from_millis(300)));
        assert_eq!(parse_duration("2m"), Some(Duration::from_secs(120)));
        assert_eq!(parse_duration("12h"), Some(Duration::from_secs(12 * 3600)));
        assert_eq!(parse_duration("2d"), Some(Duration::from_secs(2 * 86400)));
        assert_eq!(parse_duration("-1s"), None);
        assert_eq!(parse_duration("1.5s"), None);
        assert_eq!(parse_duration("1w"), None);
        assert_eq!(to_duration(Some(&json!(5)), Duration::from_secs(1)), Duration::from_secs(1));
    }

    #[test]
    fn times() {
        let time = |text: &str| DateTime::parse_from_rfc3339(text).unwrap().with_timezone(&Utc);
        assert_eq!(format_time(time("2024-03-05T10:20:30Z")), "2024-03-05T10:20:30+00:00");
        assert_eq!(format_time(time("2024-03-05T13:20:30.120+03:00")), "2024-03-05T10:20:30.12+00:00");
        assert_eq!(format_time(time("2024-03-05T10:20:30.123456789Z")), "2024-03-05T10:20:30.123456789+00:00");
    }

    #[test]
    fn strings() {
        assert_eq!(to_string(Some(&json!("abc")), ""), "abc");
        assert_eq!(to_string(Some(&json!(12345)), ""), "12345");
        assert_eq!(to_string(Some(&json!(1.5)), ""), "1.500000");
        assert_eq!(to_string(None, "default"), "default");
        assert_eq!(to_string(Some(&json!([1])), "default"), "");
    }

    #[test]
    fn integers() {
        assert_eq!(to_i64(Some(&json!(25)), 0), 25);
        assert_eq!(to_i64(Some(&json!("25")), 0), 25);
        assert_eq!(to_i64(Some(&json!("x")), 7), 7);
    }
}
