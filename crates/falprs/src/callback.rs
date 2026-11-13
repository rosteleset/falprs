//! Event callbacks of FRS and LPRS: a JSON body posted to the URL configured for a video stream.

use std::time::Duration;

use serde_json::Value;

/// Posts `body` as JSON to `url`. `Ok(true)` when the receiver answers 200 or 204, `Ok(false)` for
/// any other status, and the transport error otherwise; callers log both failures in their own words.
pub async fn post_json(http: &reqwest::Client, url: &str, body: &Value, timeout: Duration) -> reqwest::Result<bool> {
    let response = http
        .post(url)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(body.to_string())
        .timeout(timeout)
        .send()
        .await?;
    Ok(matches!(response.status().as_u16(), 200 | 204))
}
