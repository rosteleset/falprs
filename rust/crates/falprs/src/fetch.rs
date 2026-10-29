//! Acquiring frames: `data:` URLs with Base64 content or HTTP(S) downloads.

use std::collections::HashMap;
use std::time::Duration;

use base64::Engine;
use base64::engine::general_purpose::{GeneralPurpose, GeneralPurposeConfig};
use base64::engine::{DecodePaddingMode, general_purpose};
use bytes::Bytes;
use md5::{Digest, Md5};
use sha2::Sha256;

#[derive(Debug, thiserror::Error)]
pub enum FetchError {
    #[error("invalid Base64 image")]
    Base64,
    #[error("can't fetch image: {0}")]
    Http(String),
}

const BASE64: GeneralPurpose = GeneralPurpose::new(
    &base64::alphabet::STANDARD,
    GeneralPurposeConfig::new().with_decode_padding_mode(DecodePaddingMode::Indifferent),
);

/// Decodes Base64 content, with or without padding.
pub fn decode_base64(content: &str) -> Option<Vec<u8>> {
    BASE64
        .decode(content)
        .or_else(|_| general_purpose::STANDARD.decode(content.trim()))
        .ok()
}

/// Decodes `data:<type>;base64,<content>`.
fn decode_data_url(url: &str) -> Result<Vec<u8>, FetchError> {
    let content = url.find(";base64,").map(|pos| &url[pos + ";base64,".len()..]);
    let data = match content {
        Some(content) => BASE64
            .decode(content)
            .or_else(|_| general_purpose::STANDARD.decode(content.trim()))
            .map_err(|_| FetchError::Base64)?,
        None => Vec::new(),
    };
    if data.is_empty() {
        return Err(FetchError::Base64);
    }
    Ok(data)
}

/// Splits `scheme://user:password@host/...` into the URL without credentials and the credentials,
/// taking them verbatim like the C++ service does.
fn split_credentials(url: &str) -> (String, Option<(String, String)>) {
    let (Some(at), Some(scheme_end)) = (url.find('@'), url.find("://")) else {
        return (url.to_string(), None);
    };
    if scheme_end > at {
        return (url.to_string(), None);
    }
    let authority = scheme_end + 3;
    let userinfo = &url[authority..at];
    let (user, password) = userinfo.split_once(':').unwrap_or((userinfo, ""));
    let stripped = format!("{}{}", &url[..authority], &url[at + 1..]);
    (stripped, Some((user.to_string(), password.to_string())))
}

/// Parameters of a `WWW-Authenticate: Digest ...` challenge.
fn parse_digest_challenge(header: &str) -> Option<HashMap<String, String>> {
    let rest = header.trim_start();
    if rest.len() < 7 || !rest[..6].eq_ignore_ascii_case("digest") {
        return None;
    }
    let mut params = HashMap::new();
    let mut chars = rest[6..].chars().peekable();
    loop {
        while chars.peek().is_some_and(|c| *c == ',' || c.is_whitespace()) {
            chars.next();
        }
        let key: String = chars.by_ref().take_while(|c| *c != '=').collect::<String>().trim().to_ascii_lowercase();
        if key.is_empty() {
            break;
        }
        let value: String = if chars.peek() == Some(&'"') {
            chars.next();
            let mut value = String::new();
            while let Some(c) = chars.next() {
                match c {
                    '\\' => value.extend(chars.next()),
                    '"' => break,
                    _ => value.push(c),
                }
            }
            value
        } else {
            chars.by_ref().take_while(|c| *c != ',').collect::<String>().trim().to_string()
        };
        params.insert(key, value);
    }
    Some(params)
}

fn hash(algorithm: &str, data: &str) -> String {
    if algorithm.to_ascii_uppercase().starts_with("SHA-256") {
        hex::encode(Sha256::digest(data))
    } else {
        hex::encode(Md5::digest(data))
    }
}

/// The `Authorization` header answering a Digest challenge (RFC 7616, `qop=auth` when offered).
fn digest_authorization(challenge: &HashMap<String, String>, user: &str, password: &str, uri: &str) -> String {
    let get = |key: &str| challenge.get(key).map(String::as_str).unwrap_or_default();
    let (realm, nonce) = (get("realm"), get("nonce"));
    let algorithm = challenge.get("algorithm").map_or("MD5", String::as_str);
    let cnonce = uuid::Uuid::new_v4().simple().to_string();
    let mut ha1 = hash(algorithm, &format!("{user}:{realm}:{password}"));
    if algorithm.to_ascii_lowercase().ends_with("-sess") {
        ha1 = hash(algorithm, &format!("{ha1}:{nonce}:{cnonce}"));
    }
    let ha2 = hash(algorithm, &format!("GET:{uri}"));
    let use_qop = get("qop").split(',').any(|qop| qop.trim() == "auth");
    let response = if use_qop {
        hash(algorithm, &format!("{ha1}:{nonce}:00000001:{cnonce}:auth:{ha2}"))
    } else {
        hash(algorithm, &format!("{ha1}:{nonce}:{ha2}"))
    };
    let mut header = format!(
        r#"Digest username="{user}", realm="{realm}", nonce="{nonce}", uri="{uri}", algorithm={algorithm}, response="{response}""#
    );
    if use_qop {
        header += &format!(r#", qop=auth, nc=00000001, cnonce="{cnonce}""#);
    }
    if let Some(opaque) = challenge.get("opaque") {
        header += &format!(r#", opaque="{opaque}""#);
    }
    header
}

/// One GET; with credentials, a `401` carrying a Digest challenge is answered once. Like libcurl
/// with `CURLAUTH_ANYSAFE`, credentials are never sent as Basic authentication.
async fn get(client: &reqwest::Client, url: &str, credentials: Option<&(String, String)>, timeout: Duration) -> reqwest::Result<reqwest::Response> {
    let response = client.get(url).timeout(timeout).send().await?;
    let Some((user, password)) = credentials else {
        return Ok(response);
    };
    if response.status() != reqwest::StatusCode::UNAUTHORIZED {
        return Ok(response);
    }
    let challenge = response
        .headers()
        .get_all(reqwest::header::WWW_AUTHENTICATE)
        .iter()
        .find_map(|value| parse_digest_challenge(value.to_str().ok()?));
    let Some(challenge) = challenge else {
        return Ok(response);
    };
    let uri = match response.url().query() {
        Some(query) => format!("{}?{query}", response.url().path()),
        None => response.url().path().to_string(),
    };
    let authorization = digest_authorization(&challenge, user, password, &uri);
    client
        .get(url)
        .timeout(timeout)
        .header(reqwest::header::AUTHORIZATION, authorization)
        .send()
        .await
}

/// Gets an image from `url`. HTTP requests are made up to `attempts` times, retrying on network
/// errors and server errors; anything but a non-empty `200 OK` body is a failure.
pub async fn fetch_image(client: &reqwest::Client, url: &str, attempts: u32, timeout: Duration) -> Result<Bytes, FetchError> {
    if url.starts_with("data:") {
        return decode_data_url(url).map(Bytes::from);
    }
    let (url, credentials) = split_credentials(url);
    let mut last_error = String::new();
    for _ in 0..attempts.max(1) {
        match get(client, &url, credentials.as_ref(), timeout).await {
            Ok(response) if response.status() == reqwest::StatusCode::OK => {
                let body = response.bytes().await.map_err(|e| FetchError::Http(e.to_string()))?;
                if body.is_empty() {
                    return Err(FetchError::Http("empty body".to_string()));
                }
                return Ok(body);
            }
            Ok(response) if response.status().is_server_error() => last_error = format!("status {}", response.status()),
            Ok(response) => return Err(FetchError::Http(format!("status {}", response.status()))),
            Err(e) => last_error = e.to_string(),
        }
    }
    Err(FetchError::Http(last_error))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn credentials() {
        assert_eq!(split_credentials("http://cam/shot.jpg"), ("http://cam/shot.jpg".to_string(), None));
        let (url, credentials) = split_credentials("http://admin:secret@cam:8080/shot.jpg");
        assert_eq!(url, "http://cam:8080/shot.jpg");
        assert_eq!(credentials, Some(("admin".to_string(), "secret".to_string())));
        let (_, credentials) = split_credentials("http://admin@cam/shot.jpg");
        assert_eq!(credentials, Some(("admin".to_string(), String::new())));
    }

    #[test]
    fn digest_rfc2617_example() {
        // RFC 2617, section 3.5, with the client nonce of the example
        let challenge = parse_digest_challenge(
            r#"Digest realm="testrealm@host.com", qop="auth,auth-int", nonce="dcd98b7102dd2f0e8b11d0f600bfb0c093", opaque="5ccc069c403ebaf9f0171e9517f40e41""#,
        )
        .unwrap();
        assert_eq!(challenge["realm"], "testrealm@host.com");
        assert_eq!(challenge["qop"], "auth,auth-int");
        let ha1 = hash("MD5", "Mufasa:testrealm@host.com:Circle Of Life");
        let ha2 = hash("MD5", "GET:/dir/index.html");
        let response = hash("MD5", &format!("{ha1}:dcd98b7102dd2f0e8b11d0f600bfb0c093:00000001:0a4f113b:auth:{ha2}"));
        assert_eq!(response, "6629fae49393a05397450978507c4ef1");
        let header = digest_authorization(&challenge, "Mufasa", "Circle Of Life", "/dir/index.html");
        assert!(header.contains(r#"uri="/dir/index.html""#) && header.contains("qop=auth, nc=00000001"));
        assert!(header.ends_with(r#"opaque="5ccc069c403ebaf9f0171e9517f40e41""#));
    }

    #[test]
    fn data_urls() {
        assert_eq!(decode_data_url("data:image/jpeg;base64,AQID").unwrap(), vec![1, 2, 3]);
        assert_eq!(decode_data_url("data:image/jpeg;base64,AQI").unwrap(), vec![1, 2]);
        assert!(matches!(decode_data_url("data:image/jpeg;base64,!!!"), Err(FetchError::Base64)));
        assert!(matches!(decode_data_url("data:image/jpeg,AQID"), Err(FetchError::Base64)));
    }
}
