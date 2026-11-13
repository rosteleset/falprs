//! Schemas of the configs the external backend sets through the API: video stream configs,
//! default video stream configs and the FRS common config.
//!
//! A schema knows every key of its kind of config, the form and range of its value and its
//! built-in default. It is used to validate and normalize values before they are stored, to
//! read stored values, and to plan the cleanup of stored configs (`falprs check-configs`), for
//! which it also knows how the C++ service's lenient readers took each value.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{Map, Number, Value, json};

use crate::converters::parse_duration;
use crate::log;
use crate::logging::{Level, Logger};

/// The form of a value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Kind {
    /// A number in `[min, max]`, also given as a numeric string.
    Float {
        min: f64,
        max: f64,
    },
    /// An integer in `[min, max]`, also given as a numeric string or an integral float.
    Int {
        min: i64,
        max: i64,
    },
    Bool,
    Str,
    /// A duration string such as `"30s"`, or a number of seconds; stored as a string.
    Duration,
    /// One of the log level names.
    Level,
    /// FRS: an empty array or 4 numbers in percent of the frame (left, top, width, height).
    FrsWorkArea,
    /// LPRS: an array of polygons of at least 3 `[x, y]` points in percent of the frame.
    LprsWorkArea,
}

/// How the C++ service (and the Rust port before the schemas) read a value of a key.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Legacy {
    /// `convertTo*`: numbers also as strings, booleans also as numbers and words, strings also as numbers.
    Lenient,
    /// userver's typed `As<T>(default)`: only the expected JSON type.
    Typed,
}

#[derive(Debug, Clone)]
pub struct KeySpec {
    /// Read by the test comparing the schemas with the OpenAPI spec.
    #[cfg_attr(not(test), allow(dead_code))]
    pub key: String,
    pub kind: Kind,
    /// The built-in default, in its stored form.
    pub default: Value,
    pub legacy: Legacy,
}

/// Why a value is rejected: the expected form, for the error message.
#[derive(Debug, Clone, PartialEq)]
pub struct Invalid(pub String);

/// What the cleanup does with a stored value.
#[derive(Debug, Clone, PartialEq)]
pub enum Cleanup {
    /// The value is stored in its canonical form already.
    Keep,
    /// The value worked in another form; it is replaced with its canonical form.
    Normalize(Value),
    /// The value had no effect (unknown key, ignored or out-of-range value) and is removed.
    Remove(String),
}

#[derive(Debug)]
pub struct Schema {
    /// The kind of config, for messages, e.g. "FRS video stream config".
    pub name: &'static str,
    keys: Vec<KeySpec>,
    index: HashMap<String, usize>,
}

/// The result of validating a config given through the API.
#[derive(Debug, Default, PartialEq)]
pub struct Validated {
    /// The known keys with their normalized values.
    pub values: Map<String, Value>,
    /// Keys given with `null`: to be removed.
    pub nulls: Vec<String>,
    /// Unknown keys, dropped.
    pub unknown: Vec<String>,
}

fn number(value: f64) -> Value {
    Number::from_f64(value).map_or(Value::Null, Value::Number)
}

/// A number of seconds as a duration string, as the `params` form of addStream writes it.
pub fn seconds_to_duration(seconds: f64) -> Option<String> {
    if !seconds.is_finite() || seconds < 0.0 {
        return None;
    }
    let ms = (seconds * 1000.0).round();
    if ms > u64::MAX as f64 / 2.0 {
        return None;
    }
    let ms = ms as u64;
    Some(if ms.is_multiple_of(1000) {
        format!("{}s", ms / 1000)
    } else {
        format!("{ms}ms")
    })
}

fn as_float(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64(),
        Value::String(text) => text.parse::<f64>().ok().filter(|n| n.is_finite()),
        _ => None,
    }
}

fn percent(value: &Value) -> Option<f64> {
    value.as_f64().filter(|n| (0.0..=100.0).contains(n))
}

impl Kind {
    /// The canonical form of a value, or why it is rejected.
    pub fn normalize(self, value: &Value) -> Result<Value, Invalid> {
        let invalid = |expected: &str| Err(Invalid(expected.to_string()));
        match self {
            Kind::Float { min, max } => match as_float(value) {
                // A JSON number is kept as given: 300 stays 300, not 300.0
                Some(n) if (min..=max).contains(&n) => Ok(if value.is_number() { value.clone() } else { number(n) }),
                _ => invalid(&format!("a number within [{min}, {max}]")),
            },
            Kind::Int { min, max } => {
                let expected = if max == i64::MAX {
                    format!("an integer of at least {min}")
                } else {
                    format!("an integer within [{min}, {max}]")
                };
                match as_float(value) {
                    Some(n) if n.fract() == 0.0 && (min as f64..=max as f64).contains(&n) => Ok(json!(n as i64)),
                    _ => Err(Invalid(expected)),
                }
            }
            Kind::Bool => match value {
                Value::Bool(_) => Ok(value.clone()),
                _ => invalid("a boolean"),
            },
            Kind::Str => match value {
                Value::String(_) => Ok(value.clone()),
                _ => invalid("a string"),
            },
            Kind::Duration => {
                let expected = "a duration such as \"30s\" or a number of seconds";
                match value {
                    Value::String(text) if parse_duration(text).is_some() => Ok(value.clone()),
                    Value::Number(n) => n
                        .as_f64()
                        .and_then(seconds_to_duration)
                        .map(Value::String)
                        .ok_or_else(|| Invalid(expected.into())),
                    _ => invalid(expected),
                }
            }
            Kind::Level => match value.as_str().map(str::parse::<Level>) {
                Some(Ok(_)) => Ok(value.clone()),
                _ => invalid("one of the log levels trace, debug, info, warning, error, critical, none"),
            },
            Kind::FrsWorkArea => match value.as_array() {
                Some(items) if items.is_empty() || (items.len() == 4 && items.iter().all(|item| percent(item).is_some())) => Ok(value.clone()),
                _ => invalid("an empty array or 4 numbers within [0, 100]: left, top, width, height"),
            },
            Kind::LprsWorkArea => {
                let point = |point: &Value| point.as_array().is_some_and(|xy| xy.len() == 2 && xy.iter().all(|c| percent(c).is_some()));
                let polygon = |polygon: &Value| polygon.as_array().is_some_and(|points| points.len() >= 3 && points.iter().all(point));
                match value.as_array() {
                    Some(polygons) if polygons.iter().all(polygon) => Ok(value.clone()),
                    _ => invalid("an array of polygons of at least 3 [x, y] points within [0, 100]"),
                }
            }
        }
    }

    /// The value the cleanup keeps of a stored `value`, in canonical form but not checked against
    /// the range, or None if it is removed: what the old readers took, and numbers given as
    /// strings also where the typed readers ignored them, as the API accepts those now.
    fn legacy(self, legacy: Legacy, value: &Value) -> Option<Value> {
        let lenient = legacy == Legacy::Lenient;
        match self {
            Kind::Float { .. } => match value {
                Value::Number(_) => Some(value.clone()),
                Value::String(text) => text.trim().parse::<f64>().ok().filter(|n| n.is_finite()).map(number),
                _ => None,
            },
            Kind::Int { .. } => match value {
                // The lenient readers truncated a fractional number
                Value::Number(n) if lenient => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)).map(|n| json!(n)),
                Value::String(text) if lenient => text.trim().parse::<i64>().ok().map(|n| json!(n)),
                _ => as_float(value).filter(|n| n.fract() == 0.0).map(|n| json!(n as i64)),
            },
            Kind::Bool => match value {
                Value::Bool(_) => Some(value.clone()),
                Value::Number(n) if lenient => n.as_f64().map(|n| json!(n != 0.0)),
                Value::String(text) if lenient => match text.to_ascii_lowercase().as_str() {
                    "true" | "t" | "yes" | "y" | "1" => Some(json!(true)),
                    "false" | "f" | "no" | "n" | "0" => Some(json!(false)),
                    _ => None,
                },
                _ => None,
            },
            Kind::Str => match value {
                Value::String(_) => Some(value.clone()),
                Value::Number(n) if lenient && n.is_f64() => Some(json!(format!("{:.6}", n.as_f64().unwrap_or_default()))),
                Value::Number(n) if lenient => Some(json!(n.to_string())),
                _ => None,
            },
            // Both readers took duration strings only
            Kind::Duration => match value {
                Value::String(text) if parse_duration(text).is_some() => Some(value.clone()),
                _ => None,
            },
            Kind::Level => self.normalize(value).ok(),
            // Any array of numbers was taken, but only 4 of them were used
            Kind::FrsWorkArea => match value.as_array() {
                Some(items) if items.iter().all(Value::is_number) => Some(value.clone()),
                _ => None,
            },
            Kind::LprsWorkArea => {
                let point = |point: &Value| point.as_array().is_some_and(|xy| xy.len() >= 2 && xy[..2].iter().all(Value::is_number));
                let polygon = |polygon: &Value| polygon.as_array().is_some_and(|points| points.iter().all(point));
                match value.as_array() {
                    Some(polygons) if polygons.iter().all(polygon) => Some(value.clone()),
                    _ => None,
                }
            }
        }
    }
}

/// Builds a schema key by key.
pub struct SchemaBuilder {
    schema: Schema,
    legacy: Legacy,
}

impl SchemaBuilder {
    pub fn new(name: &'static str, legacy: Legacy) -> Self {
        Self {
            schema: Schema {
                name,
                keys: Vec::new(),
                index: HashMap::new(),
            },
            legacy,
        }
    }

    pub fn key(mut self, key: impl Into<String>, kind: Kind, default: Value) -> Self {
        let key = key.into();
        debug_assert!(kind.normalize(&default).is_ok(), "the default of {key} is invalid");
        self.schema.index.insert(key.clone(), self.schema.keys.len());
        self.schema.keys.push(KeySpec {
            key,
            kind,
            default,
            legacy: self.legacy,
        });
        self
    }

    /// A number within [0, 1]: a confidence, score, ratio or IoU threshold.
    pub fn fraction(self, key: impl Into<String>, default: f64) -> Self {
        self.key(key, Kind::Float { min: 0.0, max: 1.0 }, number(default))
    }

    pub fn non_negative(self, key: impl Into<String>, default: f64) -> Self {
        self.key(key, Kind::Float { min: 0.0, max: f64::MAX }, number(default))
    }

    pub fn int(self, key: impl Into<String>, min: i64, default: i64) -> Self {
        self.key(key, Kind::Int { min, max: i64::from(i32::MAX) }, json!(default))
    }

    pub fn bool(self, key: impl Into<String>, default: bool) -> Self {
        self.key(key, Kind::Bool, json!(default))
    }

    pub fn string(self, key: impl Into<String>, default: &str) -> Self {
        self.key(key, Kind::Str, json!(default))
    }

    pub fn duration(self, key: impl Into<String>, default: &str) -> Self {
        self.key(key, Kind::Duration, json!(default))
    }

    pub fn build(self) -> Schema {
        self.schema
    }
}

impl Schema {
    #[cfg(test)]
    pub fn keys(&self) -> &[KeySpec] {
        &self.keys
    }

    pub fn spec(&self, key: &str) -> Option<&KeySpec> {
        self.index.get(key).map(|&i| &self.keys[i])
    }

    fn known(&self, key: &str) -> &KeySpec {
        self.spec(key).unwrap_or_else(|| panic!("{} has no key {key}", self.name))
    }

    /// The value of a known key of a stored config: the stored one if valid, else the default.
    pub fn value(&self, config: &Value, key: &str) -> Value {
        let spec = self.known(key);
        match config.get(key) {
            None | Some(Value::Null) => spec.default.clone(),
            Some(value) => spec.kind.normalize(value).unwrap_or_else(|_| spec.default.clone()),
        }
    }

    pub fn f32(&self, config: &Value, key: &str) -> f32 {
        self.value(config, key).as_f64().unwrap_or_default() as f32
    }

    pub fn i64(&self, config: &Value, key: &str) -> i64 {
        self.value(config, key).as_i64().unwrap_or_default()
    }

    pub fn bool(&self, config: &Value, key: &str) -> bool {
        self.value(config, key).as_bool().unwrap_or_default()
    }

    pub fn string(&self, config: &Value, key: &str) -> String {
        match self.value(config, key) {
            Value::String(text) => text,
            _ => String::new(),
        }
    }

    pub fn duration(&self, config: &Value, key: &str) -> Duration {
        self.value(config, key).as_str().and_then(parse_duration).unwrap_or_default()
    }

    pub fn level(&self, config: &Value, key: &str) -> Level {
        self.value(config, key).as_str().and_then(|text| text.parse().ok()).unwrap_or(Level::Info)
    }

    /// Validates a config given through the API: an invalid value of a known key is an error
    /// naming the key; unknown keys and `null` values are reported apart.
    pub fn validate(&self, config: &Map<String, Value>) -> Result<Validated, (String, Invalid)> {
        let mut result = Validated::default();
        for (key, value) in config {
            match self.spec(key) {
                None => result.unknown.push(key.clone()),
                Some(_) if value.is_null() => result.nulls.push(key.clone()),
                Some(spec) => {
                    let normalized = spec.kind.normalize(value).map_err(|invalid| (key.clone(), invalid))?;
                    result.values.insert(key.clone(), normalized);
                }
            }
        }
        Ok(result)
    }

    /// The keys of a stored config the service can't use, with why: unknown, or an invalid value.
    pub fn problems(&self, config: &Value) -> Vec<(String, String)> {
        let Some(config) = config.as_object() else {
            return Vec::new();
        };
        config
            .iter()
            .filter_map(|(key, value)| match self.spec(key) {
                None => Some((key.clone(), format!("unknown key of the {}", self.name))),
                Some(_) if value.is_null() => None,
                Some(spec) => spec
                    .kind
                    .normalize(value)
                    .err()
                    .map(|Invalid(expected)| (key.clone(), format!("expected {expected}"))),
            })
            .collect()
    }

    /// What the cleanup does with a stored value of `key`: a value the old readers took is kept
    /// in its canonical form, so that video streams keep their behaviour; anything else is removed.
    pub fn cleanup(&self, key: &str, value: &Value) -> Cleanup {
        let Some(spec) = self.spec(key) else {
            return Cleanup::Remove(format!("unknown key of the {}", self.name));
        };
        if value.is_null() {
            return Cleanup::Remove("null".to_string());
        }
        let Some(taken) = spec.kind.legacy(spec.legacy, value) else {
            return Cleanup::Remove("the value was ignored".to_string());
        };
        match spec.kind.normalize(&taken) {
            Ok(canonical) if canonical == *value => Cleanup::Keep,
            Ok(canonical) => Cleanup::Normalize(canonical),
            Err(Invalid(expected)) => Cleanup::Remove(format!("expected {expected}")),
        }
    }
}

/// Warns about stored configs the service can't fully use, once per row, key and value: when a
/// cache loads it first and when the value changes, not on every cache update.
pub struct StoredConfigWarnings {
    logger: Arc<Logger>,
    /// The last value warned about, by `<table> <row> <key>`.
    seen: Mutex<HashMap<String, String>>,
}

impl StoredConfigWarnings {
    pub fn new(logger: Arc<Logger>) -> Arc<Self> {
        Arc::new(Self {
            logger,
            seen: Mutex::new(HashMap::new()),
        })
    }

    /// Checks the stored config of `row` (e.g. "stream group 1") in `table` against `schema`;
    /// returns how many warnings were logged.
    pub fn check(&self, schema: &Schema, table: &str, row: &str, config: &Value) -> usize {
        let mut logged = 0;
        for (key, why) in schema.problems(config) {
            let value = config[&key].to_string();
            let id = format!("{table} {row} {key}");
            let mut seen = self.seen.lock().unwrap_or_else(|e| e.into_inner());
            if seen.get(&id) != Some(&value) {
                log!(
                    self.logger,
                    Level::Warning,
                    "{table}, {row}: `{key}` = {value} is not used ({why}); run `falprs check-configs` to clean the stored configs"
                );
                seen.insert(id, value);
                logged += 1;
            }
        }
        logged
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema() -> Schema {
        SchemaBuilder::new("test config", Legacy::Lenient)
            .fraction("confidence", 0.7)
            .non_negative("blur", 300.0)
            .int("count", 0, 3)
            .bool("flag", false)
            .string("server", "127.0.0.1:8001")
            .duration("timeout", "2s")
            .key("logs-level", Kind::Level, json!("info"))
            .key("frs-area", Kind::FrsWorkArea, json!([]))
            .key("lprs-area", Kind::LprsWorkArea, json!([]))
            .build()
    }

    fn normalize(key: &str, value: Value) -> Result<Value, Invalid> {
        schema().spec(key).unwrap().kind.normalize(&value)
    }

    #[test]
    fn accepted_forms_are_normalized() {
        assert_eq!(normalize("confidence", json!(0.7)), Ok(json!(0.7)));
        assert_eq!(normalize("confidence", json!("0.7")), Ok(json!(0.7)));
        assert_eq!(normalize("confidence", json!(1)), Ok(json!(1)));
        assert_eq!(normalize("count", json!(5)), Ok(json!(5)));
        assert_eq!(normalize("count", json!("5")), Ok(json!(5)));
        assert_eq!(normalize("count", json!(5.0)), Ok(json!(5)));
        assert_eq!(normalize("count", json!("5.0")), Ok(json!(5)));
        assert_eq!(normalize("flag", json!(true)), Ok(json!(true)));
        assert_eq!(normalize("server", json!("triton:8001")), Ok(json!("triton:8001")));
        assert_eq!(normalize("timeout", json!("30s")), Ok(json!("30s")));
        assert_eq!(normalize("timeout", json!("30")), Ok(json!("30")));
        assert_eq!(normalize("timeout", json!(30)), Ok(json!("30s")));
        assert_eq!(normalize("timeout", json!(1.5)), Ok(json!("1500ms")));
        assert_eq!(normalize("logs-level", json!("trace")), Ok(json!("trace")));
        assert_eq!(normalize("frs-area", json!([10, 20, 50.5, 60])), Ok(json!([10, 20, 50.5, 60])));
        assert_eq!(normalize("frs-area", json!([])), Ok(json!([])));
        assert_eq!(
            normalize("lprs-area", json!([[[0, 0], [100, 0], [100, 50.5]]])),
            Ok(json!([[[0, 0], [100, 0], [100, 50.5]]]))
        );
    }

    #[test]
    fn rejected_forms_and_ranges() {
        for (key, value) in [
            ("confidence", json!(1.5)),
            ("confidence", json!(-0.1)),
            ("confidence", json!("high")),
            ("confidence", json!(true)),
            ("blur", json!(-1)),
            ("count", json!(5.5)),
            ("count", json!(-1)),
            ("count", json!("five")),
            ("flag", json!("true")),
            ("flag", json!(1)),
            ("server", json!(8001)),
            ("timeout", json!("30 sec")),
            ("timeout", json!(-5)),
            ("timeout", json!(true)),
            ("logs-level", json!("INFO")),
            ("frs-area", json!([10, 20, 50])),
            ("frs-area", json!([10, 20, 50, 160])),
            ("lprs-area", json!([[[0, 0], [100, 0]]])),
            ("lprs-area", json!([[[0, 0], [100], [100, 50]]])),
        ] {
            assert!(normalize(key, value.clone()).is_err(), "{key}: {value}");
        }
    }

    #[test]
    fn validation_reports_invalid_unknown_and_null_keys() {
        let schema = schema();
        let config = json!({"confidence": "0.5", "timeout": 10, "blur": null, "typo": 1});
        let validated = schema.validate(config.as_object().unwrap()).unwrap();
        assert_eq!(validated.values, json!({"confidence": 0.5, "timeout": "10s"}).as_object().unwrap().clone());
        assert_eq!(validated.nulls, vec!["blur"]);
        assert_eq!(validated.unknown, vec!["typo"]);
        let (key, Invalid(expected)) = schema.validate(json!({"count": -1}).as_object().unwrap()).unwrap_err();
        assert_eq!(key, "count");
        assert!(expected.contains("integer"), "{expected}");
    }

    #[test]
    fn stored_values_read_with_defaults() {
        let schema = schema();
        let config = json!({"confidence": "0.5", "count": "x", "timeout": 30, "flag": true, "logs-level": "INFO"});
        assert_eq!(schema.f32(&config, "confidence"), 0.5);
        assert_eq!(schema.i64(&config, "count"), 3);
        // A number of seconds is a valid form
        assert_eq!(schema.duration(&config, "timeout"), Duration::from_secs(30));
        assert!(schema.bool(&config, "flag"));
        assert_eq!(schema.level(&config, "logs-level"), Level::Info);
        assert_eq!(schema.string(&config, "server"), "127.0.0.1:8001");
        let problems = schema.problems(&config);
        let keys: Vec<_> = problems.iter().map(|(key, _)| key.as_str()).collect();
        assert_eq!(keys, vec!["count", "logs-level"]);
        assert!(schema.problems(&json!({"typo": 1}))[0].1.contains("unknown"));
    }

    #[test]
    fn cleanup_keeps_what_the_old_readers_took() {
        let schema = schema();
        let cleanup = |key: &str, value: Value| schema.cleanup(key, &value);
        assert_eq!(cleanup("confidence", json!(0.7)), Cleanup::Keep);
        assert_eq!(cleanup("confidence", json!(" 0.7")), Cleanup::Normalize(json!(0.7)));
        assert_eq!(cleanup("count", json!(5.7)), Cleanup::Normalize(json!(5)));
        assert_eq!(cleanup("flag", json!("yes")), Cleanup::Normalize(json!(true)));
        assert_eq!(cleanup("flag", json!(0)), Cleanup::Normalize(json!(false)));
        assert_eq!(cleanup("server", json!(8001)), Cleanup::Normalize(json!("8001")));
        assert_eq!(cleanup("timeout", json!("30s")), Cleanup::Keep);
        assert!(matches!(cleanup("timeout", json!(30)), Cleanup::Remove(_)), "a number was ignored");
        assert!(matches!(cleanup("flag", json!("maybe")), Cleanup::Remove(_)));
        assert!(matches!(cleanup("confidence", json!(1.5)), Cleanup::Remove(_)), "out of range");
        assert!(matches!(cleanup("frs-area", json!([1, 2, 3])), Cleanup::Remove(_)), "only 4 numbers were used");
        assert!(matches!(cleanup("typo", json!(1)), Cleanup::Remove(_)));
        assert!(matches!(cleanup("blur", Value::Null), Cleanup::Remove(_)));
        let typed = SchemaBuilder::new("typed", Legacy::Typed).int("count", 0, 3).bool("flag", false).build();
        // Numbers given as strings are normalized although the typed readers ignored them
        assert_eq!(typed.cleanup("count", &json!("5")), Cleanup::Normalize(json!(5)));
        assert!(matches!(typed.cleanup("count", &json!(5.5)), Cleanup::Remove(_)));
        assert!(matches!(typed.cleanup("flag", &json!("true")), Cleanup::Remove(_)), "typed readers ignored it");
        assert_eq!(typed.cleanup("count", &json!(5)), Cleanup::Keep);
    }

    #[test]
    fn stored_config_warnings_are_not_repeated() {
        let schema = schema();
        let warnings = StoredConfigWarnings::new(Arc::new(Logger::null()));
        let config = json!({"count": "x", "typo": 1, "blur": 5});
        assert_eq!(warnings.check(&schema, "vstreams", "video stream 1", &config), 2);
        // The next cache update of the same values
        assert_eq!(warnings.check(&schema, "vstreams", "video stream 1", &config), 0);
        // Another row, and a changed value
        assert_eq!(warnings.check(&schema, "vstreams", "video stream 2", &config), 2);
        assert_eq!(warnings.check(&schema, "vstreams", "video stream 1", &json!({"count": "y", "typo": 1})), 1);
    }

    #[test]
    fn seconds_become_durations() {
        assert_eq!(seconds_to_duration(30.0).as_deref(), Some("30s"));
        assert_eq!(seconds_to_duration(0.25).as_deref(), Some("250ms"));
        assert_eq!(seconds_to_duration(0.0).as_deref(), Some("0s"));
        assert_eq!(seconds_to_duration(-1.0), None);
        assert_eq!(seconds_to_duration(f64::NAN), None);
    }
}

/// The config schemas of `docs/openapi.yaml` describe the same keys, forms and defaults as the code.
#[cfg(test)]
mod openapi_tests {
    use serde_json::Value;

    use super::{Kind, Schema};

    fn openapi_schemas() -> Value {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/openapi.yaml");
        let spec: Value = serde_saphyr::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        spec["components"]["schemas"].clone()
    }

    fn check(schema: &Schema, documented: &[&str]) {
        let schemas = openapi_schemas();
        let mut properties = serde_json::Map::new();
        for name in documented {
            properties.extend(schemas[*name]["properties"].as_object().unwrap().clone());
        }
        let mut code_keys: Vec<_> = schema.keys().iter().map(|spec| spec.key.as_str()).collect();
        let mut doc_keys: Vec<_> = properties.keys().map(String::as_str).collect();
        code_keys.sort_unstable();
        doc_keys.sort_unstable();
        assert_eq!(doc_keys, code_keys, "the keys of {documented:?}");
        for spec in schema.keys() {
            let property = &properties[&spec.key];
            let as_f64 = |value: &Value| value.as_f64().unwrap();
            assert_eq!(as_f64_or(&property["default"]), as_f64_or(&spec.default), "the default of {}", spec.key);
            let expected_type = match spec.kind {
                Kind::Float { .. } => "number",
                Kind::Int { .. } => "integer",
                Kind::Bool => "boolean",
                Kind::Str | Kind::Duration | Kind::Level => "string",
                Kind::FrsWorkArea | Kind::LprsWorkArea => "array",
            };
            assert_eq!(property["type"], expected_type, "the type of {}", spec.key);
            match spec.kind {
                Kind::Float { min, max } => {
                    assert_eq!(as_f64(&property["minimum"]), min, "the minimum of {}", spec.key);
                    assert_eq!(property.get("maximum").map(as_f64).unwrap_or(f64::MAX), max, "the maximum of {}", spec.key);
                }
                Kind::Int { min, max } => {
                    assert_eq!(property["minimum"].as_i64(), Some(min), "the minimum of {}", spec.key);
                    let documented_max = property.get("maximum").and_then(Value::as_i64).unwrap_or(i64::from(i32::MAX));
                    assert_eq!(documented_max, max, "the maximum of {}", spec.key);
                }
                _ => {}
            }
        }
    }

    /// Numbers compare by value (300 and 300.0 are the same default), anything else as JSON.
    fn as_f64_or(value: &Value) -> Value {
        value.as_f64().map_or_else(|| value.clone(), |n| serde_json::json!(n))
    }

    #[test]
    fn frs_common_config() {
        check(&crate::frs::COMMON_SCHEMA, &["FRSCommonConfig"]);
    }

    #[test]
    fn frs_video_stream_config() {
        check(&crate::frs::VSTREAM_SCHEMA, &["FRSDefaultVStreamConfig"]);
    }

    #[test]
    fn lprs_video_stream_config() {
        check(&crate::lprs::SCHEMA, &["LPRSDefaultVStreamConfig", "LPRSInternalVStreamConfig"]);
    }
}
