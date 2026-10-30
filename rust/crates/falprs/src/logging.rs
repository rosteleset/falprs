//! File loggers compatible with the userver setup: one file per named logger, tskv format,
//! runtime-changeable level, reopening on log rotation and dropping lines when the writer is busy.

use std::collections::HashMap;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::thread;

use anyhow::{Context, Result, anyhow};

use crate::config::LoggerConfig;

const QUEUE_SIZE: usize = 65536;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum Level {
    Trace,
    Debug,
    Info,
    Warning,
    Error,
    Critical,
    None,
}

impl Level {
    const ALL: [Level; 7] = [
        Level::Trace,
        Level::Debug,
        Level::Info,
        Level::Warning,
        Level::Error,
        Level::Critical,
        Level::None,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Level::Trace => "trace",
            Level::Debug => "debug",
            Level::Info => "info",
            Level::Warning => "warning",
            Level::Error => "error",
            Level::Critical => "critical",
            Level::None => "none",
        }
    }

    fn tskv_name(self) -> &'static str {
        match self {
            Level::Trace => "TRACE",
            Level::Debug => "DEBUG",
            Level::Info => "INFO",
            Level::Warning => "WARNING",
            Level::Error => "ERROR",
            Level::Critical => "CRITICAL",
            Level::None => "NONE",
        }
    }

    fn from_u8(value: u8) -> Level {
        Level::ALL[value as usize]
    }
}

impl FromStr for Level {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        Level::ALL
            .into_iter()
            .find(|level| level.as_str() == s)
            .ok_or_else(|| anyhow!("unknown log level: {s}"))
    }
}

impl fmt::Display for Level {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

enum Command {
    Line(String),
    Reopen(SyncSender<()>),
}

enum Sink {
    File { path: String, file: File },
    Stdout,
    Stderr,
    Null,
}

impl Sink {
    fn open(path: &str) -> io::Result<Sink> {
        Ok(match path {
            "@stdout" => Sink::Stdout,
            "@stderr" => Sink::Stderr,
            "@null" => Sink::Null,
            _ => Sink::File {
                path: path.to_string(),
                file: open_append(path)?,
            },
        })
    }

    fn write_line(&mut self, line: &str) {
        // A logger has nowhere to report its own write errors
        let _ = match self {
            Sink::File { file, .. } => file.write_all(line.as_bytes()),
            Sink::Stdout => io::stdout().write_all(line.as_bytes()),
            Sink::Stderr => io::stderr().write_all(line.as_bytes()),
            Sink::Null => Ok(()),
        };
    }

    fn reopen(&mut self) {
        if let Sink::File { path, file } = self
            && let Ok(reopened) = open_append(path)
        {
            *file = reopened;
        }
    }
}

fn open_append(path: &str) -> io::Result<File> {
    if let Some(dir) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(dir)?;
    }
    OpenOptions::new().create(true).append(true).open(path)
}

fn run_writer(mut sink: Sink, commands: Receiver<Command>) {
    for command in commands {
        match command {
            Command::Line(line) => sink.write_line(&line),
            Command::Reopen(done) => {
                sink.reopen();
                let _ = done.send(());
            }
        }
    }
}

/// A named logger writing to its own file on a dedicated thread.
pub struct Logger {
    init_level: Level,
    level: AtomicU8,
    commands: SyncSender<Command>,
}

impl Logger {
    fn new(name: &str, config: &LoggerConfig) -> Result<Logger> {
        let init_level: Level = config.level.parse()?;
        let sink = Sink::open(&config.file_path).with_context(|| format!("can't open log file {}", config.file_path))?;
        let (commands, receiver) = mpsc::sync_channel(QUEUE_SIZE);
        thread::Builder::new()
            .name(format!("log-{name}"))
            .spawn(move || run_writer(sink, receiver))?;
        Ok(Logger {
            init_level,
            level: AtomicU8::new(init_level as u8),
            commands,
        })
    }

    /// A logger that discards everything, for tests.
    #[cfg(test)]
    pub fn null() -> Logger {
        let config = LoggerConfig {
            file_path: "@null".to_string(),
            level: "trace".to_string(),
        };
        Logger::new("null", &config).unwrap()
    }

    pub fn init_level(&self) -> Level {
        self.init_level
    }

    pub fn level(&self) -> Level {
        Level::from_u8(self.level.load(Ordering::Relaxed))
    }

    pub fn set_level(&self, level: Level) {
        self.level.store(level as u8, Ordering::Relaxed);
    }

    pub fn should_log(&self, level: Level) -> bool {
        level != Level::None && level >= self.level()
    }

    pub fn log(&self, level: Level, module: &str, text: fmt::Arguments<'_>) {
        if !self.should_log(level) {
            return;
        }
        let timestamp = chrono::Local::now().format("%Y-%m-%dT%H:%M:%S%.6f");
        let line = format!(
            "tskv\ttimestamp={timestamp}\tlevel={}\tmodule={}\ttext={}\n",
            level.tskv_name(),
            escape_tskv(module),
            escape_tskv(&text.to_string())
        );
        // Drop the line if the writer can't keep up, like userver's `overflow_behavior: discard`
        if let Err(TrySendError::Disconnected(_)) = self.commands.try_send(Command::Line(line)) {
            eprintln!("log writer is gone");
        }
    }

    fn reopen(&self) {
        let (done, wait) = mpsc::sync_channel(1);
        if self.commands.send(Command::Reopen(done)).is_ok() {
            let _ = wait.recv();
        }
    }
}

fn escape_tskv(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for c in value.chars() {
        match c {
            '\t' => escaped.push_str("\\t"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\\' => escaped.push_str("\\\\"),
            '=' => escaped.push_str("\\="),
            _ => escaped.push(c),
        }
    }
    escaped
}

/// All loggers from the `logging.loggers` config section, by name.
pub struct Loggers {
    loggers: HashMap<String, Arc<Logger>>,
}

impl Loggers {
    pub fn new(configs: &HashMap<String, LoggerConfig>) -> Result<Loggers> {
        let loggers = configs
            .iter()
            .map(|(name, config)| Ok((name.clone(), Arc::new(Logger::new(name, config)?))))
            .collect::<Result<_>>()?;
        Ok(Loggers { loggers })
    }

    pub fn get(&self, name: &str) -> Option<Arc<Logger>> {
        self.loggers.get(name).cloned()
    }

    pub fn default_logger(&self) -> Arc<Logger> {
        self.loggers["default"].clone()
    }

    /// Reopens every log file, blocking until all of them are reopened.
    pub fn reopen_all(&self) {
        for logger in self.loggers.values() {
            logger.reopen();
        }
    }
}

#[macro_export]
macro_rules! log {
    ($logger:expr, $level:expr, $($arg:tt)+) => {
        $logger.log($level, module_path!(), format_args!($($arg)+))
    };
}

/// The logger and key of the video stream a workflow step works for, with the stream's log level.
#[derive(Clone, Copy)]
pub struct StreamLog<'a> {
    pub logger: &'a Logger,
    pub key: &'a str,
    pub level: Level,
    /// Log regardless of the level, as FRS does for image tests.
    pub always: bool,
}

impl StreamLog<'_> {
    pub fn logs(&self, level: Level) -> bool {
        self.always || self.level <= level
    }
}

/// Logs `vstream_key = <key>;  <message>` if the video stream's log level allows `level`.
#[macro_export]
macro_rules! stream_log {
    ($stream:expr, $level:expr, $($arg:tt)+) => {
        if $stream.logs($level) {
            $crate::log!($stream.logger, $level, "vstream_key = {};  {}", $stream.key, format_args!($($arg)+))
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_log_prefixes_the_key_and_follows_the_stream_level() {
        let path = std::env::temp_dir().join(format!("falprs-stream-log-{}.txt", std::process::id()));
        let config = LoggerConfig {
            file_path: path.to_string_lossy().into_owned(),
            level: "trace".to_string(),
        };
        let logger = Logger::new("test", &config).unwrap();
        let stream = StreamLog {
            logger: &logger,
            key: "1_cam",
            level: Level::Debug,
            always: false,
        };
        crate::stream_log!(stream, Level::Trace, "hidden");
        crate::stream_log!(stream, Level::Debug, "plate count: {}", 2);
        let test = StreamLog { always: true, ..stream };
        crate::stream_log!(test, Level::Trace, "shown");
        // Reopening waits for the queued lines to be written
        logger.reopen();

        let text = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        // TSKV escapes `=` in values
        assert!(text.contains("\ttext=vstream_key \\= 1_cam;  plate count: 2\n"));
        assert!(text.contains("\ttext=vstream_key \\= 1_cam;  shown\n"));
        assert!(!text.contains("hidden"));
    }
}
