#!/usr/bin/env python3
"""Converts the userver-shaped config.yaml of the service into its config.toml.

Usage: convert_config.py <config.yaml> <config.toml>

Every setting the service used is carried over with the value it had, written explicitly even
when it equals the default. Settings that no longer exist are dropped; those that differ from
the old example config, and keys the converter does not know, are listed in one warning, and
the conversion goes on.
"""

import fnmatch
import json
import re
import sys

import yaml

# Values the service used when a setting was left out of config.yaml and whose new default differs
OLD_COMMAND_CONTROL = {"network_timeout_ms": 500, "statement_timeout_ms": 250}
OLD_MAX_POOL_SIZE = 16
OLD_MAX_REQUEST_SIZE = 1024 * 1024
OLD_NO_AUTH_GROUP_ID = {"lprs": 0, "frs": 1}

DURATION = re.compile(r"\d+(ms|s|m|h|d)?")

# Old cache component -> (subsystem, new cache name, whether it can update incrementally)
CACHES = {
    "lprs-groups-pg-cache": ("lprs", "groups", False),
    "lprs-vstreams-config-pg-cache": ("lprs", "video-streams-config", False),
    "lprs-vstream-group-pg-cache": ("lprs", "video-stream-groups", False),
    "frs-groups-pg-cache": ("frs", "groups", False),
    "frs-config-pg-cache": ("frs", "config", False),
    "frs-vstreams-config-pg-cache": ("frs", "video-streams-config", True),
    "frs-face-descriptor-pg-cache": ("frs", "face-descriptors", True),
    "frs-vstream-descriptors-pg-cache": ("frs", "video-stream-descriptors", True),
    "frs-sg-config-pg-cache": ("frs", "sg-config", False),
    "frs-sg-descriptors-pg-cache": ("frs", "sg-descriptors", True),
}

# Workflow config key -> (new key, kind), per subsystem
WORKFLOW_KEYS = {
    "lprs": {
        "allow-group-id-without-auth": ("no-auth-group-id", "int"),
        "pipelines-per-cpu": ("pipelines-per-cpu", "count"),
        "screenshots-path": ("screenshots-path", "str"),
        "screenshots-url-prefix": ("screenshots-url-prefix", "str"),
        "failed-path": ("failed-path", "str"),
        "failed-ttl": ("failed-ttl", "duration"),
        "ban-maintenance-interval": ("ban-cleanup-interval", "duration"),
        "events-log-ttl": ("events-log-ttl", "duration"),
        "events-log-maintenance-interval": ("events-log-cleanup-interval", "duration"),
        "flag-deleted-ttl": ("flag-deleted-ttl", "duration"),
        "flag-deleted-maintenance-interval": ("flag-deleted-cleanup-interval", "duration"),
    },
    "frs": {
        "allow-group-id-without-auth": ("no-auth-group-id", "int"),
        "pipelines-per-cpu": ("pipelines-per-cpu", "count"),
        "screenshots-path": ("screenshots-path", "str"),
        "screenshots-url-prefix": ("screenshots-url-prefix", "str"),
        "events-path": ("events-path", "str"),
        "barcodes-path": ("barcodes-path", "str"),
        "barcodes-url-prefix": ("barcodes-url-prefix", "str"),
        "log-faces-ttl": ("log-faces-ttl", "duration"),
        "clear-old-log-faces": ("log-faces-cleanup-interval", "duration"),
        "log-barcodes-ttl": ("log-barcodes-ttl", "duration"),
        "clear-old-log-barcodes": ("log-barcodes-cleanup-interval", "duration"),
        "events-ttl": ("events-ttl", "duration"),
        "clear-old-events": ("events-cleanup-interval", "duration"),
        "flag-deleted-ttl": ("flag-deleted-ttl", "duration"),
        "flag-deleted-maintenance-interval": ("flag-deleted-cleanup-interval", "duration"),
        "copy-events-maintenance-interval": ("copy-events-interval", "duration"),
    },
}

ANY = object()
C = "components_manager.components."
# Settings that no longer exist, as patterns of their paths, with their value in the old example
# config; ANY drops them without a warning (e.g. worker_threads, which the build script set per CPU)
DROPPED = {
    "components_manager.coro_pool.initial_size": 500,
    "components_manager.coro_pool.max_size": 1000,
    "components_manager.task_processors.*": ANY,
    "components_manager.default_task_processor": ANY,
    "*.task_processor": ANY,
    "*.fs-task-processor": ANY,
    "*.blocking_task_processor": ANY,
    "*.blocking-task-processor": ANY,
    C + "server.listener-monitor.connection.in_buffer_size": 32768,
    C + "server.listener-monitor.connection.requests_queue_size_threshold": 100,
    C + "logging.loggers.*.format": "tskv",
    C + "logging.loggers.*.overflow_behavior": "discard",
    C + "dynamic-config.defaults.POSTGRES_CONNECTION_POOL_SETTINGS.*.max_queue_size": 200,
    C + "dynamic-config.defaults.POSTGRES_CONNECTION_POOL_SETTINGS.*.min_pool_size": 8,
    C + "dynamic-config.defaults.POSTGRES_HANDLERS_COMMAND_CONTROL*": ANY,
    C + "dynamic-config.defaults.POSTGRES_QUERIES_COMMAND_CONTROL*": ANY,
    C + "dynamic-config.defaults.POSTGRES_STATEMENT_METRICS_SETTINGS*": ANY,
    C + "dns-client*": ANY,
    C + "testsuite-support*": ANY,
    C + "http-client-core*": ANY,
    C + "grpc-client-common*": ANY,
    C + "grpc-client-factory*": ANY,
    C + "triton-client*": ANY,
    C + "handler-*": ANY,
    C + "fs-cache-main.update-period": "10s",
    C + "lprs-api-http.path": ANY,
    C + "lprs-api-http.method": ANY,
    C + "frs-api-http.path": ANY,
    # motionDetection now always answers GET too
    C + "frs-api-http.method": ANY,
    C + "*-postgresql-database.max_queue_size": 200,
    C + "*-postgresql-database.min_pool_size": 8,
    C + "*-pg-cache.pgcomponent": ANY,
}


class Converter:
    def __init__(self, old):
        self.old = old or {}
        self.used = set()
        self.warnings = []

    def get(self, path, default=None):
        """The value at a path of the old config, marking a leaf value as carried over."""
        node = self.old
        for part in path:
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        if not isinstance(node, dict) or not node:
            self.used.add(".".join(path))
        return node

    def components(self, *path, default=None):
        return self.get(("components_manager", "components") + path, default)

    def warn(self, text):
        self.warnings.append(text)

    def duration(self, value, where):
        """A duration string, or None (with a warning) when the value is not one."""
        if isinstance(value, bool):
            value = None
        elif isinstance(value, int):
            value = f"{value}s"
        if isinstance(value, str) and DURATION.fullmatch(value):
            return value
        self.warn(f"{where}: {value!r} is not a duration, the default is used")
        return None

    def convert(self):
        result = {"server": self.server(), "logging": self.logging()}
        for subsystem in ("lprs", "frs"):
            if self.components(f"{subsystem}-postgresql-database") is not None:
                result[subsystem] = self.subsystem(subsystem)
        self.check_leftovers(self.old, [])
        return result

    def server(self):
        server = {}
        port = self.components("server", "listener", "port")
        if port is not None:
            server["port"] = port
        monitor_port = self.components("server", "listener-monitor", "port")
        if monitor_port is not None:
            server["monitor-port"] = monitor_port
        else:
            self.warn("server.listener-monitor is not set: the service endpoints now listen on monitor-port (default 9052)")
        static_dir = self.components("fs-cache-main", "dir")
        if static_dir is not None:
            server["static-dir"] = static_dir
        # A handler's own limit, else the server's handler defaults, else userver's default
        default_size = self.components("server", "handler-defaults", "max_request_size", default=OLD_MAX_REQUEST_SIZE)
        sizes = {}
        for subsystem in ("lprs", "frs"):
            if self.components(f"{subsystem}-postgresql-database") is not None:
                sizes[subsystem] = self.components(f"{subsystem}-api-http", "max_request_size", default=default_size)
        if len(set(sizes.values())) > 1:
            self.warn(f"the API request size limits differ ({sizes}): the larger one is used for both")
        server["max-request-size"] = max(sizes.values(), default=default_size)
        return server

    def logging(self):
        logging = {}
        loggers = self.components("logging", "loggers", default={}) or {}
        for name in loggers:
            if name not in ("default", "lprs", "frs"):
                continue
            logger = {}
            file_path = self.components("logging", "loggers", name, "file_path")
            if file_path is not None:
                logger["file"] = file_path
            # The level defaulted to info
            logger["level"] = self.components("logging", "loggers", name, "level", default="info")
            logging[name] = logger
        return logging

    def defaults(self, *path, default=None):
        return self.components("dynamic-config", "defaults", *path, default=default)

    def milliseconds(self, value, where):
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return f"{value // 1000}s" if value % 1000 == 0 else f"{value}ms"
        self.warn(f"{where}: {value!r} is not a number of milliseconds, the default is used")
        return None

    def database(self, subsystem):
        component = f"{subsystem}-postgresql-database"
        database = {"dsn": self.components(component, "dbconnection", default="")}
        # The pool size of the dynamic config won; the component's own one was ignored before
        pool_size = self.defaults("POSTGRES_CONNECTION_POOL_SETTINGS", component, "max_pool_size")
        component_pool_size = self.components(component, "max_pool_size")
        database["max-pool-size"] = pool_size or component_pool_size or OLD_MAX_POOL_SIZE
        for key, new_key in (("network_timeout_ms", "network-timeout"), ("statement_timeout_ms", "statement-timeout")):
            value = self.defaults("POSTGRES_DEFAULT_COMMAND_CONTROL", key, default=OLD_COMMAND_CONTROL[key])
            value = self.milliseconds(value, f"POSTGRES_DEFAULT_COMMAND_CONTROL.{key}")
            if value is not None:
                database[new_key] = value
        discovery = self.defaults("FALPRS_PG_TOPOLOGY_SETTINGS", component, "discovery_timeout_ms")
        if discovery is not None:
            value = self.milliseconds(discovery, f"FALPRS_PG_TOPOLOGY_SETTINGS.{component}.discovery_timeout_ms")
            if value is not None:
                database["discovery-timeout"] = value
        return database

    def cache(self, component, incremental):
        cache = {}
        where = component
        interval = self.components(component, "update-interval")
        if interval is not None:
            value = self.duration(interval, f"{where}.update-interval")
            if value is not None:
                cache["update-interval"] = value
        update_types = self.components(component, "update-types", default="only-full")
        if update_types not in ("only-full", "full-and-incremental"):
            self.warn(f"{where}.update-types: unknown value {update_types!r}, full updates only are used")
            update_types = "only-full"
        if incremental:
            if update_types == "full-and-incremental":
                full_interval = self.components(component, "full-update-interval")
                value = None if full_interval is None else self.duration(full_interval, f"{where}.full-update-interval")
                if value is not None:
                    cache["full-update-interval"] = value
                # The correction defaulted to zero
                correction = self.components(component, "update-correction", default="0s")
                value = self.duration(correction, f"{where}.update-correction")
                if value is not None:
                    cache["update-correction"] = value
            else:
                cache["full-update-interval"] = "0s"
        elif update_types == "full-and-incremental":
            self.warn(f"{where}: incremental updates are not supported by this cache, full updates only are used")
        if not incremental or update_types == "only-full":
            # Ignored before as well
            self.components(component, "full-update-interval")
            self.components(component, "update-correction")
        host_type = self.defaults("FALPRS_PG_CACHE_HOST_TYPES", component)
        if host_type is not None:
            cache["host-type"] = host_type
        timeouts = [("full-update-op-timeout", "full-update-timeout")]
        if incremental:
            timeouts.append(("incremental-update-op-timeout", "incremental-update-timeout"))
        for key, new_key in timeouts:
            timeout = self.components(component, key)
            if timeout is not None:
                value = self.duration(timeout, f"{where}.{key}")
                if value is not None:
                    cache[new_key] = value
        return cache

    def subsystem(self, subsystem):
        settings = {}
        workflow = self.components(f"{subsystem}-workflow", "config", default={}) or {}
        keys = WORKFLOW_KEYS[subsystem]
        if "allow-group-id-without-auth" not in workflow:
            settings["no-auth-group-id"] = OLD_NO_AUTH_GROUP_ID[subsystem]
        for key in workflow:
            if key not in keys:
                continue
            value = self.components(f"{subsystem}-workflow", "config", key)
            new_key, kind = keys[key]
            where = f"{subsystem}-workflow.config.{key}"
            if kind == "duration":
                value = self.duration(value, where)
            elif kind in ("int", "count"):
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    value = None
                if value is None or (kind == "count" and value < 1):
                    self.warn(f"{where}: invalid number, the default is used")
                    value = None
            elif not isinstance(value, str):
                value = None if value is None else str(value)
            if value is not None:
                settings[new_key] = value
        settings["database"] = self.database(subsystem)
        caches = {}
        for component, (cache_subsystem, name, incremental) in CACHES.items():
            if cache_subsystem == subsystem and self.components(component) is not None:
                caches[name] = self.cache(component, incremental)
        if caches:
            settings["caches"] = caches
        return settings

    def check_leftovers(self, node, path):
        """Warns about the settings neither carried over nor known as dropped."""
        dotted = ".".join(path)
        if dotted in self.used:
            return
        for pattern, example in DROPPED.items():
            if fnmatch.fnmatchcase(dotted, pattern):
                if example is not ANY and node != example:
                    self.warn(f"{dotted} = {node!r} is no longer used and is dropped")
                return
        if isinstance(node, dict) and node:
            for key, value in node.items():
                self.check_leftovers(value, path + [str(key)])
        elif path:
            self.warn(f"{dotted} = {node!r} is not known to the converter and is dropped")


def toml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    # JSON string escapes are valid in TOML basic strings
    return json.dumps(str(value), ensure_ascii=False)


def toml_tables(table, prefix, lines):
    scalars = [(key, value) for key, value in table.items() if not isinstance(value, dict)]
    tables = [(key, value) for key, value in table.items() if isinstance(value, dict)]
    if prefix and (scalars or not tables):
        lines.append("")
        lines.append(f"[{prefix}]")
    for key, value in scalars:
        lines.append(f"{key} = {toml_value(value)}")
    for key, value in tables:
        toml_tables(value, f"{prefix}.{key}" if prefix else key, lines)


def toml_dumps(config, comments=()):
    """TOML text of a config of tables, strings, integers and booleans, after comment lines."""
    lines = [f"# {comment}" for comment in comments]
    toml_tables(config, "", lines)
    return "\n".join(lines).lstrip("\n") + "\n"


def to_toml(config, source):
    return toml_dumps(config, [
        f"Converted from {source} by scripts/convert_config.py.",
        "config.toml.example lists every setting with its default.",
    ])


def convert_text(text, source="config.yaml"):
    """(TOML text, warnings) of a config.yaml text."""
    converter = Converter(yaml.safe_load(text))
    config = converter.convert()
    return to_toml(config, source), converter.warnings


def main():
    if len(sys.argv) != 3:
        print("Usage: convert_config.py <config.yaml> <config.toml>", file=sys.stderr)
        return 1
    source, destination = sys.argv[1], sys.argv[2]
    try:
        with open(source, encoding="utf-8") as f:
            text = f.read()
        toml, warnings = convert_text(text, source)
    except (OSError, yaml.YAMLError) as e:
        print(f"Error: can't read {source}: {e}", file=sys.stderr)
        return 1
    if warnings:
        print(f"Warning: converting {source}:", file=sys.stderr)
        for warning in warnings:
            print(f"  - {warning}", file=sys.stderr)
    try:
        with open(destination, "x", encoding="utf-8") as f:
            f.write(toml)
    except OSError as e:
        print(f"Error: can't write {destination}: {e}", file=sys.stderr)
        return 1
    print(f"{source} converted to {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
