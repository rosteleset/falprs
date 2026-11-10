"""The service under test: binary, database, config rendering, start and stop."""

import copy
import os
import re
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

from common import EnvironmentFailure

STRESS_DIR = Path(__file__).resolve().parent
TESTS_DIR = STRESS_DIR.parent
REPO_DIR = TESTS_DIR.parent
TEMPLATE = STRESS_DIR / "stress_config.yaml"

WORK_DIR = Path("/tmp/test_falprs")
CONFIG_PATH = WORK_DIR / "stress_config.yaml"
LOGS_DIR = WORK_DIR / "logs"
PID_PATH = WORK_DIR / "falprs.pid"
# Subsystem -> the directory holding all of its screenshots, as set in the template
SCREENSHOT_DIRS = {
    "frs": WORK_DIR / "static" / "frs",
    "lprs": WORK_DIR / "static" / "lprs",
}

WORKFLOWS = {"frs": "frs-workflow", "lprs": "lprs-workflow"}
# Argument name -> (subsystem, key in its workflow config): the TTLs and intervals the runner can override
RETENTION_KEYS = {
    "log_faces_ttl": ("frs", "log-faces-ttl"),
    "clear_old_log_faces": ("frs", "clear-old-log-faces"),
    "events_ttl": ("frs", "events-ttl"),
    "clear_old_events": ("frs", "clear-old-events"),
    "log_barcodes_ttl": ("frs", "log-barcodes-ttl"),
    "clear_old_log_barcodes": ("frs", "clear-old-log-barcodes"),
    "events_log_ttl": ("lprs", "events-log-ttl"),
    "events_log_maintenance_interval": ("lprs", "events-log-maintenance-interval"),
    "failed_ttl": ("lprs", "failed-ttl"),
}
IMPLEMENTATIONS = ("rust", "cpp")
# Durations as the service parses them: an integer with an optional unit
SERVICE_DURATION = re.compile(r"\d+(ms|s|m|h|d)?")


def implementation(environ=os.environ):
    """(implementation, binary path) from FALPRS_IMPL and FALPRS_BIN, as tests/test_prepare.sh reads them."""
    impl = environ.get("FALPRS_IMPL") or "rust"
    if impl not in IMPLEMENTATIONS:
        raise EnvironmentFailure(f"FALPRS_IMPL must be rust or cpp, not '{impl}'")
    default_bin = REPO_DIR / ("rust/target/release/falprs" if impl == "rust" else "build/falprs")
    binary = Path(os.path.realpath(environ.get("FALPRS_BIN") or default_bin))
    return impl, binary


def check_binary(binary):
    if not (binary.is_file() and os.access(binary, os.X_OK)):
        raise EnvironmentFailure(f"the service binary {binary} is not found; build it first")


def pg_port(environ=os.environ):
    return int(environ.get("pg_port") or 5432)


def load_template(path=TEMPLATE):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def components(config):
    return config["components_manager"]["components"]


def workflow_config(config, subsystem):
    return components(config)[WORKFLOWS[subsystem]]["config"]


def render_config(template, *, impl, port=5432, pipelines_per_cpu=None, retention=None, log_dir=LOGS_DIR):
    """The service config for one start of the service.

    `retention` maps keys of RETENTION_KEYS to duration strings; None values keep the template's.
    """
    config = copy.deepcopy(template)
    parts = components(config)
    for name in ("frs-postgresql-database", "lprs-postgresql-database"):
        parts[name]["dbconnection"] = parts[name]["dbconnection"].replace(":5432/", f":{port}/")
    for logger in parts["logging"]["loggers"].values():
        logger["file_path"] = str(Path(log_dir) / Path(logger["file_path"]).name)
    for arg, value in (retention or {}).items():
        if value is None:
            continue
        if arg not in RETENTION_KEYS:
            raise ValueError(f"unknown retention setting {arg}")
        if not SERVICE_DURATION.fullmatch(str(value)):
            raise EnvironmentFailure(f"invalid duration for --{arg.replace('_', '-')}: {value}")
        subsystem, key = RETENTION_KEYS[arg]
        workflow_config(config, subsystem)[key] = value
    if pipelines_per_cpu is not None:
        if impl == "cpp":
            raise EnvironmentFailure("--pipelines-per-cpu is not supported by the C++ service: it rejects the key")
        if pipelines_per_cpu < 1:
            raise EnvironmentFailure("--pipelines-per-cpu must be at least 1")
        for subsystem in WORKFLOWS:
            workflow_config(config, subsystem)["pipelines-per-cpu"] = pipelines_per_cpu
    return config


def retention_of(config):
    """The effective TTLs and intervals of a rendered config, by argument name."""
    return {arg: workflow_config(config, subsystem).get(key) for arg, (subsystem, key) in RETENTION_KEYS.items()}


def write_config(config, path=CONFIG_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)
    return path


def api_port(config):
    return int(components(config)["server"]["listener"]["port"])


def service_cpus():
    """The CPU count the service sees: its affinity, limited by the cgroup v2 CPU quota (as Rust's
    `available_parallelism`)."""
    cpus = len(os.sched_getaffinity(0))
    try:
        cgroup = Path("/proc/self/cgroup").read_text().strip().split("::", 1)[1]
        quota, period = Path(f"/sys/fs/cgroup{cgroup}/cpu.max").read_text().split()
        if quota != "max":
            cpus = min(cpus, max(1, int(quota) // int(period)))
    except (OSError, IndexError, ValueError):
        pass
    return cpus


def _run(command, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        raise EnvironmentFailure(f"{' '.join(map(str, command))} failed:\n{result.stdout}{result.stderr}")
    return result


def prepare_database():
    """Recreates the test databases, as tests/test_prepare.sh does."""
    with open(TESTS_DIR / "prepare_data.sql", encoding="utf-8") as sql:
        _run(["sudo", "-u", "postgres", "psql", "-v", "ON_ERROR_STOP=1"], stdin=sql)
    env_file = TESTS_DIR / ".env.tests"
    _run([REPO_DIR / "scripts" / "sql_frs.sh", env_file])
    _run([REPO_DIR / "scripts" / "sql_lprs.sh", env_file])


def clear_screenshots():
    """Empties the screenshot directories, keeping the directories themselves (one may be a mount point)."""
    for directory in SCREENSHOT_DIRS.values():
        directory.mkdir(parents=True, exist_ok=True)
        for entry in directory.iterdir():
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink()


def ping(port, timeout=2.0):
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/ping", timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


class Service:
    """One run of the service binary with a rendered config."""

    def __init__(self, binary, config, label, start_timeout=120.0):
        self.binary = binary
        self.config = config
        self.label = label
        self.port = api_port(config)
        self.log_dir = Path(next(iter(components(config)["logging"]["loggers"].values()))["file_path"]).parent
        self.start_timeout = start_timeout
        self.process = None
        self.stdout = None

    @property
    def pid(self):
        return self.process.pid if self.process else None

    def log_files(self):
        """The service's log files: its loggers' files and its stdout/stderr."""
        files = [Path(logger["file_path"]) for logger in components(self.config)["logging"]["loggers"].values()]
        return files + [self.log_dir / "stdout.txt"]

    def start(self):
        if ping(self.port):
            raise EnvironmentFailure(f"a service already answers on port {self.port}; stop it first")
        if self.log_dir.exists():
            shutil.rmtree(self.log_dir)
        self.log_dir.mkdir(parents=True)
        (WORK_DIR / "stress-www").mkdir(parents=True, exist_ok=True)
        path = write_config(self.config)
        self.stdout = open(self.log_dir / "stdout.txt", "wb")
        # The service writes test images to its current directory, as in tests/test_prepare.sh
        self.process = subprocess.Popen([str(self.binary), "-c", str(path)], cwd=self.binary.parent,
                                        stdout=self.stdout, stderr=subprocess.STDOUT, start_new_session=True)
        PID_PATH.write_text(f"{self.process.pid}\n")
        deadline = time.monotonic() + self.start_timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise EnvironmentFailure(f"the service exited with code {self.process.returncode} at start; "
                                         f"see {self.log_dir}")
            if ping(self.port):
                return
            time.sleep(0.5)
        self.stop()
        raise EnvironmentFailure(f"the service did not answer /ping within {self.start_timeout:.0f} s")

    def alive(self):
        return self.process is not None and self.process.poll() is None

    def stop(self, timeout=60.0):
        """SIGTERM and wait, as tests/test_clean.sh does; SIGKILL if it does not exit in `timeout`.
        Returns the exit code, None if it had to be killed."""
        if self.process is None:
            return None
        code = self.process.poll()
        if code is None:
            self.process.send_signal(signal.SIGTERM)
            try:
                code = self.process.wait(timeout)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
                code = None
        if self.stdout:
            self.stdout.close()
            self.stdout = None
        if PID_PATH.exists() and PID_PATH.read_text().strip() == str(self.process.pid):
            PID_PATH.unlink()
        return code


def tail(path, lines=200):
    """The last `lines` lines of a text file, empty if it does not exist."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - lines * 400))
            data = f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(data.splitlines()[-lines:])

