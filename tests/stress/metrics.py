"""Metrics sampling: the service process, the system, the GPU, Triton, screenshots, the emulator and the database."""

import asyncio
import csv
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import aiohttp

from common import EnvironmentFailure, RunFailure

CLK_TCK = os.sysconf("SC_CLK_TCK")
SYSTEM_CPUS = os.cpu_count() or 1
MB = 1024 * 1024
GB = 1024 * MB
ERROR_MARK = b"\tlevel=ERROR\t"
# Consecutive failed /ping answers after which the service is taken as hung
PING_FAILURES = 3

COLUMNS = [
    "time", "elapsed_s", "point", "phase",
    "service_rss_mb", "service_fds", "service_threads", "service_cpu_pct", "system_cpu_pct",
    "gpu_util_pct", "gpu_mem_mb", "triton_rss_mb",
    "frs_screenshots_mb", "frs_screenshots_files", "lprs_screenshots_mb", "lprs_screenshots_files", "free_gb",
    "frs_frames_per_s", "lprs_frames_per_s", "frs_callbacks_per_s", "lprs_callbacks_per_s",
    "frs_log_faces_per_s", "emulator_cpu_pct",
    "frs_frames_total", "lprs_frames_total", "errors_total", "ping_ok",
]


# --- /proc readers ---

def process_stats(pid):
    """(RSS bytes, open fds, OS threads, CPU seconds) of a process; None if it is gone."""
    try:
        status = Path(f"/proc/{pid}/status").read_text()
        stat = Path(f"/proc/{pid}/stat").read_text()
        fds = len(os.listdir(f"/proc/{pid}/fd"))
    except OSError:
        return None
    fields = dict(line.split(":", 1) for line in status.splitlines() if ":" in line)
    rss = int(fields.get("VmRSS", "0 kB").split()[0]) * 1024
    threads = int(fields["Threads"])
    # The command may hold spaces and parentheses: the fields after it follow the last ')'
    after = stat.rsplit(")", 1)[1].split()
    cpu = (int(after[11]) + int(after[12])) / CLK_TCK  # utime, stime
    return rss, fds, threads, cpu


def process_rss(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def system_cpu_times():
    """(busy, total) jiffies of all CPUs from /proc/stat."""
    values = [int(v) for v in Path("/proc/stat").read_text().splitlines()[0].split()[1:9]]
    idle = values[3] + values[4]  # idle + iowait
    total = sum(values)
    return total - idle, total


def find_process(name):
    """The pid of a local process called `name`, None if there is none (it may run on another host)."""
    for entry in os.scandir("/proc"):
        if entry.name.isdigit():
            try:
                if Path(entry.path, "comm").read_text().strip() == name:
                    return int(entry.name)
            except OSError:
                continue
    return None


def directory_usage(path):
    """(apparent size in bytes, file count) of everything under `path`."""
    size = files = 0
    stack = [str(path)]
    while stack:
        try:
            with os.scandir(stack.pop()) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        elif entry.is_file(follow_symlinks=False):
                            size += entry.stat(follow_symlinks=False).st_size
                            files += 1
                    except OSError:
                        continue  # removed by the service's cleanup meanwhile
        except OSError:
            continue
    return size, files


def free_space(paths):
    """The smallest free space, in bytes, of the partitions holding `paths`."""
    return min(shutil.disk_usage(p).free for p in paths)


class ErrorCounter:
    """Counts ERROR lines appended to log files since the counter was created."""

    def __init__(self, paths):
        self.paths = [Path(p) for p in paths]
        self.offsets = dict.fromkeys(self.paths, 0)
        self.count = 0

    def update(self):
        for path in self.paths:
            try:
                size = path.stat().st_size
            except OSError:
                continue
            offset = self.offsets[path]
            if size < offset:  # rotated or truncated
                offset = 0
            if size == offset:
                continue
            with open(path, "rb") as f:
                f.seek(offset)
                while offset < size:
                    data = f.read(min(size - offset, 16 * MB))
                    # Up to the last complete line, so that a line being written is counted once
                    end = data.rfind(b"\n") + 1
                    if end == 0:
                        break
                    self.count += data[:end].count(ERROR_MARK)
                    offset += end
                    f.seek(offset)
            self.offsets[path] = offset
        return self.count


class Gpu:
    """Reads `nvidia-smi -l 1` in the background; empty values if there is no nvidia-smi."""

    def __init__(self):
        self.process = None
        self.task = None
        self.util = []
        self.memory = {}

    async def start(self):
        if shutil.which("nvidia-smi") is None:
            return
        self.process = await asyncio.create_subprocess_exec(
            "nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits", "-l", "1",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        self.task = asyncio.create_task(self._read())

    async def _read(self):
        while line := await self.process.stdout.readline():
            try:
                index, util, memory = (v.strip() for v in line.decode().split(","))
                self.util.append(float(util))
                self.memory[index] = float(memory)
            except ValueError:
                continue

    def take(self):
        """(mean utilisation % since the previous call, memory used MB over all GPUs); Nones without data."""
        util = sum(self.util) / len(self.util) if self.util else None
        self.util = []
        memory = sum(self.memory.values()) if self.memory else None
        return util, memory

    async def stop(self):
        if self.process and self.process.returncode is None:
            self.process.terminate()
            await self.process.wait()
        if self.task:
            await asyncio.gather(self.task, return_exceptions=True)


async def log_faces_max_id(pg_url):
    """The largest `log_faces` id: its growth is the FRS rows per second that `avg.sh` measured."""
    process = await asyncio.create_subprocess_exec(
        "psql", pg_url, "-tAc", "select coalesce(max(id_log), 0) from log_faces",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    out, _ = await process.communicate()
    try:
        return int(out.strip())
    except ValueError:
        return None


def rate(current, previous, seconds):
    if current is None or previous is None or seconds <= 0:
        return None
    return max(0.0, (current - previous) / seconds)


def fmt(value, digits=1):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def live_line(row, label):
    """The console line: frames/s and events/s per subsystem, CPU, GPU, RSS, screenshots, free space."""
    def v(key, unit="", digits=0):
        value = row.get(key)
        return "-" if value is None else f"{value:.{digits}f}{unit}"

    shots = sum(row.get(k) or 0 for k in ("frs_screenshots_mb", "lprs_screenshots_mb"))
    elapsed = int(row["elapsed_s"])
    return (f"[{elapsed // 3600:02d}:{elapsed // 60 % 60:02d}:{elapsed % 60:02d} {label} {row['phase']}] "
            f"FRS {v('frs_frames_per_s', digits=1)} fr/s {v('frs_log_faces_per_s', digits=1)} ev/s | "
            f"LPRS {v('lprs_frames_per_s', digits=1)} fr/s {v('lprs_callbacks_per_s', digits=1)} ev/s | "
            f"CPU svc {v('service_cpu_pct', '%')} sys {v('system_cpu_pct', '%')} | GPU {v('gpu_util_pct', '%')} | "
            f"RSS {v('service_rss_mb', ' MB')} | shots {shots / 1024:.2f} GB | free {v('free_gb', ' GB', 1)}")


class Monitor:
    """Samples the metrics every `interval` into a CSV file, prints the live line and guards liveness and disk.

    A failure (the service died or stopped answering /ping, free space below `min_free`) ends `wait` early
    with RunFailure; a lost emulator with EnvironmentFailure.
    """

    def __init__(self, *, service, emulator_url, session, csv_path, screenshot_dirs, min_free_gb, pg_url,
                 interval=5.0, line_interval=10.0, point="", frs_active=True, print_fn=print):
        self.service = service
        self.emulator_url = emulator_url.rstrip("/")
        self.session = session
        self.csv_path = Path(csv_path)
        self.screenshot_dirs = screenshot_dirs
        self.min_free = min_free_gb * GB
        self.pg_url = pg_url
        self.interval = interval
        self.line_interval = line_interval
        self.point = point
        self.frs_active = frs_active
        self.print = print_fn
        self.phase = "setup"
        self.rows = []
        self.failure = None
        self.errors = ErrorCounter(service.log_files())
        self.gpu = Gpu()
        self.triton_pid = None
        self._previous = None
        self._last_line = 0.0
        self._ping_failures = 0
        self._failed = asyncio.Event()
        self._task = None
        self.started = None

    async def start(self):
        self.started = time.monotonic()
        await self.gpu.start()
        self.triton_pid = find_process("tritonserver")
        new = not self.csv_path.exists()
        self._file = open(self.csv_path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=COLUMNS)
        if new:
            self._writer.writeheader()
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        await self.gpu.stop()
        self._file.close()

    async def wait(self, seconds):
        """Sleeps `seconds` unless the run fails meanwhile."""
        try:
            await asyncio.wait_for(self._failed.wait(), seconds)
        except asyncio.TimeoutError:
            pass
        self.check()

    async def failed(self):
        """Returns once the run has failed."""
        await self._failed.wait()

    def check(self):
        if self.failure:
            raise self.failure

    def fail(self, failure):
        if not self.failure:
            self.failure = failure
            self._failed.set()

    async def _loop(self):
        next_at = time.monotonic()
        while True:
            try:
                row = await self.sample()
            except (RunFailure, EnvironmentFailure) as e:
                self.fail(e)
                return
            except Exception as e:  # a broken sampler must not end the run silently
                self.fail(EnvironmentFailure(f"metrics sampling failed: {e!r}"))
                return
            self.rows.append(row)
            self._writer.writerow({k: fmt(row.get(k), 3) for k in COLUMNS})
            self._file.flush()
            if row["mono"] - self._last_line >= self.line_interval:
                self._last_line = row["mono"]
                self.print(live_line(row, self.point))
            self.guard(row)
            if self.failure:
                return
            next_at += self.interval
            await asyncio.sleep(max(0.0, next_at - time.monotonic()))

    def guard(self, row):
        if not self.service.alive():
            self.fail(RunFailure(f"liveness: the service exited with code {self.service.process.returncode}"))
        elif self._ping_failures >= PING_FAILURES:
            self.fail(RunFailure(f"liveness: /ping did not answer {PING_FAILURES} times in a row"))
        elif row["free_gb"] is not None and row["free_gb"] * GB < self.min_free:
            self.fail(RunFailure(f"disk guard: {row['free_gb']:.2f} GB free on the screenshots partition, "
                                 f"below {self.min_free / GB:g} GB"))

    async def _emulator_stats(self):
        try:
            async with self.session.get(f"{self.emulator_url}/stats", timeout=aiohttp.ClientTimeout(total=5)) as r:
                return await r.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            raise EnvironmentFailure(f"the camera emulator does not answer: {e!r}") from e

    async def _ping(self):
        try:
            url = f"http://localhost:{self.service.port}/ping"
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as r:
                return r.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False

    async def sample(self):
        mono = time.monotonic()
        emulator, ping_ok, log_faces = await asyncio.gather(
            self._emulator_stats(), self._ping(),
            log_faces_max_id(self.pg_url) if self.frs_active else asyncio.sleep(0))
        self._ping_failures = 0 if ping_ok else self._ping_failures + 1
        usage = await asyncio.to_thread(lambda: {s: directory_usage(d) for s, d in self.screenshot_dirs.items()})
        free = free_space(self.screenshot_dirs.values())
        proc = process_stats(self.service.pid)
        busy, total = system_cpu_times()
        errors = await asyncio.to_thread(self.errors.update)
        gpu_util, gpu_mem = self.gpu.take()
        if self.triton_pid is not None and process_rss(self.triton_pid) is None:
            self.triton_pid = find_process("tritonserver")
        triton_rss = process_rss(self.triton_pid) if self.triton_pid else None

        current = {
            "mono": mono, "busy": busy, "total": total,
            "service_cpu": proc[3] if proc else None,
            "emulator_cpu": emulator["cpu_seconds"],
            "emulator_uptime": emulator["uptime"],
            "frames": emulator["frames"], "callbacks": emulator["callbacks"],
            "log_faces": log_faces if self.frs_active else None,
        }
        p = self._previous
        dt = mono - p["mono"] if p else 0.0
        row = {
            "mono": mono,
            "time": datetime.now().isoformat(timespec="seconds"),
            "elapsed_s": mono - self.started,
            "point": self.point,
            "phase": self.phase,
            "service_rss_mb": proc[0] / MB if proc else None,
            "service_fds": proc[1] if proc else None,
            "service_threads": proc[2] if proc else None,
            "system_cpu_pct": 100.0 * (busy - p["busy"]) / (total - p["total"]) if p and total > p["total"] else None,
            "gpu_util_pct": gpu_util,
            "gpu_mem_mb": gpu_mem,
            "triton_rss_mb": triton_rss / MB if triton_rss else None,
            "free_gb": free / GB,
            "errors_total": errors,
            "ping_ok": int(ping_ok),
            "frs_frames_total": emulator["frames"]["frs"],
            "lprs_frames_total": emulator["frames"]["lprs"],
        }
        for subsystem, (size, files) in usage.items():
            row[f"{subsystem}_screenshots_mb"] = size / MB
            row[f"{subsystem}_screenshots_files"] = files
        if p:
            cpu = rate(current["service_cpu"], p["service_cpu"], dt)
            # The service as a share of all CPUs, comparable with the system CPU
            row["service_cpu_pct"] = 100.0 * cpu / SYSTEM_CPUS if cpu is not None else None
            # The emulator as a share of one CPU: it runs on one thread, so 100 % is its limit
            emulator_dt = current["emulator_uptime"] - p["emulator_uptime"]
            emulator_cpu = rate(current["emulator_cpu"], p["emulator_cpu"], emulator_dt)
            row["emulator_cpu_pct"] = 100.0 * emulator_cpu if emulator_cpu is not None else None
            for subsystem in ("frs", "lprs"):
                row[f"{subsystem}_frames_per_s"] = rate(current["frames"][subsystem], p["frames"][subsystem], dt)
                row[f"{subsystem}_callbacks_per_s"] = rate(current["callbacks"][subsystem],
                                                           p["callbacks"][subsystem], dt)
            row["frs_log_faces_per_s"] = rate(current["log_faces"], p["log_faces"], dt)
        self._previous = current
        return row
