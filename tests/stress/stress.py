#!/usr/bin/env python3
"""Stress runner for FRS and LPRS.

    stress.py sweep     short runs over pipelines-per-cpu values, per subsystem, ending in a throughput table
    stress.py soak      one long run of both subsystems with stream churn, ending in a pass/fail verdict
    stress.py emulator  only the camera emulator, for running it on another host

The runner prepares the test databases, starts and stops the service, serves the streams' frames from a camera
emulator, receives their callbacks, samples the metrics and writes the results to tests/stress/results/<time>/.
Exit codes: 0 passed, 1 failed, 2 environment error. See README.md.
"""

import argparse
import asyncio
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import EnvironmentFailure, RunFailure, duration_arg, format_duration, parse_duration  # noqa: E402

STRESS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = STRESS_DIR / "results"
DEFAULT_EMULATOR_PORT = 9080


# --- arguments ---

def share_arg(text):
    """A share as `10%` or `0.1`."""
    text = text.strip()
    try:
        value = float(text[:-1]) / 100 if text.endswith("%") else float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid share: {text} (use e.g. 10% or 0.1)") from None
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError(f"a share must be within [0, 1]: {text}")
    return value


def int_list(text):
    try:
        values = [int(v) for v in text.split(",") if v.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid list of integers: {text}") from None
    if not values:
        raise argparse.ArgumentTypeError("the list is empty")
    return values


def subsystems_arg(text):
    values = [v.strip().lower() for v in text.split(",") if v.strip()]
    if not values or any(v not in ("frs", "lprs") for v in values):
        raise argparse.ArgumentTypeError(f"subsystems are frs and/or lprs, not {text}")
    return values


def service_duration_arg(text):
    """A duration the service accepts: an integer with ms/s/m/h/d or no unit."""
    import re
    if not re.fullmatch(r"\d+(ms|s|m|h|d)?", text.strip()):
        raise argparse.ArgumentTypeError(f"invalid service duration: {text} (use e.g. 30s, 10m, 1h)")
    return text.strip()


def add_emulator_args(parser):
    parser.add_argument("--camera-latency", type=duration_arg, default=0.25, metavar="DURATION",
                        help="delay before the emulator answers a frame request (default 250ms; 0 allowed)")
    parser.add_argument("--face-ratio", type=share_arg, default=0.5,
                        help="share of FRS streams whose frames have a face (default 0.5: the odd streams)")
    parser.add_argument("--plate-ratio", type=share_arg, default=0.5,
                        help="share of LPRS streams whose frames have a plate (default 0.5: the odd streams)")


def add_run_args(parser):
    import service
    group = parser.add_argument_group("emulator")
    add_emulator_args(group)
    group.add_argument("--emulator", metavar="URL",
                       help="use an emulator already running there (stress.py emulator on another host) "
                            "instead of starting one; its own latency and ratios then apply")
    group.add_argument("--emulator-port", type=int, default=DEFAULT_EMULATOR_PORT,
                       help=f"port of the emulator the runner starts (default {DEFAULT_EMULATOR_PORT})")

    group = parser.add_argument_group("streams")
    group.add_argument("--add-rate", type=int, default=50, help="streams added (and started) per second (default 50)")
    group.add_argument("--trace", action="store_true",
                       help="set logs-level: trace on every stream, for fairness.py (distorts throughput)")

    group = parser.add_argument_group("metrics and guards")
    group.add_argument("--sample-interval", type=duration_arg, default=5.0, metavar="DURATION",
                       help="metrics sampling interval (default 5s)")
    group.add_argument("--min-free", type=float, default=5.0, metavar="GB",
                       help="stop the run when the screenshots partition has less free space (default 5 GB)")

    group = parser.add_argument_group("service retention (default: the template's)")
    for arg, (subsystem, key, _) in service.RETENTION_KEYS.items():
        group.add_argument("--" + key, dest=arg, type=service_duration_arg, metavar="DURATION",
                           help=f"{subsystem.upper()} {key}")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help="where the timestamped results directory is created (default tests/stress/results)")


def parse_args(argv):
    parser = argparse.ArgumentParser(prog="stress.py", description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_subparsers(dest="mode", required=True)

    sweep = modes.add_parser("sweep", help="throughput and load over pipelines-per-cpu values, per subsystem")
    sweep.add_argument("--subsystems", type=subsystems_arg, default=["frs", "lprs"],
                       help="subsystems swept one after another (default frs,lprs)")
    sweep.add_argument("--streams", type=int, default=400, help="streams per subsystem (default 400)")
    sweep.add_argument("--pipelines-per-cpu-values", type=int_list, default=None, metavar="LIST",
                       help="values swept (default 4,8,16,32,64; the C++ service has no such setting: "
                            "under FALPRS_IMPL=cpp one point per subsystem is run without it)")
    sweep.add_argument("--warmup", type=duration_arg, default=60.0, metavar="DURATION",
                       help="warm-up after the last stream is started (default 1m)")
    sweep.add_argument("--measure", type=duration_arg, default=120.0, metavar="DURATION",
                       help="measurement window (default 2m)")
    add_run_args(sweep)

    soak = modes.add_parser("soak", help="long run of both subsystems with churn, pass/fail on leaks and errors")
    soak.add_argument("--duration", type=duration_arg, default=4 * 3600.0, metavar="DURATION",
                      help="run time after the last stream is started (default 4h)")
    soak.add_argument("--streams", type=int, default=200, help="streams per subsystem (default 200)")
    soak.add_argument("--frs-streams", type=int, help="FRS streams (default --streams)")
    soak.add_argument("--lprs-streams", type=int, help="LPRS streams (default --streams)")
    soak.add_argument("--pipelines-per-cpu", type=int, default=None,
                      help="pipelines-per-cpu of both subsystems (default 16; not supported by the C++ service)")
    soak.add_argument("--warmup", type=duration_arg, default=900.0, metavar="DURATION",
                      help="warm-up excluded from the trends (default 15m, at most half the duration)")
    soak.add_argument("--churn-interval", type=duration_arg, default=300.0, metavar="DURATION",
                      help="interval between churn rounds (default 5m; 0 disables churn)")
    soak.add_argument("--churn-share", type=share_arg, default=0.1,
                      help="share of each subsystem's streams removed and re-added per round (default 10%%)")
    soak.add_argument("--toggle-share", type=share_arg, default=0.1,
                      help="share of each subsystem's streams whose motion/workflow is stopped and restarted "
                           "per round (default 10%%)")
    soak.add_argument("--max-rss-growth", type=float, default=20.0, metavar="MB_PER_H",
                      help="largest RSS slope after warm-up (default 20 MB/h)")
    soak.add_argument("--max-error-rate", type=share_arg, default=0.001,
                      help="largest share of ERROR log lines per frame served (default 0.1%%)")
    add_run_args(soak)

    emulator = modes.add_parser("emulator", help="run only the camera emulator")
    emulator.add_argument("--host", default="0.0.0.0", help="address to listen on (default 0.0.0.0)")
    emulator.add_argument("--port", type=int, default=DEFAULT_EMULATOR_PORT,
                          help=f"port to listen on (default {DEFAULT_EMULATOR_PORT})")
    add_emulator_args(emulator)
    return parser.parse_args(argv)


# --- the camera emulator process ---

class LocalEmulator:
    """The emulator started by the runner as a separate process."""

    def __init__(self, port, latency, face_ratio, plate_ratio, log_path):
        self.port = port
        self.url = f"http://127.0.0.1:{port}"
        self.command = [sys.executable, str(Path(__file__).resolve()), "emulator", "--host", "127.0.0.1",
                        "--port", str(port), "--camera-latency", f"{latency * 1000:g}ms",
                        "--face-ratio", str(face_ratio), "--plate-ratio", str(plate_ratio)]
        self.log_path = log_path
        self.process = None

    async def start(self, session):
        self.log = open(self.log_path, "wb")
        self.process = subprocess.Popen(self.command, stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                break
            if await emulator_stats(session, self.url) is not None:
                return
            await asyncio.sleep(0.2)
        self.stop()
        raise EnvironmentFailure(f"the camera emulator did not start on port {self.port}; see {self.log_path}")

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.process:
            self.log.close()


async def emulator_stats(session, url):
    import aiohttp
    try:
        async with session.get(f"{url.rstrip('/')}/stats", timeout=aiohttp.ClientTimeout(total=3)) as r:
            return await r.json() if r.status == 200 else None
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
        return None


# --- a run ---

class Run:
    """What sweep and soak share: the results directory, the emulator, the service starts and the summary."""

    def __init__(self, args):
        import service
        self.args = args
        self.impl, self.binary = service.implementation()
        service.check_binary(self.binary)
        if self.impl == "cpp" and any(getattr(args, a, None) is not None
                                      for a in ("pipelines_per_cpu", "pipelines_per_cpu_values")):
            raise EnvironmentFailure("pipelines-per-cpu is not supported by the C++ service: it rejects the key")
        self.pg_port = service.pg_port()
        self.template = service.load_template(self.impl)
        self.cpus = service.service_cpus()
        self.retention = {arg: getattr(args, arg) for arg in service.RETENTION_KEYS}
        self.dir = make_results_dir(args.results_dir)
        self.emulator = None
        self.emulator_url = args.emulator
        self.emulator_settings = None
        self.session = None
        self.service = None
        self.log_dirs = []
        self.started = datetime.now().isoformat(timespec="seconds")
        self.write_arguments()

    def write_arguments(self):
        arguments = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(self.args).items()}
        record = {
            "command": " ".join(shlex.quote(a) for a in sys.argv),
            "arguments": arguments,
            "implementation": self.impl,
            "binary": str(self.binary),
            "pg_port": self.pg_port,
            "service_cpus": self.cpus,
            "started": self.started,
        }
        (self.dir / "arguments.json").write_text(json.dumps(record, indent=2) + "\n")

    def pg_url(self):
        return f"postgresql://test_falprs:123@localhost:{self.pg_port}/test_frs"

    async def open(self):
        import aiohttp
        connector = aiohttp.TCPConnector(limit=200)
        self.session = aiohttp.ClientSession(connector=connector)
        if self.emulator_url is None:
            a = self.args
            self.emulator = LocalEmulator(a.emulator_port, a.camera_latency, a.face_ratio, a.plate_ratio,
                                          self.dir / "emulator.log")
            await self.emulator.start(self.session)
            self.emulator_url = self.emulator.url
        self.emulator_settings = await emulator_stats(self.session, self.emulator_url)
        if self.emulator_settings is None:
            raise EnvironmentFailure(f"the camera emulator does not answer at {self.emulator_url}")
        print(f"Camera emulator {self.emulator_url}: latency {self.emulator_settings['latency'] * 1000:g} ms, "
              f"face ratio {self.emulator_settings['face_ratio']}, "
              f"plate ratio {self.emulator_settings['plate_ratio']}")

    async def close(self):
        if self.service:
            await asyncio.to_thread(self.service.stop)
            self.service = None
        if self.emulator:
            self.emulator.stop()
        if self.session:
            await self.session.close()

    def render(self, label, pipelines_per_cpu):
        import service
        config = service.render_config(self.template, impl=self.impl, port=self.pg_port,
                                       pipelines_per_cpu=pipelines_per_cpu, retention=self.retention,
                                       log_dir=service.LOGS_DIR / label)
        service.write_config(config, self.dir / f"config-{label}{service.config_suffix(config)}")
        return config

    async def start_service(self, label, pipelines_per_cpu):
        import service
        config = self.render(label, pipelines_per_cpu)
        print(f"Preparing the databases and starting {self.binary} ({label})")
        await asyncio.to_thread(service.prepare_database)
        await asyncio.to_thread(service.clear_screenshots)
        self.service = service.Service(self.binary, config, label)
        self.log_dirs.append(self.service.log_dir)
        await asyncio.to_thread(self.service.start)
        return self.service

    async def stop_service(self):
        if self.service:
            code = await asyncio.to_thread(self.service.stop)
            print(f"The service stopped with code {code}")
            self.service = None

    def monitor(self, point, frs_active):
        import metrics
        import service
        a = self.args
        return metrics.Monitor(service=self.service, emulator_url=self.emulator_url, session=self.session,
                               csv_path=self.dir / "metrics.csv", screenshot_dirs=service.SCREENSHOT_DIRS,
                               min_free_gb=a.min_free, pg_url=self.pg_url(), interval=a.sample_interval,
                               point=point, frs_active=frs_active)

    def save_log_tails(self, lines=500):
        import service
        target = self.dir / "logs-tail"
        target.mkdir(exist_ok=True)
        for log_dir in self.log_dirs:
            for path in sorted(log_dir.glob("*.txt")):
                text = service.tail(path, lines)
                if text:
                    (target / f"{log_dir.name}-{path.name}").write_text(text + "\n")
        print(f"The tails of the service logs are in {target}")

    def header(self, title):
        s = self.emulator_settings or {}
        lines = [
            f"# {title}",
            "",
            f"- Started: {self.started}, finished: {datetime.now().isoformat(timespec='seconds')}",
            f"- Command: `{' '.join(shlex.quote(a) for a in sys.argv)}`",
            f"- Implementation: {self.impl} (`{self.binary}`)",
            f"- CPUs seen by the service: {self.cpus}",
            f"- Camera emulator: {self.emulator_url}, latency {s.get('latency', 0) * 1000:g} ms, "
            f"face ratio {s.get('face_ratio')}, plate ratio {s.get('plate_ratio')}",
        ]
        return lines

    def print_fairness_hint(self):
        if not self.args.trace:
            return
        logs = " ".join(str(d / "frs-log.txt") for d in self.log_dirs if (d / "frs-log.txt").exists())
        if logs:
            print(f"Fairness of the FRS slot queue: python3 {STRESS_DIR / 'fairness.py'} {logs}")


def make_results_dir(base):
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = Path(base) / stamp
    n = 1
    while path.exists():
        n += 1
        path = Path(base) / f"{stamp}-{n}"
    path.mkdir(parents=True)
    return path


def make_streams(run, counts):
    import streams
    api = streams.Api(run.session, f"http://localhost:{run.service.port}")
    return [streams.Streams(api, subsystem, count, run.emulator_url, run.args.trace)
            for subsystem, count in counts.items() if count > 0]


def streams_api_failure(error):
    return RunFailure(f"API call failed: {error}")


async def guarded(monitor, coroutine):
    """Awaits `coroutine` unless the run fails meanwhile; an API error fails the run."""
    import streams
    task = asyncio.create_task(coroutine)
    failed = asyncio.create_task(monitor.failed())
    try:
        await asyncio.wait([task, failed], return_when=asyncio.FIRST_COMPLETED)
    finally:
        failed.cancel()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    monitor.check()
    try:
        return task.result()
    except streams.ApiError as e:
        raise streams_api_failure(e) from e


# --- sweep ---

# The services' default delay-between-frames, which the runner does not change: a stream asks for its next
# frame this long after the end of the previous frame's pipeline
DELAY_BETWEEN_FRAMES_S = 1.0
# Frames/s at this share of the stream limit or more mean that the streams, not the service, set the rate
STREAM_BOUND_SHARE = 0.8

SWEEP_COLUMNS = ["frames_per_s", "events_per_s", "service_cpu_pct", "system_cpu_pct", "gpu_util_pct",
                 "service_rss_mb", "emulator_cpu_pct"]


def sweep_point_means(rows, subsystem):
    """Means over the measurement rows of one sweep point. FRS events are `log_faces` rows (FRS calls back
    only on recognised faces); LPRS events are its callbacks (one per event)."""
    import verdict
    events = "frs_log_faces_per_s" if subsystem == "frs" else "lprs_callbacks_per_s"
    means = verdict.window_means(rows, [f"{subsystem}_frames_per_s", events, "service_cpu_pct", "system_cpu_pct",
                                        "gpu_util_pct", "service_rss_mb", "emulator_cpu_pct"])
    return {
        "frames_per_s": means[f"{subsystem}_frames_per_s"],
        "events_per_s": means[events],
        **{k: means[k] for k in SWEEP_COLUMNS[2:]},
    }


def stream_limit(streams, camera_latency):
    """The most frames/s the streams can ask for: each stream asks for a frame once per delay-between-frames plus
    the camera latency. The pipeline time and the slot wait lengthen the period further, so the real limit is
    lower."""
    return streams / (DELAY_BETWEEN_FRAMES_S + camera_latency)


def sweep_table(points, cpus, limit=None):
    def f(value, digits=1):
        return "-" if value is None else f"{value:.{digits}f}"

    lines = [
        "| Subsystem | pipelines-per-cpu | Slots | Frames/s | Stream limit | Events/s | Service CPU % "
        "| System CPU % | GPU % | RSS MB | Emulator CPU % | Status |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for p in points:
        m = p.get("means") or {}
        ppc = p["pipelines_per_cpu"]
        slots = "-" if ppc is None else str(ppc * cpus)
        frames = m.get("frames_per_s")
        status = p["status"]
        if limit and frames is not None and frames >= STREAM_BOUND_SHARE * limit:
            status += f" (stream-bound: {frames / limit:.0%} of the limit)"
        lines.append(
            f"| {p['subsystem'].upper()} | {ppc if ppc is not None else 'default'} | {slots} "
            f"| {f(frames)} | {f(limit)} | {f(m.get('events_per_s'))} | {f(m.get('service_cpu_pct'))} "
            f"| {f(m.get('system_cpu_pct'))} | {f(m.get('gpu_util_pct'))} | {f(m.get('service_rss_mb'), 0)} "
            f"| {f(m.get('emulator_cpu_pct'))} | {status} |")
    return lines


async def sweep(run):
    import streams
    args = run.args
    if run.impl == "cpp":
        if args.pipelines_per_cpu_values is not None:
            raise EnvironmentFailure("--pipelines-per-cpu-values is not supported by the C++ service: "
                                     "it rejects pipelines-per-cpu")
        values = [None]
    else:
        values = args.pipelines_per_cpu_values or [4, 8, 16, 32, 64]
    if any(v is not None and v < 1 for v in values):
        raise EnvironmentFailure("pipelines-per-cpu values must be at least 1")
    points = [{"subsystem": s, "pipelines_per_cpu": v, "status": "not run"}
              for s in args.subsystems for v in values]
    failure = None
    try:
        await run.open()
        for point in points:
            subsystem, ppc = point["subsystem"], point["pipelines_per_cpu"]
            label = f"{subsystem}-ppc{ppc if ppc is not None else 'default'}"
            print(f"=== {label}: {args.streams} streams, warm-up {format_duration(args.warmup)}, "
                  f"measurement {format_duration(args.measure)}")
            await run.start_service(label, ppc)
            monitor = run.monitor(label, frs_active=subsystem == "frs")
            await monitor.start()
            try:
                groups = make_streams(run, {subsystem: args.streams})
                await guarded(monitor, streams.set_up(groups, args.add_rate))
                monitor.phase = "warmup"
                await monitor.wait(args.warmup)
                monitor.phase = "measure"
                await monitor.wait(args.measure)
                rows = [r for r in monitor.rows if r["phase"] == "measure"]
                point["means"] = sweep_point_means(rows, subsystem)
                point["status"] = "ok"
            except RunFailure as e:
                point["status"] = f"failed: {e}"
                raise
            finally:
                await monitor.stop()
                await run.stop_service()
    except RunFailure as e:
        failure = e
        print(f"FAILED: {e}")
        run.save_log_tails()
    finally:
        await run.close()

    latency = (run.emulator_settings or {}).get("latency")
    limit = None if latency is None else stream_limit(args.streams, latency)
    lines = run.header("Stress sweep")
    lines += [
        f"- Streams per subsystem: {args.streams}, added at {args.add_rate}/s",
        f"- Warm-up {format_duration(args.warmup)}, measurement {format_duration(args.measure)}, "
        f"sampling every {format_duration(args.sample_interval)}",
        "",
        "Window means over the measurement. FRS events are `log_faces` rows; LPRS events are callbacks. "
        "Service CPU is a share of all CPUs, emulator CPU a share of one CPU.",
        "",
        f"Stream limit: the most frames/s the {args.streams} streams can ask for, streams / "
        f"(delay-between-frames {DELAY_BETWEEN_FRAMES_S:g} s + camera latency); the pipeline time lowers it further. "
        f"A point at {STREAM_BOUND_SHARE:.0%} of it or more is bound by the streams, not by the service: "
        "rerun it with more `--streams`.",
        "",
        *sweep_table(points, run.cpus, limit),
        "",
        f"Result: **{'FAILED: ' + str(failure) if failure else 'completed'}**",
        "",
    ]
    (run.dir / "summary.md").write_text("\n".join(lines))
    print("\n".join(sweep_table(points, run.cpus, limit)))
    print(f"Results: {run.dir}")
    run.print_fairness_hint()
    return 1 if failure else 0


# --- soak ---

def plateau_start(retention, setup_end):
    """Elapsed seconds after which the screenshot directories should have levelled off: the first TTL plus
    cleanup cycle after the streams are running."""
    def s(arg):
        return parse_duration(retention[arg]) or 0.0

    cycle = max(
        s("log_faces_ttl") + s("log_faces_cleanup_interval"),
        s("events_ttl") + s("events_cleanup_interval"),
        s("log_barcodes_ttl") + s("log_barcodes_cleanup_interval"),
        s("events_log_ttl") + s("events_log_cleanup_interval"),
        s("failed_ttl") + s("events_log_cleanup_interval"),
    )
    return setup_end + cycle


async def churn_loop(groups, args):
    import streams
    round_number = 0
    while True:
        await asyncio.sleep(args.churn_interval)
        readded, toggled = await streams.churn(groups, args.churn_share, args.toggle_share, round_number,
                                               args.add_rate)
        print(f"Churn round {round_number + 1}: re-added {readded} streams, toggled {toggled}")
        round_number += 1


async def soak(run):
    import service
    import streams
    import verdict
    args = run.args
    ppc = args.pipelines_per_cpu
    if run.impl == "cpp" and ppc is not None:
        raise EnvironmentFailure("--pipelines-per-cpu is not supported by the C++ service: it rejects the key")
    if run.impl == "rust" and ppc is None:
        ppc = 16
    counts = {"frs": args.frs_streams if args.frs_streams is not None else args.streams,
              "lprs": args.lprs_streams if args.lprs_streams is not None else args.streams}
    warmup = min(args.warmup, args.duration / 2)
    if warmup < args.warmup:
        print(f"The warm-up is cut to half the duration: {format_duration(warmup)}")
    failure = None
    monitor = None
    setup_end = warmup_end = None
    churn = None
    label = "soak"
    config = None
    try:
        await run.open()
        print(f"=== soak: {counts['frs']} FRS + {counts['lprs']} LPRS streams, {format_duration(args.duration)}, "
              f"pipelines-per-cpu {ppc if ppc is not None else 'default'}")
        config = (await run.start_service(label, ppc)).config
        monitor = run.monitor(label, frs_active=counts["frs"] > 0)
        await monitor.start()
        groups = make_streams(run, counts)
        last_start = await guarded(monitor, streams.set_up(groups, args.add_rate))
        setup_end = last_start - monitor.started
        warmup_end = setup_end + warmup
        monitor.phase = "warmup"
        if args.churn_interval > 0:
            churn = asyncio.create_task(churn_loop(groups, args))
        deadline = last_start + args.duration
        while (left := deadline - time.monotonic()) > 0:
            await monitor.wait(min(left, 5.0))
            if monitor.phase == "warmup" and time.monotonic() - monitor.started >= warmup_end:
                monitor.phase = "run"
            if churn and churn.done():
                error = churn.exception()
                if error:
                    raise streams_api_failure(error) if isinstance(error, streams.ApiError) else error
    except RunFailure as e:
        failure = e
        print(f"FAILED: {e}")
    finally:
        if churn:
            churn.cancel()
            await asyncio.gather(churn, return_exceptions=True)
        if monitor:
            await monitor.stop()
        await run.close()

    rows = monitor.rows if monitor else []
    effective = service.retention_of(config) if config else {}
    checks = verdict.evaluate_soak(
        rows,
        warmup_end=warmup_end if warmup_end is not None else float("inf"),
        plateau_start=plateau_start(effective, setup_end) if setup_end is not None and effective else float("inf"),
        max_rss_growth=args.max_rss_growth, max_error_rate=args.max_error_rate, failure=failure)
    ok = verdict.passed(checks)
    if not ok:
        run.save_log_tails()

    lines = run.header("Stress soak")
    lines += [
        f"- Streams: {counts['frs']} FRS + {counts['lprs']} LPRS, added at {args.add_rate}/s",
        f"- pipelines-per-cpu: {ppc if ppc is not None else 'default'}"
        + (f" ({ppc * run.cpus} slots per subsystem)" if ppc is not None else ""),
        f"- Duration {format_duration(args.duration)}, warm-up {format_duration(warmup)}, "
        f"sampling every {format_duration(args.sample_interval)}",
        f"- Churn every {format_duration(args.churn_interval)}: re-add {args.churn_share:.0%}, "
        f"toggle {args.toggle_share:.0%}" if args.churn_interval > 0 else "- Churn: off",
        "- Retention: " + ", ".join(f"{k.replace('_', '-')} {v}" for k, v in effective.items()),
        "",
    ]
    after = verdict.window(rows, warmup_end) if warmup_end is not None else []
    if after:
        means = verdict.window_means(after, ["frs_frames_per_s", "frs_log_faces_per_s", "lprs_frames_per_s",
                                             "lprs_callbacks_per_s", "service_cpu_pct", "system_cpu_pct",
                                             "gpu_util_pct", "service_rss_mb", "emulator_cpu_pct"])
        def f(key, digits=1):
            return "-" if means[key] is None else f"{means[key]:.{digits}f}"
        lines += [
            "Means after warm-up: "
            f"FRS {f('frs_frames_per_s')} frames/s, {f('frs_log_faces_per_s')} log_faces/s; "
            f"LPRS {f('lprs_frames_per_s')} frames/s, {f('lprs_callbacks_per_s')} events/s; "
            f"service CPU {f('service_cpu_pct')} %, system CPU {f('system_cpu_pct')} %, GPU {f('gpu_util_pct')} %, "
            f"RSS {f('service_rss_mb', 0)} MB, emulator CPU {f('emulator_cpu_pct')} %",
            "",
        ]
    lines += ["| Criterion | Result | Detail |", "|---|---|---|"]
    lines += [f"| {c.name} | {c.status.upper()} | {c.detail} |" for c in checks]
    lines += ["", f"Verdict: **{'PASSED' if ok else 'FAILED'}**", ""]
    (run.dir / "summary.md").write_text("\n".join(lines))
    for c in checks:
        print(f"{c.status.upper():4s}  {c.name}: {c.detail}")
    print(f"Verdict: {'PASSED' if ok else 'FAILED'}. Results: {run.dir}")
    run.print_fairness_hint()
    return 0 if ok else 1


# --- entry point ---

def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.mode == "emulator":
            import emulator
            emulator.serve(args.host, args.port, args.camera_latency, args.face_ratio, args.plate_ratio)
            return 0
        run = Run(args)
        print(f"Results directory: {run.dir}")
        return asyncio.run(sweep(run) if args.mode == "sweep" else soak(run))
    except EnvironmentFailure as e:
        print(f"Environment error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
