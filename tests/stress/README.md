# Stress tests

`stress.py` load-tests FRS and LPRS. It starts and stops the service itself and generates the streams. A separate
camera emulator serves the streams' frames with a realistic delay and receives their callbacks. The runner
samples the service, the system, the GPU and the emulator, prints a live line and writes the results to
`tests/stress/results/<YYYYmmdd-HHMMSS>/`.

| Mode | What it does | Ends with |
|---|---|---|
| `sweep` | Short runs over `pipelines-per-cpu` values, one subsystem at a time | Throughput/load table |
| `soak` | One long run of both subsystems with stream churn | Pass/fail verdict on leaks, disk, errors, liveness |
| `emulator` | Only the camera emulator, to run on another host | Runs until stopped |

## Prerequisites

- PostgreSQL on the local host. The runner recreates the `test_frs` and `test_lprs` databases as
  `tests/test_prepare.sh` does, so `sudo -u postgres psql` must work without a password. `pg_port` sets the port,
  as in the functional tests.
- Triton with the models, configured in the databases as for the functional tests. It may run on another host;
  the GPU and Triton columns are then left empty.
- The service binary: `rust/target/release/falprs` (`FALPRS_IMPL=rust`, the default). `FALPRS_BIN` points to
  another binary; `FALPRS_IMPL=cpp` needs it, set to the C++ binary built from the `main` branch.
- Python packages: `pip install -r tests/requirements.txt`, or with apt:
  `sudo apt-get install python3-aiohttp python3-yaml`.
- Nothing else may listen on port 9071. The runner also uses 9080 for the emulator (`--emulator-port`). It
  writes to `/tmp/test_falprs/`: the rendered config, the logs in `logs/<point>/`, and the screenshots in
  `static/frs` and `static/lprs`, which it empties at every service start.

Do not run the stress tests together with the functional tests: both use the same databases, port and
`/tmp/test_falprs`.

## Examples

```bash
# Sweep: FRS then LPRS, 400 streams each, pipelines-per-cpu 4,8,16,32,64, 1 min warm-up + 2 min measurement per point
python3 tests/stress/stress.py sweep

# A shorter sweep of LPRS only
python3 tests/stress/stress.py sweep --subsystems lprs --streams 200 --pipelines-per-cpu-values 8,16,32 --measure 3m

# Soak: 4 h, 200 FRS + 200 LPRS streams, pipelines-per-cpu 16, churn every 5 min
python3 tests/stress/stress.py soak

# Smoke run of the whole runner, about 3 minutes
python3 tests/stress/stress.py soak --duration 2m --streams 10

# The C++ service, for comparison (it has no pipelines-per-cpu setting: one point per subsystem in sweep)
FALPRS_IMPL=cpp python3 tests/stress/stress.py sweep

# Fairness of the FRS slot queue: trace logs on every stream, then the analyser on the FRS log
python3 tests/stress/stress.py soak --duration 5m --streams 100 --lprs-streams 0 --churn-interval 0 --trace
python3 tests/stress/fairness.py /tmp/test_falprs/logs/soak/frs-log.txt
```

At the end of a `--trace` run the runner prints the `fairness.py` command with the log paths of that run. Trace
logging itself loads the service, so do not compare throughput from `--trace` runs with other runs.

### Emulator on another host

On the emulator host (it needs this repository's `tests/images` and `aiohttp`):

```bash
python3 tests/stress/stress.py emulator --host 0.0.0.0 --port 9080 --camera-latency 250ms
```

On the service host:

```bash
python3 tests/stress/stress.py soak --emulator http://emulator-host:9080
```

The streams then point at that URL. The emulator's own `--camera-latency`, `--face-ratio` and `--plate-ratio`
apply, and the summary records them. Its counters are cumulative, so one emulator can serve several runs one
after another.

## Main arguments

Common to `sweep` and `soak`:

| Argument | Default | Meaning |
|---|---|---|
| `--camera-latency` | `250ms` | Delay before the emulator answers a frame request; `0` allowed |
| `--face-ratio`, `--plate-ratio` | `0.5` | Share of FRS/LPRS streams whose frames have a face/plate. At 0.5 these are exactly the odd streams, which `fairness.py` assumes |
| `--emulator URL` | none | Use a running emulator instead of starting one |
| `--emulator-port` | `9080` | Port of the emulator the runner starts |
| `--add-rate` | `50` | Streams added, and then started, per second |
| `--trace` | off | `logs-level: trace` on every stream, for `fairness.py` |
| `--sample-interval` | `5s` | Metrics sampling interval |
| `--min-free` | `5` GB | Disk guard: stop the run below this free space on the screenshots partition |
| `--log-faces-ttl`, `--log-faces-cleanup-interval`, `--events-ttl`, `--events-cleanup-interval`, `--log-barcodes-ttl`, `--log-barcodes-cleanup-interval` | `10m` / `1m` | FRS retention and cleanup intervals |
| `--events-log-ttl`, `--events-log-cleanup-interval`, `--failed-ttl` | `10m` / `1m` / `10m` | LPRS retention and cleanup interval |

`sweep`:

| Argument | Default | Meaning |
|---|---|---|
| `--subsystems` | `frs,lprs` | Subsystems swept, one after another |
| `--streams` | `400` | Streams per subsystem; meant to overload the service |
| `--pipelines-per-cpu-values` | `4,8,16,32,64` | Points; an error under `FALPRS_IMPL=cpp` |
| `--warmup` | `1m` | Warm-up after the last stream is started |
| `--measure` | `2m` | Measurement window; the default covers two runs of the template's 1 min cleanups |

`soak`:

| Argument | Default | Meaning |
|---|---|---|
| `--duration` | `4h` | Run time after the last stream is started |
| `--streams`, `--frs-streams`, `--lprs-streams` | `200` | Streams per subsystem |
| `--pipelines-per-cpu` | `16` | Both subsystems; an error under `FALPRS_IMPL=cpp` |
| `--warmup` | `15m` | Excluded from the trends; at most half of `--duration` |
| `--churn-interval` | `5m` | Interval between churn rounds; `0` disables churn |
| `--churn-share` | `10%` | Streams of each subsystem removed and re-added per round |
| `--toggle-share` | `10%` | Streams whose motion detection (FRS) or workflow (LPRS) is stopped and restarted per round |
| `--max-rss-growth` | `20` MB/h | Largest RSS slope after warm-up |
| `--max-error-rate` | `0.1%` | Largest number of ERROR log lines per frame served |

`python3 tests/stress/stress.py <mode> --help` lists all arguments.

## What the runner does

- It recreates the databases, renders the template into `/tmp/test_falprs/` and starts the binary: `stress_config.toml`
  for the Rust service, `stress_config.yaml` for the C++ reference (`FALPRS_IMPL=cpp`, which reads the userver-shaped
  config); the two templates hold the same settings. It waits for `/ping`, and stops the service with SIGTERM at the
  end. `sweep` repeats this for every point. Compared with `tests/test_config.toml`, the template logs at `info` and
  has short TTLs and frequent cleanup, so the service's own cleanup keeps the screenshot directories bounded.
- `--pipelines-per-cpu N` goes into both the `[frs]` and the `[lprs]` settings. The two subsystems keep separate
  slot pools, so when both run, up to `2 × N × CPUs` pipelines run at once.
- FRS streams: `addStream` with the emulator's frame and callback URLs, then `motionDetection` start. LPRS
  streams: `addStream` with `screenshot-url`, `callback-url` and `ban-duration: 0s` (no plate or special-vehicle
  bans, so every frame with a plate yields an event, a screenshot and a callback), then `startWorkflow`. The
  starts follow the adds after the service's stream caches have picked the new streams up (7 s).
- The emulator serves FRS streams with a face `einstein_003.jpg` and the others `no_faces.jpg`. It serves LPRS
  streams with a plate `test_rain_007.jpg` and the others `no_faces.jpg`.

## The live line

Every 10 s:

```
[00:12:30 soak run] FRS 150.2 fr/s 75.0 ev/s | LPRS 148.9 fr/s 74.1 ev/s | CPU svc 85% sys 92% | GPU 77% | RSS 1234 MB | shots 1.20 GB | free 40.1 GB
```

- `fr/s`: frames served by the emulator to the subsystem's streams.
- `ev/s`: FRS `log_faces` rows per second (what `avg.sh` used to measure), and LPRS callbacks per second (one per
  event). FRS calls back only on recognised faces, so its callbacks are counted but not shown.
- `CPU svc`: the service's share of all CPUs. `sys`: the system's CPU from `/proc/stat`, also as a share of all CPUs.
- `GPU`: mean `nvidia-smi` utilisation over the sampling interval; `-` without a local `nvidia-smi`.
- `shots`: FRS + LPRS screenshot directories. `free`: free space on their partition.

## Cleaning up

At the end, also when interrupted with Ctrl+C, the runner stops the service and the emulator, but leaves the test
databases and `/tmp/test_falprs` (logs, the rendered config, the last screenshots). The next run, like
`tests/test_prepare.sh`, recreates the databases, so nothing has to be cleaned in between. When done with the stress
tests, delete them with `tests/test_clean.sh`; it also stops a service left running by a runner that was killed. The
results in `tests/stress/results/` are kept.

## Results

`tests/stress/results/<YYYYmmdd-HHMMSS>/` (git-ignored) holds:

- `metrics.csv`: one row per sample. Columns: service RSS, fds, threads and CPU; system CPU; GPU utilisation and
  memory; local `tritonserver` RSS; size and file count of each screenshot directory; free space; frames/s and
  callbacks/s per subsystem; FRS `log_faces` rows/s; emulator CPU; cumulative frames and ERROR lines; `/ping`.
  `point` and `phase` (`setup`, `warmup`, `measure` or `run`) tell the rows apart. `elapsed_s` restarts at every
  sweep point.
- `summary.md`: the sweep table or the soak verdict, with the implementation, the CPU count the service sees
  (the cgroup quota counted) and the emulator's settings.
- `config-<point>.yaml`: the rendered config of every service start. `arguments.json`: the command and all
  arguments. `emulator.log`: the output of the emulator the runner started.
- `logs-tail/`: on failure, the tails of the service logs.

In the sweep table each row is one point, with means over the measurement window. `Slots` is
`pipelines-per-cpu × CPUs` of the swept subsystem. Emulator CPU is a share of one CPU: the emulator runs on one
thread, so near 100 % it is the bottleneck, not the service.

`Stream limit` is the most frames/s the streams can ask for. A stream asks for its next frame
`delay-between-frames` (1 s by default) after the end of the previous frame's pipeline, and the camera latency
adds to that, so the limit is `--streams / (1 s + camera latency)`. The pipeline time lowers the real limit
further. A point at 80 % of the limit or more is marked `stream-bound`: the streams set its frames/s, not the
service, so rerun it with more `--streams`. For example, 400 streams at 250 ms ask for at most 320 frames/s.

## Exit codes and failure criteria

| Code | Meaning |
|---|---|
| 0 | Passed (sweep: all points completed) |
| 1 | Failed: any criterion below |
| 2 | Environment error: no database, no binary, `pipelines-per-cpu` given under `FALPRS_IMPL=cpp`, the emulator did not start or stopped answering, the service did not come up, port 9071 busy |

Both modes fail when:

- **liveness**: the service exited, or `/ping` did not answer 3 samples in a row;
- **disk guard**: free space on the screenshots partition dropped below `--min-free`. The run stops at once.

`sweep` fails on nothing else: it only reports. `soak` also fails on:

- **RSS growth**: the slope of a linear fit of RSS after warm-up is above `--max-rss-growth`;
- **fd growth / thread growth**: open fds or OS threads keep growing after warm-up. The medians of the four time
  quarters rise every quarter, and the last exceeds the first by more than 5 fds or 2 threads, and by more than
  5 %;
- **screenshots plateau**: a screenshot directory keeps growing after the first TTL + cleanup cycle. Over that
  window its fitted growth is above 10 % of its mean size, and above 20 MB;
- **error rate**: the ERROR lines in all service logs exceed `--max-error-rate` of the frames served.

The trend criteria need a window of at least 10 minutes and are reported as `SKIP` on shorter runs, such as the
smoke run.

## Tuning pipelines-per-cpu

The service runs at most `pipelines-per-cpu × CPUs` recognition pipelines per subsystem. Other ready frames wait
for a slot in a FIFO queue. A pipeline holds its slot while it fetches the frame, so the camera latency counts
in the slot hold time.

- **Too few slots.** Slots sit idle while frames are being fetched, and frames/s stops below what the CPU and
  Triton could do: CPU and GPU are not saturated, but frames/s does not grow.
- **Enough slots.** The queue is fair: in a `--trace` run, `fairness.py` shows the same slot wait for streams with
  and without a face, and nearly the same frames/s per stream. The difference in period between the groups is only
  the difference in pipeline time.
- **Too many slots.** The slot wait drops to 0, and all pipelines compete for the CPU and Triton at once. Heavy
  frames (with a face) stretch much more than light ones, so streams with a face get noticeably fewer frames/s.
  Total frames/s with faces falls: in an earlier FRS measurement on 4 CPUs, 131 frames/s at 8 per CPU against
  95 at 64.
- **Starting point.** The slots needed are about the target frames/s × the pipeline time, camera latency included.
  Pick the smallest sweep point where frames/s levels off and system CPU or GPU is saturated, then check it with a
  `--trace` run and `fairness.py`. Re-check when the camera latency changes.
