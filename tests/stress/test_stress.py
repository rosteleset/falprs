"""Tests of the runner's arguments and summaries."""

import argparse

import pytest

import stress


def test_share_arg():
    assert stress.share_arg("10%") == pytest.approx(0.1)
    assert stress.share_arg("0.25") == 0.25
    with pytest.raises(argparse.ArgumentTypeError):
        stress.share_arg("150%")


def test_defaults():
    sweep = stress.parse_args(["sweep"])
    assert sweep.subsystems == ["frs", "lprs"] and sweep.streams == 400
    assert sweep.warmup == 60 and sweep.measure == 120 and sweep.pipelines_per_cpu_values is None
    soak = stress.parse_args(["soak"])
    assert soak.duration == 4 * 3600 and soak.streams == 200 and soak.warmup == 900
    assert soak.churn_interval == 300 and soak.churn_share == 0.1 and soak.max_rss_growth == 20
    assert soak.max_error_rate == 0.001 and soak.camera_latency == 0.25 and soak.add_rate == 50
    assert soak.min_free == 5 and soak.sample_interval == 5
    emulator = stress.parse_args(["emulator", "--port", "9999", "--camera-latency", "0"])
    assert emulator.port == 9999 and emulator.camera_latency == 0


def test_retention_args():
    args = stress.parse_args(["soak", "--log-faces-ttl", "5m", "--events-log-maintenance-interval", "30s"])
    assert args.log_faces_ttl == "5m" and args.events_log_maintenance_interval == "30s"
    assert args.failed_ttl is None
    with pytest.raises(SystemExit):
        stress.parse_args(["soak", "--failed-ttl", "1.5h"])


def test_plateau_start():
    retention = {"log_faces_ttl": "10m", "clear_old_log_faces": "1m", "events_ttl": "10m",
                 "clear_old_events": "1m", "log_barcodes_ttl": "10m", "clear_old_log_barcodes": "1m",
                 "events_log_ttl": "20m", "events_log_maintenance_interval": "2m", "failed_ttl": "10m"}
    assert stress.plateau_start(retention, 30) == 30 + 22 * 60


def test_sweep_point_means():
    rows = [{"frs_frames_per_s": 10.0, "frs_log_faces_per_s": 4.0, "lprs_callbacks_per_s": 99.0,
             "service_cpu_pct": 50.0, "system_cpu_pct": 60.0, "gpu_util_pct": None,
             "service_rss_mb": 100.0, "emulator_cpu_pct": 2.0},
            {"frs_frames_per_s": 20.0, "frs_log_faces_per_s": 6.0, "lprs_callbacks_per_s": 99.0,
             "service_cpu_pct": 70.0, "system_cpu_pct": 80.0, "gpu_util_pct": None,
             "service_rss_mb": 200.0, "emulator_cpu_pct": 4.0}]
    means = stress.sweep_point_means(rows, "frs")
    assert means["frames_per_s"] == 15 and means["events_per_s"] == 5 and means["gpu_util_pct"] is None
    table = stress.sweep_table([{"subsystem": "frs", "pipelines_per_cpu": 8, "status": "ok", "means": means}], 4)
    assert table[-1].startswith("| FRS | 8 | 32 | 15.0 | - | 5.0 | 60.0 | 70.0 | - | 150 | 3.0 | ok |")


def test_stream_limit_marks_stream_bound_points():
    # 400 streams at a 250 ms camera latency ask for at most 400 / 1.25 = 320 frames/s
    limit = stress.stream_limit(400, 0.25)
    assert limit == 320
    points = [{"subsystem": "frs", "pipelines_per_cpu": ppc, "status": "ok", "means": {"frames_per_s": frames}}
              for ppc, frames in ((8, 83.4), (32, 290.7))]
    table = stress.sweep_table(points, 4, limit)
    assert table[-2].startswith("| FRS | 8 | 32 | 83.4 | 320.0 |") and table[-2].endswith("| ok |")
    assert table[-1].endswith("| ok (stream-bound: 91% of the limit) |")
