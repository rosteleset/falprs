"""Tests of the trend detection and the soak verdict on synthetic series."""

import pytest

from common import RunFailure
from verdict import evaluate_soak, linear_slope, passed, reaches_plateau, sustained_growth


def test_linear_slope():
    assert linear_slope([0, 1, 2, 3], [1, 3, 5, 7]) == pytest.approx(2)
    assert linear_slope([0, 1, 2], [5, 5, 5]) == pytest.approx(0)
    assert linear_slope([0, 1, 2, 3], [1, None, 5, 7]) == pytest.approx(2)
    assert linear_slope([1], [1]) is None
    assert linear_slope([1, 1], [1, 2]) is None


def test_slope_on_noisy_series():
    ts = list(range(100))
    ys = [1000 + 0.5 * t + (7 if t % 2 else -7) for t in ts]
    assert linear_slope(ts, ys) == pytest.approx(0.5, abs=0.02)


def test_sustained_growth():
    ts = list(range(100))
    grows, medians = sustained_growth(ts, [100 + t for t in ts], absolute=5)
    assert grows and medians[0] < medians[-1]
    # Flat with noise
    assert not sustained_growth(ts, [100 + (t % 7) for t in ts], absolute=5)[0]
    # A step up that stays: not rising every quarter
    assert not sustained_growth(ts, [100 if t < 30 else 140 for t in ts], absolute=5)[0]
    # Rising, but within the tolerance
    assert not sustained_growth(ts, [100 + t // 30 for t in ts], absolute=5)[0]
    assert sustained_growth([0, 1], [1, 2], absolute=0) == (False, None)


def test_plateau():
    ts = list(range(0, 3600, 5))
    # Grows for the TTL, then is cleaned as fast as it is written
    assert reaches_plateau(ts, [1000 + (t % 60) for t in ts])[0]
    assert not reaches_plateau(ts, [t / 2 for t in ts])[0]
    assert reaches_plateau(ts, [0 for _ in ts])[0]


def rows(minutes, rss=lambda t: 500.0, fds=lambda t: 100, threads=lambda t: 40,
         frs_shots=lambda t: 1000.0, lprs_shots=lambda t: 2000.0, errors=0, frames_per_s=100):
    out = []
    for t in range(0, minutes * 60 + 1, 5):
        out.append({
            "elapsed_s": float(t), "service_rss_mb": rss(t), "service_fds": fds(t), "service_threads": threads(t),
            "frs_screenshots_mb": frs_shots(t), "lprs_screenshots_mb": lprs_shots(t),
            "frs_frames_total": t * frames_per_s, "lprs_frames_total": t * frames_per_s,
            "errors_total": errors,
        })
    return out


def evaluate(series, **kwargs):
    options = dict(warmup_end=900, plateau_start=900, max_rss_growth=20, max_error_rate=0.001)
    options.update(kwargs)
    return {c.name: c for c in evaluate_soak(series, **options)}


def test_healthy_run_passes():
    checks = evaluate(rows(240, rss=lambda t: 500 + (t % 300) / 30))
    assert all(c.status == "pass" for c in checks.values()), checks
    assert passed(checks.values())


def test_rss_leak_fails():
    checks = evaluate(rows(240, rss=lambda t: 500 + 30 * t / 3600))
    assert checks["RSS growth"].failed
    assert "+30.0 MB/h" in checks["RSS growth"].detail


def test_rss_growth_during_warmup_ignored():
    checks = evaluate(rows(240, rss=lambda t: 200 + min(t, 900) / 3))
    assert checks["RSS growth"].status == "pass"


def test_fd_and_thread_leaks_fail():
    checks = evaluate(rows(240, fds=lambda t: 100 + t // 60, threads=lambda t: 40 + t // 600))
    assert checks["fd growth"].failed and checks["thread growth"].failed


def test_screenshots_without_plateau_fail():
    checks = evaluate(rows(240, lprs_shots=lambda t: t / 2))
    assert checks["LPRS screenshots plateau"].failed
    assert checks["FRS screenshots plateau"].status == "pass"


def test_error_rate():
    # 240 min * 60 s * 200 frames/s = 2.88 M frames; 0.1 % of that is 2880
    assert evaluate(rows(240, errors=2000))["error rate"].status == "pass"
    assert evaluate(rows(240, errors=5000))["error rate"].failed


def test_liveness_and_disk_guard():
    checks = evaluate(rows(60), failure=RunFailure("liveness: the service exited with code -11"))
    assert checks["liveness"].failed and not checks["disk guard"].failed
    checks = evaluate(rows(60), failure=RunFailure("disk guard: 4.2 GB free"))
    assert checks["disk guard"].failed and not checks["liveness"].failed
    assert not passed(checks.values())


def test_short_run_skips_trends():
    checks = evaluate(rows(2, rss=lambda t: 500 + t), warmup_end=60, plateau_start=60)
    for name in ("RSS growth", "fd growth", "thread growth", "FRS screenshots plateau", "LPRS screenshots plateau"):
        assert checks[name].status == "skip"
    assert passed(checks.values())


def test_no_samples():
    checks = evaluate([], warmup_end=float("inf"), plateau_start=float("inf"))
    assert checks["error rate"].status == "skip"
