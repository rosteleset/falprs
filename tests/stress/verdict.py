"""Pure evaluation of the sampled series: window means for sweep, the pass/fail criteria of soak."""

from dataclasses import dataclass

# Shorter windows give no meaningful trend: the criteria on them are skipped
MIN_TREND_WINDOW_S = 600.0


@dataclass
class Check:
    name: str
    status: str  # "pass", "fail" or "skip"
    detail: str

    @property
    def failed(self):
        return self.status == "fail"


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def linear_slope(xs, ys):
    """Least-squares slope of ys over xs; None with fewer than two distinct points."""
    points = [(x, y) for x, y in zip(xs, ys) if y is not None]
    if len(points) < 2:
        return None
    mx = sum(x for x, _ in points) / len(points)
    my = sum(y for _, y in points) / len(points)
    sxx = sum((x - mx) ** 2 for x, _ in points)
    if sxx == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in points) / sxx


def median(values):
    values = sorted(values)
    n = len(values)
    if not n:
        return None
    return values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2


def quarter_medians(ts, values):
    """Medians of the values in the four equal time quarters of the series."""
    points = [(t, v) for t, v in zip(ts, values) if v is not None]
    if len(points) < 8:
        return None
    t0, t1 = points[0][0], points[-1][0]
    span = (t1 - t0) / 4 or 1.0
    quarters = [[] for _ in range(4)]
    for t, v in points:
        quarters[min(3, int((t - t0) / span))].append(v)
    if not all(quarters):
        return None
    return [median(q) for q in quarters]


def sustained_growth(ts, values, absolute, relative=0.05):
    """Whether a series keeps growing: the medians of its four time quarters rise every quarter and the last
    exceeds the first by more than max(`absolute`, `relative` of the first). Returns (grows, medians)."""
    medians = quarter_medians(ts, values)
    if medians is None:
        return False, None
    rising = all(b > a for a, b in zip(medians, medians[1:]))
    return rising and medians[-1] - medians[0] > max(absolute, relative * medians[0]), medians


def reaches_plateau(ts, values, relative=0.1, absolute=20.0):
    """Whether a series levels off: its fitted growth over the window stays within max(`absolute`,
    `relative` of its mean). Returns (plateau, growth over the window)."""
    slope = linear_slope(ts, values)
    if slope is None:
        return True, 0.0
    growth = slope * (ts[-1] - ts[0])
    level = mean(values) or 0.0
    return growth <= max(absolute, relative * level), growth


def window(rows, begin, end=None, key="elapsed_s"):
    return [r for r in rows if r[key] >= begin and (end is None or r[key] <= end)]


def series(rows, column, key="elapsed_s"):
    return [r[key] for r in rows], [r.get(column) for r in rows]


def window_means(rows, columns):
    return {c: mean(r.get(c) for r in rows) for c in columns}


def evaluate_soak(rows, *, warmup_end, plateau_start, max_rss_growth, max_error_rate, failure=None):
    """The soak criteria over the sampled `rows` (dicts with `elapsed_s` and the metrics columns).

    `warmup_end` and `plateau_start` are elapsed seconds; `max_rss_growth` is MB/h; `max_error_rate` a fraction
    of the frames served; `failure` the RunFailure that stopped the run, if any (liveness or disk guard).
    """
    checks = []
    message = str(failure) if failure else ""
    checks.append(Check("liveness", "fail" if message.startswith("liveness") else "pass",
                        message if message.startswith("liveness") else "the service ran and answered /ping"))
    checks.append(Check("disk guard", "fail" if message.startswith("disk guard") else "pass",
                        message if message.startswith("disk guard") else "free space stayed above the limit"))

    after = window(rows, warmup_end)
    span = after[-1]["elapsed_s"] - after[0]["elapsed_s"] if len(after) > 1 else 0.0
    if span < MIN_TREND_WINDOW_S:
        skip = f"the window after warm-up is {span / 60:.1f} min, shorter than {MIN_TREND_WINDOW_S / 60:.0f} min"
        checks += [Check("RSS growth", "skip", skip), Check("fd growth", "skip", skip),
                   Check("thread growth", "skip", skip)]
    else:
        ts, rss = series(after, "service_rss_mb")
        slope = linear_slope([t / 3600 for t in ts], rss)
        if slope is None:
            checks.append(Check("RSS growth", "skip", "no RSS samples"))
        else:
            checks.append(Check("RSS growth", "fail" if slope > max_rss_growth else "pass",
                                f"{slope:+.1f} MB/h (limit {max_rss_growth:g} MB/h)"))
        for name, column, absolute in (("fd growth", "service_fds", 5), ("thread growth", "service_threads", 2)):
            grows, medians = sustained_growth(*series(after, column), absolute)
            detail = ("quarter medians " + " -> ".join(f"{m:g}" for m in medians)) if medians else "too few samples"
            checks.append(Check(name, "fail" if grows else "pass" if medians else "skip", detail))

    plateau_rows = window(rows, plateau_start)
    plateau_span = plateau_rows[-1]["elapsed_s"] - plateau_rows[0]["elapsed_s"] if len(plateau_rows) > 1 else 0.0
    for subsystem in ("frs", "lprs"):
        name = f"{subsystem.upper()} screenshots plateau"
        if plateau_span < MIN_TREND_WINDOW_S:
            checks.append(Check(name, "skip", f"the window after the first TTL + cleanup cycle is "
                                              f"{plateau_span / 60:.1f} min, shorter than "
                                              f"{MIN_TREND_WINDOW_S / 60:.0f} min"))
            continue
        ts, size = series(plateau_rows, f"{subsystem}_screenshots_mb")
        flat, growth = reaches_plateau(ts, size)
        checks.append(Check(name, "pass" if flat else "fail",
                            f"{growth:+.1f} MB over the last {plateau_span / 60:.0f} min, mean {mean(size) or 0:.0f} MB"))

    if rows:
        last = rows[-1]
        frames = (last.get("frs_frames_total") or 0) + (last.get("lprs_frames_total") or 0)
        errors = last.get("errors_total") or 0
        error_rate = errors / frames if frames else (1.0 if errors else 0.0)
        checks.append(Check("error rate", "fail" if error_rate > max_error_rate else "pass",
                            f"{errors} ERROR lines over {frames} frames = {error_rate:.4%} "
                            f"(limit {max_error_rate:.4%})"))
    else:
        checks.append(Check("error rate", "skip", "no samples"))
    return checks


def passed(checks):
    return not any(c.failed for c in checks)
