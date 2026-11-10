"""Per-group fairness statistics of FRS recognition loops from a TRACE log.

Usage: python3 fairness.py <frs-log.txt>...

Analyses the FRS logs of a `stress.py ... --trace` run (the runner prints the command at its end; the logs are
in /tmp/test_falprs/logs/<point>/frs-log.txt). Odd stream numbers have a face in the frame, even ones don't,
as the camera emulator serves them at the default --face-ratio 0.5.
"""
import re
import statistics as st
import sys
from collections import defaultdict
from datetime import datetime

LINE = re.compile(r"timestamp=(\S+)\t.*?vstream_key = 1_tt_(\d+);(?:  (.*))?$")


def ts(text):
    return datetime.fromisoformat(text).timestamp()


def main(path):
    starts = defaultdict(list)
    ends = defaultdict(list)
    waits = defaultdict(list)
    first = last = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = LINE.search(line)
            if not m:
                continue
            t, n, text = ts(m[1]), int(m[2]), m[3] or ""
            if text.startswith("waited for a pipeline slot"):
                waits[n].append((t, float(text.split()[-1][:-2])))
            elif "End processPipeline" in line:
                ends[n].append(t)
            elif "Start processPipeline" in line:
                starts[n].append(t)
                first = t if first is None else min(first, t)
                last = t if last is None else max(last, t)
    # Steady state: all streams added and running, minus the edges
    begin = max(s[0] for s in starts.values()) + 10
    finish = last - 5
    print(f"{path}: streams {len(starts)}, window {finish - begin:.1f}s")

    groups = {"face": lambda n: n % 2 == 1, "no face": lambda n: n % 2 == 0}
    for name, member in groups.items():
        periods, wait_ms, pipeline_ms, rates = [], [], [], []
        frames = 0
        for n in starts:
            if not member(n):
                continue
            s = [t for t in starts[n] if begin <= t <= finish]
            frames += len(s)
            rates.append(len(s) / (finish - begin))
            periods += [b - a for a, b in zip(s, s[1:])]
            wait_ms += [w for t, w in waits[n] if begin <= t <= finish]
            e = sorted(ends[n])
            for a in s:
                after = next((x for x in e if x >= a), None)
                if after is not None and after - a < 30:
                    pipeline_ms.append((after - a) * 1000)
        q = lambda xs, p: st.quantiles(xs, n=100)[p - 1]
        print(
            f"  {name:8s} frames/s {frames / (finish - begin):6.1f}"
            f" | per stream fps mean {st.mean(rates):.3f} min {min(rates):.3f} max {max(rates):.3f}"
            f" | period ms median {st.median(periods) * 1000:.0f} p95 {q(periods, 95) * 1000:.0f}"
            f" | slot wait ms median {st.median(wait_ms):.0f} p95 {q(wait_ms, 95):.0f}"
            f" | pipeline ms median {st.median(pipeline_ms):.0f} p95 {q(pipeline_ms, 95):.0f}"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        main(p)
