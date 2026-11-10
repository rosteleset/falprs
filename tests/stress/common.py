"""Shared helpers of the stress runner."""

import re


class EnvironmentFailure(Exception):
    """The run could not be carried out: no database, no binary, the emulator or the service did not come up.
    Exit code 2."""


class RunFailure(Exception):
    """The run was stopped because a criterion failed (the service died, the disk guard tripped). Exit code 1."""


_DURATION = re.compile(r"^(\d+(?:\.\d+)?)(ms|s|m|h|d)?$")
_UNITS = {"ms": 0.001, "s": 1, None: 1, "m": 60, "h": 3600, "d": 86400}


def parse_duration(text):
    """Seconds in a duration such as `250ms`, `5`, `2m`, `4h` or `1d`; None if it is not one."""
    m = _DURATION.match(str(text).strip())
    if not m:
        return None
    return float(m[1]) * _UNITS[m[2]]


def duration_arg(text):
    """argparse type for durations."""
    seconds = parse_duration(text)
    if seconds is None:
        import argparse
        raise argparse.ArgumentTypeError(f"invalid duration: {text} (use e.g. 250ms, 30s, 5m, 4h)")
    return seconds


def format_duration(seconds):
    if seconds >= 3600 and seconds % 3600 == 0:
        return f"{seconds / 3600:g}h"
    if seconds >= 60 and seconds % 60 == 0:
        return f"{seconds / 60:g}m"
    if seconds < 1:
        return f"{seconds * 1000:g}ms"
    return f"{seconds:g}s"
