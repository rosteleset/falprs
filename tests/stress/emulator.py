"""Camera emulator for the stress tests.

An aiohttp server that plays the cameras and the callback receivers of the streams under test:

- ``GET /{subsystem}/{stream_id}.jpg`` answers with the stream's frame after the camera latency;
- ``POST /callback/{subsystem}`` answers 200 at once;
- ``GET /stats`` returns the counters of served frames and received callbacks per subsystem,
  the emulator's own CPU time and its settings, as JSON.

The frame depends on the subsystem and the stream number: streams with a target (a face for FRS,
a plate for LPRS) get the target image, the others an image without one. At a ratio of 0.5 the odd
stream numbers are exactly the ones with a target, which ``fairness.py`` relies on.
"""

import asyncio
import math
import os
import re
import resource
import time
from fractions import Fraction
from pathlib import Path

from aiohttp import web

IMAGES_DIR = Path(__file__).resolve().parent.parent / "images"
SUBSYSTEMS = ("frs", "lprs")
# Subsystem -> (image with a target, image without one)
IMAGES = {
    "frs": ("einstein_003.jpg", "no_faces.jpg"),
    "lprs": ("test_rain_007.jpg", "no_faces.jpg"),
}

_STREAM_NUMBER = re.compile(r"(\d+)$")


def stream_number(stream_id):
    """The number at the end of a stream id: ``tt_007`` -> 7; None without one."""
    m = _STREAM_NUMBER.search(stream_id)
    return int(m[1]) if m else None


def _picked(k, share):
    """Whether the k-th (1-based) item of a sequence is picked when a `share` of it is, spread evenly."""
    return math.floor(k * share) > math.floor((k - 1) * share)


def has_target(number, ratio):
    """Whether stream `number` gets a frame with a target when a `ratio` of the streams have one.

    Odd streams get targets first: up to 0.5 the targets are spread over the odd streams only,
    above 0.5 all odd streams have one and the rest is spread over the even ones.
    """
    ratio = Fraction(str(ratio))
    if not 0 <= ratio <= 1:
        raise ValueError(f"ratio must be within [0, 1], not {ratio}")
    if number % 2 == 1:
        return _picked((number + 1) // 2, min(ratio * 2, Fraction(1)))
    return _picked(number // 2, max(ratio * 2 - 1, Fraction(0)))


def frame_name(subsystem, number, ratio):
    """The image file served to stream `number` of `subsystem`."""
    with_target, without_target = IMAGES[subsystem]
    return with_target if has_target(number, ratio) else without_target


class Emulator:
    """The emulator's state: settings, loaded images and counters."""

    def __init__(self, latency=0.25, face_ratio=0.5, plate_ratio=0.5, images_dir=IMAGES_DIR):
        if latency < 0:
            raise ValueError("the camera latency cannot be negative")
        self.latency = latency
        self.ratios = {"frs": face_ratio, "lprs": plate_ratio}
        for ratio in self.ratios.values():
            has_target(1, ratio)  # validates the ratio
        names = {name for pair in IMAGES.values() for name in pair}
        self.images = {name: (Path(images_dir) / name).read_bytes() for name in names}
        self.frames = dict.fromkeys(SUBSYSTEMS, 0)
        self.callbacks = dict.fromkeys(SUBSYSTEMS, 0)
        self.bad_requests = 0
        self.started = time.monotonic()
        # Frame choice per (subsystem, stream number): the Fraction arithmetic is not free
        self._frame_cache = {}

    def frame(self, subsystem, number):
        key = (subsystem, number)
        name = self._frame_cache.get(key)
        if name is None:
            name = self._frame_cache[key] = frame_name(subsystem, number, self.ratios[subsystem])
        return self.images[name]

    def stats(self):
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return {
            "frames": dict(self.frames),
            "callbacks": dict(self.callbacks),
            "bad_requests": self.bad_requests,
            "cpu_seconds": usage.ru_utime + usage.ru_stime,
            "uptime": time.monotonic() - self.started,
            "pid": os.getpid(),
            "latency": self.latency,
            "face_ratio": self.ratios["frs"],
            "plate_ratio": self.ratios["lprs"],
        }

    async def handle_frame(self, request):
        subsystem = request.match_info["subsystem"]
        number = stream_number(request.match_info["stream_id"])
        if subsystem not in IMAGES or number is None:
            self.bad_requests += 1
            raise web.HTTPNotFound()
        if self.latency:
            await asyncio.sleep(self.latency)
        self.frames[subsystem] += 1
        return web.Response(body=self.frame(subsystem, number), content_type="image/jpeg")

    async def handle_callback(self, request):
        subsystem = request.match_info["subsystem"]
        if subsystem not in IMAGES:
            self.bad_requests += 1
            raise web.HTTPNotFound()
        # The body is read so that the connection can be reused, its content is not checked
        await request.read()
        self.callbacks[subsystem] += 1
        return web.Response(status=200)

    async def handle_stats(self, request):
        return web.json_response(self.stats())

    def app(self):
        app = web.Application()
        app.add_routes([
            web.get("/stats", self.handle_stats),
            web.get(r"/{subsystem}/{stream_id}.jpg", self.handle_frame),
            web.post("/callback/{subsystem}", self.handle_callback),
        ])
        return app


def frame_url(base_url, subsystem, stream_id):
    return f"{base_url.rstrip('/')}/{subsystem}/{stream_id}.jpg"


def callback_url(base_url, subsystem):
    return f"{base_url.rstrip('/')}/callback/{subsystem}"


def serve(host, port, latency, face_ratio, plate_ratio):
    """Runs the emulator until it is interrupted or terminated."""
    emulator = Emulator(latency, face_ratio, plate_ratio)
    print(f"Camera emulator on http://{host}:{port}: latency {latency * 1000:.0f} ms, "
          f"face ratio {face_ratio}, plate ratio {plate_ratio}", flush=True)
    # The access log costs CPU on every frame
    web.run_app(emulator.app(), host=host, port=port, access_log=None, print=None,
                handle_signals=True, backlog=1024)
