"""The streams under test: their ids, API bodies and paced setup through the service API."""

import asyncio
import time

import aiohttp

from emulator import callback_url, frame_url

SUBSYSTEMS = ("frs", "lprs")
# The service picks new streams up on its next streams cache update: 5 s in the template, plus a margin
CACHE_WAIT = 7.0


class ApiError(Exception):
    pass


def stream_ids(count, first=1):
    """`tt_001`, `tt_002`, ...: the ids of `count` streams of a subsystem."""
    return [f"tt_{n:03d}" for n in range(first, first + count)]


def add_body(subsystem, stream_id, emulator_url, trace=False):
    """The `addStream` body of a stream whose frames and callbacks go to the emulator."""
    if subsystem == "frs":
        body = {
            "streamId": stream_id,
            "url": frame_url(emulator_url, "frs", stream_id),
            "callback": callback_url(emulator_url, "frs"),
        }
        if trace:
            body["config"] = {"logs-level": "trace"}
        return body
    config = {
        "screenshot-url": frame_url(emulator_url, "lprs", stream_id),
        "callback-url": callback_url(emulator_url, "lprs"),
        # No plate and special-vehicle bans: every frame with a plate yields an event, a screenshot and a callback
        "ban-duration": "0s",
    }
    if trace:
        config["logs-level"] = "trace"
    return {"streamId": stream_id, "config": config}


def start_call(subsystem, stream_id):
    """(API method, body) that starts processing a stream."""
    if subsystem == "frs":
        return "motionDetection", {"streamId": stream_id, "start": True}
    return "startWorkflow", {"streamId": stream_id}


def stop_call(subsystem, stream_id):
    """(API method, body) that stops processing a stream."""
    if subsystem == "frs":
        return "motionDetection", {"streamId": stream_id, "start": False}
    return "stopWorkflow", {"streamId": stream_id}


def batches(items, size):
    size = max(1, int(size))
    return [items[i:i + size] for i in range(0, len(items), size)]


def window(ids, share, round_number):
    """The `share` of `ids` taken in round `round_number`: consecutive windows that walk around the list,
    so that successive rounds touch different streams."""
    if not ids or share <= 0:
        return []
    size = max(1, round(len(ids) * share))
    start = (round_number * size) % len(ids)
    return [ids[(start + i) % len(ids)] for i in range(min(size, len(ids)))]


class Api:
    """Calls of the service API."""

    def __init__(self, session, base_url):
        self.session = session
        self.base_url = base_url.rstrip("/")

    async def call(self, subsystem, method, body):
        url = f"{self.base_url}/{subsystem}/api/{method}"
        try:
            async with self.session.post(url, json=body, timeout=aiohttp.ClientTimeout(total=30)) as response:
                text = await response.text()
                if response.status not in (200, 204):
                    raise ApiError(f"{subsystem} {method} {body.get('streamId', '')}: HTTP {response.status} {text}")
                return text
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            raise ApiError(f"{subsystem} {method} {body.get('streamId', '')}: {e!r}") from e


class Streams:
    """The streams of one subsystem."""

    def __init__(self, api, subsystem, count, emulator_url, trace=False):
        self.api = api
        self.subsystem = subsystem
        self.ids = stream_ids(count)
        self.emulator_url = emulator_url
        self.trace = trace

    async def add(self, stream_id):
        await self.api.call(self.subsystem, "addStream", add_body(self.subsystem, stream_id, self.emulator_url, self.trace))

    async def remove(self, stream_id):
        await self.api.call(self.subsystem, "removeStream", {"streamId": stream_id})

    async def start(self, stream_id):
        await self.api.call(self.subsystem, *start_call(self.subsystem, stream_id))

    async def stop(self, stream_id):
        await self.api.call(self.subsystem, *stop_call(self.subsystem, stream_id))


async def paced(calls, rate):
    """Awaits `calls` (coroutine factories) in batches of `rate` per second. Returns the monotonic time
    the last batch finished."""
    for batch in batches(calls, rate):
        started = time.monotonic()
        await asyncio.gather(*(call() for call in batch))
        await asyncio.sleep(max(0.0, 1.0 - (time.monotonic() - started)))
    return time.monotonic()


async def set_up(groups, rate, cache_wait=CACHE_WAIT, log=print):
    """Adds the streams of all `groups`, waits for the service's caches, then starts them, all at `rate` per second.
    Returns the monotonic time the last stream was started: the warm-up counts from there."""
    adds = [(lambda s=s, i=i: s.add(i)) for s in groups for i in s.ids]
    log(f"Adding {len(adds)} streams at {rate}/s")
    await paced(adds, rate)
    await asyncio.sleep(cache_wait)
    starts = [(lambda s=s, i=i: s.start(i)) for s in groups for i in s.ids]
    log(f"Starting {len(starts)} streams at {rate}/s")
    return await paced(starts, rate)


async def churn(groups, share, toggle_share, round_number, rate, cache_wait=CACHE_WAIT, toggle_pause=2.0):
    """One churn round: removes and re-adds a `share` of the streams of each group, and stops and restarts
    processing of another `toggle_share` of them (motion for FRS, the workflow for LPRS)."""
    readd = {s: window(s.ids, share, round_number) for s in groups}
    taken = {s: set(ids) for s, ids in readd.items()}
    # The toggled streams start half a list away from the re-added ones
    toggle = {}
    for s in groups:
        shifted = s.ids[len(s.ids) // 2:] + s.ids[:len(s.ids) // 2]
        toggle[s] = [i for i in window(shifted, toggle_share, round_number) if i not in taken[s]]

    await paced([(lambda s=s, i=i: s.remove(i)) for s, ids in readd.items() for i in ids], rate)
    await paced([(lambda s=s, i=i: s.add(i)) for s, ids in readd.items() for i in ids], rate)
    await paced([(lambda s=s, i=i: s.stop(i)) for s, ids in toggle.items() for i in ids], rate)
    await asyncio.sleep(max(cache_wait, toggle_pause))
    await paced([(lambda s=s, i=i: s.start(i)) for s, ids in toggle.items() for i in ids]
                + [(lambda s=s, i=i: s.start(i)) for s, ids in readd.items() for i in ids], rate)
    return sum(map(len, readd.values())), sum(map(len, toggle.values()))
