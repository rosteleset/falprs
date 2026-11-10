"""Tests of the camera emulator, with an aiohttp test client."""

import asyncio
import time

import pytest
from aiohttp.test_utils import TestClient, TestServer

import emulator
from emulator import IMAGES_DIR, Emulator, frame_name, has_target, stream_number


def test_stream_number():
    assert stream_number("tt_001") == 1
    assert stream_number("tt_1234") == 1234
    assert stream_number("tt_") is None


def test_half_ratio_is_odd_streams():
    assert [has_target(n, 0.5) for n in range(1, 9)] == [True, False] * 4
    assert frame_name("frs", 1, 0.5) == "einstein_003.jpg"
    assert frame_name("frs", 2, 0.5) == "no_faces.jpg"
    assert frame_name("lprs", 3, 0.5) == "test_rain_007.jpg"
    assert frame_name("lprs", 4, 0.5) == "no_faces.jpg"


@pytest.mark.parametrize("ratio", [0, 0.1, 0.25, 0.3, 0.5, 0.7, 0.75, 0.9, 1])
def test_ratio_share(ratio):
    for count in (40, 100, 400):
        hits = sum(has_target(n, ratio) for n in range(1, count + 1))
        assert abs(hits - ratio * count) <= 1


def test_low_ratio_uses_odd_streams_only():
    assert not any(has_target(n, 0.3) for n in range(2, 401, 2))


def test_high_ratio_keeps_all_odd_streams():
    assert all(has_target(n, 0.8) for n in range(1, 401, 2))


def test_invalid_ratio():
    with pytest.raises(ValueError):
        has_target(1, 1.5)


def run(coroutine):
    return asyncio.run(coroutine)


async def with_client(state, body):
    async with TestClient(TestServer(state.app())) as client:
        return await body(client)


def test_frames_per_stream():
    state = Emulator(latency=0)

    async def body(client):
        images = {}
        for subsystem, stream in (("frs", "tt_001"), ("frs", "tt_002"), ("lprs", "tt_001"), ("lprs", "tt_002")):
            response = await client.get(f"/{subsystem}/{stream}.jpg")
            assert response.status == 200
            assert response.content_type == "image/jpeg"
            images[subsystem, stream] = await response.read()
        return images

    images = run(with_client(state, body))
    read = lambda name: (IMAGES_DIR / name).read_bytes()
    assert images["frs", "tt_001"] == read("einstein_003.jpg")
    assert images["frs", "tt_002"] == read("no_faces.jpg")
    assert images["lprs", "tt_001"] == read("test_rain_007.jpg")
    assert images["lprs", "tt_002"] == read("no_faces.jpg")


def test_face_ratio_applies():
    state = Emulator(latency=0, face_ratio=0, plate_ratio=1)

    async def body(client):
        frs = await (await client.get("/frs/tt_001.jpg")).read()
        lprs = await (await client.get("/lprs/tt_002.jpg")).read()
        return frs, lprs

    frs, lprs = run(with_client(state, body))
    assert frs == (IMAGES_DIR / "no_faces.jpg").read_bytes()
    assert lprs == (IMAGES_DIR / "test_rain_007.jpg").read_bytes()


def test_latency_respected():
    state = Emulator(latency=0.3)

    async def body(client):
        started = time.monotonic()
        responses = await asyncio.gather(*(client.get(f"/frs/tt_{n:03d}.jpg") for n in range(1, 11)))
        elapsed = time.monotonic() - started
        assert all(r.status == 200 for r in responses)
        await asyncio.gather(*(r.read() for r in responses))
        return elapsed

    elapsed = run(with_client(state, body))
    # Concurrent requests wait in parallel: the latency is per request, not a queue
    assert 0.3 <= elapsed < 0.9


def test_zero_latency():
    state = Emulator(latency=0)

    async def body(client):
        started = time.monotonic()
        await (await client.get("/frs/tt_001.jpg")).read()
        return time.monotonic() - started

    assert run(with_client(state, body)) < 0.2


def test_unknown_paths():
    state = Emulator(latency=0)

    async def body(client):
        assert (await client.get("/xyz/tt_001.jpg")).status == 404
        assert (await client.get("/frs/camera.jpg")).status == 404
        assert (await client.post("/callback/xyz", json={})).status == 404
        return (await (await client.get("/stats")).json())

    stats = run(with_client(state, body))
    assert stats["bad_requests"] == 3
    assert stats["frames"] == {"frs": 0, "lprs": 0}


def test_callbacks_and_counters():
    state = Emulator(latency=0, face_ratio=0.25, plate_ratio=0.75)

    async def body(client):
        # The bodies are read: an unread one holds its connection and the server's shutdown waits for it
        for n in range(1, 4):
            await (await client.get(f"/frs/tt_{n:03d}.jpg")).read()
        await (await client.get("/lprs/tt_001.jpg")).read()
        for _ in range(2):
            response = await client.post("/callback/lprs", json={"plates": []})
            assert response.status == 200
        response = await client.post("/callback/frs", json={"faceId": 1})
        assert response.status == 200
        return await (await client.get("/stats")).json()

    stats = run(with_client(state, body))
    assert stats["frames"] == {"frs": 3, "lprs": 1}
    assert stats["callbacks"] == {"frs": 1, "lprs": 2}
    assert stats["latency"] == 0
    assert stats["face_ratio"] == 0.25
    assert stats["plate_ratio"] == 0.75
    assert stats["cpu_seconds"] > 0
    assert stats["uptime"] > 0


def test_urls():
    assert emulator.frame_url("http://h:9080/", "frs", "tt_001") == "http://h:9080/frs/tt_001.jpg"
    assert emulator.callback_url("http://h:9080", "lprs") == "http://h:9080/callback/lprs"
