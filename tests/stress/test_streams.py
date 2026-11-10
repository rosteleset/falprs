"""Tests of the stream ids, API bodies and churn windows."""

import streams
from streams import add_body, start_call, stop_call, stream_ids, window

EMU = "http://127.0.0.1:9080"


def test_stream_ids():
    assert stream_ids(3) == ["tt_001", "tt_002", "tt_003"]
    assert stream_ids(2, first=999) == ["tt_999", "tt_1000"]
    assert stream_ids(0) == []


def test_frs_add_body():
    assert add_body("frs", "tt_001", EMU) == {
        "streamId": "tt_001",
        "url": f"{EMU}/frs/tt_001.jpg",
        "callback": f"{EMU}/callback/frs",
    }
    assert add_body("frs", "tt_001", EMU, trace=True)["config"] == {"logs-level": "trace"}


def test_lprs_add_body():
    body = add_body("lprs", "tt_002", EMU)
    assert body == {"streamId": "tt_002", "config": {
        "screenshot-url": f"{EMU}/lprs/tt_002.jpg",
        "callback-url": f"{EMU}/callback/lprs",
        "ban-duration": "0s",
    }}
    assert add_body("lprs", "tt_002", EMU, trace=True)["config"]["logs-level"] == "trace"


def test_start_stop_calls():
    assert start_call("frs", "tt_001") == ("motionDetection", {"streamId": "tt_001", "start": True})
    assert stop_call("frs", "tt_001") == ("motionDetection", {"streamId": "tt_001", "start": False})
    assert start_call("lprs", "tt_001") == ("startWorkflow", {"streamId": "tt_001"})
    assert stop_call("lprs", "tt_001") == ("stopWorkflow", {"streamId": "tt_001"})


def test_batches():
    assert streams.batches(list(range(5)), 2) == [[0, 1], [2, 3], [4]]
    assert streams.batches([], 50) == []


def test_window_walks_around():
    ids = stream_ids(20)
    first, second = window(ids, 0.1, 0), window(ids, 0.1, 1)
    assert first == ["tt_001", "tt_002"] and second == ["tt_003", "tt_004"]
    assert window(ids, 0.1, 10) == first
    assert len(window(ids, 0.25, 3)) == 5
    assert window(stream_ids(3), 0.1, 0) == ["tt_001"]
    assert window(ids, 0, 0) == [] and window([], 0.1, 0) == []
