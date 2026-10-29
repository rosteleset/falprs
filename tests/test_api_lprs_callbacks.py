import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import requests
from PIL import Image

FALPRS_URL = "http://localhost:9071"
API_URL = FALPRS_URL + "/lprs/api/"
STREAM_ID = "streamId"
CONFIG = "config"
LOGS_LEVEL = "logs-level"
DATE = "date"
EVENT_ID = "eventId"
PLATES = "plates"
HAS_SPECIAL = "hasSpecial"
TYPE = "type"
NUMBER = "number"

CONF_SCREENSHOT_URL = "screenshot-url"
CONF_CALLBACK_URL = "callback-url"
CONF_DELAY_BETWEEN_FRAMES = "delay-between-frames"
CONF_BAN_DURATION = "ban-duration"
CONF_BAN_DURATION_AREA = "ban-duration-area"

TYPE_RU_1 = "ru_1"
PLATE_NUMBER = "O588OA68"

# Test-side server: serves screenshots chosen by the test and records callbacks
RECEIVER_HOST = "localhost"
RECEIVER_PORT = 9073
RECEIVER_URL = f"http://{RECEIVER_HOST}:{RECEIVER_PORT}"

# Must match ban-maintenance-interval in test_config.yaml
BAN_MAINTENANCE_INTERVAL = 5
# Must match update-interval of lprs-vstreams-config-pg-cache in test_config.yaml
CONFIG_CACHE_UPDATE_INTERVAL = 5

IMAGES_DIR = Path(__file__).parent / "images"


def encode_jpeg(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def load_images():
    original = Image.open(IMAGES_DIR / "test_001.jpg").convert("RGB")
    # The same frame with all content moved to the right, so the plate changes its position
    shift = original.width // 3
    shifted = Image.new("RGB", (original.width + shift, original.height), (128, 128, 128))
    shifted.paste(original, (shift, 0))
    blank = Image.new("RGB", original.size, (128, 128, 128))
    return {
        "plate": encode_jpeg(original),
        "plate_shifted": encode_jpeg(shifted),
        "blank": encode_jpeg(blank),
    }


class Receiver:
    def __init__(self):
        self.lock = threading.Lock()
        self.screenshots = {}
        self.callbacks = {}

    def show(self, stream_id, image):
        with self.lock:
            self.screenshots[stream_id] = image

    def screenshot(self, stream_id):
        with self.lock:
            return self.screenshots.get(stream_id)

    def add_callback(self, stream_id, payload):
        with self.lock:
            self.callbacks.setdefault(stream_id, []).append(payload)

    def received(self, stream_id):
        with self.lock:
            return list(self.callbacks.get(stream_id, []))


def make_handler(receiver):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            # /screenshot/<streamId>
            parts = self.path.strip("/").split("/")
            image = receiver.screenshot(parts[1]) if len(parts) == 2 and parts[0] == "screenshot" else None
            if image is None:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(image)))
            self.end_headers()
            self.wfile.write(image)

        def do_POST(self):
            # /callback/<streamId>
            parts = self.path.strip("/").split("/")
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if len(parts) == 2 and parts[0] == "callback":
                receiver.add_callback(parts[1], json.loads(body))
            self.send_response(204)
            self.end_headers()

        def log_message(self, format, *args):
            pass

    return Handler


@pytest.fixture(scope="module")
def images():
    return load_images()


@pytest.fixture(scope="module")
def receiver():
    receiver = Receiver()
    server = ThreadingHTTPServer((RECEIVER_HOST, RECEIVER_PORT), make_handler(receiver))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield receiver
    server.shutdown()
    server.server_close()


def add_stream(stream_id, ban_duration, ban_duration_area):
    config = {
        CONF_SCREENSHOT_URL: f"{RECEIVER_URL}/screenshot/{stream_id}",
        CONF_CALLBACK_URL: f"{RECEIVER_URL}/callback/{stream_id}",
        CONF_DELAY_BETWEEN_FRAMES: "300ms",
        CONF_BAN_DURATION: ban_duration,
        CONF_BAN_DURATION_AREA: ban_duration_area,
        LOGS_LEVEL: "trace",
    }
    response = requests.post(API_URL + "addStream", json={STREAM_ID: stream_id, CONFIG: config})
    assert response.status_code == 204
    # startWorkflow ignores a video stream until the config cache picks it up
    time.sleep(CONFIG_CACHE_UPDATE_INTERVAL + 1)


def start_workflow(stream_id):
    response = requests.post(API_URL + "startWorkflow", json={STREAM_ID: stream_id})
    assert response.status_code == 204


def stop_workflow(stream_id):
    response = requests.post(API_URL + "stopWorkflow", json={STREAM_ID: stream_id})
    assert response.status_code == 204


def remove_stream(stream_id):
    response = requests.post(API_URL + "removeStream", json={STREAM_ID: stream_id})
    assert response.status_code == 204


@pytest.fixture
def stream(request, receiver):
    # Each test uses its own video stream, so plate bans of different tests never interfere
    stream_id = "cb_" + request.node.name
    yield stream_id
    stop_workflow(stream_id)
    remove_stream(stream_id)


def wait_for_callbacks(receiver, stream_id, count, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(receiver.received(stream_id)) >= count:
            return receiver.received(stream_id)
        time.sleep(0.1)
    return receiver.received(stream_id)


def plate_numbers(callback):
    return [plate[NUMBER] for plate in callback.get(PLATES, [])]


# Callback payload for a recognised plate
def test_callback_payload(receiver, images, stream):
    receiver.show(stream, images["plate"])
    add_stream(stream, "60s", "60s")
    start_workflow(stream)

    callbacks = wait_for_callbacks(receiver, stream, 1)
    assert len(callbacks) >= 1
    callback = callbacks[0]
    assert callback[STREAM_ID] == stream
    assert isinstance(callback[DATE], str)
    assert isinstance(callback[EVENT_ID], int)
    assert callback[HAS_SPECIAL] is False
    assert callback[PLATES] == [{TYPE: TYPE_RU_1, NUMBER: PLATE_NUMBER}]


# A plate in a full ban is not reported again
def test_full_ban(receiver, images, stream):
    receiver.show(stream, images["plate"])
    add_stream(stream, "60s", "60s")
    start_workflow(stream)

    assert len(wait_for_callbacks(receiver, stream, 1)) == 1
    time.sleep(4)
    assert len(receiver.received(stream)) == 1


# Seeing a plate during its full ban extends the full ban, so a later move is still ignored
def test_full_ban_extended(receiver, images, stream):
    receiver.show(stream, images["plate"])
    add_stream(stream, "2s", "60s")
    start_workflow(stream)

    assert len(wait_for_callbacks(receiver, stream, 1)) == 1
    # Without the extension the full ban would have ended after 2 seconds
    time.sleep(4)
    receiver.show(stream, images["plate_shifted"])
    time.sleep(3)
    assert len(receiver.received(stream)) == 1


# After the full ban the plate is in an area ban: ignored in place, reported again once it moves
def test_area_ban(receiver, images, stream):
    receiver.show(stream, images["plate"])
    add_stream(stream, "2s", "60s")
    start_workflow(stream)

    assert len(wait_for_callbacks(receiver, stream, 1)) == 1
    # Let the full ban end without extending it
    receiver.show(stream, images["blank"])
    time.sleep(3)
    receiver.show(stream, images["plate"])
    time.sleep(3)
    assert len(receiver.received(stream)) == 1

    receiver.show(stream, images["plate_shifted"])
    callbacks = wait_for_callbacks(receiver, stream, 2)
    assert len(callbacks) == 2
    assert plate_numbers(callbacks[1]) == [PLATE_NUMBER]


# Once the area ban ends, the plate is reported again even in the same position
def test_area_ban_expires(receiver, images, stream):
    receiver.show(stream, images["plate"])
    add_stream(stream, "1s", "2s")
    start_workflow(stream)

    assert len(wait_for_callbacks(receiver, stream, 1)) == 1
    # The ban list is cleaned up only by the periodic ban maintenance
    receiver.show(stream, images["blank"])
    time.sleep(2 + BAN_MAINTENANCE_INTERVAL + 1)
    receiver.show(stream, images["plate"])
    callbacks = wait_for_callbacks(receiver, stream, 2)
    assert len(callbacks) == 2
    assert plate_numbers(callbacks[1]) == [PLATE_NUMBER]


# A zero ban duration disables plate bans
def test_ban_disabled(receiver, images, stream):
    receiver.show(stream, images["plate"])
    add_stream(stream, "0s", "60s")
    start_workflow(stream)

    callbacks = wait_for_callbacks(receiver, stream, 2)
    assert len(callbacks) >= 2
    assert all(plate_numbers(callback) == [PLATE_NUMBER] for callback in callbacks)
