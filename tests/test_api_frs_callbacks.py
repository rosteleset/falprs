import io
import json
import threading
import time
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import requests
from PIL import Image

FALPRS_URL = "http://localhost:9071"
API_URL = FALPRS_URL + "/frs/api/"
STREAM_ID = "streamId"
URL = "url"
CALLBACK = "callback"
CALLBACK_BARCODES = "callbackBarcodes"
BARCODES = "barcodes"
CONFIG = "config"
LOGS_LEVEL = "logs-level"
DATA = "data"
FACE_ID = "faceId"
FACES = "faces"
EVENT_ID = "eventId"
START = "start"
DURATION = "duration"
DATE = "date"
DATE_START = "dateStart"
DATE_END = "dateEnd"
SCREENSHOT = "screenshot"
GROUP_NAME = "groupName"
GROUP_ID = "groupId"
ACCESS_API_TOKEN = "accessApiToken"

CONF_DELAY_BETWEEN_FRAMES = "delay-between-frames"

# Test-side server: serves frames chosen by the test and records callbacks
RECEIVER_HOST = "localhost"
RECEIVER_PORT = 9074
RECEIVER_URL = f"http://{RECEIVER_HOST}:{RECEIVER_PORT}"

# Must cover update-interval of the FRS video stream, face and special group caches in test_config.yaml
CACHE_UPDATE_INTERVAL = 5

IMAGES_DIR = Path(__file__).parent / "images"


def load_images():
    names = ("einstein_001", "hanks_001", "no_faces", "qr_code_001")
    return {name: (IMAGES_DIR / f"{name}.jpg").read_bytes() for name in names}


class Receiver:
    def __init__(self):
        self.lock = threading.Lock()
        self.frames = {}
        self.callbacks = {}

    def show(self, name, image):
        with self.lock:
            self.frames[name] = image

    def frame(self, name):
        with self.lock:
            return self.frames.get(name)

    def add_callback(self, name, payload):
        with self.lock:
            self.callbacks.setdefault(name, []).append(payload)

    def received(self, name):
        with self.lock:
            return list(self.callbacks.get(name, []))


def make_handler(receiver):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            # /frame/<name>
            parts = self.path.strip("/").split("/")
            image = receiver.frame(parts[1]) if len(parts) == 2 and parts[0] == "frame" else None
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
            # /callback/<name>
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


def api(method, data, status=204, token=None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = requests.post(API_URL + method, json=data, headers=headers)
    assert response.status_code == status, response.text
    return response.json().get(DATA) if status == 200 else None


def add_stream(stream_id, faces=(), config=None):
    config = {CONF_DELAY_BETWEEN_FRAMES: "300ms", LOGS_LEVEL: "trace", **(config or {})}
    data = {
        STREAM_ID: stream_id,
        URL: f"{RECEIVER_URL}/frame/{stream_id}",
        CALLBACK: f"{RECEIVER_URL}/callback/{stream_id}",
        CALLBACK_BARCODES: f"{RECEIVER_URL}/callback/{stream_id}_barcodes",
        CONFIG: config,
    }
    if faces:
        data[FACES] = list(faces)
    api("addStream", data)
    # Faces are registered for a video stream only once the config cache knows it
    wait_for_caches()


def wait_for_caches():
    # The service sees new video streams, faces and special groups only after their caches update
    time.sleep(CACHE_UPDATE_INTERVAL + 1)


def register_face(stream_id, image_name):
    return api("registerFace", {STREAM_ID: stream_id, URL: f"{FALPRS_URL}/{image_name}.jpg"}, 200)[FACE_ID]


def motion(stream_id, start, duration=None):
    data = {STREAM_ID: stream_id, START: start}
    if duration:
        data[DURATION] = duration
    api("motionDetection", data)


@pytest.fixture
def stream(request, receiver):
    # Each test uses its own video stream and removes it with its faces afterwards
    stream_id = "fcb_" + request.node.name
    created_faces = []
    yield stream_id, created_faces
    motion(stream_id, False)
    if created_faces:
        api("deleteFaces", {FACES: created_faces})
    api("removeStream", {STREAM_ID: stream_id})


def wait_for_callbacks(receiver, name, count, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(receiver.received(name)) >= count:
            return receiver.received(name)
        time.sleep(0.1)
    return receiver.received(name)


def today_interval():
    now = datetime.now()
    return {
        DATE_START: (now - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
        DATE_END: (now + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
    }


# A recognised attached face is reported with its descriptor and the logged event
def test_recognition_callback(receiver, images, stream):
    stream_id, created_faces = stream
    add_stream(stream_id)
    face_id = register_face(stream_id, "einstein_001")
    created_faces.append(face_id)
    receiver.show(stream_id, images["einstein_001"])
    wait_for_caches()
    motion(stream_id, True)

    callbacks = wait_for_callbacks(receiver, stream_id, 1)
    assert len(callbacks) >= 1
    assert callbacks[0] == {FACE_ID: face_id, EVENT_ID: callbacks[0][EVENT_ID]}
    assert isinstance(callbacks[0][EVENT_ID], int)

    best = api("bestQuality", {EVENT_ID: callbacks[0][EVENT_ID]}, 200)
    assert best[SCREENSHOT].startswith(FALPRS_URL + "/frs/screenshots/")
    assert best["width"] > 0 and best["height"] > 0
    assert requests.get(best[SCREENSHOT]).status_code == 200

    events = api("getEvents", {STREAM_ID: stream_id, **today_interval()}, 200)
    assert any(event.get(FACE_ID) == face_id for event in events)


# An unknown face is logged as an event without a descriptor and not reported
def test_unknown_face_no_callback(receiver, images, stream):
    stream_id, created_faces = stream
    add_stream(stream_id)
    created_faces.append(register_face(stream_id, "einstein_001"))
    receiver.show(stream_id, images["hanks_001"])
    wait_for_caches()
    motion(stream_id, True)

    time.sleep(4)
    assert receiver.received(stream_id) == []
    events = api("getEvents", {STREAM_ID: stream_id, **today_interval()}, 200)
    assert len(events) >= 1
    assert all(FACE_ID not in event for event in events)


# A frame without faces produces neither events nor callbacks
def test_no_faces(receiver, images, stream):
    stream_id, created_faces = stream
    add_stream(stream_id)
    created_faces.append(register_face(stream_id, "einstein_001"))
    receiver.show(stream_id, images["no_faces"])
    wait_for_caches()
    motion(stream_id, True)

    time.sleep(3)
    assert receiver.received(stream_id) == []
    response = requests.post(API_URL + "getEvents", json={STREAM_ID: stream_id, **today_interval()})
    assert response.status_code == 204


# An open door stops the workflow
def test_door_is_open_stops_workflow(receiver, images, stream):
    stream_id, created_faces = stream
    add_stream(stream_id)
    created_faces.append(register_face(stream_id, "einstein_001"))
    receiver.show(stream_id, images["einstein_001"])
    wait_for_caches()
    motion(stream_id, True)

    assert len(wait_for_callbacks(receiver, stream_id, 1)) >= 1
    api("doorIsOpen", {STREAM_ID: stream_id})
    # A frame in progress may still finish
    time.sleep(2)
    count = len(receiver.received(stream_id))
    time.sleep(3)
    assert len(receiver.received(stream_id)) == count


# A workflow started with a duration stops by itself
def test_motion_duration(receiver, images, stream):
    stream_id, created_faces = stream
    add_stream(stream_id)
    created_faces.append(register_face(stream_id, "einstein_001"))
    receiver.show(stream_id, images["einstein_001"])
    wait_for_caches()
    motion(stream_id, True, "2s")

    assert len(wait_for_callbacks(receiver, stream_id, 1)) >= 1
    time.sleep(3)
    count = len(receiver.received(stream_id))
    time.sleep(3)
    assert len(receiver.received(stream_id)) == count


# A face of a special group seen by a video stream of the stream group is reported to the special group
def test_special_group_callback(receiver, images, stream):
    stream_id, _ = stream
    # A deleted special group keeps its name until the marked rows are purged, so names are unique per run
    group = api("addSpecialGroup", {GROUP_NAME: f"callbacks_{time.time_ns()}"}, 200)
    try:
        token = group[ACCESS_API_TOKEN]
        wait_for_caches()
        sg_name = f"sg_{group[GROUP_ID]}"
        api("sgUpdateGroup", {CALLBACK: f"{RECEIVER_URL}/callback/{sg_name}"}, token=token)
        face_id = api("sgRegisterFace", {URL: f"{FALPRS_URL}/einstein_001.jpg"}, 200, token=token)[FACE_ID]

        add_stream(stream_id)
        receiver.show(stream_id, images["einstein_001"])
        wait_for_caches()
        motion(stream_id, True)

        callbacks = wait_for_callbacks(receiver, sg_name, 1)
        assert len(callbacks) >= 1
        callback = callbacks[0]
        assert set(callback) == {FACE_ID, SCREENSHOT, DATE}
        assert callback[FACE_ID] == face_id
        assert isinstance(callback[DATE], str)
        assert requests.get(callback[SCREENSHOT]).status_code == 200
        # Special group faces are not attached to the video stream, so it gets no callback of its own
        assert receiver.received(stream_id) == []
    finally:
        api("deleteSpecialGroup", {GROUP_ID: group[GROUP_ID]})


# A QR code in a frame of a video stream with barcode processing is logged and reported
def test_barcode_callback(receiver, images, stream):
    stream_id, _ = stream
    add_stream(stream_id, config={"flag-process-barcodes": True, "flag-process-faces": False})
    receiver.show(stream_id, images["qr_code_001"])
    motion(stream_id, True)

    callbacks = wait_for_callbacks(receiver, stream_id + "_barcodes", 1)
    assert len(callbacks) >= 1
    callback = callbacks[0]
    assert set(callback) == {STREAM_ID, DATE, EVENT_ID, BARCODES}
    assert callback[STREAM_ID] == stream_id
    assert isinstance(callback[EVENT_ID], int)
    assert len(callback[BARCODES]) == 1
    barcode = callback[BARCODES][0]
    assert barcode["text"] == "https://example.com/test"
    assert barcode["format"] == "QR Code"
    # The code's corners in the 640x640 frame: top left, top right, bottom right, bottom left
    expected = [78, 78, 562, 78, 562, 562, 78, 562]
    assert all(abs(a - b) <= 2 for a, b in zip(barcode["position"], expected, strict=True))
    # Faces are not processed, so the face callback stays silent
    assert receiver.received(stream_id) == []

    event = api("getBarcodeEvent", {EVENT_ID: callback[EVENT_ID]}, 200)
    assert event[BARCODES] == callback[BARCODES]
    assert event[SCREENSHOT].startswith(FALPRS_URL + "/frs/barcodes/")
    assert requests.get(event[SCREENSHOT]).status_code == 200


def mean_difference(original, screenshot, box):
    """Mean absolute difference of the grey levels of two images in a box."""
    pairs = zip(original.convert("L").crop(box).getdata(), screenshot.convert("L").crop(box).getdata(), strict=True)
    differences = [abs(a - b) for a, b in pairs]
    return sum(differences) / len(differences)


# With a title the screenshot gets the date in the upper left corner and the title in the lower left one
def test_osd_title(receiver, images, stream):
    stream_id, created_faces = stream
    add_stream(stream_id, config={"title": "Подъезд №1", "title-height-ratio": 0.05})
    created_faces.append(register_face(stream_id, "einstein_001"))
    receiver.show(stream_id, images["einstein_001"])
    wait_for_caches()
    motion(stream_id, True)

    callbacks = wait_for_callbacks(receiver, stream_id, 1)
    assert len(callbacks) >= 1
    best = api("bestQuality", {EVENT_ID: callbacks[0][EVENT_ID]}, 200)
    assert_has_osd(images["einstein_001"], best[SCREENSHOT], 0.05)


def assert_has_osd(original_image, screenshot_url, title_height_ratio):
    screenshot = Image.open(io.BytesIO(requests.get(screenshot_url).content)).convert("RGB")
    original = Image.open(io.BytesIO(original_image)).convert("RGB")
    assert screenshot.size == original.size
    width, height = original.size
    text_height = int(title_height_ratio * height)
    # JPEG re-encoding changes grey levels slightly everywhere; white text with a black outline a lot
    upper_left = (10, 10, width // 2, 10 + text_height)
    lower_left = (10, height - 10 - text_height, width // 4, height - 10)
    middle = (width // 3, height // 3, width // 3 + width // 4, height // 3 + text_height)
    noise = mean_difference(original, screenshot, middle)
    assert mean_difference(original, screenshot, upper_left) > noise + 20
    assert mean_difference(original, screenshot, lower_left) > noise + 20


# A barcode screenshot gets the OSD too
def test_osd_barcode_screenshot(receiver, images, stream):
    stream_id, _ = stream
    config = {"title": "Ворота", "title-height-ratio": 0.05, "flag-process-barcodes": True, "flag-process-faces": False}
    add_stream(stream_id, config=config)
    receiver.show(stream_id, images["qr_code_001"])
    motion(stream_id, True)

    callbacks = wait_for_callbacks(receiver, stream_id + "_barcodes", 1)
    assert len(callbacks) >= 1
    event = api("getBarcodeEvent", {EVENT_ID: callbacks[0][EVENT_ID]}, 200)
    assert_has_osd(images["qr_code_001"], event[SCREENSHOT], 0.05)
