import time
from pathlib import Path

import requests

FALPRS_URL = "http://localhost:9071"
MONITOR_URL = "http://localhost:9072"
LOG_LEVEL_URL = MONITOR_URL + "/service/log-level/"
LOG_ROTATE_URL = MONITOR_URL + "/service/on-log-rotate/"

# Must match the loggers section of test_config.yaml
LOGS_DIR = Path("/tmp/test_falprs/logs")
LOG_FILES = ["log.txt", "lprs-log.txt", "frs-log.txt"]
INIT_LEVELS = {None: "error", "lprs": "debug", "frs": "debug"}

INIT_LOG_LEVEL = "init-log-level"
CURRENT_LOG_LEVEL = "current-log-level"

IMAGES_DIR = Path(__file__).parent / "images"


def get_log_level(logger=None):
    params = {"logger": logger} if logger else None
    response = requests.get(LOG_LEVEL_URL, params=params)
    assert response.status_code == 200
    return response.json()


def set_log_level(level, logger=None):
    params = {"logger": logger} if logger else None
    response = requests.put(LOG_LEVEL_URL + level, params=params)
    assert response.status_code == 200
    return response.json()


def wait_for_file(path, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.1)
    return path.exists()


def test_ping():
    response = requests.get(FALPRS_URL + "/ping")
    assert response.status_code == 200


def test_log_files_exist():
    for name in LOG_FILES:
        assert (LOGS_DIR / name).is_file(), name


# logrotate moves the files away and asks the service to reopen them
def test_on_log_rotate():
    for name in LOG_FILES:
        (LOGS_DIR / name).rename(LOGS_DIR / (name + ".1"))

    response = requests.post(LOG_ROTATE_URL)
    assert response.status_code == 200

    for name in LOG_FILES:
        assert wait_for_file(LOGS_DIR / name), name


# The path the example logrotate config uses has no trailing slash
def test_on_log_rotate_without_trailing_slash():
    response = requests.post(LOG_ROTATE_URL.rstrip("/"))
    assert response.status_code == 200


def test_get_log_level():
    for logger, level in INIT_LEVELS.items():
        assert get_log_level(logger) == {INIT_LOG_LEVEL: level, CURRENT_LOG_LEVEL: level}


def test_get_log_level_with_level_is_rejected():
    response = requests.get(LOG_LEVEL_URL + "info")
    assert response.status_code == 400


def test_set_and_reset_log_level():
    for logger, level in INIT_LEVELS.items():
        assert set_log_level("warning", logger) == {INIT_LOG_LEVEL: level, CURRENT_LOG_LEVEL: "warning"}
        assert get_log_level(logger)[CURRENT_LOG_LEVEL] == "warning"
        assert set_log_level("reset", logger) == {INIT_LOG_LEVEL: level, CURRENT_LOG_LEVEL: level}
        assert get_log_level(logger)[CURRENT_LOG_LEVEL] == level


def test_static_file():
    response = requests.get(FALPRS_URL + "/test_001.jpg")
    assert response.status_code == 200
    assert response.content == (IMAGES_DIR / "test_001.jpg").read_bytes()


def test_static_file_missing():
    response = requests.get(FALPRS_URL + "/no_such_file.jpg")
    assert response.status_code == 404
