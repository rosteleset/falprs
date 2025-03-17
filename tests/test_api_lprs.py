import pytest
import requests
import time
from datetime import datetime, timedelta

FALPRS_URL = "http://localhost:9071"
API_URL = FALPRS_URL + "/lprs/api/"
DATA = "data"
STREAM_ID = "streamId"
URL = "url"
VEHICLES = "vehicles"
DATE = "date"
CONFIG = "config"
BOX = "box"
PLATES = "plates"
IS_SPECIAL = "isSpecial"
CONFIDENCE = "confidence"
SCREENSHOT_URL = "screenshotUrl"
LOGS_LEVEL = "logs-level"
TYPE = "type"
SCORE = "score"
NUMBER = "number"
KPTS = "kpts"

CONF_SCREENSHOT_URL = "screenshot-url"
CONF_WORK_AREA = "work-area"
CONF_MIN_PLATE_HEIGHT = "min-plate-height"
CONF_VEHICLE_CONFIDENCE = "vehicle-confidence"
CONF_PLATE_CONFIDENCE = "plate-confidence"
CONF_CHAR_SCORE = "char-score"
CONF_FLAG_PROCESS_SPECIAL = "flag-process-special"

TYPE_RU_1 = "ru_1"
TYPE_RU_1a = "ru_1a"

order = 0

def add_stream(stream_id, screenshot_url, work_area = None, min_plate_height = None, vehicle_confidence = None, plate_confidence = None,
               char_score = None, flag_process_special = None):
    url = API_URL + "addStream"
    config = {CONF_SCREENSHOT_URL: screenshot_url, LOGS_LEVEL: "trace"}
    if work_area != None:
        config[CONF_WORK_AREA] = work_area
    if min_plate_height != None:
        config[CONF_MIN_PLATE_HEIGHT] = min_plate_height
    if vehicle_confidence != None:
        config[CONF_VEHICLE_CONFIDENCE] = vehicle_confidence
    if plate_confidence != None:
        config[CONF_PLATE_CONFIDENCE] = plate_confidence
    if char_score != None:
        config[CONF_CHAR_SCORE] = char_score
    if flag_process_special != None:
        config[CONF_FLAG_PROCESS_SPECIAL] = flag_process_special
    data = {STREAM_ID: stream_id, CONFIG: config}
    response = requests.post(url, json=data)
    assert response.status_code == 204

def start_stop_workflow(stream_id):
    url = API_URL + "startWorkflow"
    data = {STREAM_ID: stream_id}
    response = requests.post(url, json=data)
    assert response.status_code == 204
    time.sleep(0.7)
    url = API_URL + "stopWorkflow"
    data = {STREAM_ID: stream_id}
    response = requests.post(url, json=data)
    assert response.status_code == 204
    global tp
    tp = datetime.now()

def check_event_vehicle(vehicle, type = TYPE_RU_1):
    assert BOX in vehicle
    assert len(vehicle[BOX]) == 4
    assert PLATES in vehicle
    assert len(vehicle[PLATES]) >= 1
    assert IS_SPECIAL in vehicle
    assert vehicle[IS_SPECIAL] == False
    assert CONFIDENCE in vehicle
    assert 0.0 <= vehicle[CONFIDENCE] and vehicle[CONFIDENCE] <= 1.0

    plate = vehicle[PLATES][0]
    assert BOX in plate
    assert len(plate[BOX]) == 4
    assert KPTS in plate
    assert len(plate[KPTS]) == 8
    assert TYPE in plate
    assert plate[TYPE] == type
    assert SCORE in plate
    assert 0.0 <= plate[SCORE] and plate[SCORE] <= 1.0

def run_single(stream_id, number, type = TYPE_RU_1):
    start_stop_workflow(stream_id)
    url = API_URL + "getEventData"
    global tp
    data = {STREAM_ID: stream_id, DATE: tp.strftime("%Y-%m-%d %H:%M:%S")}
    response = requests.post(url, json=data)
    assert response.status_code == 200

    data = response.json()
    assert DATE in data[DATA]
    assert SCREENSHOT_URL in data[DATA]

    assert VEHICLES in data[DATA]
    assert len(data[DATA][VEHICLES]) == 1
    vehicle = data[DATA][VEHICLES][0]
    check_event_vehicle(vehicle, type)
    has_number = False
    for item in vehicle[PLATES]:
        has_number = has_number or (item[NUMBER] == number)
    assert has_number == True

def run_double(stream_id, number1, number2, type = TYPE_RU_1):
    start_stop_workflow(stream_id)
    url = API_URL + "getEventData"
    global tp
    data = {STREAM_ID: stream_id, DATE: tp.strftime("%Y-%m-%d %H:%M:%S")}
    response = requests.post(url, json=data)
    assert response.status_code == 200

    data = response.json()
    assert DATE in data[DATA]
    assert SCREENSHOT_URL in data[DATA]

    assert VEHICLES in data[DATA]
    assert len(data[DATA][VEHICLES]) == 2
    vehicle0 = data[DATA][VEHICLES][0]
    check_event_vehicle(vehicle0)
    vehicle1 = data[DATA][VEHICLES][1]
    check_event_vehicle(vehicle1, type)
    valid_numbers = [number1, number2]
    assert vehicle0[PLATES][0][NUMBER] in valid_numbers
    assert vehicle1[PLATES][0][NUMBER] in valid_numbers
    assert vehicle0[PLATES][0][NUMBER] != vehicle1[PLATES][0][NUMBER]

def run_special_single(stream_id):
    start_stop_workflow(stream_id)
    url = API_URL + "getEventData"
    global tp
    data = {STREAM_ID: stream_id, DATE: tp.strftime("%Y-%m-%d %H:%M:%S")}
    response = requests.post(url, json=data)
    assert response.status_code == 200

    data = response.json()
    assert DATE in data[DATA]
    assert VEHICLES in data[DATA]
    assert len(data[DATA][VEHICLES]) == 1
    vehicle = data[DATA][VEHICLES][0]
    assert vehicle[IS_SPECIAL] == True

def run_special(stream_id):
    start_stop_workflow(stream_id)
    url = API_URL + "getEventData"
    global tp
    data = {STREAM_ID: stream_id, DATE: tp.strftime("%Y-%m-%d %H:%M:%S")}
    response = requests.post(url, json=data)
    assert response.status_code == 200

    data = response.json()
    assert DATE in data[DATA]
    assert SCREENSHOT_URL in data[DATA]

    assert VEHICLES in data[DATA]
    assert len(data[DATA][VEHICLES]) > 0
    special_count = 0
    for item in data[DATA][VEHICLES]:
        if item[IS_SPECIAL] == True:
            special_count += 1
    assert special_count == 1

# ping
@pytest.mark.order(++order)
def test_ping():
   # waiting for FALPRS to start
    print("[Wait 1 second for FALPRS to start...]")
    time.sleep(1)

    url = FALPRS_URL + "/ping"
    response = requests.get(url)
    assert response.status_code == 200

# listStreams: should be empty
@pytest.mark.order(++order)
def test_list_streams_empty():
    url = API_URL + "listStreams"
    response = requests.post(url)
    assert response.status_code == 204

# addStream
@pytest.mark.order(++order)
def test_add_stream():
    add_stream("1", FALPRS_URL + "/test_001.jpg")

# addStream with the same data
@pytest.mark.order(++order)
def test_add_stream2():
    test_add_stream()

# listStreams
@pytest.mark.order(++order)
def test_list_streams():
    url = API_URL + "listStreams"
    response = requests.post(url)
    assert response.status_code == 200

    data = response.json()
    assert len(data[DATA]) == 1
    assert data[DATA][0][STREAM_ID] == "1"
    assert CONFIG in data[DATA][0]
    assert CONF_SCREENSHOT_URL in data[DATA][0][CONFIG]

# Add one more video stream
@pytest.mark.order(++order)
def test_add_stream3():
    add_stream("2", FALPRS_URL + "/test_002.jpg")

# listStreams 2
@pytest.mark.order(++order)
def test_list_streams2():
    url = API_URL + "listStreams"
    response = requests.post(url)
    assert response.status_code == 200

    data = response.json()
    assert len(data[DATA]) == 2
    ids = set()
    for item in data[DATA]:
        ids.add(item[STREAM_ID])
        assert CONFIG in item
        assert CONF_SCREENSHOT_URL in item[CONFIG]
    assert ids == set(["1", "2"])

# removeStream
@pytest.mark.order(++order)
def test_remove_stream():
    url = API_URL + "removeStream"
    data = {STREAM_ID: "1"}
    response = requests.post(url, json=data)
    assert response.status_code == 204

# listStreams 3
@pytest.mark.order(++order)
def test_list_streams3():
    url = API_URL + "listStreams"
    response = requests.post(url)
    assert response.status_code == 200

    data = response.json()
    assert len(data[DATA]) == 1
    ids = set()
    for item in data[DATA]:
        ids.add(item[STREAM_ID])
        assert CONFIG in item
        assert CONF_SCREENSHOT_URL in item[CONFIG]
    assert ids == set(["2"])

# addStream with streamId="1" again
@pytest.mark.order(++order)
def test_add_stream4():
    test_add_stream()

# listStreams 4
@pytest.mark.order(++order)
def test_list_streams4():
    test_list_streams2

# add more video streams
@pytest.mark.order(++order)
def test_add_more_streams():
    add_stream("3", FALPRS_URL + "/test_003.jpg")

    w = 2592.0
    h = 1520.0
    add_stream("4", FALPRS_URL + "/test_003.jpg", [
        [
            [0, 0],
            [500 / w * 100, 0],
            [500 / w * 100, 800 / h * 100],
            [0, 800 / h * 100]
        ]
    ])
    add_stream("5", FALPRS_URL + "/test_003.jpg", [
        [
            [1970 / w * 100, 60 / h * 100],
            [2590 / w * 100, 60 / h * 100],
            [2590 / w * 100, 830 / h * 100],
            [1970 / w * 100, 830 / h * 100]
        ]
    ])
    add_stream("6", FALPRS_URL + "/test_003.jpg", [
        [
            [1970 / w * 100, 60 / h * 100],
            [2590 / w * 100, 60 / h * 100],
            [2590 / w * 100, 830 / h * 100],
            [1970 / w * 100, 830 / h * 100]
        ],
        [
            [0, 0],
            [500 / w * 100, 0],
            [500 / w * 100, 800 / h * 100],
            [0, 800 / h * 100]
        ]
    ])
    add_stream("a1", FALPRS_URL + "/test_angle_001.jpg")
    add_stream("a2", FALPRS_URL + "/test_angle_002.jpg")
    add_stream("a3", FALPRS_URL + "/test_angle_003.jpg", min_plate_height=30)
    add_stream("a4", FALPRS_URL + "/test_angle_004.jpg")
    add_stream("a5", FALPRS_URL + "/test_angle_005.jpg")
    add_stream("a6", FALPRS_URL + "/test_angle_006.jpg")
    add_stream("a7", FALPRS_URL + "/test_angle_007.jpg")
    add_stream("a8", FALPRS_URL + "/test_angle_008.jpg")
    add_stream("a9", FALPRS_URL + "/test_angle_009.jpg")
    add_stream("a10", FALPRS_URL + "/test_angle_010.jpg")
    add_stream("a11", FALPRS_URL + "/test_angle_011.jpg")
    add_stream("a12", FALPRS_URL + "/test_angle_012.jpg")
    add_stream("a13", FALPRS_URL + "/test_angle_013.jpg")
    add_stream("a14", FALPRS_URL + "/test_angle_014.jpg")
    add_stream("a15", FALPRS_URL + "/test_angle_015.jpg")
    add_stream("a16", FALPRS_URL + "/test_angle_016.jpg")
    add_stream("a17", FALPRS_URL + "/test_angle_017.jpg")
    add_stream("a18", FALPRS_URL + "/test_angle_018.jpg")
    add_stream("a19", FALPRS_URL + "/test_angle_019.jpg")
    add_stream("a20", FALPRS_URL + "/test_angle_020.jpg")
    add_stream("a21", FALPRS_URL + "/test_angle_021.jpg")
    add_stream("a22", FALPRS_URL + "/test_angle_022.jpg")
    add_stream("a23", FALPRS_URL + "/test_angle_023.jpg")

    add_stream("c1", FALPRS_URL + "/test_001.jpg", min_plate_height=100)
    add_stream("c2", FALPRS_URL + "/test_001.jpg", vehicle_confidence=0.99)
    add_stream("c3", FALPRS_URL + "/test_001.jpg", plate_confidence=0.9)
    add_stream("c4", FALPRS_URL + "/test_001.jpg", char_score=0.99)

    add_stream("b1", FALPRS_URL + "/test_blur_001.jpg")
    add_stream("b2", FALPRS_URL + "/test_blur_002.jpg")
    add_stream("b3", FALPRS_URL + "/test_blur_003.jpg")
    add_stream("b4", FALPRS_URL + "/test_blur_004.jpg")
    add_stream("b5", FALPRS_URL + "/test_blur_005.jpg")
    add_stream("b6", FALPRS_URL + "/test_blur_006.jpg")

    add_stream("d1", FALPRS_URL + "/test_dirty_001.jpg")
    add_stream("d2", FALPRS_URL + "/test_dirty_002.jpg")
    add_stream("d3", FALPRS_URL + "/test_dirty_003.jpg")
    add_stream("d4", FALPRS_URL + "/test_dirty_004.jpg")
    add_stream("d5", FALPRS_URL + "/test_dirty_005.jpg")
    add_stream("d6", FALPRS_URL + "/test_dirty_006.jpg")
    add_stream("d7", FALPRS_URL + "/test_dirty_007.jpg")
    add_stream("d8", FALPRS_URL + "/test_dirty_008.jpg")
    add_stream("d9", FALPRS_URL + "/test_dirty_009.jpg")
    add_stream("d10", FALPRS_URL + "/test_dirty_010.jpg")

    add_stream("r1", FALPRS_URL + "/test_rain_001.jpg", vehicle_confidence=0.3)
    add_stream("r2", FALPRS_URL + "/test_rain_002.jpg", vehicle_confidence=0.3)
    add_stream("r3", FALPRS_URL + "/test_rain_003.jpg", vehicle_confidence=0.3)
    add_stream("r4", FALPRS_URL + "/test_rain_004.jpg", vehicle_confidence=0.3)
    add_stream("r5", FALPRS_URL + "/test_rain_005.jpg", vehicle_confidence=0.3, min_plate_height=20)
    add_stream("r6", FALPRS_URL + "/test_rain_006.jpg", vehicle_confidence=0.3)
    add_stream("r7", FALPRS_URL + "/test_rain_007.jpg", vehicle_confidence=0.3)
    add_stream("r8", FALPRS_URL + "/test_rain_008.jpg", vehicle_confidence=0.3)
    add_stream("r9", FALPRS_URL + "/test_rain_009.jpg", vehicle_confidence=0.3)
    add_stream("r10", FALPRS_URL + "/test_rain_010.jpg", vehicle_confidence=0.3)
    add_stream("r11", FALPRS_URL + "/test_rain_011.jpg", vehicle_confidence=0.3)
    add_stream("r12", FALPRS_URL + "/test_rain_012.jpg", vehicle_confidence=0.3)

    add_stream("sq1", FALPRS_URL + "/test_square_001.jpg")
    add_stream("sq2", FALPRS_URL + "/test_square_002.jpg")
    add_stream("sq3", FALPRS_URL + "/test_square_003.jpg")
    add_stream("sq4", FALPRS_URL + "/test_square_004.jpg")
    add_stream("sq5", FALPRS_URL + "/test_square_005.jpg")
    add_stream("sq6", FALPRS_URL + "/test_square_006.jpg")
    add_stream("sq7", FALPRS_URL + "/test_square_007.jpg")
    add_stream("sq8", FALPRS_URL + "/test_square_008.jpg")
    add_stream("sq9", FALPRS_URL + "/test_square_009.jpg")

    add_stream("s1", FALPRS_URL + "/test_special_001.jpg", flag_process_special = True)
    add_stream("s2", FALPRS_URL + "/test_special_002.jpg", flag_process_special = True)
    add_stream("s3", FALPRS_URL + "/test_special_003.jpg", flag_process_special = True)
    add_stream("s4", FALPRS_URL + "/test_special_004.jpg", flag_process_special = True)

    w = 1010
    h = 505
    add_stream("s5", FALPRS_URL + "/test_special_003.jpg", [
        [
            [900 / w * 100, 0],
            [1010 / w * 100, 0],
            [1010 / w * 100, 505 / h * 100],
            [900 / w * 100, 505 / h * 100]
        ]
    ], flag_process_special = True)
    add_stream("s6", FALPRS_URL + "/test_special_003.jpg", [
        [
            [200 / w * 100, 50 / h * 100],
            [410 / w * 100, 50 / h * 100],
            [410 / w * 100, 450 / h * 100],
            [200 / w * 100, 450 / h * 100]
        ]
    ], flag_process_special = True)
    add_stream("s7", FALPRS_URL + "/test_special_007.jpg", flag_process_special = True)
    add_stream("s8", FALPRS_URL + "/test_special_008.jpg", flag_process_special = True)
    add_stream("s9", FALPRS_URL + "/test_special_009.jpg", flag_process_special = True)
    add_stream("s10", FALPRS_URL + "/test_special_010.jpg", flag_process_special = True)

# start / stop workflow
@pytest.mark.order(++order)
def test_start_stop_workflow():
    # waiting for cache
    print("[Wait 6 seconds for cache...]")
    time.sleep(6)

    start_stop_workflow("1")
    
# getEventData ISO
@pytest.mark.order(++order)
def test_get_event_data_iso():
    url = API_URL + "getEventData"
    global tp
    data = {STREAM_ID: "1", DATE: tp.isoformat()}
    response = requests.post(url, json=data)
    assert response.status_code == 200

    data = response.json()
    assert DATE in data[DATA]
    assert SCREENSHOT_URL in data[DATA]

    assert VEHICLES in data[DATA]
    assert len(data[DATA][VEHICLES]) == 1
    vehicle = data[DATA][VEHICLES][0]
    check_event_vehicle(vehicle)
    assert vehicle[PLATES][0][NUMBER] == "O588OA68"

# getEventData
@pytest.mark.order(++order)
def test_get_event_data():
    url = API_URL + "getEventData"
    global tp
    data = {STREAM_ID: "1", DATE: tp.strftime("%Y-%m-%d %H:%M:%S")}
    response = requests.post(url, json=data)
    assert response.status_code == 200

    data = response.json()
    assert DATE in data[DATA]
    assert SCREENSHOT_URL in data[DATA]

    assert VEHICLES in data[DATA]
    assert len(data[DATA][VEHICLES]) == 1
    vehicle = data[DATA][VEHICLES][0]
    check_event_vehicle(vehicle)
    assert vehicle[PLATES][0][NUMBER] == "O588OA68"

# start / stop workflow 2
@pytest.mark.order(++order)
def test_start_stop_workflow2():
    start_stop_workflow("3")

# getEventData 2
@pytest.mark.order(++order)
def test_get_event_data2():
    url = API_URL + "getEventData"
    global tp
    data = {STREAM_ID: "3", DATE: tp.strftime("%Y-%m-%d %H:%M:%S")}
    response = requests.post(url, json=data)
    assert response.status_code == 200

    data = response.json()
    assert DATE in data[DATA]
    assert SCREENSHOT_URL in data[DATA]

    assert VEHICLES in data[DATA]
    assert len(data[DATA][VEHICLES]) == 2
    vehicle0 = data[DATA][VEHICLES][0]
    check_event_vehicle(vehicle0)
    vehicle1 = data[DATA][VEHICLES][1]
    check_event_vehicle(vehicle1)
    valid_numbers = ["O434PA68", "O105OK68"]
    assert vehicle0[PLATES][0][NUMBER] in valid_numbers
    assert vehicle1[PLATES][0][NUMBER] in valid_numbers
    assert vehicle0[PLATES][0][NUMBER] != vehicle1[PLATES][0][NUMBER]

# start / stop workflow 3 (one region)
@pytest.mark.order(++order)
def test_start_stop_workflow3():
    start_stop_workflow("4")

# getEventData 3 (one region)
@pytest.mark.order(++order)
def test_get_event_data3():
    url = API_URL + "getEventData"
    global tp
    data = {STREAM_ID: "4", DATE: tp.strftime("%Y-%m-%d %H:%M:%S")}
    response = requests.post(url, json=data)
    assert response.status_code == 200

    data = response.json()
    assert DATE in data[DATA]
    assert SCREENSHOT_URL in data[DATA]

    assert VEHICLES in data[DATA]
    assert len(data[DATA][VEHICLES]) == 1
    vehicle = data[DATA][VEHICLES][0]
    check_event_vehicle(vehicle)
    assert vehicle[PLATES][0][NUMBER] == "O105OK68"

# start / stop workflow 4 (one region)
@pytest.mark.order(++order)
def test_start_stop_workflow4():
    start_stop_workflow("5")

# getEventData 4 (one region)
@pytest.mark.order(++order)
def test_get_event_data4():
    url = API_URL + "getEventData"
    global tp
    data = {STREAM_ID: "5", DATE: tp.strftime("%Y-%m-%d %H:%M:%S")}
    response = requests.post(url, json=data)
    assert response.status_code == 200

    data = response.json()
    assert DATE in data[DATA]
    assert SCREENSHOT_URL in data[DATA]

    assert VEHICLES in data[DATA]
    assert len(data[DATA][VEHICLES]) == 1
    vehicle = data[DATA][VEHICLES][0]
    check_event_vehicle(vehicle)
    assert vehicle[PLATES][0][NUMBER] == "O434PA68"

# getEventData in the future
@pytest.mark.order(++order)
def test_get_event_data5():
    url = API_URL + "getEventData"
    global tp
    data = {STREAM_ID: "5", DATE: (tp + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")}
    response = requests.post(url, json=data)
    assert response.status_code == 204

# getEventData in the past
@pytest.mark.order(++order)
def test_get_event_data6():
    url = API_URL + "getEventData"
    global tp
    data = {STREAM_ID: "5", DATE: (tp - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")}
    response = requests.post(url, json=data)
    assert response.status_code == 204

# start / stop workflow 6 (two regions)
@pytest.mark.order(++order)
def test_start_stop_workflow6():
    start_stop_workflow("6")

# getEventData 7 (two regions)
@pytest.mark.order(++order)
def test_get_event_data7():
    url = API_URL + "getEventData"
    global tp
    data = {STREAM_ID: "6", DATE: tp.strftime("%Y-%m-%d %H:%M:%S")}
    response = requests.post(url, json=data)
    assert response.status_code == 200

    data = response.json()
    assert DATE in data[DATA]
    assert SCREENSHOT_URL in data[DATA]

    assert VEHICLES in data[DATA]
    assert len(data[DATA][VEHICLES]) == 2
    vehicle0 = data[DATA][VEHICLES][0]
    check_event_vehicle(vehicle0)
    vehicle1 = data[DATA][VEHICLES][1]
    check_event_vehicle(vehicle1)
    valid_numbers = ["O434PA68", "O105OK68"]
    assert vehicle0[PLATES][0][NUMBER] in valid_numbers
    assert vehicle1[PLATES][0][NUMBER] in valid_numbers
    assert vehicle0[PLATES][0][NUMBER] != vehicle1[PLATES][0][NUMBER]

# test angle 1
@pytest.mark.order(++order)
def test_angle1():
    run_single("a1", "O016TK68")

# test angle 2
@pytest.mark.order(++order)
def test_angle2():
    run_single("a2", "E823CA73")

# test angle 3
@pytest.mark.order(++order)
def test_angle3():
    run_single("a3", "O072TK68")

# test angle 4
@pytest.mark.order(++order)
def test_angle4():
    run_single("a4", "C449AC68")

# test angle 5
@pytest.mark.order(++order)
def test_angle5():
    run_single("a5", "O590EE68")

# test angle 6
@pytest.mark.order(++order)
def test_angle6():
    run_single("a6", "O529PX68")

# test angle 7
@pytest.mark.order(++order)
def test_angle7():
    run_single("a7", "O851BY68")

# test angle 8
@pytest.mark.order(++order)
def test_angle8():
    run_single("a8", "H751YY68")

# test angle 9
@pytest.mark.order(++order)
def test_angle9():
    run_single("a9", "E823CA73")

# test angle 10
@pytest.mark.order(++order)
def test_angle10():
    run_single("a10", "C059AX68")

# test angle 11
@pytest.mark.order(++order)
def test_angle11():
    run_double("a11", "O343PA68", "O082XH68")

# test angle 12
@pytest.mark.order(++order)
def test_angle12():
    run_single("a12", "O529PX68")

# test angle 13
@pytest.mark.order(++order)
def test_angle13():
    run_single("a13", "O978XM68")

# test angle 14
@pytest.mark.order(++order)
def test_angle14():
    run_single("a14", "O016TK68")

# test angle 15
@pytest.mark.order(++order)
def test_angle15():
    run_single("a13", "O978XM68")

# test angle 16
@pytest.mark.order(++order)
def test_angle16():
    run_single("a16", "C848AT68")

# test angle 17
@pytest.mark.order(++order)
def test_angle17():
    run_single("a17", "O082XH68")

# test angle 18
@pytest.mark.order(++order)
def test_angle18():
    run_single("a18", "C059AX68")

# test angle 19
@pytest.mark.order(++order)
def test_angle19():
    run_single("a19", "O529PX68")

# test angle 20
@pytest.mark.order(++order)
def test_angle20():
    run_single("a20", "O354CP31")

# test angle 21
@pytest.mark.order(++order)
def test_angle21():
    run_single("a21", "O590EE68")

# test angle 22
@pytest.mark.order(++order)
def test_angle22():
    run_single("a22", "C059AX68")

# test angle 23
@pytest.mark.order(++order)
def test_angle23():
    run_single("a23", "O590EE68")

# test config 1
@pytest.mark.order(++order)
def test_config_params1():
    stream_id = "c1"
    start_stop_workflow(stream_id)
    url = API_URL + "getEventData"
    global tp
    data = {STREAM_ID: stream_id, DATE: tp.strftime("%Y-%m-%d %H:%M:%S")}
    response = requests.post(url, json=data)
    assert response.status_code == 204

# test config 2
@pytest.mark.order(++order)
def test_config_params2():
    stream_id = "c2"
    start_stop_workflow(stream_id)
    url = API_URL + "getEventData"
    global tp
    data = {STREAM_ID: stream_id, DATE: tp.strftime("%Y-%m-%d %H:%M:%S")}
    response = requests.post(url, json=data)
    assert response.status_code == 204

# test config 3
@pytest.mark.order(++order)
def test_config_params3():
    stream_id = "c3"
    start_stop_workflow(stream_id)
    url = API_URL + "getEventData"
    global tp
    data = {STREAM_ID: stream_id, DATE: tp.strftime("%Y-%m-%d %H:%M:%S")}
    response = requests.post(url, json=data)
    assert response.status_code == 204

# test config 4
@pytest.mark.order(++order)
def test_config_params4():
    stream_id = "c4"
    start_stop_workflow(stream_id)
    url = API_URL + "getEventData"
    global tp
    data = {STREAM_ID: stream_id, DATE: tp.strftime("%Y-%m-%d %H:%M:%S")}
    response = requests.post(url, json=data)
    assert response.status_code == 204

# test blur 1
@pytest.mark.order(++order)
def test_blur1():
    run_double("b1", "M680OO68", "O289KX68")

# test blur 2
@pytest.mark.order(++order)
def test_blur2():
    run_single("b2", "O016TK68")

# test blur 3
@pytest.mark.order(++order)
def test_blur3():
    run_single("b3", "O283PA68")

# test blur 4
@pytest.mark.order(++order)
def test_blur4():
    run_double("b4", "M680OO68", "O354CP31")

# test blur 5
@pytest.mark.order(++order)
def test_blur5():
    run_single("b5", "O283PA68")

# test blur 6
@pytest.mark.order(++order)
def test_blur6():
    run_single("b6", "O887XB68")

# test dirty 1
@pytest.mark.order(++order)
def test_dirty1():
    run_single("d1", "C848AT68")

# test dirty 2
@pytest.mark.order(++order)
def test_dirty2():
    run_single("d2", "M680OO68")

# test dirty 3
@pytest.mark.order(++order)
def test_dirty3():
    run_single("d3", "O588OA68")

# test dirty 4
@pytest.mark.order(++order)
def test_dirty4():
    run_single("d4", "O016TK68")

# test dirty 5
@pytest.mark.order(++order)
def test_dirty5():
    run_single("d5", "C887XB68")

# test dirty 6
@pytest.mark.order(++order)
def test_dirty6():
    run_single("d6", "C253BH68")

# test dirty 7
@pytest.mark.order(++order)
def test_dirty7():
    run_single("d7", "E823CA73")

# test dirty 8
@pytest.mark.order(++order)
def test_dirty8():
    run_single("d8", "C848AT68")

# test dirty 9
@pytest.mark.order(++order)
def test_dirty9():
    run_single("d9", "O016TK68")

# test dirty 10
@pytest.mark.order(++order)
def test_dirty10():
    run_single("d10", "O978XM68")

# test rain 1
@pytest.mark.order(++order)
def test_rain1():
    run_double("r1", "O529PX68", "O590EE68")

# test rain 2
@pytest.mark.order(++order)
def test_rain2():
    run_double("r2", "O082XH68", "O590EE68")

# test rain 3
@pytest.mark.order(++order)
def test_rain3():
    run_single("r3", "O590EE68")

# test rain 4
@pytest.mark.order(++order)
def test_rain4():
    run_double("r4", "C059AX68", "O590EE68")

# test rain 5
@pytest.mark.order(++order)
def test_rain5():
    run_single("r5", "O887XB68")

# test rain 6
@pytest.mark.order(++order)
def test_rain6():
    run_single("r6", "O978XM68")

# test rain 7
@pytest.mark.order(++order)
def test_rain7():
    run_single("r7", "O978XM68")

# test rain 8
@pytest.mark.order(++order)
def test_rain8():
    run_single("r8", "C449AC68")

# test rain 9
@pytest.mark.order(++order)
def test_rain9():
    run_double("r9", "C253BH68", "O590EE68")

# test rain 10
@pytest.mark.order(++order)
def test_rain10():
    run_double("r10", "O887XB68", "O590EE68")

# test rain 11
@pytest.mark.order(++order)
def test_rain11():
    run_double("r11", "C449AC68", "O590EE68")

# test rain 12
@pytest.mark.order(++order)
def test_rain12():
    run_single("r12", "O283PA68")

# test square 1
@pytest.mark.order(++order)
def test_square1():
    run_single("sq1", "H701HE27", type = TYPE_RU_1a)

# test square 2
@pytest.mark.order(++order)
def test_square2():
    run_single("sq2", "K799YT27", type = TYPE_RU_1a)

# test square 3
@pytest.mark.order(++order)
def test_square3():
    run_single("sq3", "E260EE166", type = TYPE_RU_1a)

# test square 4
@pytest.mark.order(++order)
def test_square4():
    run_single("sq4", "A748XP27", type = TYPE_RU_1a)

# test square 5
@pytest.mark.order(++order)
def test_square5():
    run_single("sq5", "B555PX777", type = TYPE_RU_1a)

# test square 6
@pytest.mark.order(++order)
def test_square6():
    run_single("sq6", "K554YH31", type = TYPE_RU_1a)

# test square 7
@pytest.mark.order(++order)
def test_square7():
    run_single("sq7", "E820HB124", type = TYPE_RU_1a)

# test square 8
@pytest.mark.order(++order)
def test_square8():
    run_single("sq8", "Y886OY125", type = TYPE_RU_1a)

# test square 9
@pytest.mark.order(++order)
def test_square9():
    run_single("sq9", "B411XX152", type = TYPE_RU_1a)

# test special 1
@pytest.mark.order(++order)
def test_special1():
    run_special_single("s1")

# test special 2
@pytest.mark.order(++order)
def test_special2():
    run_special_single("s2")

# test special 3
@pytest.mark.order(++order)
def test_special3():
    run_special("s3")

# test special 4
@pytest.mark.order(++order)
def test_special4():
    run_special("s4")

# test special 5
@pytest.mark.order(++order)
def test_special5():
    stream_id = "s5"
    start_stop_workflow(stream_id)
    url = API_URL + "getEventData"
    global tp
    data = {STREAM_ID: stream_id, DATE: tp.strftime("%Y-%m-%d %H:%M:%S")}
    response = requests.post(url, json=data)
    assert response.status_code == 204

# test special 6
@pytest.mark.order(++order)
def test_special6():
    run_special("s6")

# test special 7
@pytest.mark.order(++order)
def test_special7():
    run_special("s7")

# test special 8
@pytest.mark.order(++order)
def test_special8():
    run_special("s8")

# test special 9
@pytest.mark.order(++order)
def test_special9():
    run_special("s9")

# test special 10
@pytest.mark.order(++order)
def test_special10():
    run_special("s10")
