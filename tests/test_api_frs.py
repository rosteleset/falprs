import pytest
import requests
import time
from datetime import datetime, timedelta

FALPRS_URL = "http://localhost:9071"
API_URL = FALPRS_URL + "/frs/api/"
DATA = "data"
STREAM_ID = "streamId"
URL = "url"
FACE_ID = "faceId"
FACES = "faces"
FACE_IMAGE = "faceImage"
LEFT = "left"
TOP = "top"
WIDTH = "width"
HEIGHT = "height"
START = "start"
DATE = "date"
SCREENSHOT_URL = "screenshot"
GROUP_NAME = "groupName"
MAX_DESCRIPTOR_COUNT = "maxDescriptorCount"
GROUP_ID = "groupId"
ACCESS_API_TOKEN = "accessApiToken"
DATE_START = "dateStart"
DATE_END = "dateEnd"
SIMILARITY_THRESHOLD = "similarityThreshold"
UUID = "uuid"
EVENT_ID = "eventId"
SIMILARITY = "similarity"

order = 0
face_id1 = 0
face_id2 = 0
tp = 0
sg_api_token = ""
sg_id = 0

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
    url = API_URL + "addStream"
    data = {STREAM_ID: "1", URL: FALPRS_URL + "/einstein_001.jpg"}
    response = requests.post(url, json=data)
    assert response.status_code == 204

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

# Add one more video stream
@pytest.mark.order(++order)
def test_add_stream3():
    url = API_URL + "addStream"
    data = {STREAM_ID: "2", URL: FALPRS_URL + "/einstein_001.jpg"}
    response = requests.post(url, json=data)
    assert response.status_code == 204

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
    assert ids == set(["2"])

# addStream with streamId="1" again
@pytest.mark.order(++order)
def test_add_stream4():
    test_add_stream()

# listStreams 4
@pytest.mark.order(++order)
def test_list_streams4():
    test_list_streams2

# registerFace
@pytest.mark.order(++order)
def test_register_face():
    # waiting for cache
    print("[Wait 6 seconds for cache...]")
    time.sleep(6)

    url = API_URL + "registerFace"
    data = {STREAM_ID: "1", URL: FALPRS_URL + "/einstein_002.jpg"}
    response = requests.post(url, json=data)
    assert response.status_code == 200

    data = response.json()
    global face_id1
    face_id1 = data[DATA][FACE_ID]
    assert face_id1 > 0
    assert len(data[DATA][FACE_IMAGE]) > 0
    assert LEFT in data[DATA]
    assert TOP in data[DATA]
    assert WIDTH in data[DATA]
    assert HEIGHT in data[DATA]

# listAllFaces
@pytest.mark.order(++order)
def test_list_all_faces():
    url = API_URL + "listAllFaces"
    response = requests.post(url)
    assert response.status_code == 200

    data = response.json()
    assert len(data[DATA]) == 1
    global face_id1
    assert data[DATA][0] == face_id1

# listStreams 5
@pytest.mark.order(++order)
def test_list_streams5():
    url = API_URL + "listStreams"
    response = requests.post(url)
    assert response.status_code == 200

    data = response.json()
    assert len(data[DATA]) == 2
    ids = set()
    faces = set()
    for item in data[DATA]:
        ids.add(item[STREAM_ID])
        if item[STREAM_ID] == "1":
            faces.update(set(item[FACES]))
    assert ids == set(["1", "2"])
    global face_id1
    assert faces == set([face_id1])

# add non-existent face
@pytest.mark.order(++order)
def test_add_faces():
    url = API_URL + "addFaces"
    data = {STREAM_ID: "1", FACES: [100, 101]}
    response = requests.post(url, json=data)
    assert response.status_code == 204

# add existent face to non-existent video stream
@pytest.mark.order(++order)
def test_add_faces2():
    url = API_URL + "addFaces"
    global face_id1
    data = {STREAM_ID: "10", FACES: [face_id1]}
    response = requests.post(url, json=data)
    assert response.status_code == 204

# listStreams 6
@pytest.mark.order(++order)
def test_list_streams6():
    test_list_streams5()

# add existent face to existent video stream
@pytest.mark.order(++order)
def test_add_faces3():
    url = API_URL + "addFaces"
    global face_id1
    data = {STREAM_ID: "2", FACES: [face_id1]}
    response = requests.post(url, json=data)
    assert response.status_code == 204

# listStreams 7
@pytest.mark.order(++order)
def test_list_streams7():
    url = API_URL + "listStreams"
    response = requests.post(url)
    assert response.status_code == 200

    data = response.json()
    assert len(data[DATA]) == 2
    ids = set()
    faces1 = set()
    faces2 = set()
    for item in data[DATA]:
        ids.add(item[STREAM_ID])
        if item[STREAM_ID] == "1":
            faces1.update(set(item[FACES]))
        if item[STREAM_ID] == "2":
            faces2.update(set(item[FACES]))
    assert ids == set(["1", "2"])
    global face_id1
    assert faces1 == set([face_id1])
    assert faces2 == set([face_id1])

# removeFaces
@pytest.mark.order(++order)
def test_remove_faces():
    url = API_URL + "removeFaces"
    global face_id1
    data = {STREAM_ID: "1", FACES: [face_id1, 2, 3, 4, 5]}
    response = requests.post(url, json=data)
    assert response.status_code == 204

# listStreams 8
@pytest.mark.order(++order)
def test_list_streams8():
    url = API_URL + "listStreams"
    response = requests.post(url)
    assert response.status_code == 200

    data = response.json()
    assert len(data[DATA]) == 2
    ids = set()
    faces1 = set()
    faces2 = set()
    for item in data[DATA]:
        ids.add(item[STREAM_ID])
        if item[STREAM_ID] == "1":
            if FACES in item:
                faces1.update(set(item[FACES]))
        if item[STREAM_ID] == "2":
            if FACES in item:
                faces2.update(set(item[FACES]))
    assert ids == set(["1", "2"])
    assert faces1 == set()
    global face_id1
    assert faces2 == set([face_id1])

# listAllFaces 2
@pytest.mark.order(++order)
def test_list_all_faces2():
    test_list_all_faces()

# registerFace 2
@pytest.mark.order(++order)
def test_register_face2():
    # waiting for cache
    print("[Wait 6 seconds for cache...]")
    time.sleep(6)

    url = API_URL + "registerFace"
    data = {STREAM_ID: "1", URL: FALPRS_URL + "/hanks_001.jpg"}
    response = requests.post(url, json=data)
    assert response.status_code == 200

    data = response.json()
    global face_id2
    face_id2 = data[DATA][FACE_ID]
    assert face_id2 > 0
    assert len(data[DATA][FACE_IMAGE]) > 0
    assert LEFT in data[DATA]
    assert TOP in data[DATA]
    assert WIDTH in data[DATA]
    assert HEIGHT in data[DATA]

# listAllFaces 3
@pytest.mark.order(++order)
def test_list_all_faces3():
    url = API_URL + "listAllFaces"
    response = requests.post(url)
    assert response.status_code == 200

    data = response.json()
    global face_id1
    global face_id2
    assert set(data[DATA]) == set([face_id1, face_id2])

# listStreams 9
@pytest.mark.order(++order)
def test_list_streams9():
    url = API_URL + "listStreams"
    response = requests.post(url)
    assert response.status_code == 200

    data = response.json()
    assert len(data[DATA]) == 2
    ids = set()
    faces1 = set()
    faces2 = set()
    for item in data[DATA]:
        ids.add(item[STREAM_ID])
        if item[STREAM_ID] == "1":
            if FACES in item:
                faces1.update(set(item[FACES]))
        if item[STREAM_ID] == "2":
            if FACES in item:
                faces2.update(set(item[FACES]))
    assert ids == set(["1", "2"])
    global face_id1
    global face_id2
    assert faces1 == set([face_id2])
    assert faces2 == set([face_id1])

# deleteFaces
@pytest.mark.order(++order)
def test_delete_faces():
    url = API_URL + "deleteFaces"
    global face_id1
    data = {FACES: [face_id1]}
    response = requests.post(url, json=data)
    assert response.status_code == 204

# listAllFaces 4
@pytest.mark.order(++order)
def test_list_all_faces4():
    url = API_URL + "listAllFaces"
    response = requests.post(url)
    assert response.status_code == 200

    data = response.json()
    global face_id2
    assert set(data[DATA]) == set([face_id2])

# listStreams 10
@pytest.mark.order(++order)
def test_list_streams10():
    url = API_URL + "listStreams"
    response = requests.post(url)
    assert response.status_code == 200

    data = response.json()
    assert len(data[DATA]) == 2
    ids = set()
    faces1 = set()
    faces2 = set()
    for item in data[DATA]:
        ids.add(item[STREAM_ID])
        if item[STREAM_ID] == "1":
            if FACES in item:
                faces1.update(set(item[FACES]))
        if item[STREAM_ID] == "2":
            if FACES in item:
                faces2.update(set(item[FACES]))
    assert ids == set(["1", "2"])
    global face_id2
    assert faces1 == set([face_id2])
    assert faces2 == set()

# removeStream 2
@pytest.mark.order(++order)
def test_remove_stream2():
    url = API_URL + "removeStream"
    data = {STREAM_ID: "1"}
    response = requests.post(url, json=data)
    assert response.status_code == 204

# listStreams 11
@pytest.mark.order(++order)
def test_list_streams11():
    url = API_URL + "listStreams"
    response = requests.post(url)
    assert response.status_code == 200

    data = response.json()
    assert len(data[DATA]) == 1
    assert data[DATA][0][STREAM_ID] == "2"
    assert not (FACES in data[DATA][0])

# listAllFaces 5
@pytest.mark.order(++order)
def test_list_all_faces5():
    test_list_all_faces4()

# deleteFaces 2
@pytest.mark.order(++order)
def test_delete_faces2():
    url = API_URL + "deleteFaces"
    global face_id2
    data = {FACES: [face_id2]}
    response = requests.post(url, json=data)
    assert response.status_code == 204

# listAllFaces 6
@pytest.mark.order(++order)
def test_list_all_faces6():
    url = API_URL + "listAllFaces"
    response = requests.post(url)
    assert response.status_code == 204

# removeStream 3
@pytest.mark.order(++order)
def test_remove_stream3():
    url = API_URL + "removeStream"
    data = {STREAM_ID: "2"}
    response = requests.post(url, json=data)
    assert response.status_code == 204

# listStreams 12
@pytest.mark.order(++order)
def test_list_streams12():
    url = API_URL + "listStreams"
    response = requests.post(url)
    assert response.status_code == 204

# addStream with streamId="1" again
@pytest.mark.order(++order)
def test_add_stream5():
    test_add_stream()

# start / stop motion
@pytest.mark.order(++order)
def test_start_stop_motion():
    # waiting for cache
    print("[Wait 6 seconds for cache...]")
    time.sleep(6)

    url = API_URL + "motionDetection"
    data = {STREAM_ID: "1", START: True}
    response = requests.post(url, json=data)
    assert response.status_code == 204
    time.sleep(0.3)
    url = API_URL + "motionDetection"
    data = {STREAM_ID: "1", START: "f"}
    response = requests.post(url, json=data)
    assert response.status_code == 204

# bestQuality ISO
@pytest.mark.order(++order)
def test_best_quality_iso():
    url = API_URL + "bestQuality"
    global tp
    tp = datetime.now()
    data = {STREAM_ID: "1", DATE: tp.isoformat()}
    response = requests.post(url, json=data)
    assert response.status_code == 200

    data = response.json()
    assert not (FACE_ID in data[DATA])
    assert SCREENSHOT_URL in data[DATA]
    assert LEFT in data[DATA]
    assert TOP in data[DATA]
    assert WIDTH in data[DATA]
    assert HEIGHT in data[DATA]

# bestQuality
@pytest.mark.order(++order)
def test_best_quality():
    url = API_URL + "bestQuality"
    global tp
    data = {STREAM_ID: "1", DATE: tp.strftime("%Y-%m-%d %H:%M:%S")}
    response = requests.post(url, json=data)
    assert response.status_code == 200

    data = response.json()
    assert not (FACE_ID in data[DATA])
    assert SCREENSHOT_URL in data[DATA]
    assert LEFT in data[DATA]
    assert TOP in data[DATA]
    assert WIDTH in data[DATA]
    assert HEIGHT in data[DATA]

# registerFace 3
@pytest.mark.order(++order)
def test_register_face3():
    url = API_URL + "registerFace"
    data = {STREAM_ID: "1", URL: FALPRS_URL + "/einstein_002.jpg"}
    response = requests.post(url, json=data)
    assert response.status_code == 200

    data = response.json()
    global face_id1
    face_id1 = data[DATA][FACE_ID]
    assert face_id1 > 0
    assert len(data[DATA][FACE_IMAGE]) > 0
    assert LEFT in data[DATA]
    assert TOP in data[DATA]
    assert WIDTH in data[DATA]
    assert HEIGHT in data[DATA]

# start / stop motion 2
@pytest.mark.order(++order)
def test_start_stop_motion2():
    test_start_stop_motion()

# bestQuality ISO 2
@pytest.mark.order(++order)
def test_best_quality_iso2():
    url = API_URL + "bestQuality"
    tp = datetime.now()
    data = {STREAM_ID: "1", DATE: tp.isoformat()}
    response = requests.post(url, json=data)
    assert response.status_code == 200
    data = response.json()
    assert SCREENSHOT_URL in data[DATA]
    assert LEFT in data[DATA]
    assert TOP in data[DATA]
    assert WIDTH in data[DATA]
    assert HEIGHT in data[DATA]

# addSpecialGroup
@pytest.mark.order(++order)
def test_add_special_group():
    url = API_URL + "addSpecialGroup"
    data = {GROUP_NAME: "Test Group 1"}
    response = requests.post(url, json=data)
    assert response.status_code == 200
    data = response.json()
    assert GROUP_ID in data[DATA]
    assert ACCESS_API_TOKEN in data[DATA]

    global sg_id
    global sg_api_token
    sg_id = data[DATA][GROUP_ID]
    sg_api_token = data[DATA][ACCESS_API_TOKEN]

# listSpecialGroups
@pytest.mark.order(++order)
def test_list_special_groups():
    url = API_URL + "listSpecialGroups"
    response = requests.post(url)
    assert response.status_code == 200
    data = response.json()
    assert len(data[DATA]) == 1
    assert GROUP_ID in data[DATA][0]
    assert GROUP_NAME in data[DATA][0]
    assert ACCESS_API_TOKEN in data[DATA][0]
    assert MAX_DESCRIPTOR_COUNT in data[DATA][0]
    global sg_id
    global sg_api_token
    assert sg_id == data[DATA][0][GROUP_ID]
    assert sg_api_token == data[DATA][0][ACCESS_API_TOKEN]

# updateSpecialGroup
@pytest.mark.order(++order)
def test_update_special_group():
    url = API_URL + "updateSpecialGroup"
    data = {GROUP_NAME: "New name"}
    response = requests.post(url, json=data)
    assert response.status_code == 400

# updateSpecialGroup 2
@pytest.mark.order(++order)
def test_update_special_group2():
    url = API_URL + "updateSpecialGroup"
    global sg_id
    data = {GROUP_ID: sg_id + 10, GROUP_NAME: "New name 1"}
    response = requests.post(url, json=data)
    assert response.status_code == 204

# listSpecialGroups 2
@pytest.mark.order(++order)
def test_list_special_groups2():
    url = API_URL + "listSpecialGroups"
    response = requests.post(url)
    assert response.status_code == 200
    data = response.json()
    assert len(data[DATA]) == 1
    assert GROUP_ID in data[DATA][0]
    assert GROUP_NAME in data[DATA][0]
    assert ACCESS_API_TOKEN in data[DATA][0]
    assert MAX_DESCRIPTOR_COUNT in data[DATA][0]
    global sg_id
    global sg_api_token
    assert sg_id == data[DATA][0][GROUP_ID]
    assert sg_api_token == data[DATA][0][ACCESS_API_TOKEN]
    assert "Test Group 1" == data[DATA][0][GROUP_NAME]

# updateSpecialGroup 3
@pytest.mark.order(++order)
def test_update_special_group3():
    url = API_URL + "updateSpecialGroup"
    global sg_id
    data = {GROUP_ID: sg_id, GROUP_NAME: "Group 1", MAX_DESCRIPTOR_COUNT: 500}
    response = requests.post(url, json=data)
    assert response.status_code == 204

# listSpecialGroups 3
@pytest.mark.order(++order)
def test_list_special_groups3():
    url = API_URL + "listSpecialGroups"
    response = requests.post(url)
    assert response.status_code == 200
    data = response.json()
    assert len(data[DATA]) == 1
    assert GROUP_ID in data[DATA][0]
    assert GROUP_NAME in data[DATA][0]
    assert ACCESS_API_TOKEN in data[DATA][0]
    assert MAX_DESCRIPTOR_COUNT in data[DATA][0]
    global sg_id
    global sg_api_token
    assert sg_id == data[DATA][0][GROUP_ID]
    assert sg_api_token == data[DATA][0][ACCESS_API_TOKEN]
    assert "Group 1" == data[DATA][0][GROUP_NAME]
    assert 500 == data[DATA][0][MAX_DESCRIPTOR_COUNT]

# sgRenewToken
@pytest.mark.order(++order)
def test_renew_special_group_token():
    # waiting for cache
    print("[Wait 6 seconds for cache...]")
    time.sleep(6)

    url = API_URL + "sgRenewToken"
    global sg_api_token
    headers = {"Authorization": "Bearer " + sg_api_token}
    response = requests.post(url, headers=headers)
    assert response.status_code == 200
    data = response.json()
    sg_api_token = data[DATA][ACCESS_API_TOKEN]

# sgRenewToken 2
@pytest.mark.order(++order)
def test_renew_special_group_token2():
    url = API_URL + "sgRenewToken"
    global sg_api_token
    headers = {"Authorization": "Bearer 1234567"}
    response = requests.post(url, headers=headers)
    assert response.status_code == 401

# sgRegisterFace
@pytest.mark.order(++order)
def test_sg_register_face():
    # waiting for cache
    print("[Wait 6 seconds for cache...]")
    time.sleep(6)

    url = API_URL + "sgRegisterFace"
    global sg_api_token
    headers = {"Authorization": "Bearer " + sg_api_token}
    data = {URL: FALPRS_URL + "/einstein_003.jpg"}
    response = requests.post(url, headers=headers, json=data)
    assert response.status_code == 200

    data = response.json()
    global face_id1
    face_id1 = data[DATA][FACE_ID]
    assert face_id1 > 0
    assert len(data[DATA][FACE_IMAGE]) > 0
    assert LEFT in data[DATA]
    assert TOP in data[DATA]
    assert WIDTH in data[DATA]
    assert HEIGHT in data[DATA]

# sgListFaces
@pytest.mark.order(++order)
def test_sg_list_faces():
    url = API_URL + "sgListFaces"
    global sg_api_token
    headers = {"Authorization": "Bearer " + sg_api_token}
    response = requests.post(url, headers=headers)
    assert response.status_code == 200

    data = response.json()
    global face_id1
    assert len(data[DATA]) == 1
    assert data[DATA][0][FACE_ID] == face_id1
    assert len(data[DATA][0][FACE_IMAGE]) > 0

# sgSearchFaces
@pytest.mark.order(++order)
def test_sg_search_faces():
    # waiting for cache
    print("[Wait 6 seconds for cache...]")
    time.sleep(6)

    url = API_URL + "sgSearchFaces"
    global sg_api_token
    global face_id1
    global tp
    headers = {"Authorization": "Bearer " + sg_api_token}
    data = {FACES: [face_id1], DATE_START: tp.strftime("%Y-%m-%d"), DATE_END: tp.strftime("%Y-%m-%d"), SIMILARITY_THRESHOLD: 0.48}
    response = requests.post(url, headers=headers, json=data)
    assert response.status_code == 200
    data = response.json()
    assert len(data[DATA]) == 2
    for item in data[DATA]:
        assert DATE in item
        assert UUID in item
        assert EVENT_ID in item
        assert URL in item
        assert FACE_ID in item
        assert SIMILARITY in item

# sgSearchFaces with empty faces
@pytest.mark.order(++order)
def test_sg_search_faces2():
    url = API_URL + "sgSearchFaces"
    global sg_api_token
    global tp
    headers = {"Authorization": "Bearer " + sg_api_token}
    data = {FACES: [], DATE_START: tp.strftime("%Y-%m-%d"), DATE_END: tp.strftime("%Y-%m-%d"), SIMILARITY_THRESHOLD: 0.5}
    response = requests.post(url, headers=headers, json=data)
    assert response.status_code == 400

# sgSearchFaces with non-existent face
@pytest.mark.order(++order)
def test_sg_search_faces3():
    url = API_URL + "sgSearchFaces"
    global sg_api_token
    global face_id1
    global tp
    headers = {"Authorization": "Bearer " + sg_api_token}
    data = {FACES: [face_id1 + 100], DATE_START: tp.strftime("%Y-%m-%d"), DATE_END: tp.strftime("%Y-%m-%d"), SIMILARITY_THRESHOLD: 0.5}
    response = requests.post(url, headers=headers, json=data)
    assert response.status_code == 204

# sgSearchFaces in the future
@pytest.mark.order(++order)
def test_sg_search_faces4():
    url = API_URL + "sgSearchFaces"
    global sg_api_token
    global face_id1
    global tp
    headers = {"Authorization": "Bearer " + sg_api_token}
    data = {FACES: [face_id1], DATE_START: (tp + timedelta(days=1)).strftime("%Y-%m-%d"), DATE_END: (tp + timedelta(days=2)).strftime("%Y-%m-%d"), SIMILARITY_THRESHOLD: 0.5}
    response = requests.post(url, headers=headers, json=data)
    assert response.status_code == 204

# sgSearchFaces in the past
@pytest.mark.order(++order)
def test_sg_search_faces5():
    url = API_URL + "sgSearchFaces"
    global sg_api_token
    global face_id1
    global tp
    headers = {"Authorization": "Bearer " + sg_api_token}
    data = {FACES: [face_id1], DATE_START: (tp - timedelta(days=2)).strftime("%Y-%m-%d"), DATE_END: (tp - timedelta(days=1)).strftime("%Y-%m-%d"), SIMILARITY_THRESHOLD: 0.5}
    response = requests.post(url, headers=headers, json=data)
    assert response.status_code == 204

# sgSearchFaces without dateStart
@pytest.mark.order(++order)
def test_sg_search_faces6():
    url = API_URL + "sgSearchFaces"
    global sg_api_token
    global tp
    headers = {"Authorization": "Bearer " + sg_api_token}
    data = {FACES: [1, 2, 3], DATE_END: tp.strftime("%Y-%m-%d"), SIMILARITY_THRESHOLD: 0.5}
    response = requests.post(url, headers=headers, json=data)
    assert response.status_code == 400

# sgSearchFaces without dateEnd
@pytest.mark.order(++order)
def test_sg_search_faces7():
    url = API_URL + "sgSearchFaces"
    global sg_api_token
    global tp
    headers = {"Authorization": "Bearer " + sg_api_token}
    data = {FACES: [1, 2, 3], DATE_START: tp.strftime("%Y-%m-%d"), SIMILARITY_THRESHOLD: 0.5}
    response = requests.post(url, headers=headers, json=data)
    assert response.status_code == 400

# sgSearchFaces with high similarity threshold
@pytest.mark.order(++order)
def test_sg_search_faces8():
    url = API_URL + "sgSearchFaces"
    global sg_api_token
    global face_id1
    global tp
    headers = {"Authorization": "Bearer " + sg_api_token}
    data = {FACES: [face_id1], DATE_START: tp.strftime("%Y-%m-%d"), DATE_END: tp.strftime("%Y-%m-%d"), SIMILARITY_THRESHOLD: 0.9}
    response = requests.post(url, headers=headers, json=data)
    assert response.status_code == 204

# sgSearchFaces with larger range
@pytest.mark.order(++order)
def test_sg_search_faces9():
    url = API_URL + "sgSearchFaces"
    global sg_api_token
    global face_id1
    global tp
    headers = {"Authorization": "Bearer " + sg_api_token}
    data = {FACES: [face_id1], DATE_START: (tp - timedelta(days=4)).strftime("%Y-%m-%d"), DATE_END: (tp + timedelta(days=5)).strftime("%Y-%m-%d"), SIMILARITY_THRESHOLD: 0.48}
    response = requests.post(url, headers=headers, json=data)
    assert response.status_code == 200
    data = response.json()
    assert len(data[DATA]) == 2
    for item in data[DATA]:
        assert DATE in item
        assert UUID in item
        assert EVENT_ID in item
        assert URL in item
        assert FACE_ID in item
        assert SIMILARITY in item

# sgDeleteFaces non-existent face
@pytest.mark.order(++order)
def test_sg_delete_faces():
    url = API_URL + "sgDeleteFaces"
    global sg_api_token
    global face_id1
    global tp
    headers = {"Authorization": "Bearer " + sg_api_token}
    data = {FACES: [face_id1 + 200]}
    response = requests.post(url, headers=headers, json=data)
    assert response.status_code == 204

# sgListFaces 2
@pytest.mark.order(++order)
def test_sg_list_faces2():
    test_sg_list_faces()

# sgDeleteFaces existent
@pytest.mark.order(++order)
def test_sg_delete_faces2():
    url = API_URL + "sgDeleteFaces"
    global sg_api_token
    global face_id1
    global tp
    headers = {"Authorization": "Bearer " + sg_api_token}
    data = {FACES: [face_id1]}
    response = requests.post(url, headers=headers, json=data)
    assert response.status_code == 204

# sgListFaces 3
@pytest.mark.order(++order)
def test_sg_list_faces3():
    url = API_URL + "sgListFaces"
    global sg_api_token
    headers = {"Authorization": "Bearer " + sg_api_token}
    response = requests.post(url, headers=headers)
    assert response.status_code == 204

# addSpecialGroup existent
@pytest.mark.order(++order)
def test_add_special_group2():
    url = API_URL + "addSpecialGroup"
    data = {GROUP_NAME: "Group 1"}
    response = requests.post(url, json=data)
    assert response.status_code == 400

# deleteSpecialGroup non-existent
@pytest.mark.order(++order)
def test_delete_special_group():
    url = API_URL + "deleteSpecialGroup"
    global sg_id
    data = {GROUP_ID: sg_id + 50}
    response = requests.post(url, json=data)
    assert response.status_code == 204

# listSpecialGroups 4
@pytest.mark.order(++order)
def test_list_special_groups4():
    test_list_special_groups3()

# deleteSpecialGroup non-existent
@pytest.mark.order(++order)
def test_delete_special_group2():
    url = API_URL + "deleteSpecialGroup"
    global sg_id
    data = {GROUP_ID: sg_id}
    response = requests.post(url, json=data)
    assert response.status_code == 204

# listSpecialGroups 5
@pytest.mark.order(++order)
def test_list_special_groups5():
    url = API_URL + "listSpecialGroups"
    response = requests.post(url)
    assert response.status_code == 204

# removeStream 4
@pytest.mark.order(++order)
def test_remove_stream4():
    url = API_URL + "removeStream"
    data = {STREAM_ID: "1"}
    response = requests.post(url, json=data)
    assert response.status_code == 204

# listAllFaces 7
@pytest.mark.order(++order)
def test_list_all_faces7():
    url = API_URL + "listAllFaces"
    response = requests.post(url)
    assert response.status_code == 200

    data = response.json()
    assert len(data[DATA]) == 1
    global face_id1
    face_id1 = data[DATA][0]

# deleteFaces 3
@pytest.mark.order(++order)
def test_delete_faces3():
    test_delete_faces()

# listAllFaces 8
@pytest.mark.order(++order)
def test_list_all_faces8():
    test_list_all_faces6()
