#!/bin/bash

curl --request POST \
  --url http://localhost:9051/frs/api/motionDetection \
  --header 'Content-Type: application/json' \
  --data '{
	      "streamId": "test001",
              "start": true
          }'

sleep 0.3

curl --request POST \
  --url http://localhost:9051/frs/api/motionDetection \
  --header 'Content-Type: application/json' \
  --data '{
	      "streamId": "test001",
              "start": false
          }'
