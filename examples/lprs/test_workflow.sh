#!/bin/bash

curl --request POST \
  --url http://localhost:9051/lprs/api/startWorkflow \
  --header 'Content-Type: application/json' \
  --data '{
	      "streamId": "test001"
          }'

sleep 0.3

curl --request POST \
  --url http://localhost:9051/lprs/api/stopWorkflow \
  --header 'Content-Type: application/json' \
  --data '{
	      "streamId": "test001"
          }'
