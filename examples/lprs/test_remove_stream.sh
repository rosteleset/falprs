#!/bin/bash

curl --request POST \
  --url http://localhost:9051/lprs/api/removeStream \
  --header 'Content-Type: application/json' \
  --data '{
              "streamId": "test001"
          }'
