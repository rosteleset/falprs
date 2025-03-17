#!/bin/bash

curl --request POST \
  --url http://localhost:9051/frs/api/removeStream \
  --header 'Content-Type: application/json' \
  --data '{
              "streamId": "test001"
          }'
