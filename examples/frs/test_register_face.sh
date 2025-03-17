#!/bin/bash
curl --request POST \
  --url http://localhost:9051/frs/api/registerFace \
  --header 'Content-Type: application/json' \
  --data '{
              "streamId": "test001",
              "url": "http://localhost:9051/einstein_002.jpg"
          }' \
  -o /dev/null
