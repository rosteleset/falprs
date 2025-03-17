#!/bin/bash
curl --request POST \
  --url http://localhost:9051/frs/api/addStream \
  --header 'Content-Type: application/json' \
  --data '{
              "streamId": "test001",
              "url": "http://localhost:9051/einstein_001.jpg",
              "callback": "http://localhost:12347/callback"
          }'
