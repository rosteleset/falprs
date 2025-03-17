#!/bin/bash
curl --request POST \
  --url http://localhost:9051/lprs/api/addStream \
  --header 'Content-Type: application/json' \
  --data '{
              "streamId": "test001",
              "config": {
                  "screenshot-url": "http://localhost:9051/test001.jpg",
                  "callback-url": "http://localhost:12346/callback"
              }
          }'
