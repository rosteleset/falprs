#!/bin/bash

curl --request POST \
  --url http://localhost:9051/lprs/api/listStreams \
  --header 'Content-Type: application/json'
