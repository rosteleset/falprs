#!/bin/bash

curl --request POST \
  --url http://localhost:9051/frs/api/listAllFaces \
  --header 'Content-Type: application/json'
