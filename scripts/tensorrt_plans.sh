#!/bin/bash

# Creates TensorRT plans

set -e

BASEDIR=$(realpath `dirname $0`)

# Load configuration from file if exists
if [ -f "$BASEDIR/.env" ]; then
    source $BASEDIR/.env
elif [ -f "$BASEDIR/../.env" ]; then
    source $BASEDIR/../.env
fi

# External variables used in the script
# TRITON_VERSION - NVIDIA Triton Inference Server version
# FALPRS_WORKDIR - FALPRS working directory
# ARCFACE_SHA1 - ONNX model file checksum when migrating from a deprecated FRS project (optional)

export TRITON_VERSION="${TRITON_VERSION:-24.09}"
export FALPRS_WORKDIR="${FALPRS_WORKDIR:-/opt/falprs}"
export ARCFACE_SHA1="${ARCFACE_SHA1:-3642b396053aa5e9cd4518de66baf0d26c9e1467}"
python3 $BASEDIR/tensorrt_plans.py
