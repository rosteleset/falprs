#!/bin/bash

# External variables used in the script
# TRITON_VERSION - NVIDIA Triton Inference Server version
# FALPRS_WORKDIR - FALPRS working directory

BASEDIR=$(realpath `dirname $0`)

# Load configuration from file if exists
if [ -f "$BASEDIR/.env" ]; then
    source $BASEDIR/.env
elif [ -f "$BASEDIR/../.env" ]; then
    source $BASEDIR/../.env
fi

export TRITON_VERSION="${TRITON_VERSION:-24.09}"
export FALPRS_WORKDIR="${FALPRS_WORKDIR:-/opt/falprs}"

sudo docker run --gpus all -d --restart unless-stopped --net=host -v $FALPRS_WORKDIR/model_repository:/models nvcr.io/nvidia/tritonserver:$TRITON_VERSION-py3 sh -c "tritonserver --model-repository=/models"
