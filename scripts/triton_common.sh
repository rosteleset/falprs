#!/bin/bash

# Shared by triton_service.sh and triton_recreate.sh: the environment and the command that starts
# the Triton Inference Server container. Sourced, not run.
#
# External variables used here
# TRITON_VERSION - NVIDIA Triton Inference Server container image version
# FALPRS_WORKDIR - FALPRS working directory

TRITON_CONTAINER=falprs-triton
TRITON_IMAGE=nvcr.io/nvidia/tritonserver

# Load configuration from file if exists
load_triton_env() {
    local basedir=$1
    if [ -f "$basedir/.env" ]; then
        source "$basedir/.env"
    elif [ -f "$basedir/../.env" ]; then
        source "$basedir/../.env"
    fi
    export FALPRS_WORKDIR="${FALPRS_WORKDIR:-/opt/falprs}"
}

# Starts the container. --ipc=host shares /dev/shm with the host, so that FALPRS can pass the
# detector inputs through Triton system shared memory instead of sending them over gRPC.
run_triton() {
    sudo docker run --gpus all -d --restart unless-stopped --net=host --ipc=host --name "$TRITON_CONTAINER" \
        -v "$FALPRS_WORKDIR/model_repository:/models" "$TRITON_IMAGE:$TRITON_VERSION-py3" \
        sh -c "tritonserver --model-repository=/models"
}
