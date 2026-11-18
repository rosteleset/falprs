#!/bin/bash

# Starts the Triton Inference Server container. To replace a container that already exists
# (e.g. one started without --ipc=host), use triton_recreate.sh.
#
# External variables used in the script
# TRITON_VERSION - NVIDIA Triton Inference Server container image version
# FALPRS_WORKDIR - FALPRS working directory

BASEDIR=$(realpath "$(dirname "$0")")
source "$BASEDIR/triton_common.sh"
load_triton_env "$BASEDIR"
export TRITON_VERSION="${TRITON_VERSION:-24.09}"

run_triton
