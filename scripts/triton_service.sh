#!/bin/bash

# External variables used in the script

# TRITON_VERSION - NVIDIA Triton Inference Server version
# FALPRS_WORKDIR - FALPRS working directory

TRITON_VERSION="${TRITON_VERSION:=22.12}"
FALPRS_WORKDIR="${FALPRS_WORKDIR:=/opt/falprs}"

sudo docker run --gpus all -d --restart unless-stopped --net=host -v $FALPRS_WORKDIR/model_repository:/models nvcr.io/nvidia/tritonserver:$TRITON_VERSION-py3 sh -c "tritonserver --model-repository=/models"
