#!/bin/bash

# Update script for the FALPRS project
# Optimizes update process to minimize downtime.

set -e

BASEDIR=$(realpath "$(dirname "$0")")

# Load configuration from file if exists
if [ -f "$BASEDIR/.env" ]; then
    source $BASEDIR/.env
elif [ -f "$BASEDIR/../.env" ]; then
    source $BASEDIR/../.env
fi

# External variables used in the script
# PG_VERSION - PostgreSQL database system version
# TRITON_VERSION - NVIDIA Triton Inference Server container image version
# FALPRS_WORKDIR - FALPRS working directory

# Set default values if not provided
# Auto-detect Ubuntu version if not set
if [ -f /etc/os-release ]; then
    . /etc/os-release
    UBUNTU_VERSION=$VERSION_ID
fi

if [ -z "$PG_VERSION" ]; then
    case $UBUNTU_VERSION in
        "24.04")
            PG_VERSION=16
            ;;
        "26.04")
            PG_VERSION=18
            ;;
        *)
            PG_VERSION=16
            ;;
    esac
fi
export TRITON_VERSION=${TRITON_VERSION:-24.09}
export FALPRS_WORKDIR=${FALPRS_WORKDIR:-/opt/falprs}
export FALPRS_REPOSITORY_URL=${FALPRS_REPOSITORY_URL:-https://github.com/rosteleset/falprs}

# Extract project version from CMakeLists.txt
CMAKELISTS_FILE="$BASEDIR/../CMakeLists.txt"
REQUIRED_VERSION=""
if [ -f "$CMAKELISTS_FILE" ]; then
    REQUIRED_VERSION=$(grep -oiP 'project\s*\(\s*falprs\s+VERSION\s+\K[0-9]+(\.[0-9]+)+' "$CMAKELISTS_FILE" | head -n 1 | tr -d '[:space:]')
fi

# Determine installed version
INSTALLED_VERSION=""
if [ -x "$FALPRS_WORKDIR/falprs" ]; then
    if RAW_OUT=$("$FALPRS_WORKDIR/falprs" --version 2>/dev/null); then
        TRIMMED_OUT=$(echo "$RAW_OUT" | tr -d '[:space:]')
        if [[ "$TRIMMED_OUT" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
            INSTALLED_VERSION="$TRIMMED_OUT"
        fi
    fi
fi

# Check if rebuild is needed
NEED_BUILD=true
if [ -n "$INSTALLED_VERSION" ] && [ -n "$REQUIRED_VERSION" ] && [ "$INSTALLED_VERSION" = "$REQUIRED_VERSION" ]; then
    echo "Current FALPRS version: $INSTALLED_VERSION"
    echo "Required FALPRS version: $REQUIRED_VERSION"
    echo "FALPRS executable is up to date."
    NEED_BUILD=false
else
    if [ -n "$INSTALLED_VERSION" ]; then
        echo "Current FALPRS version: $INSTALLED_VERSION"
    else
        echo "Current FALPRS version: unknown"
    fi
    echo "Required FALPRS version: $REQUIRED_VERSION"
    echo "FALPRS executable needs to be rebuilt."
    NEED_BUILD=true
fi

# Build project if needed without stopping service
if [ "$NEED_BUILD" = true ]; then
    echo "Building the project..."
    $BASEDIR/build_falprs.sh
fi

# TensorRT Planning phase (before stopping services)
echo "Planning TensorRT plans..."
cd $BASEDIR/..

PLAN_STATUS=0
if [ -n "$ARCFACE_SHA1" ]; then
    sudo TRITON_VERSION=$TRITON_VERSION FALPRS_WORKDIR=$FALPRS_WORKDIR ARCFACE_SHA1=$ARCFACE_SHA1 python3 ./scripts/tensorrt_plans.py plan || PLAN_STATUS=$?
else
    sudo TRITON_VERSION=$TRITON_VERSION FALPRS_WORKDIR=$FALPRS_WORKDIR python3 ./scripts/tensorrt_plans.py plan || PLAN_STATUS=$?
fi

NEEDS_GEN=false
MODELS_TO_GEN=""
if [ $PLAN_STATUS -eq 0 ]; then
    NEEDS_GEN=false
elif [ $PLAN_STATUS -eq 10 ]; then
    NEEDS_GEN=true
    if [ -f "/tmp/falprs_models_to_gen" ]; then
        MODELS_TO_GEN=$(cat /tmp/falprs_models_to_gen)
        rm -f /tmp/falprs_models_to_gen
    fi
else
    echo "Error during TensorRT planning."
    exit $PLAN_STATUS
fi

if [[ "$NEED_BUILD" == true || "$NEEDS_GEN" == true ]]; then
    FALPRS_WAS_ACTIVE=false
    if systemctl is-active --quiet falprs.service; then
        FALPRS_WAS_ACTIVE=true
    fi
    if [ "$FALPRS_WAS_ACTIVE" = true ]; then
        echo "Stopping falprs service..."
        sudo systemctl stop falprs.service
    fi
    if [ "$NEED_BUILD" = true ]; then
        echo "Copying new executable..."
        cp "$BASEDIR/../build/falprs" "$FALPRS_WORKDIR"
    fi
fi

if [ "$NEEDS_GEN" = true ]; then
    TRITON_CONTAINER_ID=$(sudo docker ps -q --filter "ancestor=nvcr.io/nvidia/tritonserver:$TRITON_VERSION-py3")
    if [ -n "$TRITON_CONTAINER_ID" ]; then
        echo "Stopping Triton Inference Server container..."
        sudo docker stop $TRITON_CONTAINER_ID
    fi

    echo "Creating TensorRT neural network model plans..."
    GEN_ERROR=0
    if [ -n "$ARCFACE_SHA1" ]; then
        sudo TRITON_VERSION=$TRITON_VERSION FALPRS_WORKDIR=$FALPRS_WORKDIR ARCFACE_SHA1=$ARCFACE_SHA1 python3 ./scripts/tensorrt_plans.py generate $MODELS_TO_GEN || GEN_ERROR=$?
    else
        sudo TRITON_VERSION=$TRITON_VERSION FALPRS_WORKDIR=$FALPRS_WORKDIR python3 ./scripts/tensorrt_plans.py generate $MODELS_TO_GEN || GEN_ERROR=$?
    fi

    if [ "$GEN_ERROR" -ne 0 ]; then
        echo "Error: TensorRT generation failed."
        exit $GEN_ERROR
    fi

    # Restore Triton Inference Server to its initial state
    if [ -n "$TRITON_CONTAINER_ID" ]; then
        echo "Starting Triton Inference Server container..."
        sudo docker start $TRITON_CONTAINER_ID
    fi
fi

# Update DB schema
echo "Updating database schema..."
$BASEDIR/sql_frs.sh
$BASEDIR/sql_lprs.sh

# Restore FALPRS to its initial state
if [ "$FALPRS_WAS_ACTIVE" = true ]; then
    echo "Starting falprs service..."
    sudo systemctl start falprs.service
fi

echo "Project updated successfully."
