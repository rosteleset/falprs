#!/bin/bash

# Update script for the FALPRS project
# Stops services, builds the project, regenerates TensorRT plans, and restarts services.

set -e

BASEDIR=$(realpath `dirname $0`)

# Load configuration from file if exists
if [ -f "$BASEDIR/.env" ]; then
    source $BASEDIR/.env
elif [ -f "$BASEDIR/../.env" ]; then
    source $BASEDIR/../.env
fi

# External variables used in the script
# PG_VERSION - PostgreSQL database system version
# TRITON_VERSION - NVIDIA Triton Inference Server version
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

echo "Stopping services..."

# Stop falprs service
if systemctl is-active --quiet falprs.service; then
    echo "Stopping falprs service..."
    sudo systemctl stop falprs.service
fi

# Stop Triton container
echo "Stopping Triton Inference Server container..."
# Find container by image
CONTAINER_ID=$(docker ps -q --filter "ancestor=nvcr.io/nvidia/tritonserver:$TRITON_VERSION-py3")
if [ -n "$CONTAINER_ID" ]; then
    sudo docker stop $CONTAINER_ID
fi

# Build the project
echo "Building the project..."
# Ensure we are in the project root for consistency
cd $BASEDIR/..
sudo PG_VERSION=$PG_VERSION TRITON_VERSION=$TRITON_VERSION FALPRS_WORKDIR=$FALPRS_WORKDIR ./scripts/build_falprs.sh

# Create TensorRT plans
echo "Creating TensorRT neural network model plans..."
if [ -n "$ARCFACE_SHA1" ]; then
    sudo TRITON_VERSION=$TRITON_VERSION FALPRS_WORKDIR=$FALPRS_WORKDIR ARCFACE_SHA1=$ARCFACE_SHA1 python3 ./scripts/tensorrt_plans.py
else
    sudo TRITON_VERSION=$TRITON_VERSION FALPRS_WORKDIR=$FALPRS_WORKDIR python3 ./scripts/tensorrt_plans.py
fi

# Start Triton container
echo "Starting Triton Inference Server container..."
if [ -n "$CONTAINER_ID" ]; then
    sudo docker start $CONTAINER_ID
else
    sudo TRITON_VERSION=$TRITON_VERSION FALPRS_WORKDIR=$FALPRS_WORKDIR ./scripts/triton_service.sh
fi

# Update DB schema
echo "Updating database schema..."
PG_USER_FRS=$PG_USER_FRS PG_PASSWD_FRS=$PG_PASSWD_FRS PG_HOST_FRS=$PG_HOST_FRS PG_PORT_FRS=$PG_PORT_FRS PG_DB_FRS=frs ./scripts/sql_frs.sh
PG_USER_LPRS=$PG_USER_LPRS PG_PASSWD_LPRS=$PG_PASSWD_LPRS PG_HOST_LPRS=$PG_HOST_LPRS PG_PORT_LPRS=$PG_PORT_LPRS PG_DB_LPRS=lprs ./scripts/sql_lprs.sh

# Start falprs service
echo "Starting falprs service..."
sudo systemctl start falprs.service

echo "Project updated successfully."
