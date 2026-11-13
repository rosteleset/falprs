#!/bin/bash

# Update script for the FALPRS project
# Optimizes update process to minimize downtime.
#
# Without arguments all the phases run one after another. A phase can also be run on its own,
# e.g. to update the nodes of a cluster one at a time behind a load balancer:
#   --build    builds the executable and downloads the ONNX models; the running services
#              and the working directory are left as they are
#   --install  on a node taken out of service: stops FALPRS when it has to be replaced and Triton
#              when TensorRT plans have to be generated, installs the executable, converts the
#              config, generates the plans and starts Triton again; FALPRS is left stopped and
#              the databases are not touched
#   --finish   updates the database schemas, cleans the stored configs and starts FALPRS

set -e

BASEDIR=$(realpath "$(dirname "$0")")

PHASE=all
case "${1:-}" in
    "") ;;
    --build) PHASE=build ;;
    --install) PHASE=install ;;
    --finish) PHASE=finish ;;
    *) echo "Usage: $0 [--build | --install | --finish]" >&2; exit 1 ;;
esac

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
FALPRS_BINARY="$BASEDIR/../target/release/falprs"

# Extract the project version
REQUIRED_VERSION=""
CARGO_FILE="$BASEDIR/../Cargo.toml"
if [ -f "$CARGO_FILE" ]; then
    REQUIRED_VERSION=$(awk '/^\[workspace\.package\]/ { in_package = 1; next } /^\[/ { in_package = 0 }
        in_package && /^version[[:space:]]*=/ { gsub(/[^0-9.]/, "", $0); print; exit }' "$CARGO_FILE")
fi

# Determine the installed implementation; installations without the file are C++ ones.
# A C++ installation is replaced with the Rust one even when the version numbers match.
INSTALLED_IMPL=cpp
if [ -f "$FALPRS_WORKDIR/falprs.impl" ]; then
    INSTALLED_IMPL=$(tr -d '[:space:]' < "$FALPRS_WORKDIR/falprs.impl")
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
if [ -n "$INSTALLED_VERSION" ] && [ -n "$REQUIRED_VERSION" ] && [ "$INSTALLED_VERSION" = "$REQUIRED_VERSION" ] \
    && [ "$INSTALLED_IMPL" = "rust" ]; then
    echo "Current FALPRS version: $INSTALLED_VERSION ($INSTALLED_IMPL)"
    echo "Required FALPRS version: $REQUIRED_VERSION (rust)"
    echo "FALPRS executable is up to date."
    NEED_BUILD=false
else
    if [ -n "$INSTALLED_VERSION" ]; then
        echo "Current FALPRS version: $INSTALLED_VERSION ($INSTALLED_IMPL)"
    else
        echo "Current FALPRS version: unknown"
    fi
    echo "Required FALPRS version: $REQUIRED_VERSION (rust)"
    echo "FALPRS executable needs to be rebuilt."
    NEED_BUILD=true
fi

TENSORRT_ENV=(TRITON_VERSION="$TRITON_VERSION" FALPRS_WORKDIR="$FALPRS_WORKDIR")
if [ -n "$ARCFACE_SHA1" ]; then
    TENSORRT_ENV+=(ARCFACE_SHA1="$ARCFACE_SHA1")
fi

# Build project if needed without stopping service
if [ "$PHASE" = all ] && [ "$NEED_BUILD" = true ]; then
    echo "Building the project..."
    $BASEDIR/build_falprs.sh
fi
if [ "$PHASE" = build ]; then
    if [ "$NEED_BUILD" = true ]; then
        echo "Building the project..."
        $BASEDIR/build_falprs.sh --no-install
    fi
    # The planning downloads the missing ONNX models and lists the plans the install will generate
    echo "Planning TensorRT plans..."
    cd $BASEDIR/..
    PLAN_STATUS=0
    sudo "${TENSORRT_ENV[@]}" python3 ./scripts/tensorrt_plans.py plan || PLAN_STATUS=$?
    rm -f /tmp/falprs_models_to_gen
    if [ $PLAN_STATUS -ne 0 ] && [ $PLAN_STATUS -ne 10 ]; then
        echo "Error during TensorRT planning."
        exit $PLAN_STATUS
    fi
    echo "Build phase completed."
    exit 0
fi

FALPRS_WAS_ACTIVE=false
if [ "$PHASE" = all ] || [ "$PHASE" = install ]; then
    # The executable to install has to be built for the required version
    if [ "$NEED_BUILD" = true ] && [ "$PHASE" = install ]; then
        BUILT_VERSION=$("$FALPRS_BINARY" --version 2>/dev/null | tr -d '[:space:]' || true)
        if [ "$BUILT_VERSION" != "$REQUIRED_VERSION" ]; then
            echo "Error: $FALPRS_BINARY is not built for version $REQUIRED_VERSION; run $0 --build first."
            exit 1
        fi
    fi

    # Convert an old config.yaml if the build did not do it already
    "$BASEDIR/prepare_config.sh"

    # A service unit installed for config.yaml starts the service with config.toml from now on
    SERVICE_UNIT=/etc/systemd/system/falprs.service
    if [ -f "$SERVICE_UNIT" ] && grep -q -- "-c config.yaml" "$SERVICE_UNIT"; then
        echo "Switching $SERVICE_UNIT to config.toml"
        sudo sed -i 's/-c config\.yaml/-c config.toml/' "$SERVICE_UNIT"
        sudo systemctl daemon-reload
    fi

    # TensorRT Planning phase (before stopping services)
    echo "Planning TensorRT plans..."
    cd $BASEDIR/..

    PLAN_STATUS=0
    sudo "${TENSORRT_ENV[@]}" python3 ./scripts/tensorrt_plans.py plan || PLAN_STATUS=$?

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
        if systemctl is-active --quiet falprs.service; then
            FALPRS_WAS_ACTIVE=true
        fi
        if [ "$FALPRS_WAS_ACTIVE" = true ]; then
            echo "Stopping falprs service..."
            sudo systemctl stop falprs.service
        fi
        if [ "$NEED_BUILD" = true ]; then
            echo "Copying new executable..."
            cp "$FALPRS_BINARY" "$FALPRS_WORKDIR/falprs"
            echo rust > "$FALPRS_WORKDIR/falprs.impl"
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
        sudo "${TENSORRT_ENV[@]}" python3 ./scripts/tensorrt_plans.py generate $MODELS_TO_GEN || GEN_ERROR=$?

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

    if [ "$PHASE" = install ]; then
        echo "Install phase completed; $0 --finish updates the databases and starts the falprs service."
        exit 0
    fi
fi

# The finish phase on its own needs the required executable installed by the install phase
if [ "$PHASE" = finish ] && [ "$NEED_BUILD" = true ]; then
    echo "Error: $FALPRS_WORKDIR/falprs is not the Rust implementation of version $REQUIRED_VERSION; run $0 --install first."
    exit 1
fi

# Update DB schema
echo "Updating database schema..."
$BASEDIR/sql_frs.sh
$BASEDIR/sql_lprs.sh

# Clean the video stream, default video stream and common configs stored in the databases:
# normalize the values the service took, remove invalid and unknown ones; the changed rows are
# saved to config-cleanup-<date>.json in the working directory first
echo "Checking the configs stored in the databases..."
(cd "$FALPRS_WORKDIR" && "$FALPRS_WORKDIR/falprs" -c config.toml check-configs --fix)

# Restore FALPRS to its initial state; the finish phase on its own always starts it
if [ "$FALPRS_WAS_ACTIVE" = true ] || [ "$PHASE" = finish ]; then
    echo "Starting falprs service..."
    sudo systemctl start falprs.service
fi

echo "Project updated successfully."
