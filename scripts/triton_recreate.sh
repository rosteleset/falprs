#!/bin/bash

# Replaces the existing Triton Inference Server container with one started as triton_service.sh
# starts it: with --ipc=host, which FALPRS needs to pass the detector inputs through shared memory,
# and named falprs-triton. The container is found by that name, otherwise by its image; when several
# Triton containers exist and none has that name, the script stops without removing any.
#
# External variables used in the script
# TRITON_VERSION - NVIDIA Triton Inference Server container image version; by default the version of
#                  the existing container, since the TensorRT plans are built for it
# FALPRS_WORKDIR - FALPRS working directory

set -e

BASEDIR=$(realpath "$(dirname "$0")")
source "$BASEDIR/triton_common.sh"
load_triton_env "$BASEDIR"

# ID, image and name of every container of the Triton image, running or not
containers=$(sudo docker ps -a --format '{{.ID}} {{.Image}} {{.Names}}' | awk -v image="$TRITON_IMAGE:" 'index($2, image) == 1')
named=$(echo "$containers" | awk -v name="$TRITON_CONTAINER" '$3 == name')
if [ -n "$named" ]; then
    old=$named
elif [ -z "$containers" ]; then
    old=""
elif [ "$(echo "$containers" | wc -l)" -eq 1 ]; then
    old=$containers
else
    echo "Several Triton containers and none named $TRITON_CONTAINER; remove the ones not used and run again:" >&2
    echo "$containers" >&2
    exit 1
fi

if [ -n "$old" ]; then
    read -r old_id old_image old_name <<< "$old"
    old_version=${old_image#"$TRITON_IMAGE:"}
    old_version=${old_version%-py3}
    if [ -z "$TRITON_VERSION" ]; then
        TRITON_VERSION=$old_version
    elif [ "$TRITON_VERSION" != "$old_version" ]; then
        echo "TRITON_VERSION is $TRITON_VERSION, the container $old_name runs $old_version;" >&2
        echo "the TensorRT plans are built for one version: set TRITON_VERSION to the one they were built for." >&2
        exit 1
    fi
    echo "Removing the container $old_name ($old_image)"
    sudo docker rm -f "$old_id" > /dev/null
else
    echo "No Triton container found, starting a new one"
fi
export TRITON_VERSION="${TRITON_VERSION:-24.09}"

run_triton > /dev/null
echo "Started $TRITON_CONTAINER ($TRITON_IMAGE:$TRITON_VERSION-py3), waiting for it to be ready"
for _ in $(seq 1 120); do
    if curl -sf localhost:8000/v2/health/ready > /dev/null; then
        echo "Triton is ready"
        exit 0
    fi
    sleep 1
done
echo "Triton is not ready after 2 minutes; see: sudo docker logs $TRITON_CONTAINER" >&2
exit 1
