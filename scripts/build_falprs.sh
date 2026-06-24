#!/bin/bash

# Build script for the FALPRS project
# Creates executable file and prepares a working directory

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

export TRITON_VERSION="${TRITON_VERSION:-24.09}"
export FALPRS_WORKDIR="${FALPRS_WORKDIR:-/opt/falprs}"

apt-get update
apt-get install -y build-essential ccache cmake git libboost-dev libboost-context-dev libboost-coroutine-dev libboost-filesystem-dev libboost-iostreams-dev libboost-locale-dev libboost-program-options-dev libboost-regex-dev libboost-stacktrace-dev zlib1g-dev nasm clang-format libssl-dev libyaml-cpp-dev libjemalloc-dev libpq-dev postgresql-server-dev-$PG_VERSION rapidjson-dev python3-dev python3-jinja2 python3-protobuf python3-venv python3-voluptuous python3-yaml libgtest-dev libnghttp2-dev libev-dev libldap2-dev libkrb5-dev libzstd-dev libopencv-dev libbz2-dev libre2-dev libcrypto++-dev libfmt-dev libc-ares-dev libcurl4-openssl-dev libcctz-dev

cd ~
if [ ! -d "triton-client" ]; then
  git clone https://github.com/triton-inference-server/client.git triton-client
fi

cd triton-client
git checkout .

ver_le() {
  [[ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n1)" == "$1" ]]
}

TRITON_TAG="r$TRITON_VERSION"
if [[ "$TRITON_TAG" > "r25.07" ]] && ver_le "$UBUNTU_VERSION" "24.04"; then
  TRITON_TAG="r25.07"
fi

rm -rf build && mkdir -p build && cd build
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=20 \
    -DCMAKE_INSTALL_PREFIX:PATH=~/triton-client/build/install \
    -DTRITON_ENABLE_CC_HTTP=ON \
    -DTRITON_ENABLE_CC_GRPC=OFF \
    -DTRITON_ENABLE_PYTHON_HTTP=OFF \
    -DTRITON_ENABLE_PYTHON_GRPC=OFF \
    -DTRITON_ENABLE_GPU=OFF \
    -DTRITON_ENABLE_EXAMPLES=OFF \
    -DTRITON_ENABLE_TESTS=OFF \
    -DTRITON_COMMON_REPO_TAG=$TRITON_TAG \
    -DTRITON_THIRD_PARTY_REPO_TAG=$TRITON_TAG \
    ..
make cc-clients -j`nproc`

cd $BASEDIR/..
rm -rf build && mkdir -p build && cd build
cmake  \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSERVER_PG_SERVER_INCLUDE_DIR=/usr/include/postgresql/$PG_VERSION/server \
    -DUSERVER_PG_SERVER_LIBRARY_DIR=/usr/lib/postgresql/$PG_VERSION/lib \
    -DUSERVER_PG_LIBRARY_DIR=/usr/lib/postgresql/$PG_VERSION/lib \
    ..
make -j`nproc`

# Copy files to the working directory
mkdir -p $FALPRS_WORKDIR
mkdir -p $FALPRS_WORKDIR/static
cp falprs $FALPRS_WORKDIR
cd $BASEDIR/..
cp --update=none ./examples/lprs/test001.jpg $FALPRS_WORKDIR/static/
cp --update=none ./examples/frs/einstein_001.jpg $FALPRS_WORKDIR/static/
cp --update=none ./examples/frs/einstein_002.jpg $FALPRS_WORKDIR/static/

python3 $BASEDIR/merge_yaml.py config.yaml.example $FALPRS_WORKDIR/config.yaml

CPU=$(nproc)
awk -v u_lprs="$PG_USER_LPRS" -v p_lprs="$PG_PASSWD_LPRS" -v h_lprs="$PG_HOST_LPRS" -v pt_lprs="$PG_PORT_LPRS" -v d_lprs="$PG_DB_LPRS" \
    -v u_frs="$PG_USER_FRS" -v p_frs="$PG_PASSWD_FRS" -v h_frs="$PG_HOST_FRS" -v pt_frs="$PG_PORT_FRS" -v d_frs="$PG_DB_FRS" \
    -v cpu="$CPU" '
function build_conn(u,p,h,pt,d) {
    return "postgresql://" u ":" p "@" h ":" pt "/" d
}

/^[[:space:]]*main-task-processor:/ { section="main" }
/^[[:space:]]*fs-task-processor:/   { section="fs" }
section=="main" && /^[[:space:]]*worker_threads:/ {
    sub(/[0-9]+/, cpu)
}
section=="fs" && /^[[:space:]]*worker_threads:/ {
    sub(/[0-9]+/, cpu*4)
}

$0 ~ /^[[:space:]]*lprs-postgresql-database:/ { section="lprs" }
$0 ~ /^[[:space:]]*frs-postgresql-database:/  { section="frs"  }

/^[[:space:]]*lprs-postgresql-database:/ || /^[[:space:]]*frs-postgresql-database:/ {
    print
    next
}

section=="lprs" && /^[[:space:]]*dbconnection:/ {
    sub(/dbconnection:.*/, "dbconnection: \x27" build_conn(u_lprs,p_lprs,h_lprs,pt_lprs,d_lprs) "\x27")
    print
    next
}

section=="frs" && /^[[:space:]]*dbconnection:/ {
    sub(/dbconnection:.*/, "dbconnection: \x27" build_conn(u_frs,p_frs,h_frs,pt_frs,d_frs) "\x27")
    print
    next
}

/^[^[:space:]]/ { section="" }

{ print }
' $FALPRS_WORKDIR/config.yaml > /tmp/config.yaml.new && mv /tmp/config.yaml.new $FALPRS_WORKDIR/config.yaml
