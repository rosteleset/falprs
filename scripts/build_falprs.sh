#!/bin/bash

# External variables used in the script

# LLVM_VERSION - LLVM Project version
# PG_VERSION - PostgreSQL database system version
# TRITON_VERSION - NVIDIA Triton Inference Server version
# FALPRS_WORKDIR - FALPRS working directory

LLVM_VERSION="${LLVM_VERSION:=15}"
PG_VERSION="${PG_VERSION:=14}"
TRITON_VERSION="${TRITON_VERSION:=22.12}"
FALPRS_WORKDIR="${FALPRS_WORKDIR:=/opt/falprs}"

BASEDIR=$(realpath `dirname $0`)
apt-get update
apt-get install -y build-essential ccache cmake git libboost-dev libboost-context-dev libboost-coroutine-dev libboost-filesystem-dev libboost-iostreams-dev libboost-locale-dev libboost-program-options-dev libboost-regex-dev libboost-stacktrace-dev zlib1g-dev nasm clang-$LLVM_VERSION lldb-$LLVM_VERSION lld-$LLVM_VERSION clangd-$LLVM_VERSION libssl-dev libyaml-cpp-dev libjemalloc-dev libpq-dev postgresql-server-dev-$PG_VERSION rapidjson-dev python3-dev python3-jinja2 python3-protobuf python3-venv python3-voluptuous python3-yaml libgtest-dev libnghttp2-dev libev-dev libldap2-dev libkrb5-dev libzstd-dev libopencv-dev
if [[ $LLVM_VERSION == "15" ]]; then
    apt-get install libstdc++-12-dev
fi

export CC=clang-$LLVM_VERSION
export CXX=clang++-$LLVM_VERSION

cd ~
git clone https://github.com/triton-inference-server/client.git triton-client

cd triton-client
git checkout r$TRITON_VERSION

# Get rid of re2 dependency (we don't need GRPC)
sed -i 's/_cc_client_depends re2/_cc_client_depends ""/' CMakeLists.txt

mkdir -p build && cd build
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
    -DTRITON_COMMON_REPO_TAG=r$TRITON_VERSION \
    -DTRITON_THIRD_PARTY_REPO_TAG=r$TRITON_VERSION \
    ..
make cc-clients -j`nproc`

cd $BASEDIR/..
mkdir -p build && cd build
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
cp -n config.yaml.example $FALPRS_WORKDIR/config.yaml
cp -r ./model_repository $FALPRS_WORKDIR
cp -n ./examples/lprs/test001.jpg $FALPRS_WORKDIR/static/
cp -n ./examples/frs/einstein_001.jpg $FALPRS_WORKDIR/static/
cp -n ./examples/frs/einstein_002.jpg $FALPRS_WORKDIR/static/
