#!/bin/bash

# External variables used in the script

# TRITON_VERSION - NVIDIA Triton Inference Server version
# FALPRS_WORKDIR - FALPRS working directory

TRITON_VERSION="${TRITON_VERSION:=22.12}"
FALPRS_WORKDIR="${FALPRS_WORKDIR:=/opt/falprs}"
ARCFACE_SHA1="${ARCFACE_SHA1:=3642b396053aa5e9cd4518de66baf0d26c9e1467}"

mkdir -p $FALPRS_WORKDIR/model_repository/arcface/1/
mkdir -p $FALPRS_WORKDIR/model_repository/genet/1/
mkdir -p $FALPRS_WORKDIR/model_repository/lpdnet_yolo/1/
mkdir -p $FALPRS_WORKDIR/model_repository/lprnet_yolo/1/
mkdir -p $FALPRS_WORKDIR/model_repository/scrfd/1/
mkdir -p $FALPRS_WORKDIR/model_repository/vcnet_vit/1/
mkdir -p $FALPRS_WORKDIR/model_repository/vdnet_yolo/1/

docker pull nvcr.io/nvidia/tritonserver:$TRITON_VERSION-py3

BASEDIR=$(realpath `dirname $0`)
cd $BASEDIR/..
mkdir -p ./temp
cd ./temp

# Download models
apt-get install -y wget

# arcface - glint360k_r50.onnx or glint_r50.onnx
arcface_id=1aO2QfGAd8cVsZ-V-X5Wb8YJsm0BdscRT
arcface_onnx=glint360k_r50.onnx
if [[ $ARCFACE_SHA1 == "4fd7dce20b6987ba89910eda8614a33eb3593216" ]]; then
    # glint_r50.onnx
    arcface_id=102F98ufVggXyXbWKXCF6tWdIk21FIvhS
    arcface_onnx=glint_r50.onnx
fi
wget --content-disposition --no-clobber "https://drive.usercontent.google.com/download?id=$arcface_id&confirm=y" -O $arcface_onnx

# genet - genet_small_custom_ft.onnx
wget --content-disposition --no-clobber 'https://drive.usercontent.google.com/download?id=1tIBqGBPb5Pgss0b2wIOqNv9BpcSar76-&confirm=y'

# lpdnet_yolo - lpdnet_yolo.onnx
wget --content-disposition --no-clobber 'https://drive.usercontent.google.com/download?id=1k9aUWGW61JPnAG3j7LlAQR4rn1avsX1L&confirm=y'

# lprnet_yolo - lprnet_yolo.onnx
wget --content-disposition --no-clobber 'https://drive.usercontent.google.com/download?id=1I-GlfHeAFUnOaOyH03p7Y3507ok9eSOP&confirm=y'

# scrfd - scrfd_10g_bnkps.onnx
wget --content-disposition --no-clobber 'https://drive.usercontent.google.com/download?id=1ug1uimJzuwDqbxQPYCWEDAYumDXaj1f2&confirm=y'

# vcnet_vit - vcnet_vit.onnx
wget --content-disposition --no-clobber 'https://drive.usercontent.google.com/download?id=178NdNvKhOSAURJg8bTP5IlNRyzBigr3v&confirm=y'

# vdnet_yolo - vdnet_yolo.onnx
wget --content-disposition --no-clobber 'https://drive.usercontent.google.com/download?id=1BPwVSvI1qytIO2WlCzdz6lGXVo_IiQ2E&confirm=y'

docker run --gpus all --rm -v $BASEDIR/../temp:/source -v $FALPRS_WORKDIR/model_repository:/destination --entrypoint=bash nvcr.io/nvidia/tritonserver:$TRITON_VERSION-py3 -c "
	/usr/src/tensorrt/bin/trtexec --onnx=/source/$arcface_onnx --saveEngine=/destination/arcface/1/model.plan --shapes=input.1:1x3x112x112
	/usr/src/tensorrt/bin/trtexec --onnx=/source/genet_small_custom_ft.onnx --saveEngine=/destination/genet/1/model.plan
	/usr/src/tensorrt/bin/trtexec --onnx=/source/lpdnet_yolo.onnx --minShapes=images:1x3x640x640 --optShapes=images:8x3x640x640 --maxShapes=images:8x3x640x640 --saveEngine=/destination/lpdnet_yolo/1/lpdnet_yolo.engine
	/usr/src/tensorrt/bin/trtexec --onnx=/source/lprnet_yolo.onnx --minShapes=images:1x3x160x160 --optShapes=images:8x3x160x160 --maxShapes=images:8x3x160x160 --saveEngine=/destination/lprnet_yolo/1/lprnet_yolo.engine
	/usr/src/tensorrt/bin/trtexec --onnx=/source/scrfd_10g_bnkps.onnx --saveEngine=/destination/scrfd/1/model.plan --shapes=input.1:1x3x320x320
        /usr/src/tensorrt/bin/trtexec --onnx=/source/vcnet_vit.onnx --minShapes=input:1x3x224x224 --optShapes=input:8x3x224x224 --maxShapes=input:8x3x224x224 --saveEngine=/destination/vcnet_vit/1/vcnet_vit.engine
	/usr/src/tensorrt/bin/trtexec --onnx=/source/vdnet_yolo.onnx --minShapes=images:1x3x640x640 --optShapes=images:8x3x640x640 --maxShapes=images:8x3x640x640 --saveEngine=/destination/vdnet_yolo/1/vdnet_yolo.engine
	"
