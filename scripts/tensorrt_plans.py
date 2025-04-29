import subprocess
import os
from pathlib import Path

# Get GPUs Compute Capability
cc_set = set()
gpus = []
try:
    r = subprocess.run(['nvidia-smi', '--query-gpu=name,compute_cap', '--format=csv,noheader'], stdout=subprocess.PIPE)
    gpus = list(filter(None, r.stdout.decode('utf-8').split("\n")))
except:
    print("Error executing nvidia-smi.")
    exit(-1)

if len(gpus) == 0:
    print("No GPUs found.")
    exit(0)

# Remove duplicates
gpu_info = {}
for i, gpu in enumerate(gpus):
    device_name = gpu.split(',')[0].lstrip(' ').rstrip(' ')
    device_name = device_name.split(' ')
    if len(device_name) > 2:
        device_name = device_name[-2:]
    device_name = ''.join(device_name).lower()
    cc = gpu.split(',')[1].lstrip(' ').rstrip(' ')
    if not (cc in cc_set):
        gpu_info[i] = [device_name, cc]
        cc_set.add(cc)

TRITON_VERSION = 'TRITON_VERSION'
triton_version = os.environ.get(TRITON_VERSION) if os.environ.get(TRITON_VERSION) is not None else "22.12"

FALPRS_WORKDIR = 'FALPRS_WORKDIR'
falprs_workdir = os.environ.get(FALPRS_WORKDIR) if os.environ.get(FALPRS_WORKDIR) is not None else "/opt/falprs"

ARCFACE_SHA1 = 'ARCFACE_SHA1'
arcface_sha1 = os.environ.get(ARCFACE_SHA1) if os.environ.get(ARCFACE_SHA1) is not None else "3642b396053aa5e9cd4518de66baf0d26c9e1467"

# Pull Triron Inference Server docker container
try:
    subprocess.run(['docker', 'pull', f"nvcr.io/nvidia/tritonserver:{triton_version}-py3"])
except:
    print("Error executing docker.")
    exit(-1)

try:
    subprocess.run(['apt-get', 'install', '-y', 'wget'])
except:
    print("Error installing wget.")
    exit(-1)

tmp_dir = os.path.dirname(__file__) + '/../temp'
Path(tmp_dir).mkdir(parents=True, exist_ok=True)

# Download models
# arcface - glint360k_r50.onnx or glint_r50.onnx
arcface_id = '1aO2QfGAd8cVsZ-V-X5Wb8YJsm0BdscRT'
arcface_onnx = 'glint360k_r50.onnx'
if arcface_sha1 == '4fd7dce20b6987ba89910eda8614a33eb3593216':
    # glint_r50.onnx
    arcface_id = '102F98ufVggXyXbWKXCF6tWdIk21FIvhS'
    arcface_onnx = 'glint_r50.onnx'
try:
    subprocess.run(['wget', '--content-disposition', '--no-clobber',
                    f"https://drive.usercontent.google.com/download?id={arcface_id}&confirm=y",
                    '-O',
                    tmp_dir + "/" + arcface_onnx])
except:
    print("Error downloading arcface model.")
    exit(-1)

# genet - genet_small_custom_ft.onnx
try:
    subprocess.run(['wget', '--content-disposition', '--no-clobber',
                    'https://drive.usercontent.google.com/download?id=1tIBqGBPb5Pgss0b2wIOqNv9BpcSar76-&confirm=y',
                    '-O',
                    tmp_dir + '/genet_small_custom_ft.onnx'])
except:
    print("Error downloading genet model.")
    exit(-1)

# lpdnet_yolo - lpdnet_yolo.onnx
try:
    subprocess.run(['wget', '--content-disposition', '--no-clobber',
                    'https://drive.usercontent.google.com/download?id=1k9aUWGW61JPnAG3j7LlAQR4rn1avsX1L&confirm=y',
                    '-O',
                    tmp_dir + '/lpdnet_yolo.onnx'])
except:
    print("Error downloading lpdnet_yolo model.")
    exit(-1)

# lprnet_yolo - lprnet_yolo.onnx
try:
    subprocess.run(['wget', '--content-disposition', '--no-clobber',
                    'https://drive.usercontent.google.com/download?id=1I-GlfHeAFUnOaOyH03p7Y3507ok9eSOP&confirm=y',
                    '-O',
                    tmp_dir + '/lprnet_yolo.onnx'])
except:
    print("Error downloading lprnet_yolo model.")
    exit(-1)

# scrfd - scrfd_10g_bnkps.onnx
try:
    subprocess.run(['wget', '--content-disposition', '--no-clobber',
                    'https://drive.usercontent.google.com/download?id=1ug1uimJzuwDqbxQPYCWEDAYumDXaj1f2&confirm=y',
                    '-O',
                    tmp_dir + '/scrfd_10g_bnkps.onnx'])
except:
    print("Error downloading scrfd model.")
    exit(-1)

# vcnet_vit - vcnet_vit.onnx
try:
    subprocess.run(['wget', '--content-disposition', '--no-clobber',
                    'https://drive.usercontent.google.com/download?id=178NdNvKhOSAURJg8bTP5IlNRyzBigr3v&confirm=y',
                    '-O',
                    tmp_dir + '/vcnet_vit.onnx'])
except:
    print("Error downloading vcnet_vit model.")
    exit(-1)

# vdnet_yolo - vdnet_yolo.onnx
try:
    subprocess.run(['wget', '--content-disposition', '--no-clobber',
                    'https://drive.usercontent.google.com/download?id=1BPwVSvI1qytIO2WlCzdz6lGXVo_IiQ2E&confirm=y',
                    '-O',
                    tmp_dir + '/vdnet_yolo.onnx'])
except:
    print("Error downloading vdnet_yolo model.")
    exit(-1)

model_templates = {
    'arcface': "model_{suffix}.plan",
    'genet': "model_{suffix}.plan",
    'lpdnet_yolo': "lpdnet_yolo_{suffix}.engine",
    'lprnet_yolo': "lprnet_yolo_{suffix}.engine",
    'scrfd': "model_{suffix}.plan",
    'vcnet_vit': "vcnet_vit_{suffix}.engine",
    'vdnet_yolo': "vdnet_yolo_{suffix}.engine"
}

try:
    rep_dir = falprs_workdir + '/model_repository'
    Path(rep_dir).mkdir(parents=True, exist_ok=True)
except:
    print("Error creating model repository path.")
    exit(-1)

cc_model_filenames = {}
commands = []

# Prepare trtexec commands
for i, gpu in gpu_info.items():
    suffix = ""

    if len(gpu_info) > 1:
        suffix = "_" + gpu[0]
        cc = gpu[1]
        for key, value in model_templates.items():
            if not (key in cc_model_filenames):
                cc_model_filenames[key] = []
            cc_model_filenames[key].append({
                'key': cc,
                'value': value.format(suffix=gpu[0])
            })

    commands.append(f"CUDA_DEVICE_ORDER=PCI_BUS_ID /usr/src/tensorrt/bin/trtexec --device={i} --onnx=/source/{arcface_onnx} --saveEngine=/destination/arcface/1/model{suffix}.plan --shapes=input.1:1x3x112x112")
    commands.append(f"CUDA_DEVICE_ORDER=PCI_BUS_ID /usr/src/tensorrt/bin/trtexec --device={i} --onnx=/source/genet_small_custom_ft.onnx --saveEngine=/destination/genet/1/model{suffix}.plan")
    commands.append(f"CUDA_DEVICE_ORDER=PCI_BUS_ID /usr/src/tensorrt/bin/trtexec --device={i} --onnx=/source/lpdnet_yolo.onnx --minShapes=images:1x3x640x640 --optShapes=images:8x3x640x640 --maxShapes=images:8x3x640x640 --saveEngine=/destination/lpdnet_yolo/1/lpdnet_yolo{suffix}.engine")
    commands.append(f"CUDA_DEVICE_ORDER=PCI_BUS_ID /usr/src/tensorrt/bin/trtexec --device={i} --onnx=/source/lprnet_yolo.onnx --minShapes=images:1x3x160x160 --optShapes=images:8x3x160x160 --maxShapes=images:8x3x160x160 --saveEngine=/destination/lprnet_yolo/1/lprnet_yolo{suffix}.engine")
    commands.append(f"CUDA_DEVICE_ORDER=PCI_BUS_ID /usr/src/tensorrt/bin/trtexec --device={i} --onnx=/source/scrfd_10g_bnkps.onnx --saveEngine=/destination/scrfd/1/model{suffix}.plan --shapes=input.1:1x3x320x320")
    commands.append(f"CUDA_DEVICE_ORDER=PCI_BUS_ID /usr/src/tensorrt/bin/trtexec --device={i} --onnx=/source/vcnet_vit.onnx --minShapes=input:1x3x224x224 --optShapes=input:8x3x224x224 --maxShapes=input:8x3x224x224 --saveEngine=/destination/vcnet_vit/1/vcnet_vit{suffix}.engine")
    commands.append(f"CUDA_DEVICE_ORDER=PCI_BUS_ID /usr/src/tensorrt/bin/trtexec --device={i} --onnx=/source/vdnet_yolo.onnx --minShapes=images:1x3x640x640 --optShapes=images:8x3x640x640 --maxShapes=images:8x3x640x640 --saveEngine=/destination/vdnet_yolo/1/vdnet_yolo{suffix}.engine")

# Generate cc_model_filenames section for multiple GPUs
for key in model_templates:
    cc_models = []
    if key in cc_model_filenames:
        for m in cc_model_filenames[key]:
            cc_models.append(f"  {{\n    key: \"{m['key']}\"\n    value: \"{m['value']}\"\n  }}")
    cc_option = ""
    if len(cc_models) > 0:
        cc_option = "cc_model_filenames [\n" + ',\n'.join(cc_models) + "\n]"

    template_dir = os.path.dirname(__file__) + '/../model_repository_templates/'
    file_name = template_dir + key + "/config.pbtxt"
    try:
        f = open(file_name, mode='r')
        data = f.read()
        f.close()
    except:
        print(f"Error opening file {file_name}")
        exit(-1)

    plan_path = falprs_workdir + "/model_repository/" + key
    file_name = plan_path + "/config.pbtxt"
    Path(plan_path + "/1").mkdir(parents=True, exist_ok=True)
    data = data.replace('%cc_model_filenames%', cc_option)
    try:
        f = open(file_name, mode='w')
        f.write(data)
        f.close()
    except:
        print(f"Error saving file {file_name}")
        exit(-1)

try:
    subprocess.run(['docker', 'run', '--gpus', 'all', '--rm',
                    '-v', f"{tmp_dir}:/source", '-v', f"{falprs_workdir}/model_repository:/destination",
                    '--entrypoint=bash', f"nvcr.io/nvidia/tritonserver:{triton_version}-py3", '-c', "\n".join(commands)])
except:
    print("Error creating TensorRT plans.")
    exit(-1)
