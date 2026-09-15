import os
import sys
import subprocess
import hashlib
import json
import shutil
from pathlib import Path

REGULAR_MODELS = [
    {
        'key': 'barcode_detection',
        'onnx': 'barcode_detection.onnx',
        'id': '19Lf1lXFUgLGtsZLjLkcTfyQC38KyI20n',
        'template': 'barcode_detection_{suffix}.engine',
        'shape_args': '--minShapes=images:1x3x320x320 --optShapes=images:8x3x320x320 --maxShapes=images:8x3x320x320',
        'output_pattern': 'barcode_detection{suffix}.engine',
    },
    {
        'key': 'genet',
        'onnx': 'genet_small_custom_ft.onnx',
        'id': '1tIBqGBPb5Pgss0b2wIOqNv9BpcSar76-',
        'template': 'model_{suffix}.plan',
        'shape_args': '',
        'output_pattern': 'model{suffix}.plan',
    },
    {
        'key': 'lpcnet_vit',
        'onnx': 'lpcnet_vit.onnx',
        'id': '14aaFuG6y_26SJt7F-4OM9FeVMoEmk1yW',
        'template': 'lpcnet_vit_{suffix}.engine',
        'shape_args': '--minShapes=input:1x3x224x224 --optShapes=input:8x3x224x224 --maxShapes=input:8x3x224x224',
        'output_pattern': 'lpcnet_vit{suffix}.engine',
    },
    {
        'key': 'lpdnet_yolo',
        'onnx': 'lpdnet_yolo_v2.onnx',
        'id': '1o--KonChXshsxMSVaEn6VX9pE0I4TxRA',
        'template': 'lpdnet_yolo_{suffix}.engine',
        'shape_args': '--minShapes=images:1x3x640x640 --optShapes=images:8x3x640x640 --maxShapes=images:8x3x640x640',
        'output_pattern': 'lpdnet_yolo{suffix}.engine',
    },
    {
        'key': 'lprnet_yolo',
        'onnx': 'lprnet_yolo_v2.onnx',
        'id': '1-uUaYmLQM8CHN6IYIDXKZqGwvraTkKU1',
        'template': 'lprnet_yolo_{suffix}.engine',
        'shape_args': '--minShapes=images:1x3x320x320 --optShapes=images:8x3x320x320 --maxShapes=images:8x3x320x320',
        'output_pattern': 'lprnet_yolo{suffix}.engine',
    },
    {
        'key': 'scrfd',
        'onnx': 'scrfd_10g_bnkps.onnx',
        'id': '1ug1uimJzuwDqbxQPYCWEDAYumDXaj1f2',
        'template': 'model_{suffix}.plan',
        'shape_args': '--shapes=input.1:1x3x320x320',
        'output_pattern': 'model{suffix}.plan',
    },
    {
        'key': 'vcnet_vit',
        'onnx': 'vcnet_vit.onnx',
        'id': '178NdNvKhOSAURJg8bTP5IlNRyzBigr3v',
        'template': 'vcnet_vit_{suffix}.engine',
        'shape_args': '--minShapes=input:1x3x224x224 --optShapes=input:8x3x224x224 --maxShapes=input:8x3x224x224',
        'output_pattern': 'vcnet_vit{suffix}.engine',
    },
    {
        'key': 'vdnet_yolo',
        'onnx': 'vdnet_yolo.onnx',
        'id': '1BPwVSvI1qytIO2WlCzdz6lGXVo_IiQ2E',
        'template': 'vdnet_yolo_{suffix}.engine',
        'shape_args': '--minShapes=images:1x3x640x640 --optShapes=images:8x3x640x640 --maxShapes=images:8x3x640x640',
        'output_pattern': 'vdnet_yolo{suffix}.engine',
    },
]

MODEL_TEMPLATES = {
    'arcface': "model_{suffix}.plan",
    'barcode_detection': "barcode_detection_{suffix}.engine",
    'genet': "model_{suffix}.plan",
    'lpdnet_yolo': "lpdnet_yolo_{suffix}.engine",
    'lpcnet_vit': "lpcnet_vit_{suffix}.engine",
    'lprnet_yolo': "lprnet_yolo_{suffix}.engine",
    'scrfd': "model_{suffix}.plan",
    'vcnet_vit': "vcnet_vit_{suffix}.engine",
    'vdnet_yolo': "vdnet_yolo_{suffix}.engine",
}


def get_arcface_info(arcface_sha1):
    if arcface_sha1 == '4fd7dce20b6987ba89910eda8614a33eb3593216':
        return {
            'key': 'arcface',
            'onnx': 'glint_r50.onnx',
            'id': '102F98ufVggXyXbWKXCF6tWdIk21FIvhS',
            'template': 'model_{suffix}.plan',
            'shape_args': '--shapes=input.1:1x3x112x112',
            'output_pattern': 'model{suffix}.plan',
        }
    else:
        return {
            'key': 'arcface',
            'onnx': 'glint360k_r50.onnx',
            'id': '1aO2QfGAd8cVsZ-V-X5Wb8YJsm0BdscRT',
            'template': 'model_{suffix}.plan',
            'shape_args': '--shapes=input.1:1x3x112x112',
            'output_pattern': 'model{suffix}.plan',
        }


def get_gpu_info():
    cc_set = set()
    gpus = []
    try:
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,compute_cap', '--format=csv,noheader'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        gpus = list(filter(None, r.stdout.decode('utf-8').split("\n")))
    except Exception:
        print("Error executing nvidia-smi.")
        sys.exit(1)

    if len(gpus) == 0:
        print("No GPUs found.")
        return {}

    gpu_info = {}
    for i, gpu in enumerate(gpus):
        device_name = gpu.split(',')[0].strip()
        device_name_parts = device_name.split(' ')
        if len(device_name_parts) > 2:
            device_name_parts = device_name_parts[-2:]
        device_name = ''.join(device_name_parts).lower()
        cc = gpu.split(',')[1].strip()
        if cc not in cc_set:
            gpu_info[i] = [device_name, cc]
            cc_set.add(cc)

    return gpu_info


def calc_sha1(filepath):
    h = hashlib.sha1()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_onnx_sha1(workdir):
    sha1_file = os.path.join(workdir, '.onnx_sha1')
    if not os.path.isfile(sha1_file):
        return {}
    res = {}
    with open(sha1_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                res[parts[0]] = parts[1]
    return res


def plan():
    triton_version = os.environ.get('TRITON_VERSION', '24.09')
    falprs_workdir = os.environ.get('FALPRS_WORKDIR', '/opt/falprs')
    arcface_sha1 = os.environ.get('ARCFACE_SHA1', '3642b396053aa5e9cd4518de66baf0d26c9e1467')

    gpu_info = get_gpu_info()
    if not gpu_info:
        manifest = {
            'needs_generation': False,
            'models_to_regenerate': [],
            'current_onnx_sha1': {},
            'arcface_sha1': arcface_sha1,
            'triton_version': triton_version,
            'gpu_info': {},
            'commands': [],
            'tmp_dir': '',
        }
        manifest_path = os.path.join(falprs_workdir, '.tensorrt_generation_plan.json')
        Path(falprs_workdir).mkdir(parents=True, exist_ok=True)
        with open(manifest_path + '.tmp', 'w') as f:
            json.dump(manifest, f, indent=2)
        os.replace(manifest_path + '.tmp', manifest_path)
        return

    try:
        subprocess.run(['docker', 'pull', f"nvcr.io/nvidia/tritonserver:{triton_version}-py3"], check=True)
    except Exception:
        print("Error executing docker.")
        sys.exit(1)

    if not shutil.which('wget'):
        try:
            subprocess.run(['apt-get', 'install', '-y', 'wget'], check=True)
        except Exception:
            print("Error installing wget.")
            sys.exit(1)

    tmp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'temp'))
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)

    arcface_info = get_arcface_info(arcface_sha1)
    all_models_to_download = [arcface_info] + REGULAR_MODELS

    calculated_sha1 = {}
    for model in all_models_to_download:
        dest_path = os.path.join(tmp_dir, model['onnx'])
        try:
            subprocess.run(
                ['wget', '--content-disposition',
                 f"https://drive.usercontent.google.com/download?id={model['id']}&confirm=y",
                 '-O', dest_path],
                check=True
            )
        except Exception:
            print(f"Error downloading {model['key']} model.")
            sys.exit(1)

        calculated_sha1[model['onnx']] = calc_sha1(dest_path)

    old_onnx_sha1 = load_onnx_sha1(falprs_workdir)

    print("Planning TensorRT plans...")
    models_to_regenerate = set()
    commands = []

    # Check regular models
    for model in REGULAR_MODELS:
        onnx_name = model['onnx']
        curr_sha1 = calculated_sha1[onnx_name]
        sha1_changed = (onnx_name not in old_onnx_sha1) or (old_onnx_sha1[onnx_name] != curr_sha1)

        status_str = "changed" if sha1_changed else "unchanged"
        print(f"{onnx_name}: {status_str}")

        for i, gpu in gpu_info.items():
            suffix = "" if len(gpu_info) == 1 else f"_{gpu[0]}"
            plan_filename = model['output_pattern'].format(suffix=suffix)
            plan_path = os.path.join(falprs_workdir, 'model_repository', model['key'], '1', plan_filename)
            plan_exists = os.path.isfile(plan_path)

            if sha1_changed or not plan_exists:
                models_to_regenerate.add(model['key'])
                shape_part = f" {model['shape_args']}" if model['shape_args'] else ""
                cmd = f"CUDA_DEVICE_ORDER=PCI_BUS_ID /usr/src/tensorrt/bin/trtexec --device={i} --onnx=/source/{model['onnx']}{shape_part} --saveEngine=/destination/{model['key']}/1/{plan_filename}"
                commands.append(cmd)

    # Check ArcFace
    arcface_sha1_file = os.path.join(falprs_workdir, '.arcface_sha1')
    saved_arcface_sha1 = None
    if os.path.isfile(arcface_sha1_file):
        try:
            with open(arcface_sha1_file, 'r') as f:
                saved_arcface_sha1 = f.read().strip()
        except Exception:
            pass

    arcface_changed = (saved_arcface_sha1 != arcface_sha1)
    for i, gpu in gpu_info.items():
        suffix = "" if len(gpu_info) == 1 else f"_{gpu[0]}"
        plan_filename = arcface_info['output_pattern'].format(suffix=suffix)
        plan_path = os.path.join(falprs_workdir, 'model_repository', 'arcface', '1', plan_filename)
        plan_exists = os.path.isfile(plan_path)

        if arcface_changed or not plan_exists:
            models_to_regenerate.add('arcface')
            shape_part = f" {arcface_info['shape_args']}" if arcface_info['shape_args'] else ""
            cmd = f"CUDA_DEVICE_ORDER=PCI_BUS_ID /usr/src/tensorrt/bin/trtexec --device={i} --onnx=/source/{arcface_info['onnx']}{shape_part} --saveEngine=/destination/arcface/1/{plan_filename}"
            commands.append(cmd)

    needs_generation = len(models_to_regenerate) > 0

    if not needs_generation:
        print("All TensorRT plans are up to date.")
        print("Triton Inference Server does not need to be restarted.")
    else:
        print("TensorRT plans need to be regenerated for:")
        for key in sorted(models_to_regenerate):
            print(f"  {key}")

    manifest = {
        'needs_generation': needs_generation,
        'models_to_regenerate': sorted(list(models_to_regenerate)),
        'current_onnx_sha1': {
            m['onnx']: calculated_sha1[m['onnx']] for m in REGULAR_MODELS
        },
        'arcface_sha1': arcface_sha1,
        'triton_version': triton_version,
        'gpu_info': {str(k): v for k, v in gpu_info.items()},
        'commands': commands,
        'tmp_dir': tmp_dir,
    }

    manifest_path = os.path.join(falprs_workdir, '.tensorrt_generation_plan.json')
    Path(falprs_workdir).mkdir(parents=True, exist_ok=True)
    with open(manifest_path + '.tmp', 'w') as f:
        json.dump(manifest, f, indent=2)
    os.replace(manifest_path + '.tmp', manifest_path)


def generate():
    falprs_workdir = os.environ.get('FALPRS_WORKDIR', '/opt/falprs')
    manifest_path = os.path.join(falprs_workdir, '.tensorrt_generation_plan.json')

    if not os.path.isfile(manifest_path):
        print(f"Manifest file {manifest_path} not found. Run planning first.")
        sys.exit(1)

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    if not manifest.get('needs_generation', False):
        print("No TensorRT plans generation required.")
        return

    triton_version = manifest['triton_version']
    tmp_dir = manifest['tmp_dir']
    gpu_info = manifest['gpu_info']  # keys are str device index
    commands = manifest['commands']

    # Update config.pbtxt for model repository
    cc_model_filenames = {}
    if len(gpu_info) > 1:
        for i, gpu in gpu_info.items():
            device_name = gpu[0]
            cc = gpu[1]
            for key, tmpl in MODEL_TEMPLATES.items():
                if key not in cc_model_filenames:
                    cc_model_filenames[key] = []
                cc_model_filenames[key].append({
                    'key': cc,
                    'value': tmpl.format(suffix=device_name),
                })

    template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model_repository_templates'))
    rep_dir = os.path.join(falprs_workdir, 'model_repository')
    Path(rep_dir).mkdir(parents=True, exist_ok=True)

    for key in MODEL_TEMPLATES:
        cc_models = []
        if key in cc_model_filenames:
            for m in cc_model_filenames[key]:
                cc_models.append(f"  {{\n    key: \"{m['key']}\"\n    value: \"{m['value']}\"\n  }}")
        cc_option = ""
        if len(cc_models) > 0:
            cc_option = "cc_model_filenames [\n" + ',\n'.join(cc_models) + "\n]"

        src_config_file = os.path.join(template_dir, key, 'config.pbtxt')
        try:
            with open(src_config_file, 'r') as f:
                data = f.read()
        except Exception:
            print(f"Error opening file {src_config_file}")
            sys.exit(1)

        dst_model_dir = os.path.join(rep_dir, key)
        dst_ver_dir = os.path.join(dst_model_dir, '1')
        Path(dst_ver_dir).mkdir(parents=True, exist_ok=True)

        dst_config_file = os.path.join(dst_model_dir, 'config.pbtxt')
        data = data.replace('%cc_model_filenames%', cc_option)
        try:
            with open(dst_config_file, 'w') as f:
                f.write(data)
        except Exception:
            print(f"Error saving file {dst_config_file}")
            sys.exit(1)

    if commands:
        cmd_script = "\n".join(commands)
        try:
            subprocess.run([
                'docker', 'run', '--gpus', 'all', '--rm',
                '-v', f"{tmp_dir}:/source",
                '-v', f"{rep_dir}:/destination",
                '--entrypoint=bash',
                f"nvcr.io/nvidia/tritonserver:{triton_version}-py3",
                '-c', cmd_script
            ], check=True)
        except Exception:
            print("Error creating TensorRT plans.")
            sys.exit(1)

    # Atomic update of .onnx_sha1
    sha1_lines = [f"{k} {manifest['current_onnx_sha1'][k]}\n" for k in sorted(manifest['current_onnx_sha1'].keys())]
    tmp_sha1_path = os.path.join(falprs_workdir, '.onnx_sha1.tmp')
    with open(tmp_sha1_path, 'w') as f:
        f.writelines(sha1_lines)
    os.replace(tmp_sha1_path, os.path.join(falprs_workdir, '.onnx_sha1'))

    # Atomic update of .arcface_sha1
    tmp_arcface_path = os.path.join(falprs_workdir, '.arcface_sha1.tmp')
    with open(tmp_arcface_path, 'w') as f:
        f.write(manifest['arcface_sha1'] + '\n')
    os.replace(tmp_arcface_path, os.path.join(falprs_workdir, '.arcface_sha1'))

    # Remove temporary manifest
    if os.path.isfile(manifest_path):
        os.remove(manifest_path)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if mode == 'plan':
        plan()
    elif mode == 'generate':
        generate()
    elif mode == 'all':
        plan()
        falprs_workdir = os.environ.get('FALPRS_WORKDIR', '/opt/falprs')
        manifest_path = os.path.join(falprs_workdir, '.tensorrt_generation_plan.json')
        if os.path.isfile(manifest_path):
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            if manifest.get('needs_generation', False):
                generate()
    else:
        print(f"Unknown mode: {mode}. Use 'plan', 'generate', or 'all'.")
        sys.exit(1)


if __name__ == '__main__':
    main()
