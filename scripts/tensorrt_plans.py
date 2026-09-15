import os
import sys
import subprocess
import hashlib
import shutil
from pathlib import Path

REGULAR_MODELS = [
    {
        'key': 'barcode_detection',
        'onnx': 'barcode_detection.onnx',
        'sha1': 'a114dc0a69aface32041d69f86c1ced13169f974',
        'id': '19Lf1lXFUgLGtsZLjLkcTfyQC38KyI20n',
        'template': 'barcode_detection_{suffix}.engine',
        'shape_args': '--minShapes=images:1x3x320x320 --optShapes=images:8x3x320x320 --maxShapes=images:8x3x320x320',
        'output_pattern': 'barcode_detection{suffix}.engine',
    },
    {
        'key': 'genet',
        'onnx': 'genet_small_custom_ft.onnx',
        'sha1': '4c31a127efd1c7b13a925fbae93785019be3e473',
        'id': '1tIBqGBPb5Pgss0b2wIOqNv9BpcSar76-',
        'template': 'model_{suffix}.plan',
        'shape_args': '',
        'output_pattern': 'model{suffix}.plan',
    },
    {
        'key': 'lpcnet_vit',
        'onnx': 'lpcnet_vit.onnx',
        'sha1': 'c2d226380502d384786300884d368a026666815d',
        'id': '14aaFuG6y_26SJt7F-4OM9FeVMoEmk1yW',
        'template': 'lpcnet_vit_{suffix}.engine',
        'shape_args': '--minShapes=input:1x3x224x224 --optShapes=input:8x3x224x224 --maxShapes=input:8x3x224x224',
        'output_pattern': 'lpcnet_vit{suffix}.engine',
    },
    {
        'key': 'lpdnet_yolo',
        'onnx': 'lpdnet_yolo_v2.onnx',
        'sha1': 'e81fd42d26d0e1e9c9e66c6e259c61c29fc50d05',
        'id': '1o--KonChXshsxMSVaEn6VX9pE0I4TxRA',
        'template': 'lpdnet_yolo_{suffix}.engine',
        'shape_args': '--minShapes=images:1x3x640x640 --optShapes=images:8x3x640x640 --maxShapes=images:8x3x640x640',
        'output_pattern': 'lpdnet_yolo{suffix}.engine',
    },
    {
        'key': 'lprnet_yolo',
        'onnx': 'lprnet_yolo_v2.onnx',
        'sha1': 'c66ffb13f5bfaf30a5b37733c26419206011bb33',
        'id': '1-uUaYmLQM8CHN6IYIDXKZqGwvraTkKU1',
        'template': 'lprnet_yolo_{suffix}.engine',
        'shape_args': '--minShapes=images:1x3x320x320 --optShapes=images:8x3x320x320 --maxShapes=images:8x3x320x320',
        'output_pattern': 'lprnet_yolo{suffix}.engine',
    },
    {
        'key': 'scrfd',
        'onnx': 'scrfd_10g_bnkps.onnx',
        'sha1': 'e94f6c810fcf5b17602b10c6dc24bd39fd568a52',
        'id': '1ug1uimJzuwDqbxQPYCWEDAYumDXaj1f2',
        'template': 'model_{suffix}.plan',
        'shape_args': '--shapes=input.1:1x3x320x320',
        'output_pattern': 'model{suffix}.plan',
    },
    {
        'key': 'vcnet_vit',
        'onnx': 'vcnet_vit.onnx',
        'sha1': 'd16db48bf83c111a501a6bc3279897dd8822257e',
        'id': '178NdNvKhOSAURJg8bTP5IlNRyzBigr3v',
        'template': 'vcnet_vit_{suffix}.engine',
        'shape_args': '--minShapes=input:1x3x224x224 --optShapes=input:8x3x224x224 --maxShapes=input:8x3x224x224',
        'output_pattern': 'vcnet_vit{suffix}.engine',
    },
    {
        'key': 'vdnet_yolo',
        'onnx': 'vdnet_yolo.onnx',
        'sha1': 'a5a7a8701818c565bebbc9b2288da2b8130a864e',
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


def ensure_wget():
    if not shutil.which('wget'):
        try:
            subprocess.run(['apt-get', 'install', '-y', 'wget'], check=True)
        except Exception:
            print("Error installing wget.")
            sys.exit(1)


def ensure_model_onnx(model, tmp_dir):
    dest_path = os.path.join(tmp_dir, model['onnx'])
    expected_sha1 = model.get('sha1')

    if os.path.isfile(dest_path):
        if expected_sha1:
            curr_sha1 = calc_sha1(dest_path)
            if curr_sha1 == expected_sha1:
                return False
        else:
            return False

    ensure_wget()
    download_url = f"https://drive.usercontent.google.com/download?id={model['id']}&confirm=y"
    try:
        subprocess.run(
            ['wget', '--content-disposition', download_url, '-O', dest_path],
            check=True
        )
    except Exception:
        print(f"Error downloading {model['key']} model.")
        sys.exit(1)

    return True


def prepare_and_check(triton_version, falprs_workdir, arcface_sha1, forced_models=None):
    gpu_info = get_gpu_info()
    if not gpu_info:
        return False, [], set(), {}, ""

    try:
        subprocess.run(['docker', 'pull', f"nvcr.io/nvidia/tritonserver:{triton_version}-py3"], check=True)
    except Exception:
        print("Error executing docker.")
        sys.exit(1)

    tmp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'temp'))
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)

    arcface_info = get_arcface_info(arcface_sha1)
    arcface_onnx_updated = ensure_model_onnx(arcface_info, tmp_dir)

    regular_onnx_updated = {}
    for model in REGULAR_MODELS:
        regular_onnx_updated[model['key']] = ensure_model_onnx(model, tmp_dir)

    commands = []
    models_to_regenerate = set()
    forced_set = set(forced_models) if forced_models else None

    # Check ArcFace plans
    for i, gpu in gpu_info.items():
        suffix = "" if len(gpu_info) == 1 else f"_{gpu[0]}"
        plan_filename = arcface_info['output_pattern'].format(suffix=suffix)
        plan_path = os.path.join(falprs_workdir, 'model_repository', 'arcface', '1', plan_filename)
        plan_exists = os.path.isfile(plan_path)

        force_this = forced_set is not None and 'arcface' in forced_set
        if force_this or arcface_onnx_updated or not plan_exists:
            models_to_regenerate.add('arcface')
            shape_part = f" {arcface_info['shape_args']}" if arcface_info['shape_args'] else ""
            cmd = f"CUDA_DEVICE_ORDER=PCI_BUS_ID /usr/src/tensorrt/bin/trtexec --device={i} --onnx=/source/{arcface_info['onnx']}{shape_part} --saveEngine=/destination/arcface/1/{plan_filename}"
            commands.append(cmd)

    # Check regular model plans
    for model in REGULAR_MODELS:
        onnx_updated = regular_onnx_updated[model['key']]
        force_this = forced_set is not None and model['key'] in forced_set
        for i, gpu in gpu_info.items():
            suffix = "" if len(gpu_info) == 1 else f"_{gpu[0]}"
            plan_filename = model['output_pattern'].format(suffix=suffix)
            plan_path = os.path.join(falprs_workdir, 'model_repository', model['key'], '1', plan_filename)
            plan_exists = os.path.isfile(plan_path)

            if force_this or onnx_updated or not plan_exists:
                models_to_regenerate.add(model['key'])
                shape_part = f" {model['shape_args']}" if model['shape_args'] else ""
                cmd = f"CUDA_DEVICE_ORDER=PCI_BUS_ID /usr/src/tensorrt/bin/trtexec --device={i} --onnx=/source/{model['onnx']}{shape_part} --saveEngine=/destination/{model['key']}/1/{plan_filename}"
                commands.append(cmd)

    needs_generation = len(commands) > 0
    return needs_generation, commands, models_to_regenerate, gpu_info, tmp_dir


def update_configs(falprs_workdir, gpu_info):
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


def plan():
    triton_version = os.environ.get('TRITON_VERSION', '24.09')
    falprs_workdir = os.environ.get('FALPRS_WORKDIR', '/opt/falprs')
    arcface_sha1 = os.environ.get('ARCFACE_SHA1', '3642b396053aa5e9cd4518de66baf0d26c9e1467')

    needs_generation, _, models_to_regenerate, gpu_info, _ = prepare_and_check(triton_version, falprs_workdir, arcface_sha1)
    if not gpu_info:
        sys.exit(0)

    if not needs_generation:
        print("All TensorRT plans are up to date.")
        print("Triton Inference Server does not need to be restarted.")
        sys.exit(0)
    else:
        print("TensorRT plans need to be regenerated for:")
        model_list = sorted(list(models_to_regenerate))
        for key in model_list:
            print(f"  {key}")
        model_str = ' '.join(model_list)
        print(f"MODELS_LIST: {model_str}")
        try:
            with open('/tmp/falprs_models_to_gen', 'w') as f:
                f.write(model_str)
        except Exception:
            pass
        sys.exit(10)


def generate():
    triton_version = os.environ.get('TRITON_VERSION', '24.09')
    falprs_workdir = os.environ.get('FALPRS_WORKDIR', '/opt/falprs')
    arcface_sha1 = os.environ.get('ARCFACE_SHA1', '3642b396053aa5e9cd4518de66baf0d26c9e1467')

    forced_models = sys.argv[2:] if len(sys.argv) > 2 else None

    needs_generation, commands, _, gpu_info, tmp_dir = prepare_and_check(triton_version, falprs_workdir, arcface_sha1, forced_models=forced_models)
    if not gpu_info or not needs_generation:
        print("No TensorRT plans generation required.")
        return

    update_configs(falprs_workdir, gpu_info)

    if commands:
        cmd_script = "\n".join(commands)
        try:
            subprocess.run([
                'docker', 'run', '--gpus', 'all', '--rm',
                '-v', f"{tmp_dir}:/source",
                '-v', f"{falprs_workdir}/model_repository:/destination",
                '--entrypoint=bash',
                f"nvcr.io/nvidia/tritonserver:{triton_version}-py3",
                '-c', cmd_script
            ], check=True)
        except Exception:
            print("Error creating TensorRT plans.")
            sys.exit(1)


def run_all():
    triton_version = os.environ.get('TRITON_VERSION', '24.09')
    falprs_workdir = os.environ.get('FALPRS_WORKDIR', '/opt/falprs')
    arcface_sha1 = os.environ.get('ARCFACE_SHA1', '3642b396053aa5e9cd4518de66baf0d26c9e1467')

    needs_generation, commands, _, gpu_info, tmp_dir = prepare_and_check(triton_version, falprs_workdir, arcface_sha1)
    if not gpu_info:
        return

    if not needs_generation:
        print("All TensorRT plans are up to date.")
        return

    update_configs(falprs_workdir, gpu_info)

    if commands:
        cmd_script = "\n".join(commands)
        try:
            subprocess.run([
                'docker', 'run', '--gpus', 'all', '--rm',
                '-v', f"{tmp_dir}:/source",
                '-v', f"{falprs_workdir}/model_repository:/destination",
                '--entrypoint=bash',
                f"nvcr.io/nvidia/tritonserver:{triton_version}-py3",
                '-c', cmd_script
            ], check=True)
        except Exception:
            print("Error creating TensorRT plans.")
            sys.exit(1)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if mode == 'plan':
        plan()
    elif mode == 'generate':
        generate()
    elif mode == 'all':
        run_all()
    else:
        print(f"Unknown mode: {mode}. Use 'plan', 'generate', or 'all'.")
        sys.exit(1)


if __name__ == '__main__':
    main()
