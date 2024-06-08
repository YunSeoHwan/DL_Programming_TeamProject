import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from argparse import Namespace
import subprocess

device = torch.device("cpu")

def run_command(command):
    try:
        subprocess.check_call(command, shell=True)
        print(f"Command '{command}' executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing '{command}': {e}")

def git_clone(repo_url, clone_to_path):
    command = f"git clone {repo_url} {clone_to_path}"
    run_command(command)

def pip_install_requirements(requirements_path):
    command = f"{sys.executable} -m pip install -r {requirements_path}"
    run_command(command)
    
def replace_legacy(legacy_file_path):
    with open(legacy_file_path, 'r') as file:
        lines = file.readlines()

    replacements = {
        'mask = np.zeros_like(nmap).astype(np.bool)': 'mask = np.zeros_like(nmap).astype(np.bool_)',
        'plt.tight_layout(True)': 'plt.tight_layout()',
        "matplotlib.use('TkAgg')": "matplotlib.use('Agg')"
    }

    modified_lines = []
    changed_lines_info = []

    for line_number, line in enumerate(lines):
        original_line = line
        for old_value, new_value in replacements.items():
            if old_value in line:
                line = line.replace(old_value, new_value)
                changed_lines_info.append((line_number + 1, original_line.strip(), line.strip()))
        modified_lines.append(line)

    with open(legacy_file_path, 'w') as file:
        file.writelines(modified_lines)

    for line_info in changed_lines_info:
        print(f"Line {line_info[0]}: '{line_info[1]}' -> '{line_info[2]}'")
          
def install_libraries(root_path='', ignore_asking=False):
    if not ignore_asking:
        user_input = input("Would you like to download and install the required libraries? (y/n): ").strip().lower()
        if user_input != 'y':
            print("Installation aborted by the user.")
            return
    print("Start downloading sea-thru...")
    git_clone("https://github.com/hainh/sea-thru.git", root_path + "sea-thru")
    print('Done\n')
    print("Start downloading monodepth2...")
    git_clone("https://github.com/nianticlabs/monodepth2.git", root_path + "monodepth2")
    print('Done\n')
    
    print("Start downloading sea-thru requirements..")
    pip_install_requirements(root_path + "sea-thru/requirements.txt")
    print('Done\n')
    
    print("replace legacy...")
    replace_legacy(root_path + "sea-thru/seathru.py")
    print('Done\n')

    monodepth2_path = root_path+"monodepth2"
    sea_thru_path = root_path+"sea-thru"

    # 시스템 경로에 추가
    sys.path.append(monodepth2_path)
    sys.path.append(sea_thru_path)

    print('Finish installing!\n')



def load_monodepth2_model(model_name="mono_640x192"):
    import networks
    from utils import download_model_if_doesnt_exist

    download_model_if_doesnt_exist(model_name)
    model_path = os.path.join("models", model_name)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    encoder = networks.ResnetEncoder(18, False).to(device)
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(3)).to(device)

    loaded_dict_enc = torch.load(encoder_path, map_location=device)
    filtered_dict_enc = {k : v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc, strict=False)

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict, strict=False)

    encoder.eval()
    depth_decoder.eval()
    
    return encoder, depth_decoder

def run_sea_thru(img, depth_map):
    import seathru

    args = Namespace(
        image=img,
        depth_map=depth_map,
        output='restored_image.png',
        f=2.0,
        l=0.5,
        p=0.01,
        min_depth=0.1,
        max_depth=1.0,
        spread_data_fraction=0.01,
        size=320,
        output_graphs=False,
        preprocess_for_monodepth=False,
        monodepth=True,
        monodepth_add_depth=2.0,
        monodepth_multiply_depth=10.0,
        equalize_image=False
    )
    try:
        restored_img = seathru.run_pipeline(img / 255.0, depth_map, args)
    except TypeError as e:
        print(f"Error: {e}")
        restored_img = img
    return restored_img

def apply_sea_thru(image, encoder, depth_decoder):
    from layers import disp_to_depth
    input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_height, original_width, _ = input_image.shape
    feed_height, feed_width = 192, 640

    input_image_resized = cv2.resize(input_image, (feed_width, feed_height), interpolation=cv2.INTER_LANCZOS4)
    input_image_pytorch = transforms.ToTensor()(input_image_resized).unsqueeze(0)

    with torch.no_grad():
        features = encoder(input_image_pytorch.to(device))
        outputs = depth_decoder(features)
        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(disp, (original_height, original_width), mode="bilinear", align_corners=False)

    scaled_disp, depth = disp_to_depth(disp_resized, 0.1, 100)
    depth_np = depth.squeeze().cpu().numpy()

    image_float32 = image.astype(np.float32)
    restored_img = run_sea_thru(image_float32, depth_np)
    return restored_img