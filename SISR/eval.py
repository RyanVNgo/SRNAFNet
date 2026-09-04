
import os
import sys
import time
import argparse

import torch
import numpy as np
from torchvision import transforms
from PIL import Image

import metrics
import utils
import models
import data

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def main():
    print('')
    parser = get_argparser()

    if len(sys.argv) < 2:
        parser.print_help()
        return
    args = parser.parse_args(sys.argv[1:])

    model_path = args.model_path
    if os.path.exists(model_path) == False:
        print(f'Model path does not exist. Exiting.')
        return

    yaml_path = args.yaml_path
    if args.yaml_path is None:
        dir = os.path.dirname(model_path)
        filename = os.path.basename(model_path)
        filename = os.path.splitext(filename)[0] + '.yaml'
        yaml_path = os.path.join(dir, filename)
        
    dataset_path = args.dataset_path
    if os.path.exists(dataset_path) == False:
        print(f'Dataset path does not exist. Exiting.')
        return

    scaling_factor = args.scaling_factor
    if scaling_factor == None:
        scaling_factor = 2
    
    print(f'Initializing model')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using Device: {device}')
    model = models.create_sisr_model(utils.parse_options(yaml_path))
    model.load_model(model_path)
    model.set_eval()
    model.to_device(device)

    print(f'Finding Images...')
    image_paths = valid_image_paths_in_directory(dataset_path)
    print(f'Images found: {len(image_paths)}')

    print(f'\nStarting Evaluation') 
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    psnr_list = []
    ssim_list = []
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for idx, image_path in enumerate(image_paths):
        print(f'    Evaluating Image [{idx+1}]: {image_path}')
        hr_image = Image.open(image_path).convert('RGB')
        hr_image = crop_to_scale(hr_image, scaling_factor)
        lr_image = downscale_image(hr_image, scaling_factor)
        
        lr_tensor = to_tensor(lr_image).unsqueeze(0).to(device)
        start.record()
        with torch.no_grad():
            sr_tensor = model.predict(lr_tensor)
        end.record()
        torch.cuda.synchronize()

        inference_time = start.elapsed_time(end)
        times.append(inference_time)

        sr_tensor = sr_tensor.squeeze(0).clamp(0,1).detach().cpu()
        sr_tensor = (sr_tensor * 255).byte()
        sr_image = transforms.functional.to_pil_image(sr_tensor)

        psnr = metrics.PSNR(np.array(sr_image), np.array(hr_image), 255.0)
        ssim = metrics.SSIM(np.array(sr_image), np.array(hr_image))
        print(f'        Inference Time: {inference_time : .4f}ms')
        print(f'        PSNR: {psnr:.4f} | SSIM: {ssim:.4f}')
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        del sr_tensor
        del lr_tensor

    print(f'\nAverage metrics over {len(image_paths)} images:')
    print(f'    Inference Time: {sum(times[1:]) / len(times) : .4f}ms')
    print(f'    PSNR: {sum(psnr_list) / len(psnr_list) : .4f}')
    print(f'    SSIM: {sum(ssim_list) / len(ssim_list) : .4f}')
    print(f'')

    return


def crop_to_scale(image, scale):
    width, height = image.size
    new_width = (width // scale) * scale
    new_height = (height // scale) * scale
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height
    return image.crop((left, top, right, bottom))


def downscale_image(image, factor):
    width, height = image.size
    return image.resize((width // factor, height // factor), Image.BICUBIC)


def valid_image_paths_in_directory(dir_path):
    image_paths = []
    for filename in os.listdir(dir_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            if 'LR' not in filename:
                filepath = os.path.join(dir_path, filename)
                image_paths.append(filepath)
    return image_paths


def get_argparser():
    parser = argparse.ArgumentParser(
        prog='eval.py',
        description='Script for running models on a dataset to gather metrics',
    )

    parser.add_argument('-m', dest='model_path', type=str, required=True, help='Path to model .pth file')
    parser.add_argument('-c', dest='yaml_path', type=str, required=False, help='Path to model .yaml file')
    parser.add_argument('-d', dest='dataset_path', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('-f', dest='scaling_factor', type=int, required=True, help='Scaling factor')
    return parser


if __name__ == "__main__":
    main()


