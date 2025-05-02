
import os
import sys
import argparse

import torch
import numpy as np
from torchvision import transforms
from PIL import Image

import metrics
import utils
import models
import data


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
    for idx, image_path in enumerate(image_paths):
        print(f'    Evaluating Image [{idx+1}]: {image_path}')
        hr_image = Image.open(image_path).convert('RGB')
        lr_path = find_lowres(image_path)
        lr_image = downscale_image(hr_image)
        if lr_path is not None:
            print(f'    LR Pair Found: {lr_path}')
            lr_image = Image.open(lr_path).convert('RGB')
        else:
            lr_image = downscale_image(hr_image)
        
        lr_tensor = to_tensor(lr_image).unsqueeze(0).to(device)
        with torch.no_grad():
            sr_tensor = model.predict(lr_tensor)

        sr_tensor = sr_tensor.squeeze(0).clamp(0,1).detach().cpu()
        sr_tensor = (sr_tensor * 255).byte()
        sr_image = transforms.functional.to_pil_image(sr_tensor)

        psnr = metrics.PSNR(np.array(sr_image), np.array(hr_image))
        ssim = metrics.SSIM(np.array(sr_image), np.array(hr_image))
        print(f'        PSNR: {psnr:.4f} | SSIM: {ssim:.4f}')
        psnr_list.append(psnr)
        ssim_list.append(ssim)

    print(f'\nAverage metrics over {len(image_paths)} images:')
    print(f'    PSNR: {sum(psnr_list) / len(psnr_list) : .4f}')
    print(f'    SSIM: {sum(ssim_list) / len(ssim_list) : .4f}')
    print(f'')

    return


def find_lowres(image_path):
    dir = os.path.dirname(image_path)
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    if name[-2:] == 'HR':
        name = name[:-2] + 'LR'
        filename = name + ext
    else:
        return None

    lr_path = os.path.join(dir, filename)

    if os.path.exists(lr_path):
        return lr_path
    else:
        return None


def downscale_image(image):
    width, height = image.size
    return image.resize((width // 2, height // 2), Image.BICUBIC)


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
    return parser


if __name__ == "__main__":
    main()


