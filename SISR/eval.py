
import os
import sys
import argparse

import numpy as np
from PIL import Image

import metrics


def main():
    print('')
    parser = get_argparser()

    if len(sys.argv) < 2:
        parser.print_help()
        return
    args = parser.parse_args(sys.argv[1:])

    sr_path = args.sr_path
    if os.path.exists(sr_path) == False:
        print(f'SR path does not exist. Exiting.')
        return

    hr_path = args.hr_path
    if os.path.exists(hr_path) == False:
        print(f'HR path does not exist. Exiting.')
        return

    # Open images
    sr_image = Image.open(sr_path).convert('RGB')
    hr_image = Image.open(hr_path).convert('RGB')
    sr_data = np.array(sr_image)
    hr_data = np.array(hr_image)
    print(f'Opening SR Image: {sr_path}')
    print(f'Opening HR Image: {hr_path}\n')

    if sr_data.shape != hr_data.shape:
        print(f'Image shapes do not match. Exiting.')
        return

    print(f'Images Shape: {sr_data.shape}\n')

    psnr = metrics.PSNR(sr_data, hr_data)
    ssim = metrics.SSIM(sr_data, hr_data)
    print(f'SR <-> HR Evaluation')
    print(f'    PSNR: {psnr:.4f}')
    print(f'    SSIM: {ssim:.4f}\n')

    width, height = hr_image.size
    bicubic = hr_image.resize((width // 2, height // 2), Image.BICUBIC)
    bicubic = bicubic.resize((width, height), Image.BICUBIC)
    bicubic = np.array(bicubic)
    psnr_bicubic = metrics.PSNR(bicubic, hr_data)
    ssim_bicubic = metrics.SSIM(bicubic, hr_data)
    print(f'Bicubic <-> HR Evaluation')
    print(f'    PSNR: {psnr_bicubic:.4f}')
    print(f'    SSIM: {ssim_bicubic:.4f}\n')

    print(f'Performance Difference')
    print(f'    PSNR of SR is {psnr - psnr_bicubic:.4f} vs Bicubic')
    print(f'    SSIM of SR is {ssim - ssim_bicubic:.4f} vs Bicubic\n')

    return


def get_argparser():
    parser = argparse.ArgumentParser(
        prog='eval.py',
        description='Evaluation script for comparing images based on metrics',
    )

    parser.add_argument(
        '-s', 
        dest='sr_path', 
        type=str, 
        required=True, 
        help='Path to super-resolution image'
    )
    parser.add_argument(
        '-t', 
        dest='hr_path',
        type=str, 
        required=True, 
        help='Path to high resolution image'
    )
    return parser


if __name__ == "__main__":
    main()


