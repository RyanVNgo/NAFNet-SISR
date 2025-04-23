
import os
import sys
import time
import argparse

import torch
from torchvision import transforms
from PIL import Image

import utils
import models
import data


def main():
    print(f"Running {os.path.basename(__file__)}")
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
        
    if os.path.exists(yaml_path) == False:
        print(f'Yaml path does not exist. Exiting.')
        return
    
    input_path = args.input_path
    if os.path.exists(input_path) == False:
        print(f'Input path does not exist. Exiting.')
        return

    output_path = args.output_path
    if os.path.exists(os.path.dirname(output_path)) == False:
        print(f'Output path does not exist. Exiting.')
        return

    print(f'Starting Demo')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'    Using Device: {device}')

    print(f'    Opening image at {input_path}')
    input_img = Image.open(input_path).convert('RGB')

    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    print(f'    Converting input image to tensor')
    input_tensor = to_tensor(input_img).unsqueeze(0).to(device)

    print(f'    Initializing model')
    model = models.create_sisr_model(utils.parse_options(yaml_path))
    model.load_model(model_path)
    model.set_eval()
    model.to_device(device)

    print(f'    Predicting...')
    start_time = time.time()
    with torch.no_grad():
        output_tensor = model.predict(input_tensor)
    print(f'    Predict Time: {time.time() - start_time : .6f} s')

    print(f'    Converting output to image')
    output_img = output_tensor.squeeze(0).clamp(0, 1).cpu()
    output_pil = to_pil(output_img)

    print(f'    Saving output image')
    output_pil.save(output_path)

    print(f'    Output image saved at {output_path}')
    print(f'Demo Complete')


def get_argparser():
    parser = argparse.ArgumentParser(
        prog='NAFNet-SISR Demo',
        description='Training script for project models',
    )

    parser.add_argument('-m', dest='model_path',type=str, required=True, help='Path to model .pth file')
    parser.add_argument('-c', dest='yaml_path', type=str, required=False, help='Path to model .yaml file')
    parser.add_argument('-i', dest='input_path', type=str, required=True, help='Path to input image')
    parser.add_argument('-o', dest='output_path',type=str, required=True, help='Path to save output image')
    return parser


if __name__ == "__main__":
    main()


