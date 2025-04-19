
import os
import sys
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn

from utils import base_utils
import models
from data import custom_dataset


def main():
    print(f"Running {os.path.basename(__file__)}\n")
    parser = get_argparser()

    if len(sys.argv) < 2:
        parser.print_help()
        return

    args = parser.parse_args(sys.argv[1:])

    # Placeholder for loading dataset
    data_path = args.data_path
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'Invalid dataset path: {data_path}\n')

    dataset = custom_dataset.SISRDataset(data_path)
    if len(dataset) == 0:
        raise FileNotFoundError(f'No valid images found in dataset path: {data_path}')

    print(f'SISR Dataset loaded successfully from directory: {data_path}')
    print(f'    Dataset length: {len(dataset)}')
    print(f'    Dataset input shape: {dataset[0]['input'].shape}')
    print(f'    Dataset target shape: {dataset[0]['target'].shape}\n')

    # Placeholder for validating models
    sisr_models = []
    for net_type in models.sisr_network_types():
        sisr_models.append(models.SISRModel(net_type))

    for idx, model in enumerate(sisr_models):
        print(f'Model {idx}:')
        print(f'    Network type: {model.network_type()}')
        print(f'    Channel depth: {model.input_channel_depth()}')
        print(f'    Current Device: {model.curr_device()}')

        runs = 20
        print(f'    Integrity test:')
        print(f'        Performing {runs} predicitions...')

        model.set_eval()
        predict_times = []
        for _ in range(runs):
            input = dataset[random.randint(0, len(dataset) - 1)]['input']
            input = input.unsqueeze(0).to(model.curr_device())
            start_time = time.time()
            with torch.no_grad():
                pred = model.predict(input)
            end_time = time.time()
            predict_times.append((end_time - start_time) * 1000)

        avg_predict_time = round(np.mean(predict_times[2:]), 3)
        print(f'    Avg Predict Time for shape {dataset[0]['input'].shape}: {avg_predict_time}ms\n')


def get_argparser():
    parser = argparse.ArgumentParser(
        prog='NAFNet-SISR Train',
        description='Training script for project models',
    )

    parser.add_argument('-d', '--dataset', dest='data_path', help='Directory of training images')
    return parser


if __name__ == "__main__":
    main()

