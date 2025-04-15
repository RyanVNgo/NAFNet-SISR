
import os
import time
import random

import numpy as np
import torch
import torch.nn as nn

from utils import base_utils
import models
from data import data_loader


def main():
    print(f"Running {os.path.basename(__file__)}")

    # Placeholder to verify model variations
    batch_size = 1
    input_channels = 3
    patch_size = 256
    input_dim = [batch_size, input_channels, patch_size, patch_size]

    sisr_models = []
    for net_type in models.sisr_network_types():
        sisr_models.append(models.SISRModel(net_type, c_in = input_channels))

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
        preds = []
        for _ in range(runs):
            input = torch.rand(input_dim).to(model.curr_device())
            start_time = time.time()
            with torch.no_grad():
                pred = model.predict(input)
            end_time = time.time()
            preds.append(pred.cpu().detach().numpy())
            predict_times.append((end_time - start_time) * 1000)

        avg_predict_time = round(np.mean(predict_times[2:]), 3)
        print(f'    Avg Predict Time for shape {input_dim}: {avg_predict_time}ms\n')


if __name__ == "__main__":
    main()

