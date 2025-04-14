
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

    sisr_models = []
    for net_type in models.sisr_network_types():
        sisr_models.append(models.SISRModel(net_type, c_in = 3))

    for idx, model in enumerate(sisr_models):
        print(f'Model {idx}:')
        print(f'    Network type: {model.network_type()}')
        print(f'    Channel depth: {model.input_channel_depth()}')
        print(f'    Current Device: {model.curr_device()}')
        print(f'    Integrity test:')

        predict_times = []
        for _ in range(20):
            input = torch.rand(1, 3, 256, 256).to(model.curr_device())
            start_time = time.time()
            preds = model.predict(input).detach().numpy()
            end_time = time.time()
            predict_times.append((end_time - start_time) * 1000)

        avg_predict_time = round(np.mean(predict_times[2:]), 3)
        print(f'    Predict Avg (ms): {avg_predict_time}\n')


if __name__ == "__main__":
    main()

