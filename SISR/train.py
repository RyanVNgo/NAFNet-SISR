
import os

import numpy as np
import torch
import torch.nn as nn

from utils import base_utils
import models
from data import data_loader


def main():
    print(f"Running {os.path.basename(__file__)}")

    network_type = 'NAFNet'
    model = models.SISRModel(net_type = network_type, c_in = 3)

    sisr_models = []
    for net_type in models.sisr_network_types():
        sisr_models.append(models.SISRModel(net_type, c_in = 3))

    for idx, model in enumerate(sisr_models):
        print(f'Model {idx}:')
        print(f'    Network type: {model.network_type()}')
        print(f'    Channel depth: {model.input_channel_depth()}\n')


if __name__ == "__main__":
    main()

