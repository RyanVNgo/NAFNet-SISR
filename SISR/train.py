
import os

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

    input = torch.rand(1, 3, 128, 128)
    for idx, model in enumerate(sisr_models):
        print(f'Model {idx}:')
        print(f'    Network type: {model.network_type()}')
        print(f'    Channel depth: {model.input_channel_depth()}')
        print(f'    Integrity test:')
        output = model.run(input).detach().numpy()
        print(f'        {output.shape}\n')


if __name__ == "__main__":
    main()

