
import os

import numpy as np
import torch
import torch.nn as nn

from utils import base_utils
from model import models
from data import data_loader


def main():
    print(f"Running {os.path.basename(__file__)}")

    channels = 3
    model = models.BaseModel(channels)

    for module in model.get_modules():
        print(module)


if __name__ == "__main__":
    main()

