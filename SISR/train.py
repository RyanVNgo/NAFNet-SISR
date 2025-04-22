
import os
import sys
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn

import utils
import models
from data import custom_dataset


def main():
    print(f"Running {os.path.basename(__file__)}\n")
    parser = get_argparser()

    if len(sys.argv) < 2:
        parser.print_help()
        return

    args = parser.parse_args(sys.argv[1:])
    options = utils.parse_options(args.opt)

    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.device('cuda')

    model = models.create_sisr_model(options.get('model'))

    training_options = options.get('training', None)
    optimizer = setup_optimizer(model.get_parameters(), training_options.get('optimizer'))
    scheduler = setup_scheduler(optimizer, training_options.get('scheduler'))
    iterations = training_options.get('iterations', 1)

    return


def setup_scheduler(optimizer, options):
    scheduler_type = options.get('type', 'AdamW')
    match scheduler_type:
        case 'CosineAnnealingLR':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=options.get('t_max'),
                eta_min=options.get('eta_min', 0.0)
            )


def setup_optimizer(params, options):
    optim_type= options.get('type', 'AdamW')
    lr = options.get('lr', 1e-3)
    match optim_type:
        case 'Adam':
            return torch.optim.Adam(
                params=params,
                lr=lr,
                betas=options.get('betas', [0.9, 0.999]),
                weight_decay=options.get('weight_decay', 0.0)
            )
        case 'AdamW':
            return torch.optim.AdamW(
                params=params,
                lr=lr,
                betas=options.get('betas', [0.9, 0.999]),
                weight_decay=options.get('weight_decay', 0.01)
            )
        case 'RMSprop':
            return torch.optim.RMSprop(
                params=params,
                lr=lr,
                weight_decay=options.get('weight_decay', 0.0),
                momentum=options.get('momentum', 0.0)
            )
        case 'SGD':
            return torch.optim.SGD(
                params=params,
                lr=lr,
                momentum=options.get('momentum', 0.0),
                dampening=options.get('dampening', 0.0),
                weight_decay=options.get('weight_decay', 0.0)
            )


def get_argparser():
    parser = argparse.ArgumentParser(
        prog='NAFNet-SISR Train',
        description='Training script for project models',
    )

    parser.add_argument('-opt', type=str, required=True, help='Path to yaml file')
    return parser


if __name__ == "__main__":
    main()

