
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
import data

def main():
    print(f"Running {os.path.basename(__file__)}\n")
    parser = get_argparser()

    if len(sys.argv) < 2:
        parser.print_help()
        return

    args = parser.parse_args(sys.argv[1:])
    options = utils.parse_options(args.opt)

    print('Setting up Model...')
    model = models.create_sisr_model(options.get('model'))
    training_options = options.get('training', None)
    print('Setting up optimizer...')
    optimizer = setup_optimizer(model.get_parameters(), training_options.get('optimizer'))
    print('Setting up scheduler...')
    scheduler = setup_scheduler(optimizer, training_options.get('scheduler'))
    iterations = training_options.get('iterations', 1)
    print('Setting up criterions...')
    criterions = setup_criterions(training_options.get('losses'))
    batch_size = training_options.get('batch_size', 16)

    print('Setting up data loaders...\n')
    dataset_opts = options.get('datasets')
    dataloaders = data.setup_dataloaders(dataset_opts, batch_size, model.curr_device())
    train_loader = dataloaders['train']
    valid_loader = dataloaders['valid']

    print(f'Starting training for network: {model.network_type()}')
    print(f'    Optimizer: {optimizer.__class__.__name__}')
    print(f'    Scheduler: {scheduler.__class__.__name__}')
    print(f'    Loss Functions:')
    for loss_fn in criterions:
        print(f'        {loss_fn.__class__.__name__}')
    print(f'    Training dataset count: {len(train_loader.dataset)}')
    print(f'    Validation dataset count: {len(valid_loader.dataset)}')
    print(f'    Batch Size: {batch_size}')
    print(f'    Iterations: {iterations}')
    print(f'        (Effective Epochs): {iterations / len(train_loader)}')

    device = model.curr_device()
    start_time = time.time()
    for i in range(iterations):
        iter_start_time = time.time()

        train_iter = iter(train_loader)
        try:
            train_batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            train_batch = next(train_iter)

        model.set_train()
        output = model.predict(train_batch['input'].to(device))

        train_loss = 0.0
        for loss_fn in criterions:
            train_loss += loss_fn(output, train_batch['target'].to(device))
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        scheduler.step()

        model.set_eval()
        valid_loss = 0.0
        for valid_batch in valid_loader:
            with torch.no_grad():
                output = model.predict(valid_batch['input'].to(device))

            for loss_fn in criterions:
                valid_loss += loss_fn(output, valid_batch['target'].to(device))

        valid_loss = valid_loss / len(valid_loader)

        iter_elapsed_time = time.time() - iter_start_time
        total_elapsed_time = time.time() - start_time
        print(f'---')
        print(f'Iteration {i+1} / {iterations}')
        print(f'    LR: {scheduler.get_last_lr()[0]:.9f}')
        print(f'        Train Loss: {train_loss:0.6f}')
        print(f'        Valid Loss: {valid_loss:0.6f}')
        print(f'    Iteration Time: {iter_elapsed_time:2f}s')
        print(f'    Total elapsed Time: {total_elapsed_time:2f}s')

    elapsed_time = time.time() - start_time
    print(f'\nTraining Complete')
    print(f'    Total Training Time: {elapsed_time:.2f}s')

    return


def setup_criterions(options):
    criterions = []
    for type in options.keys():
        crit = models.create_loss(type, options.get(type))
        if crit is not None:
            criterions.append(crit)
    return criterions


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

