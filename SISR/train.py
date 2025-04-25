
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
    valid_interval = training_options.get('valid_interval', 32)

    print('Setting up data loaders...\n')
    dataset_opts = options.get('datasets')
    dataloaders = data.setup_dataloaders(dataset_opts, batch_size, model.curr_device())

    model_save_path = define_model_save_path(options)

    print(f'Starting training for network: {model.network_type()}')
    print(f'    Device: {model.curr_device()}')
    print(f'    Optimizer: {optimizer.__class__.__name__}')
    print(f'    Scheduler: {scheduler.__class__.__name__}')
    print(f'    Loss Functions:')
    for loss_fn in criterions:
        print(f'        {loss_fn.__class__.__name__}')
    print(f'    Training dataset count: {len(dataloaders['train'].dataset)}')
    print(f'    Validation dataset count: {len(dataloaders['valid'].dataset)}')
    print(f'    Batch Size: {batch_size}')
    print(f'    Iterations: {iterations}')
    print(f'    Validation Interval: {valid_interval}')
    print(f'        (Effective Epochs): {iterations / len(dataloaders['train'])}')
    print(f'    Model will be saved to:\n       {model_save_path}')

    model = train_for_iterations(model, dataloaders, optimizer, scheduler, criterions, iterations, valid_interval)

    model.save_model(model_save_path)
    model_config_save_path = define_model_config_save_path(model_save_path)
    model.save_config(model_config_save_path)

    return


def train_for_iterations(model, dataloaders, optimizer, scheduler, criterions, iterations, valid_interval):
    device = model.curr_device()
    train_loader = dataloaders['train']
    valid_loader = dataloaders['valid']
    train_iter = iter(train_loader)
    start_time = time.time()
    for i in range(iterations):
        iter_start_time = time.time()
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
        nn.utils.clip_grad_norm(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        iter_elapsed_time = time.time() - iter_start_time
        total_elapsed_time = time.time() - start_time

        print(f'---')
        print(f'Iteration {i+1} / {iterations}')
        print(f'    LR: {scheduler.get_last_lr()[0]:.9f}')
        print(f'        Train Loss: {train_loss:0.6f}')
        if torch.cuda.is_available():
            print(f'    Memory Usage (reserved): {torch.cuda.memory_reserved() / (1024**2):.2f} MB')
        print(f'    Iteration Time: {iter_elapsed_time:2f}s')
        print(f'    Total elapsed Time: {total_elapsed_time:2f}s')

        valid_loss = None
        if i % valid_interval == valid_interval - 1:
            print(f'---')
            print(f'Validating model...')
            valid_loss = validate_model(model, valid_loader, criterions)
            print(f'    Validation Loss: {valid_loss}')
        # time.sleep(0.15)

    elapsed_time = time.time() - start_time
    print(f'\nTraining Complete')
    print(f'    Total Training Time: {elapsed_time:.2f}s')

    return model


def validate_model(model, valid_loader, criterions):
    model.set_eval()
    device = model.curr_device()
    valid_loss = 0.0
    for valid_batch in valid_loader:
        with torch.no_grad():
            output = model.predict(valid_batch['input'].to(device))
        for loss_fn in criterions:
            valid_loss += loss_fn(output, valid_batch['target'].to(device))
    return valid_loss / len(valid_loader)


def define_model_save_path(options):
    save_path = os.path.abspath(options['model'].get('save_path', './'))
    print(save_path)
    if os.path.exists(save_path) == False:
        save_path = os.path.abspath(os.getcwd())
    file_name = options['name'] + '.pth'

    model_save_path = os.path.join(save_path, file_name)
    if os.path.exists(model_save_path):
        index = 1;
        while os.path.exists(model_save_path):
            file_name = f'{options['name']}_({index}).pth'
            model_save_path = os.path.join(save_path, file_name)
            index += 1

    return model_save_path


def define_model_config_save_path(model_save_path):
    dir = os.path.dirname(model_save_path)
    filename = os.path.basename(model_save_path)
    filename = os.path.splitext(filename)[0] + '.yaml'
    model_config_save_path = os.path.join(dir, filename)
    return model_config_save_path


def setup_criterions(options):
    criterions = []
    for type in options.keys():
        crit = models.create_loss(type, options.get(type))
        if crit is not None:
            criterions.append(crit)
    return criterions


def setup_scheduler(optimizer, options):
    scheduler_type = options.get('type', 'CosineAnnealingLR')
    match scheduler_type:
        case 'CosineAnnealingLR':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=options.get('t_max'),
                eta_min=options.get('eta_min', 1e-6)
            )


def setup_optimizer(params, options):
    optim_type = options.get('type', 'AdamW')
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
        prog='train.py',
        description='Training script for project models',
    )

    parser.add_argument('-opt', type=str, required=True, help='Path to yaml file')
    return parser


if __name__ == "__main__":
    main()

