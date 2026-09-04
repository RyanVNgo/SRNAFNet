
import os
import sys
import time
import datetime
import random
import argparse
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

import utils
import models
import data
import metrics


def main():
    parser = get_argparser()

    if len(sys.argv) < 2:
        parser.print_help()
        return

    args = parser.parse_args(sys.argv[1:])
    options = utils.parse_options(args.opt)

    print('Setting up Model...')
    model_name = options.get('name')
    model_options = options.get('model')
    model = models.create_sisr_model(model_options)
    model_save_path = utils.model_save_path(model_options.get('save_path', None), model_name)
    config_save_path = utils.config_save_path(model_save_path)

    training_options = options.get('training', None)
    print('Setting up optimizer...')
    optimizer = setup_optimizer(model.get_parameters(), training_options.get('optimizer'))
    print('Setting up scheduler...')
    scheduler = setup_scheduler(optimizer, training_options.get('scheduler'))
    iterations = training_options.get('iterations', 1)
    print('Setting up criterions...')
    criterions = setup_criterions(training_options.get('losses'), model.curr_device())
    batch_size = training_options.get('batch_size', 16)
    valid_interval = training_options.get('valid_interval', 32)

    print('Setting up data loaders...\n')
    dataset_options = options.get('datasets')
    dataloaders = data.setup_dataloaders(dataset_options, batch_size, model.curr_device())

    logging_dir = os.path.abspath('./logs/' + model_name)
    log_writer = SummaryWriter(log_dir=logging_dir)

    print(f'Starting training for network: {model.network_type()}')
    print(f'    Device: {model.curr_device()}')
    print(f'    Optimizer: {optimizer.__class__.__name__}')
    print(f'    Scheduler: {scheduler.__class__.__name__}')
    print(f'    Loss Functions:')
    for loss_fn in criterions:
        print(f'        {loss_fn.__class__.__name__}')
    train_dataset_count = len(dataloaders['train'].dataset)
    valid_dataset_count = len(dataloaders['valid'].dataset)
    print(f'    Training dataset count: {train_dataset_count}')
    print(f'    Validation dataset count: {valid_dataset_count}')
    print(f'    Batch Size: {batch_size}')
    print(f'    Iterations: {iterations}')
    print(f'    Validation Interval: {valid_interval}')
    print(f'    Model will be saved to:\n       {model_save_path}')

    model.set_model_save_path(model_save_path)
    model.set_config_save_path(config_save_path)

    model = train_for_iterations(
        model, 
        dataloaders, 
        optimizer, 
        scheduler, 
        criterions, 
        iterations, 
        valid_interval,
        log_writer
    )

    log_writer.close()
    model.save_model(model_save_path)
    model.save_config(config_save_path)
    return


def train_for_iterations(model, dataloaders, optimizer, scheduler, criterions, iterations, valid_interval, log_writer):
    time_window = deque(maxlen=64)
    start_time = time.time()

    device = model.curr_device()
    train_loader = dataloaders['train']
    valid_loader = dataloaders['valid']
    train_iter = iter(train_loader)
    scaler = GradScaler()

    # discrim = models.Discriminator(iterations, device)

    for i in range(iterations):
        iter_start_time = time.time()
        try:
            train_batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            train_batch = next(train_iter)

        input = train_batch['input'].to(device)
        target = train_batch['target'].to(device)

        model.set_train()
        with autocast(device):
            pred = model.predict(input)
            loss = 0.0
            for loss_fn in criterions:
                loss += loss_fn(pred, target)


        # d_loss = discrim.update(pred, target)
        # g_loss = discrim.adversarial_loss(pred, target)
        # loss += g_loss * 1e-3

        # log_writer.add_scalar('Discrim/D_Loss', d_loss.item(), i)
        # log_writer.add_scalar('Discrim/Adv_Loss', g_loss.item(), i)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(model.get_parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        iter_elapsed_time = time.time() - iter_start_time
        total_elapsed_time = time.time() - start_time
        time_window.append(iter_elapsed_time)

        psnr = metrics.PSNR_Tensor(pred, target)

        log_writer.add_scalar('Loss/Train', loss.item(), i)
        log_writer.add_scalar('PSNR/Train', psnr, i)

        print(f'---')
        print(f'Iteration {i+1} / {iterations}')
        print(f'    LR: {scheduler.get_last_lr()[0]:.9f}')
        print(f'        Train Loss: {loss:0.6f}')
        print(f'    Metrics:')
        print(f'        PSNR: {psnr:.4f}')
        if device == 'cuda':
            free, total = torch.cuda.mem_get_info(device)
            print(f'    Memory Usage: {(total - free) / (1024**2):.2f} MB / {total / (1025**2):.2f} MB')
        print(f'    Iteration Time: {iter_elapsed_time:2f}s')
        print(f'    Total elapsed Time: {total_elapsed_time:2f}s')

        est_rem_time = (iterations - i) * (sum(time_window) / len(time_window))
        pred_end_time = time.time() + est_rem_time
        end_time_str = datetime.datetime.fromtimestamp(pred_end_time).strftime('%Y-%m-%d %H:%M:%S')
        print(f'    Predicted End Time: {end_time_str}')

        valid_loss = None
        if i % valid_interval == valid_interval - 1:
            print(f'---')
            print(f'Validating model...')
            valid_loss, img_ex_grid, v_psnr = validate_model(model, valid_loader, criterions)
            print(f'    Validation Loss: {valid_loss}')
            print(f'    PSNR: {v_psnr:0.4f}')
            log_writer.add_scalar('PSNR/Valid', v_psnr, i)
            log_writer.add_scalar('Loss/Valid', valid_loss, i)
            log_writer.add_image('LR_SR_HR', img_ex_grid, i)

        if i % 1000 == 999:
            model.save_model()
            model.save_config()

    elapsed_time = time.time() - start_time
    print(f'\nTraining Complete')
    print(f'    Total Training Time: {elapsed_time:.2f}s')

    return model


def validate_model(model, valid_loader, criterions):
    model.set_eval()
    device = model.curr_device()
    valid_loss = 0.0
    input, pred, target = None, None, None
    psnrs = []
    for valid_batch in valid_loader:
        input = valid_batch['input'].to(device)
        target = valid_batch['target'].to(device)
        with torch.no_grad():
            pred = model.predict(input)
            psnrs.append(metrics.PSNR_Tensor(pred, target))
        for loss_fn in criterions:
            valid_loss += loss_fn(pred, target)
    ex_grid = prepare_image_preview(input[0], pred[0], target[0])
    return (valid_loss / len(valid_loader)), ex_grid, (sum(psnrs) / len(psnrs))


def prepare_image_preview(lr_img, sr_img, hr_img):
    hr_img = hr_img
    sr_img = sr_img
    sf = hr_img.shape[-1] / lr_img.shape[-1]
    lr_img = nn.functional.interpolate(lr_img.unsqueeze(0), scale_factor=sf).squeeze(0)
    return torch.cat((lr_img, sr_img, hr_img), dim=2)


def setup_criterions(options, device):
    criterions = []
    for type in options.keys():
        crit = models.create_loss(type, options.get(type), device)
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

