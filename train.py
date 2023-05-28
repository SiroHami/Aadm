import argparse
import imaplib
import os
import time
import random
import sys
import logging
import numpy as np
from progress.bar import Bar as Bar

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from utils.utils import accuracy, get_data, prepare_logger, initialize_logging,  prepare_model
from .datasets import get_dataset_metainfo
from .models import *


sys.path.append(os.path.join(os.path.dirname(__file__), "progress"))

def add_trtain_parser_arguments(parser):
    """
    Create python script parameters (for training/classification specific subpart).

    Parameters:
    ----------
    parser : ArgumentParser
        ArgumentParser instance.
    """
    #general settings
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        help="model name (default: resnet18)")
    parser.add_argument(
        "--use-pretrained",
        action="store_true",
        help="use pretrained model")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="path to latest checkpoint (default: none)")
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=0,
        help="number of GPUs to use")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="number of data loading workers (default: 4)")
    #training
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="input batch size for training (default: 256)")
    parser.add_argument(
        "--batch-size-scale",
        type=int,
        default=1,
        help="batch size scale factor (default: 1)")
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="number of epochs to train (default: 100)")
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="starting epoch number (default: 1)")
    parser.add_argument(
        "--attempt",
        type=int,
        default=1,
        help="current attempt number (default: 1)")
    #optimizer
    parser.add_argument(
        "--optimizer-name",
        type=str,
        default="sgd",
        help="optimizer name (default: sgd)")
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="initial learning rate (default: 0.1)")
    parser.add_argument(
        "--lr-mode",
        type=str,
        default="cosine",
        help="learning rate scheduler mode (default: cosine)")
    parser.add_argument(
        "--lr-decay",
        type=float,
        default=0.1,
        help="learning rate decay (default: 0.1)")
    parser.add_argument(
        "--lr-decay-period",
        type=int,
        default=0,
        help="learning rate decay period (default: 0)")
    parser.add_argument(
        "--lr-decay-epoch",
        type=str,
        default="40,60",
        help="epoches at which learning rate decays (default: 40,60)")
    parser.add_argument(
        "--lr-warmup",
        type=int,
        default=1e-8,
        help="learning rate warmup (default: 1e-8)")
    parser.add_argument(
        "--warmup-mode",
        type=str,
        default="linear",
        help="warmup mode (default: linear)")
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="momentum (default: 0.9)")
    parser.add_argument(
        "--wd",
        type=float,
        default=1e-4,
        help="weight decay (default: 1e-4)")
    parser.add_argument(
        "--gamma-wd-multi",
        type=float,
        default=1.0,
        help="weight decay multiplier for gamma and beta in BN layers (default: 1.0)")
    parser.add_argument(
        "--beta-wd-multi",
        type=float,
        default=1.0,
        help="weight decay multiplier for gamma and beta in BN layers (default: 1.0)")
    parser.add_argument(
        "--bias-wd-multi",
        type=float,
        default=1.0,
        help="weight decay multiplier for bias in conv and FC layers (default: 1.0)")
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=0.0,
        help="gradient clipping (default: 0.0)")
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="label smoothing (default: 0.0)")
    #agumentation
    parser.add_argument(
        "--cutout",
        type=float,
        default=0.0,
        help="cutout method cutout enabled if > 0.0 (default: 0.0)")
    parser.add_argument(
        "--cutout-epoch-tail",
        type=int,
        default=0,
        help="cutout epoch tail (default: 0)")
    parser.add_argument(
        "--mixup",
        type=float,
        default=0.0,
        help="mixup method mixup enabled if > 0.0 (default: 0.0)")
    parser.add_argument(
        "--mixup-epoch-tail",
        type=int,
        default=0,
        help="mixup epoch tail (default: 0)")  
    parser.add_argument(
        "--cutmix",
        type=float,
        default=0.0,
        help="cutmix method cutmix enabled if > 0.0 (default: 0.0)")
    parser.add_argument(
        "--cutmix-epoch-tail",
        type=int,
        default=0,
        help="cutmix epoch tail (default: 0)")
    parser.add_argument(
        "--auto-augment",
        type=str,
        default="",
        help="auto augmentation method (default: '')")
    parser.add_argument(
        "--auto-augment-epoch-tail",
        type=int,
        default=0,
        help="auto augmentation epoch tail (default: 0)")
    parser.add_argument(
        "--fast-auto-augment",
        type=str,
        default="",
        help="fast auto augmentation method (default: '')")
    parser.add_argument(
        "--fast-auto-augment-epoch-tail",
        type=int,
        default=0,
        help="fast auto augmentation epoch tail (default: 0)")
    parser.add_argument(
        "CTAugment",
        type=str,
        default="",
        help="CTAugment method (default: '')")
    parser.add_argument(
        "--CTAugment-epoch-tail",
        type=int,
        default=0,
        help="CTAugment epoch tail (default: 0)")
    parser.add_argument(
        "--Augment Anchoring",
        type=str,
        default="",
        help="Augment Anchoring method (default: '')")
    parser.add_argument(
        "pseudo-label",
        type=str,
        default="",
        help="pseudo-label method (default: '')")
    parser.add_argument(
        "--guess-label",
        type=str,
        default="",
        help="guess-label method (default: '')")
    #interval
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="logging interval (default: 50)")
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1,
        help="model saving interval (default: 1)")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="",
        help="model and log saving directory (default: '')")
    parser.add_argument(
        "--logging-file-name",
        type=str,
        default="",
        help="logging file name (default: '')")
    #seed
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="random seed (default: -1)")
    parser.add_argument(
        "--log-packages",
        type=str,
        default="torch, torchvision",
        help="list of python packages for logging (default: 'torch, torchvision')")
    parser.add_argument(
        "--log-pip-packages",
        type=str,
        default="",
        help="list of pip packages for logging (default: '')")
    
def parse_args():
    """
    Parse python script parameters (common part).

    Returns:
    -------
    ArgumentParser
        Resulted args.
    """ 
    parser = argparse.ArgumentParser(
        description="Train a model for image classification (use PyTorch)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="dataset name (default: cifar10), options= ['cifar10', 'cifar100', 'imagenet', 'SVHN']")
    parser.add_argument(
        "--work-dir",
        type=str,
        default=os.path.join("..", "image_clf_data"),
        help="path to working directory (default: ../image_clf_data)")
    
    args, _ = parser.parse_known_args()
    dataset_metainfo = get_dataset_metainfo(args.dataset, args.work_dir)
    dataset_metainfo.add_dataset_parser_arguments(
        parser=parser,
        work_dir=args.work_dir)
    
    add_trtain_parser_arguments(parser)

    args = parser.parse_args()
    return args

def parse_args():
    """
    parse python script parameters (common part).

    Returns:
    -------
    ArgumentParser
        Resulted args.
    """
    parser = argparse.ArgumentParser(
        description="Train a model for image classification (use PyTorch)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="dataset name (default: cifar10), options= ['cifar10', 'cifar100', 'imagenet', 'SVHN']")
    parser.add_argument(
        "--work-dir",
        type=str,
        default=os.path.join("..", "image_clf_data"),
        help="path to working directory (default: ../image_clf_data)")

    args, _ = parser.parse_known_args()
    dataset_metainfo = get_dataset_metainfo(args.dataset, args.work_dir)
    dataset_metainfo.add_dataset_parser_arguments(
        parser=parser,
        work_dir=args.work_dir)

    args = parser.parse_args()
    return args

def init_rand(seed):
    """
    Initialize random state.

    Parameters:
    ----------
    seed : int
        Random seed.

    Returns:
    -------
    int
        Used random seed.
    """
    if seed <= 0:
        seed = np.random.randint(10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    return seed

def prepare_train(net,
                  optimizer,
                  wd,
                  momentum,
                  lr_mode,
                  lr,
                  lr_decay_period,
                  lr_decay_epoch,
                  lr_decay,
                  num_epochs,
                  state_file_path,):
    """
    Prepare model for training.

    Parameters:
    ----------
    net : Module
        Model.
    optimizer : Optimizer
        Optimizer.
    wd : float
        Weight decay.
    momentum : float
        Momentum.
    lr_mode : str
        Learning rate mode.
    lr : float
        Learning rate.
    lr_decay_period : int
        Learning rate decay period.
    lr_decay_epoch : int
        Learning rate decay epoch.
    lr_decay : float
        Learning rate decay.
    num_epochs : int
        Number of epochs.
    state_file_path : str
        Path to file to store model state.

    Returns:
    -------
    Module
        Model.
    Optimizer
        Optimizer.
    LRScheduler
        Learning rate scheduler.
    int
        Start epoch.
    """
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "sgd":
        optimizer_fun = optim.SGD(
        params=net.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=wd)
    else:
        raise Exception("Unsupported optimizer: {}".format(optimizer_name))
    
    if state_file_path:
        net, optimizer_fun, start_epoch = torch.load(
            net=net,
            optimizer=optimizer_fun,
            state_file_path=state_file_path)
    else:
        start_epoch = 0
    
    cudnn.benchmark = True

    lr_mode = lr_mode.lower()

    if lr_mode == "step":
        lr_scheduler = lr_scheduler.StepLR(
            optimizer=optimizer_fun,
            step_size=lr_decay_period,
            gamma=lr_decay,
            last_epoch=start_epoch-1)
    elif lr_mode == "multistep":
        lr_decay_epochs = [int(i) for i in lr_decay_epoch.split(',')]
        lr_scheduler = lr_scheduler.MultiStepLR(
            optimizer=optimizer_fun,
            milestones=lr_decay_epochs,
            gamma=lr_decay,
            last_epoch=start_epoch-1)
    elif lr_mode == "cosine":
        lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer_fun,
            T_max=num_epochs,
            eta_min=0,
            last_epoch=start_epoch-1)
    elif lr_mode == "plateau":
        lr_scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer_fun,
            mode='min',
            factor=lr_decay,
            patience=lr_decay_period,
            verbose=True,
            threshold=1e-4,
            threshold_mode='rel',
            cooldown=0,
            min_lr=0,
            eps=1e-8)
    else:
        raise Exception("Unsupported lr mode: {}".format(lr_mode))
    
    return net, optimizer_fun, lr_scheduler, start_epoch

def save_params(file_path,
                net,
                optimizer,
                lr_scheduler,
                epoch,
                best_err,
                best_epoch,
                best_params_file_path):
    """
    Save model parameters.

    Parameters:
    ----------
    file_path : str
        Path to file to store model parameters.
    net : Module
        Model.
    optimizer : Optimizer
        Optimizer.
    lr_scheduler : LRScheduler
        Learning rate scheduler.
    epoch : int
        Epoch.
    best_err : float
        Best error.
    best_epoch : int
        Best epoch.
    best_params_file_path : str
        Path to file to store best model parameters.
    """
    if isinstance(net, torch.nn.DataParallel):
        net = net.module
    torch.save(net.state_dict(), file_path)
    if best_params_file_path is not None:
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        torch.save(net.state_dict(), best_params_file_path)
    state = {
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": epoch,
        "best_err": best_err,
        "best_epoch": best_epoch}
    torch.save(state, file_path)

def train_epoch(num_epochs,
                net,
                epoch,
                train_data,
                use_cuda,
                valid_data,
                batch_size,
                loss_fun,
                optimizer,
                lr_scheduler,
                log_interval):
    """
    Epoch training.

    Parameters:
    ----------
    num_epochs : int
        Number of epochs.
    net : Module
        Model.
    epoch : int
        Epoch.
    train_data : DataLoader
        Data loader with training data.
    use_cuda : bool
        Whether to use CUDA.
    valid_data : DataLoader
        Data loader with validation data.
    batch_size : int
        Batch size.
    loss_fun : Loss
        Loss function.
    optimizer : Optimizer
        Optimizer.
    lr_scheduler : LRScheduler
        Learning rate scheduler.
    log_interval : int
        Logging interval.
    """
    time_start = time.time()
    net.train()
    train_loss = 0.0
    train_acc_top1 = 0.0
    train_acc_top5 = 0.0
    valid_acc_top1 = 0.0
    valid_acc_top5 = 0.0
    num_samples = 0
    num_batches = 0
    for batch_idx, (data, target) in enumerate(train_data):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = net(data)
        loss = loss_fun(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        acc_top1, acc_top5 = accuracy(output, target, topk=(1, 5))
        train_acc_top1 += acc_top1.item()
        train_acc_top5 += acc_top5.item()

        if log_interval and not (imaplib + 1) % log_interval:
            speed = batch_size * log_interval / (time.time() - time_start)
            logging.info("Epoch[{}/{}] Batch[{}/{}] Speed: {:.2f} samples/sec Loss: {:.4f} Top1: {:.2f}% Top5: {:.2f}%".format(
                epoch + 1, num_epochs, batch_idx + 1, len(train_data), speed, train_loss / (batch_idx + 1),
                train_acc_top1 / (batch_idx + 1), train_acc_top5 / (batch_idx + 1)))
            time_start = time.time()
        num_samples += data.shape[0]
        num_batches += 1

    train_loss /= num_batches
    train_acc_top1 /= num_batches
    train_acc_top5 /= num_batches

    if valid_data is not None:
        valid_acc_top1, valid_acc_top5 = validate(
            net=net,
            valid_data=valid_data,
            use_cuda=use_cuda,
            batch_size=batch_size)
        logging.info("Epoch[{}/{}] Train Loss: {:.4f} Top1: {:.2f}% Top5: {:.2f}% Valid Top1: {:.2f}% Top5: {:.2f}%".format(
            epoch + 1, num_epochs, train_loss, train_acc_top1, train_acc_top5, valid_acc_top1, valid_acc_top5))
    else:
        logging.info("Epoch[{}/{}] Train Loss: {:.4f} Top1: {:.2f}% Top5: {:.2f}%".format(
            epoch + 1, num_epochs, train_loss, train_acc_top1, train_acc_top5))
    
    if lr_scheduler is not None:
        lr_scheduler.step()

def validate(net,
             valid_data,
             use_cuda,
             batch_size):
    """
    Validation.

    Parameters:
    ----------
    net : Module
        Model.
    valid_data : DataLoader
        Data loader with validation data.
    use_cuda : bool
        Whether to use CUDA.
    batch_size : int
        Batch size.

    Returns:
    -------
    tuple of two floats
        Top1 and Top5 accuracies.
    """
    net.eval()
    valid_acc_top1 = 0.0
    valid_acc_top5 = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(valid_data):
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            output = net(data)
            acc_top1, acc_top5 = accuracy(output, target, topk=(1, 5))
            valid_acc_top1 += acc_top1.item()
            valid_acc_top5 += acc_top5.item()
            num_batches += 1
    valid_acc_top1 /= num_batches
    valid_acc_top5 /= num_batches
    return valid_acc_top1, valid_acc_top5

def train(num_epochs,
          net,
          train_data,
          valid_data,
          loss_fun,
          optimizer,
          lr_scheduler,
          init_lr,
          lr_decay_epoch,
          use_cuda,
          batch_size,
          log_interval,
          model_dir,
          model_prefix):
    """
    Train model.

    Parameters:
    ----------
    num_epochs : int
        Number of epochs.
    net : Module
        Model.
    train_data : DataLoader
        Data loader with training data.
    valid_data : DataLoader
        Data loader with validation data.
    loss_fun : Loss
        Loss function.
    optimizer : Optimizer
        Optimizer.
    lr_scheduler : LRScheduler
        Learning rate scheduler.
    init_lr : float
        Initial learning rate.
    lr_decay_epoch : int
        Epoch interval to decrease learning rate.
    use_cuda : bool
        Whether to use CUDA.
    batch_size : int
        Batch size.
    log_interval : int
        Logging interval.
    model_dir : str
        Directory with models.
    model_prefix : str
        Prefix of model names.
    """
    logging.info("Epoch[0] Initial lr: {}".format(optimizer.param_groups[0]["lr"]))
    for epoch in range(num_epochs):
        train_epoch(
            num_epochs=num_epochs,
            net=net,
            epoch=epoch,
            train_data=train_data,
            use_cuda=use_cuda,
            valid_data=valid_data,
            batch_size=batch_size,
            loss_fun=loss_fun,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            log_interval=log_interval)
        if (epoch + 1) % lr_decay_epoch == 0:
            lr = init_lr * (0.1 ** ((epoch + 1) // lr_decay_epoch))
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            logging.info("Epoch[{}] Set lr to {}".format(epoch + 1, lr))
        save_params(
            file_path=os.path.join(model_dir, "{}-Epoch-{}.pth".format(model_prefix, epoch + 1)),
            net=net,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epoch=epoch,
            best_err=None,
            best_epoch=None,
            best_params_file_path=None)
        
def main():
    """
    Main body of script.
    """
    args = parse_args()
    logging.info("args = %s", args)
    args.seed = init_rand(seed=args.seed)
    logging.info("args.seed = %d", args.seed)

    _, log_file_path = initialize_logging(path=args.log_dir, prefix="train")
    logging.info("log_file_path = %s", log_file_path)

    use_cuda = (args.gpu is not None)
    if use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        logging.info("CUDA_VISIBLE_DEVICES = [%s]", os.environ["CUDA_VISIBLE_DEVICES"])
        cudnn.benchmark = True
        cudnn.enabled = True
        logging.info("cudnn.enabled = %s", cudnn.enabled)
    
    net = prepare_model(
        net_name=args.net,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        use_cuda=use_cuda)
    logging.info("net = %s", net)
    if use_cuda:
        net = net.cuda()

    train_data, valid_data = get_data(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_prefetcher=True,
        use_cuda=use_cuda)
    logging.info("train_data = %s", train_data)
    logging.info("valid_data = %s", valid_data)

    loss_fun = nn.CrossEntropyLoss()
    logging.info("loss_fun = %s", loss_fun)

    optimizer = optim.SGD(
        params=net.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    logging.info("optimizer = %s", optimizer)

    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=args.lr_decay_epoch,
        gamma=args.lr_decay)
    logging.info("lr_scheduler = %s", lr_scheduler)

    train(
        num_epochs=args.num_epochs,
        net=net,
        train_data=train_data,
        valid_data=valid_data,
        loss_fun=loss_fun,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        init_lr=args.lr,
        lr_decay_epoch=args.lr_decay_epoch,
        use_cuda=use_cuda,
        batch_size=args.batch_size,
        log_interval=args.log_interval,
        model_dir=args.model_dir,
        model_prefix=args.model_prefix)
    
if __name__ == "__main__":
    main()