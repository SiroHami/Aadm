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
from datasets.dataset_utils import get_dataset_metainfo, get_test_data_source, get_val_data_source

from utils.utils import accuracy, get_data, get_metric_name, get_train_data_source, prepare_logger, initialize_logging,  prepare_pt_context, prepare_model, report_accuracy, validate

from train_log_param_saver import TrainLogParamSaver

from models import *


sys.path.append(os.path.join(os.path.dirname(__file__), "progress"))

def add_train_parser_arguments(parser):
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
        "--CTAugment",
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
        "--pseudo-label",
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
    parser.add_argument(
        "--log_dir",
        type=str,
        default="",
        help="log directory (default: '')")
    parser.add_argument(
        "--script-args",
        type=str,
        default="",
        help="script arguments (default: '')")
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
        description="Train a model for image classification (PyTorch)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--dataset",
        type=str,
        default="ImageNet1K",
        help="dataset name. options are ImageNet1K, CUB200_2011, CIFAR10, CIFAR100, SVHN")
    parser.add_argument(
        "--work-dir",
        type=str,
        default=os.path.join("..", "imgclsmob_data"),
        help="path to working directory only for dataset root path preset")

    args, _ = parser.parse_known_args()
    dataset_metainfo = get_dataset_metainfo(dataset_name=args.dataset)
    dataset_metainfo.add_dataset_parser_arguments(
        parser=parser,
        work_dir_path=args.work_dir)

    add_train_parser_arguments(parser)

    args = parser.parse_args()
    return args


def init_rand(seed):
    """
    Initialize all random generators by seed.

    Parameters:
    ----------
    seed : int
        Seed value.

    Returns:
    -------
    int
        Generated seed value.
    """
    if seed <= 0:
        seed = np.random.randint(10000)
    else:
        cudnn.deterministic = True
        logging.warning(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, which can slow down "
            "your training considerably! You may see unexpected behavior when restarting from checkpoints.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


def prepare_trainer(net,
                    optimizer_name,
                    wd,
                    momentum,
                    lr_mode,
                    lr,
                    lr_decay_period,
                    lr_decay_epoch,
                    lr_decay,
                    num_epochs,
                    state_file_path):
    """
    Prepare trainer.

    Parameters:
    ----------
    net : Module
        Model.
    optimizer_name : str
        Name of optimizer.
    wd : float
        Weight decay rate.
    momentum : float
        Momentum value.
    lr_mode : str
        Learning rate scheduler mode.
    lr : float
        Learning rate.
    lr_decay_period : int
        Interval for periodic learning rate decays.
    lr_decay_epoch : str
        Epoches at which learning rate decays.
    lr_decay : float
        Decay rate of learning rate.
    num_epochs : int
        Number of training epochs.
    state_file_path : str
        Path for file with trainer state.

    Returns:
    -------
    Optimizer
        Optimizer.
    LRScheduler
        Learning rate scheduler.
    int
        Start epoch.
    """
    optimizer_name = optimizer_name.lower()
    if (optimizer_name == "sgd") or (optimizer_name == "nag"):
        optimizer = torch.optim.SGD(
            params=net.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=wd,
            nesterov=(optimizer_name == "nag"))
    else:
        raise ValueError("Usupported optimizer: {}".format(optimizer_name))

    if state_file_path:
        checkpoint = torch.load(state_file_path)
        if type(checkpoint) == dict:
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint["epoch"]
        else:
            start_epoch = None
    else:
        start_epoch = None

    cudnn.benchmark = True

    lr_mode = lr_mode.lower()
    if lr_decay_period > 0:
        lr_decay_epoch = list(range(lr_decay_period, num_epochs, lr_decay_period))
    else:
        lr_decay_epoch = [int(i) for i in lr_decay_epoch.split(",")]
    if (lr_mode == "step") and (lr_decay_period != 0):
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=lr_decay_period,
            gamma=lr_decay,
            last_epoch=-1)
    elif (lr_mode == "multistep") or ((lr_mode == "step") and (lr_decay_period == 0)):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=lr_decay_epoch,
            gamma=lr_decay,
            last_epoch=-1)
    elif lr_mode == "cosine":
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=num_epochs,
            last_epoch=(num_epochs - 1))
    else:
        raise ValueError("Usupported lr_scheduler: {}".format(lr_mode))

    return optimizer, lr_scheduler, start_epoch


def save_params(file_stem,
                state):
    """
    Save current model/trainer parameters.

    Parameters:
    ----------
    file_stem : str
        File stem (with path).
    state : dict
        Whole state of model & trainer.
    trainer : Trainer
        Trainer.
    """
    torch.save(
        obj=state["state_dict"],
        f=(file_stem + ".pth"))
    torch.save(
        obj=state,
        f=(file_stem + ".states"))


def train_epoch(epoch,
                net,
                train_metric,
                train_data,
                use_cuda,
                L,
                optimizer,
                # lr_scheduler,
                batch_size,
                log_interval):
    """
    Train model on particular epoch.

    Parameters:
    ----------
    epoch : int
        Epoch number.
    net : Module
        Model.
    train_metric : EvalMetric
        Metric object instance.
    train_data : DataLoader
        Data loader.
    use_cuda : bool
        Whether to use CUDA.
    L : Loss
        Loss function.
    optimizer : Optimizer
        Optimizer.
    batch_size : int
        Training batch size.
    log_interval : int
        Batch count period for logging.

    Returns:
    -------
    float
        Loss value.
    """
    tic = time.time()
    net.train()
    train_metric.reset()
    train_loss = 0.0

    btic = time.time()
    for i, (data, target) in enumerate(train_data):
        if use_cuda:
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        output = net(data)
        loss = L(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        train_metric.update(
            labels=target,
            preds=output)

        if log_interval and not (i + 1) % log_interval:
            speed = batch_size * log_interval / (time.time() - btic)
            btic = time.time()
            train_accuracy_msg = report_accuracy(metric=train_metric)
            logging.info("Epoch[{}] Batch [{}]\tSpeed: {:.2f} samples/sec\t{}\tlr={:.5f}".format(
                epoch + 1, i, speed, train_accuracy_msg, optimizer.param_groups[0]["lr"]))

    throughput = int(batch_size * (i + 1) / (time.time() - tic))
    logging.info("[Epoch {}] speed: {:.2f} samples/sec\ttime cost: {:.2f} sec".format(
        epoch + 1, throughput, time.time() - tic))

    train_loss /= (i + 1)
    train_accuracy_msg = report_accuracy(metric=train_metric)
    logging.info("[Epoch {}] training: {}\tloss={:.4f}".format(
        epoch + 1, train_accuracy_msg, train_loss))

    return train_loss


def train_net(batch_size,
              num_epochs,
              start_epoch1,
              train_data,
              val_data,
              net,
              optimizer,
              lr_scheduler,
              lp_saver,
              log_interval,
              num_classes,
              val_metric,
              train_metric,
              use_cuda):
    """
    Main procedure for training model.

    Parameters:
    ----------
    batch_size : int
        Training batch size.
    num_epochs : int
        Number of training epochs.
    start_epoch1 : int
        Number of starting epoch (1-based).
    train_data : DataLoader
        Data loader (training subset).
    val_data : DataLoader
        Data loader (validation subset).
    net : Module
        Model.
    optimizer : Optimizer
        Optimizer.
    lr_scheduler : LRScheduler
        Learning rate scheduler.
    lp_saver : TrainLogParamSaver
        Model/trainer state saver.
    log_interval : int
        Batch count period for logging.
    num_classes : int
        Number of model classes.
    val_metric : EvalMetric
        Metric object instance (validation subset).
    train_metric : EvalMetric
        Metric object instance (training subset).
    use_cuda : bool
        Whether to use CUDA.
    """
    assert (num_classes > 0)

    L = nn.CrossEntropyLoss()
    if use_cuda:
        L = L.cuda()

    assert (type(start_epoch1) == int)
    assert (start_epoch1 >= 1)
    if start_epoch1 > 1:
        logging.info("Start training from [Epoch {}]".format(start_epoch1))
        validate(
            metric=val_metric,
            net=net,
            val_data=val_data,
            use_cuda=use_cuda)
        val_accuracy_msg = report_accuracy(metric=val_metric)
        logging.info("[Epoch {}] validation: {}".format(start_epoch1 - 1, val_accuracy_msg))

    gtic = time.time()
    for epoch in range(start_epoch1 - 1, num_epochs):
        lr_scheduler.step()

        train_loss = train_epoch(
            epoch=epoch,
            net=net,
            train_metric=train_metric,
            train_data=train_data,
            use_cuda=use_cuda,
            L=L,
            optimizer=optimizer,
            # lr_scheduler,
            batch_size=batch_size,
            log_interval=log_interval)

        validate(
            metric=val_metric,
            net=net,
            val_data=val_data,
            use_cuda=use_cuda)
        val_accuracy_msg = report_accuracy(metric=val_metric)
        logging.info("[Epoch {}] validation: {}".format(epoch + 1, val_accuracy_msg))

        if lp_saver is not None:
            state = {
                "epoch": epoch + 1,
                "state_dict": net.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            lp_saver_kwargs = {"state": state}
            val_acc_values = val_metric.get()[1]
            train_acc_values = train_metric.get()[1]
            val_acc_values = val_acc_values if type(val_acc_values) == list else [val_acc_values]
            train_acc_values = train_acc_values if type(train_acc_values) == list else [train_acc_values]
            lp_saver.epoch_test_end_callback(
                epoch1=(epoch + 1),
                params=(val_acc_values + train_acc_values + [train_loss, optimizer.param_groups[0]["lr"]]),
                **lp_saver_kwargs)

    logging.info("Total time cost: {:.2f} sec".format(time.time() - gtic))
    if lp_saver is not None:
        opt_metric_name = get_metric_name(val_metric, lp_saver.acc_ind)
        logging.info("Best {}: {:.4f} at {} epoch".format(
            opt_metric_name, lp_saver.best_eval_metric_value, lp_saver.best_eval_metric_epoch))


def main():
    """
    Main body of script.
    """
    args = parse_args()
    args.seed = init_rand(seed=args.seed)

    _, log_file_exist = initialize_logging(
        logging_dir_path=args.save_dir,
        logging_file_name=args.logging_file_name,
        script_args=args,
        log_packages=args.log_packages,
        log_pip_packages=args.log_pip_packages)

    use_cuda, batch_size = prepare_pt_context(
        num_gpus=args.num_gpus,
        batch_size=args.batch_size)

    net = prepare_model(
        model_name=args.model,
        use_pretrained=args.use_pretrained,
        pretrained_model_file_path=args.resume.strip(),
        use_cuda=use_cuda)
    real_net = net.module if hasattr(net, "module") else net
    assert (hasattr(real_net, "num_classes"))
    num_classes = real_net.num_classes

    ds_metainfo = get_dataset_metainfo(dataset_name=args.dataset)
    ds_metainfo.update(args=args)

    train_data = get_train_data_source(
        ds_metainfo=ds_metainfo,
        batch_size=batch_size,
        num_workers=args.num_workers)
    val_data = get_val_data_source(
        ds_metainfo=ds_metainfo,
        batch_size=batch_size,
        num_workers=args.num_workers)

    optimizer, lr_scheduler, start_epoch = prepare_trainer(
        net=net,
        optimizer_name=args.optimizer_name,
        wd=args.wd,
        momentum=args.momentum,
        lr_mode=args.lr_mode,
        lr=args.lr,
        lr_decay_period=args.lr_decay_period,
        lr_decay_epoch=args.lr_decay_epoch,
        lr_decay=args.lr_decay,
        num_epochs=args.num_epochs,
        state_file_path=args.resume_state)

    if args.save_dir and args.save_interval:
        param_names = ds_metainfo.val_metric_capts + ds_metainfo.train_metric_capts + ["Train.Loss", "LR"]
        lp_saver = TrainLogParamSaver(
            checkpoint_file_name_prefix="{}_{}".format(ds_metainfo.short_label, args.model),
            last_checkpoint_file_name_suffix="last",
            best_checkpoint_file_name_suffix=None,
            last_checkpoint_dir_path=args.save_dir,
            best_checkpoint_dir_path=None,
            last_checkpoint_file_count=2,
            best_checkpoint_file_count=2,
            checkpoint_file_save_callback=save_params,
            checkpoint_file_exts=(".pth", ".states"),
            save_interval=args.save_interval,
            num_epochs=args.num_epochs,
            param_names=param_names,
            acc_ind=ds_metainfo.saver_acc_ind,
            # bigger=[True],
            # mask=None,
            score_log_file_path=os.path.join(args.save_dir, "score.log"),
            score_log_attempt_value=args.attempt,
            best_map_log_file_path=os.path.join(args.save_dir, "best_map.log"))
    else:
        lp_saver = None

    train_net(
        batch_size=batch_size,
        num_epochs=args.num_epochs,
        start_epoch1=args.start_epoch,
        train_data=train_data,
        val_data=val_data,
        net=net,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        lp_saver=lp_saver,
        log_interval=args.log_interval,
        num_classes=num_classes,
        val_metric=get_composite_metric(ds_metainfo.val_metric_names, ds_metainfo.val_metric_extra_kwargs),
        train_metric=get_composite_metric(ds_metainfo.train_metric_names, ds_metainfo.train_metric_extra_kwargs),
        use_cuda=use_cuda)


if __name__ == "__main__":
    main()