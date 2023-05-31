"""
Run Pytorch model
"""

import logging
import os
import argparse
import time
import numpy as np
import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data

from datasets.dataset_utils import get_dataset_info, get_train_data_info, get_val_data_info
from logger import initialize_logging
from metrics import get_metric
from train_log_param_saver import TrainLogParamSaver
from utils import prepare_pt_context, report_accuracy, validate


def add_train_cls_parser_agument(parser):
    """
    Create python argparse parser for training

    parameters:
    ----------
    parser : argparse.ArgumentParser
        input parser
    """
    #baisc setting
    parser.add_argument('--model', type=str, required=True,
                        help='model name, see model_provider for available options')
    parser.add_argument('--label', type=str or int, default='full', choices = [0, 250, 500, 1000, 2000, 4000, 'full'],
                        help='label dataset name, see dataset_info for available options')
    parser.add_argument('--dataset', type=str, required=True,
                        help='dataset name, see dataset_info for available options')
    parser.add_argument('--resume', type=str, default=None,
                        help='path to checkpoint to resume training')
    parser.add_argument('--resume-idr', type=str, default='',
                        help='path to checkpoint to resume training')
    #gpu setting
    parser.add_argument('--num-gpus', type=int, default=0,
                        help='number of GPUs to use by default')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of data loading workers')
    #training setting
    parser.add_argument('--batch-size', type=int, default=64,
                        help='training batch size per device (CPU/GPU)')
    parser.add_argument('--num-epochs', type=int, default=200,
                        help='number of training epochs')
    #optimizer setting
    parser.add_argument('--optimizer-name', type=str, default='sgd',
                        help='optimizer name')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr-decay', type=float, default=0.2,
                        help='decay rate of learning rate')
    parser.add_argument('--lr-decay-epoch', type=int, default=0,
                        help='epoch at which learning rate decays')
    parser.add_argument('--decay-epoch', type=str, default='30,60',
                        help='epoch at which learning rate decays')
    parser.add_argument('--target-lr', type=float, default=1e-8,
                        help='end learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum value for optimizer')
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='weight decay rate')
    #augmentation setting
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='label smoothing')
    parser.add_agumnet('--mixup', type=float, default=0.0,
                        help='mixup alpha, mixup enabled if > 0.0')
    parser.add_argument('--mixup-off-epoch', type=int, default=0,
                        help='how many last epochs to train without mixup, if mixup enabled')
    parser.add_argument('--mixup-alpha', type=float, default=1.0,
                        help='mixup alpha, mixup enabled if > 0.0')
    #log setting
    parser.add_argument('--log-interval', type=int, default=50,
                        help='number of batches to wait before logging')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='saving parameters epoch interval, best model will always be saved')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                        help='directory name to save')
    parser.add_argumnet('--save-name', type=str, default='train.log',
                        help='file name to save')
    parser.add_argument('--log-packages', type=str, default='torch, torchvision',
                        help='list of python packages for logging')
    parser.add_argument('--log-pip-packages', type=str, default='',
                        help='lost of pip packages for logging')
    #seed setting
    parser.add_argument('--seed', type=int, default=17,
                        help='random seed to use. Default=17')
    #other setting
    parser.add_argument('--tune-layer', type=str, default='',
                        help='select layers for fine tuning')
    
def parse_args():
    """
    Parse python argparse arguments

    Returns:
    ----------
    args : argparse.Namespace
        input arguments
    """
    parser = argparse.ArgumentParser(description='DeepLearning Models with Pytorch')
    parser.add_argument('--dataset', type=str, required=True,
                        help='dataset name, see dataset_info for available options')
    parser.add_argument('--work-dir', type=str, default=os.path.join("..", "data"),
                        help='the dir to working dir for dataset root path')
    args, _ = parser.parse_known_args()
    dataset_info = get_dataset_info(args.dataset)
    dataset_info.add_dataset_parser_agument(parser,
                                            working_dir=args.work_dir)
    add_train_cls_parser_agument(parser)
    args = parser.parse_args()
    return args

def set_seed(seed):
    """
    Set random seed

    Parameters:
    ----------
    seed : int
        random seed
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
    torch.cuda.manual_seed_all(seed)
    return seed

def load_model(net,
                 optimizer_name,
                 wd,
                 momentum,
                 lr_mode,
                 lr,
                 lr_decay,
                 lr_decay_epoch,
                 lr_decay_period,
                 num_epochs,
                 file_path):
    """
    Load trainer for training

    Parameters:
    ----------
    net : Module
        network
    optimizer_name : str
        optimizer name
    wd : float
        weight decay rate
    momentum : float
        momentum value for optimizer
    lr_mode : str
        learning rate scheduler mode
    lr : float
        learning rate
    lr_decay : float
        decay rate of learning rate
    lr_decay_epoch : int
        epoch at which learning rate decays
    lr_decay_period : int
        epoch period at which learning rate decays
    num_epochs : int
        number of training epochs
    file_path : str
        checkpoint file path
    """
    optimizer_name = optimizer_name.lower()
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=lr,
                                    momentum=momentum,
                                    weight_decay=wd,
                                    nesterov=(optimizer_name == 'sgd'))
    else:
        raise ValueError("Unsupported optimizer: {}".format(optimizer_name))
    
    if file_path:
        checkpoint = torch.load(file_path)
        if type(checkpoint) == dict:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
        else:
            start_epoch = None
    else:
        start_epoch = None
    
    cudnn.benchmark = True

    lr_mode = lr_mode.lower()
    if lr_decay_epoch > 0:
        lr_decay_epoch = list(range(lr_decay_epoch, num_epochs, lr_decay_period))
    else:
        lr_decay_epoch = [int(i) for i in lr_decay_epoch.split(',')]
    if (lr_mode == "step") and (lr_decay_period !=0):
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=lr_decay_period,
            gamma=lr_decay,
            last_epoch=start_epoch-1)
    elif lr_mode == "multistep" or ((lr_mode == "step") and (lr_decay_period == 0)):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=lr_decay_epoch,
            gamma=lr_decay,
            last_epoch=start_epoch-1)
    elif lr_mode == "cosine":
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=num_epochs,
            eta_min=lr_decay,
            last_epoch=start_epoch-1)
    elif lr_mode == "poly":
        lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer=optimizer,
            max_iter=num_epochs,
            power=lr_decay,
            last_epoch=start_epoch-1)
    else:
        raise ValueError("Unsupported lr mode: {}".format(lr_mode))
    
def calculation_net_weight(net):
    """
    Calculation network weight

    Parameters:
    ----------
    net : Module
        network
    """
    net.train()
    num_params = filter(lambda p: p.requires_grad, net.parameters())
    weight = 0
    for param in num_params:
        weight += param.numel()
    return weight

def save_params(file_stem, state):
    """
    Save parameters to file

    Parameters:
    ----------
    file_stem : str
        file stem
    state : dict
        state dictionary
    """
    file_path = file_stem + '.params'
    logging.info("Saving parameters to {}".format(file_path))
    torch.save(state, file_path)

def train_epoch(epoch,
                net,
                train_metric,
                train_data,
                use_cuda,
                criterion,
                optimizer,
                lr_scheduler,
                batch_size,
                log_interval):
    """
    Train model in one epoch

    Parameters:
    ----------
    epoch : int
        current training epoch
    net : Module
        network
    train_metric : EvalMetric
        training metric
    train_data : DataLoader
        training data loader
    use_cuda : bool
        whether to use CUDA
    criterion : Module
        loss function
    optimizer : Optimizer
        optimizer
    lr_scheduler : LRScheduler
        learning rate scheduler
    batch_size : int
        batch size
    log_interval : int
        logging interval
    """
    tic = time.time()
    train_metric.reset()
    train_loss = 0.0
    net.train()

    for i, (data, label) in enumerate(train_data):
        if use_cuda:
            data = data.cuda()
            label = label.cuda()
        output = net(data)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        train_metric.update(label, output)

        if (i + 1) % log_interval == 0:
            name1, loss1 = train_metric.get()
            logging.info('Epoch[{}] Batch [{}]\tSpeed: {:.2f} samples/sec\t{}={:.6f}\tloss={:.6f}'.format(
                epoch, i, batch_size / (time.time() - tic), name1, loss1, train_loss / (i + 1)))
    name1, loss1 = train_metric.get()
    logging.info('[Epoch {}] training: {}={:.6f}\tloss={:.6f}'.format(epoch, name1, loss1, train_loss / (i + 1)))
    if lr_scheduler is not None:
        lr_scheduler.step()
    
    return train_loss / (i + 1)

def train_net(batch_size,
              num_epochs,
              start_epoch,
              train_data,
              val_data,
              net,
              optimizer,
              lr_scheduler,
              lp_saver,
              log_interval,
              train_metric,
              val_metric,
              use_cuda):
    """
    Train model

    Parameters:
    ---------- 
    batch_size : int
        batch size
    num_epochs : int
        number of training epochs
    start_epoch : int
        start epoch
    train_data : DataLoader
        training data loader
    val_data : DataLoader
        validation data loader
    net : Module
        network
    optimizer : Optimizer
        optimizer
    lr_scheduler : LRScheduler
        learning rate scheduler
    lp_saver : LRScheduler
        learning rate scheduler
    log_interval : int
        logging interval
    train_metric : EvalMetric
        training metric
    val_metric : EvalMetric
        validation metric
    use_cuda : bool
        whether to use CUDA
    """
    criterion = nn.CrossEntropyLoss()
    if use_cuda:
        criterion = criterion.cuda()

    if start_epoch > 1:
        logging.info("Start training from [Epoch {}]".format(start_epoch))
        validate(
            metric = val_metric,
            net = net,
            val_data = val_data,
            use_cuda = use_cuda)
        val_acc_message = report_accuracy(val_metric)
        logging.info('[Epoch {}] validation: {}'.format(start_epoch - 1, val_acc_message))

    tic = time.time()
    for epoch in range(start_epoch - 1, num_epochs):
        lr_scheduler.step()

        train_loss = train_epoch(
            epoch = epoch,
            net = net,
            train_metric = train_metric,
            train_data = train_data,
            use_cuda = use_cuda,
            criterion = criterion,
            optimizer = optimizer,
            lr_scheduler = lr_scheduler,
            batch_size = batch_size,
            log_interval = log_interval)
        
        validate(
            metric = val_metric,
            net = net,
            val_data = val_data,
            use_cuda = use_cuda)
        val_acc_message = report_accuracy(val_metric)
        logging.info('[Epoch {}] validation: {}'.format(epoch, val_acc_message))

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

    logging.info("Total time cost: {:.2f} sec".format(time.time() - tic))
    if lp_saver is not None:
        opt_metric_name = get_metric(val_metric, lp_saver.acc_ind)
        logging.info("Best {}: {:.4f} at {} epoch".format(
            opt_metric_name, lp_saver.best_eval_metric_value, lp_saver.best_eval_metric_epoch))


def main():
    """
    main script
    """
    args = parse_args()
    args.seed = set_seed(seed=args.seed, verbose=True)

    _, log_file_exist = initialize_logging(
        logging_dir_path=args.save_dir,
        logging_file_name=args.logging_file_name,
        script_args=args,
        log_packages=args.log_packages,
        log_pip_packages=args.log_pip_packages)

    use_cuda, batch_size = prepare_pt_context(
        num_gpus=args.num_gpus,
        batch_size=args.batch_size)
    
    net = load_model(
        model_name=args.model,
        use_pretrained=args.use_pretrained,
        pretrained_model_file_path=args.pretrained_model_file_path,
        use_cuda=use_cuda)
    real_net = net.module if hasattr(net, "module") else net
    num_classes = real_net.num_classes

    dataset_info = get_dataset_info(dataset_name=args.dataset_name)
    dataset_info.updatae(args=args)

    train_data = get_train_data_info(
        dataset_name=dataset_info,
        batch_size=batch_size,
        num_workers=args.num_workers)
    val_data = get_val_data_info(
        dataset_name=dataset_info,
        batch_size=batch_size,
        num_workers=args.num_workers)
    
    optimizer, lr_scheduler, start_epoch = load_model(
        net=net,
        optimizer_name=args.optimizer_name,
        momentum=args.momentum,
        lr_mode=args.lr_mode,
        lr=args.lr,
        wd=args.wd,
        lr_decay=args.lr_decay,
        lr_decay_period=args.lr_decay_period,
        lr_decay_epoch=args.lr_decay_epoch,
        num_epochs=args.num_epochs,
        file_path=args.resume_state)
    
    if args.save_dir and args.save_interval:
        param_names = dataset_info.val_metric + dataset_info.train_metric + ["train_loss", "lr"]
        lp_saver = TrainLogParamSaver(
            checkpoint_file_name_prefix="{}_{}".format(dataset_info.short_label, args.model),
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
            acc_ind=dataset_info.saver_acc_ind,
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
        start_epoch=args.start_epoch,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        lp_saver=lp_saver,
        log_interval=args.log_interval,
        net=net,
        train_data=train_data,
        val_data=val_data,
        num_classes=num_classes,
        train_metric=get_metric(dataset_info.train_metric, dataset_info.train_metric_extra),
        val_metric=get_metric(dataset_info.val_metric, dataset_info.val_metric_extra),
        use_cuda=use_cuda)
    
if __name__ == "__main__":
    main()

