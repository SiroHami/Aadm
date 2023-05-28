
import argparse
import os
import shutil
import time
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

import models.ResNet as model
import datasets.cifar10_dataset as dataset
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig, fix_seed

parser = argparse.ArgumentParser(description='PyTorch MixMatch')
#add options
parser.add_argument('--epochs', default=1024, tpye=int, metavar='N',
                    help='number of epochs')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=64, type=str, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', default=0.002, type=float, metavar='LR',
                    help='initial learning rate')
#Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
#Miscs
parser.add_argument('--manualSeed', default=0, type=int, 
                    help='menaul seed')
#Device
parser.add_argument('--gpu', default='0', type=str,
                    help=f'id(s) for gpu')
#Method Option
parser.add_argument('--n-labeled', type=int, default=250,
                    help='Number of labeled data')
parser.add_argument('--train-iteration', type=int, default=1024,
                    help='Number of iteration per each')
parser.add_argument('--out', default='result',
                    help='Directory to ouput result')
parser.add_argument('--alpha', default=0.75, type=float,
                    help='')
parser.add_argument('--lambda-u', default=75, type=float,
                    help='')
parser.add_argument('--T', default=0.5, type=float,
                    help='')
parser.add_argument('--eam-decay', default=0.999, type=float)

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

#use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

#fix seed
if args.manualSeed is None:
    args.manualSeed =  fix_seed(random.randint(1, 100000))
fix_seed(args.manualSeed)

#best test accuracy
best_acc = 0

def main():
    global best_acc

    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    #Load & Transform Dataset
    print(f'==> Load Dataset')
    tansform_train=transforms.Compose([
        dataset.ToTensor()
    ])

    transform_val = transforms.Compose([
        dataset.ToTensor()
    ])

    train_labeled_set, train_unlabeled_set, val_set, test_set = dataset.get_cifart10('./data', args.n_labeled, transform_labeled_train=transform_labeled_train, 
                                                                                     transform_unlabeled_train=transform_unlabeled_train, transform_val=transform_val)
    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_works=0)

    print('==> creating ResNet')

    def create_model(ema=False):
        model = model.ResNet(num_classes=10)
        model = model.cuda()
        
        if ema:
            for param in model.parameters():
                param.detach_()

        return model
    
    model = create_model()
    ema_model = create_model(ema=True)

    cudnn.benchmark = True
    print('Total params: %.fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    train_criterion = SemiLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    ema_optimizer= WeightEMA(model, eam_model, alptha=args.ema_decay)
    start_epoch = 0

    #Resume
    title = 'noisy-cifar-10'
    if args.resume:
        #Load Checkpoint
        print('==> Resuming from chekpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found'
        args.out = os.path(args.dirname)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
        logger.set_names(['Train Loss', 'Train Loss X', 'Train Loss U', 'Valid Loss', 'Valid Acc', 'Test Loss', 'Test Acc.'])

    writer = SummaryWriter(args.out)
    step = 0
    test_acc = []

    #train and validate
    for epoch in range(start_epoch, args.epochs):

        print('\nEpoch: [%d | %d] LR: %f' % (epoch+1, state['lr']))

        train_loss, train_loss_x, train_loss_u = train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, train_criterion, epoch, use_cuda)
        _, train_acc = validate(labeled_trainloader, ema_model, criterion, use_cuda, mode='Train Stats')
        val_loss, val_acc = validate(val_loader, ema_model, criterion, epoch, use_cuda, mode='Valid Stats')
        test_loss, test_acc = validate(test_loader, ema_model, criterion, epoch, use_cuda, mode='Test Stats')

        step = args.train_iteration * (epoch +1)

        writer.add_scalar('losses/train_loss', train_loss, step)
        writer.add_scalar('losses/valid_loss', val_loss, step)
        writer.add_scalar('losses/test_loss', test_loss, step)

        writer.add_scalar('accuracy/train_acc', train_acc, step)
        writer.add_scalar('accuracy/val_acc', val_acc, step)
        writer.add_scalar('accuracy/test_acc', test_acc, step) 

        # append logger file
        logger.append([train_loss, train_loss_x, train_loss_u, val_loss, val_acc, test_loss, test_acc])  

        # save model
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'acc': val_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best)
        test_accs.append(test_acc)
    logger.close()
    writer.close()

    print('Best acc:')
    print(best_acc)

    print('Mean acc:')
    print(np.mean(test_accs[-20:]))
   


def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, cirterion, epoch, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    looses_u = AverageMeter()
    ws = AverageMeter()
    ema = time.time()

    bar = Bar('Training', max=args.train_iteration)
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    model.train()
    for batch_idx in range(args.train_iteration):
        try:
            inputs_x, targets_x = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            input_x, target_x = labeled_train_iter.next()

        try:
            (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2),_ = unlabeled_train_iter.next()

        # measure data loading time
        data_time.update(time.time() - end)

        batch_size = inputs_x.size(0)

        #transform label to one-hot
        target_x = torch.zeros(batch_size, 10).scatter_(1, targets_x.view(-1,1).long(), 1)

        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u = inputs_u.cuda()
            inputs_u2 = inputs_u2.cuda()

        with torch.no_grad():
            # compute guessed labels of unlabel samples
            outputs_u = model(inputs_u)
            outputs_u2 = model(inputs_u2)
            p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
            pt = p**(1/args.T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()