# This is the code to train our EML net for salency detection.
# The backbone used is ResNet50.
#
# Author: Sen Jia
# Date: 09 / Mar / 2020
#
import argparse
import os
import pathlib as pl

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import SaliconLoader
import EMLLoss
import resnet

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data_folder', type=pl.Path,default=r'D:\data\animals_catalogy\salicon',
                    help='the folder of salicon data')
parser.add_argument('output_folder', type=str,default=r'D:\data\animals_catalogy\salicon\out',
                    help='the folder used to save the trained model')
parser.add_argument('--model_path', default=r'C:\Users\28235\.cache\torch\hub\checkpoints\resnet50-0676ba61.pth', type=pl.Path,
                    help='the path of the pre-trained model')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--decay_epoch', default=5, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--val_epoch', default=5, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--val_thread', default=5, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=500, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--mse', action="store_true",
                    help='apply MSE as a loss function')
parser.add_argument('--gpu', default='0', type=str,
                    help='The index of the gpu you want to use')
argv = [r'D:\data\animals_catalogy\salicon',r'D:\data\animals_catalogy\salicon\out',]
args = parser.parse_args(argv)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

def main():
    global args
    print(args.data_folder)
    model = resnet.resnet50(args.model_path)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # Data loading code
    train_loader = torch.utils.data.DataLoader(
        SaliconLoader.ImageList(args.data_folder, transforms.Compose([
            transforms.ToTensor(),
        ]),
        train=True,
        ),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    if args.mse:
        args.output_folder = args.output_folder + "_mse"
        criterion = nn.MSELoss()
    else:
        args.lr *= 0.1
        args.output_folder = args.output_folder + "_eml"
        criterion = EMLLoss.Loss()

    args.output_folder = pl.Path(args.output_folder)
    print(args.output_folder)
    if not args.output_folder.is_dir():
        args.output_folder.mkdir()

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        train(train_loader, model, criterion, optimizer, epoch)

    state = {
        'state_dict' : model.state_dict(),
        }

    save_path = args.output_folder / ("model.pth.tar")
    save_model(state, save_path)

def save_model(state, path):
    torch.save(state, path)

def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    model.train()

    for i, (input, fixmap, smap) in enumerate(train_loader):

        input = input
        fixmap = fixmap
        smap = smap
        
        decoded = model(input)

        if args.mse: 
            loss = criterion(decoded, smap)
        else:
            loss = criterion(decoded, fixmap, smap)

        losses.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.10f} ({loss.avg:.10f})'.format(
                   epoch, i, len(train_loader),
                   loss=losses))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // args.decay_epoch
    lr = args.lr*(0.1**factor)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main1():
    main()
