import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from data_loader import ImagerLoader
import numpy as np
import random
import math
import torchvision.utils as vutils
from model import GazeLSTM,PinBallLoss

source_path = "../imgs/"
val_file = "validation.txt"
train_file = "train.txt"

test_file = "test.txt"


workers = 30;
epochs = 80
batch_size = 80
best_error = 100 # init with a large value
lr = 1e-4

test = True
checkpoint_test = 'gaze360_model.pth.tar'
network_name = 'Gaze360'

from tensorboardX import SummaryWriter
foo = SummaryWriter(comment=network_name)


count_test = 0
count = 0




def main():
    global args, best_error

    model_v = GazeLSTM()
    model = torch.nn.DataParallel(model_v).cuda()
    model.cuda()


    cudnn.benchmark = True

    image_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    train_loader = torch.utils.data.DataLoader(
        ImagerLoader(source_path,train_file,transforms.Compose([
            transforms.RandomResizedCrop(size=224,scale=(0.8,1)),transforms.ToTensor(),image_normalize,
        ])),
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        ImagerLoader(source_path,val_file,transforms.Compose([
            transforms.Resize((224,224)),transforms.ToTensor(),image_normalize,
        ])),
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)



    criterion = PinBallLoss().cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr)

    if test==True:

        test_loader = torch.utils.data.DataLoader(
            ImagerLoader(source_path,test_file,transforms.Compose([
                transforms.Resize((224,224)),transforms.ToTensor(),image_normalize,
            ])),
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True)
        checkpoint = torch.load(checkpoint_test)
        model.load_state_dict(checkpoint['state_dict'])
        angular_error = validate(test_loader, model, criterion)
        print('Angular Error is',angular_error)


    for epoch in range(0, epochs):


        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        angular_error = validate(val_loader, model, criterion)

        # remember best angular error in validation and save checkpoint
        is_best = angular_error < best_error
        best_error = min(angular_error, best_error)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_error,
        }, is_best)


def train(train_loader, model, criterion,optimizer, epoch):
    global count
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    prediction_error = AverageMeter()
    angular = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i,  (source_frame,target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)
        source_frame = source_frame.cuda(async=True)
        target = target.cuda(async=True)


        source_frame_var = torch.autograd.Variable(source_frame)
        target_var = torch.autograd.Variable(target)

        # compute output
        output,ang_error = model(source_frame_var)


        loss = criterion(output, target_var,ang_error)

        angular_error = compute_angular_error(output,target_var)
        pred_error = ang_error[:,0]*180/math.pi
        pred_error = torch.mean(pred_error,0)

        angular.update(angular_error, source_frame.size(0))

        losses.update(loss.item(), source_frame.size(0))

        prediction_error.update(pred_error, source_frame.size(0))


        foo.add_scalar("loss", losses.val, count)
        foo.add_scalar("angular", angular.val, count)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        count = count +1

        print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Angular {angular.val:.3f} ({angular.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prediction Error {prediction_error.val:.4f} ({prediction_error.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,angular=angular,prediction_error=prediction_error))

def validate(val_loader, model, criterion):
    global count_test
    batch_time = AverageMeter()
    losses = AverageMeter()
    prediction_error = AverageMeter()
    model.eval()
    end = time.time()
    angular = AverageMeter()

    for i, (source_frame,target) in enumerate(val_loader):

        source_frame = source_frame.cuda(async=True)
        target = target.cuda(async=True)

        source_frame_var = torch.autograd.Variable(source_frame,volatile=True)
        target_var = torch.autograd.Variable(target,volatile=True)
        with torch.no_grad():
            # compute output
            output,ang_error = model(source_frame_var)

            loss = criterion(output, target_var,ang_error)
            angular_error = compute_angular_error(output,target_var)
            pred_error = ang_error[:,0]*180/math.pi
            pred_error = torch.mean(pred_error,0)

            angular.update(angular_error, source_frame.size(0))
            prediction_error.update(pred_error, source_frame.size(0))

            losses.update(loss.item(), source_frame.size(0))

            batch_time.update(time.time() - end)
            end = time.time()


        print('Epoch: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Angular {angular.val:.4f} ({angular.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    i, len(val_loader), batch_time=batch_time,
                   loss=losses,angular=angular))

    foo.add_scalar("predicted error", prediction_error.avg, count)
    foo.add_scalar("angular-test", angular.avg, count)
    foo.add_scalar("loss-test", losses.avg, count)
    return angular.avg




def save_checkpoint(state, is_best, filename='checkpoint_'+network_name+'.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_'+network_name+'.pth.tar')


def spherical2cartesial(x):
    
    output = torch.zeros(x.size(0),3)
    output[:,2] = -torch.cos(x[:,1])*torch.cos(x[:,0])
    output[:,0] = torch.cos(x[:,1])*torch.sin(x[:,0])
    output[:,1] = torch.sin(x[:,1])

    return output


def compute_angular_error(input,target):

    input = spherical2cartesial(input)
    target = spherical2cartesial(target)

    input = input.view(-1,3,1)
    target = target.view(-1,1,3)
    output_dot = torch.bmm(target,input)
    output_dot = output_dot.view(-1)
    output_dot = torch.clamp(output_dot,-0.999999,0.999999)
    output_dot = torch.acos(output_dot)
    output_dot = output_dot.data
    output_dot = 180*torch.mean(output_dot)/math.pi
    return output_dot




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








if __name__ == '__main__':
    main()
