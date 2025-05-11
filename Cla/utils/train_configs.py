import torch
import math
import sys
import torch.nn as nn
from prefetch_generator import BackgroundGenerator


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""

    max_k = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def count_parameters(models):
    return sum(p.numel() for p in models.parameters() if p.requires_grad)


def adjust_learning_rate(optimizer, lr_scheduler, args, epoch, nBatch, batch):
    warmup_epoch = args.warmup_epoch
    warmup = args.warmup
    if epoch < warmup_epoch and warmup is True:
        warmup_steps = nBatch * args.warmup_epoch
        lr_step = args.lr / (warmup_steps - 1)
        current_step = epoch * nBatch + batch
        lr = lr_step * current_step
    else:
        lr = lr_scheduler.get_last_lr()[0]

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_runner(model, mydevice, trainloader, criterion, optimizer, lr_scheduler, args, epoch, scaler=None):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    nBatch = len(trainloader)

    for i, (inputs, labels) in enumerate(trainloader):
        adjust_learning_rate(optimizer, lr_scheduler, args, epoch, nBatch, i)

        inputs, labels = inputs.to(mydevice), labels.to(mydevice)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            prec1, prec5 = accuracy(outputs, labels, topk=(1, 5))

        if not math.isfinite(loss.item()) or math.isnan(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

    return top1.avg, top5.avg, losses.avg


def val_runner(model, mydevice, val_loader):
    criterion = nn.CrossEntropyLoss()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    with torch.no_grad():
        for _, (inputs, labels) in BackgroundGenerator(enumerate(val_loader)):
            inputs, labels = inputs.to(mydevice), labels.to(mydevice)

            output = model(inputs)
            loss = criterion(output, labels)

            prec1, prec5 = accuracy(output, labels, topk=(1, 5))

            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

    return top1.avg, top5.avg, losses.avg
