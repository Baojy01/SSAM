import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader

from utils import GetData, train_runner, val_runner, dataset_path_cla, load_model


models = ['MobileNet_V2_075', 'MobileNet_V2_10', 'MobileNet_V2_14', ' MobileNet_V2_20',
          'EfficientFormerV2_S0', 'EfficientFormerV2_S1', 'EfficientFormerV2_S2',
          'FasterNet_t0', 'FasterNet_t1', 'FasterNet_t2',
          'EdgeViT_XXS', 'EdgeViT_XS', 'EdgeViT_S',
          'EdgeNext_XXS', 'EdgeNext_XS', 'EdgeNext_S',
          'MobileViT_XXS', 'MobileViT_XS', 'MobileViT_S',
          'SSAMNet_Tiny', 'SSAMNet_Small', 'SSAMNet_Base', 'SSAMNet_Large']

parser = argparse.ArgumentParser(description='PyTorch image training')
parser.add_argument('--seed', default=42, type=int, help='seed for initializing training')
parser.add_argument('--device', default='cuda:0', help='device')
parser.add_argument('--dataset', default='NEU-CLS', choices=['SSD-10', 'NEU-CLS'],
                    help='dataset for training')
parser.add_argument('--arch', default='SSAMNet_Tiny', metavar='ARCH', help='models architecture')
parser.add_argument('--workers', default=2, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--warmup', default=True, type=bool, help='whether using warmup or not (default: True)')
parser.add_argument('--warmup_epoch', default=5, type=int, metavar='N', help='use warmup epoch number (default: 5)')
parser.add_argument('--batch_size', default=64, type=int, metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', default=0.001, type=float, metavar='LR', help='initial learning rate')  # 0.001
parser.add_argument('--lr_gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight_decay', default=0.01, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', default=False, type=bool, help='directory if using pre-trained models')
parser.add_argument("--amp", default=True, type=bool, help="Use torch.cuda.amp for mixed precision training")
args = parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False


def main():
    if args.seed is not None:
        set_seed(args.seed)

    data_path, num_classes = dataset_path_cla(args)
    args.num_classes = num_classes
    args.data_dir = data_path

    train_set, val_set = GetData(args)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                            pin_memory=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device))

    model = load_model(args.arch, args.num_classes)
    model = model.to(device)

    results_dir = './results/%s/' % (
            args.arch + '_' + args.dataset + '_bs_' + str(args.batch_size) + '_epochs_' + str(args.epochs))
    save_dir = './checkpoint/%s/' % (
            args.arch + '_' + args.dataset + '_bs_' + str(args.batch_size) + '_epochs_' + str(args.epochs))

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs-args.warmup_epoch, eta_min=1e-5)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    log_dir = os.path.join(save_dir, "last_model.pth")
    train_dir = os.path.join(results_dir, "train.csv")
    val_dir = os.path.join(results_dir, "val.csv")

    best_top1, best_top5 = 0.0, 0.0
    Loss_train, Accuracy_train_top1, Accuracy_train_top5 = [], [], []
    Loss_val, Accuracy_val_top1, Accuracy_val_top5 = [], [], []

    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
        elif os.path.isfile(log_dir):
            print("There is no checkpoint found at '{}', then loading default checkpoint at '{}'.".format(args.resume, log_dir))
            checkpoint = torch.load(log_dir)  # default load last_model.pth
        else:
            raise FileNotFoundError()

        model.load_state_dict(checkpoint['models'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

        if args.start_epoch < args.epochs:
            best_top1, best_top5 = checkpoint['best_top1'], checkpoint['best_top5']
            print('Loading models successfully, current start_epoch={}.'.format(args.start_epoch))
            trainF = open(train_dir, 'a+')
            valF = open(val_dir, 'a+')
        else:
            raise ValueError(
                'epochs={}, but start_epoch={} in the saved models, please reset epochs larger!'.format(args.epochs, args.start_epoch))
    else:
        trainF = open(results_dir + 'train.csv', 'w')
        valF = open(results_dir + 'val.csv', 'w')
        trainF.write('{},{}\n'.format('epoch', 'loss', 'top1', 'top5', 'lr'))
        valF.write('{},{},{},{},{},{}\n'.format('epoch', 'val_loss', 'val_top1', 'val_top5', 'best_top1', 'best_top5'))


    lrs = []
    for epoch in range(args.start_epoch, args.epochs):

        time_star = time.time()
        top1, top5, loss = train_runner(model, device, train_loader, criterion, optimizer, lr_scheduler, args, epoch, scaler=scaler)
        val_top1, val_top5, val_loss = val_runner(model, device, val_loader)
        time_end = time.time()

        lr = optimizer.param_groups[0]["lr"]

        if args.warmup:
            if epoch >= args.warmup_epoch - 1:
                lr_scheduler.step()
        else:
            lr_scheduler.step()

        lrs.append(lr)
        # save weights
        save_files = {
            'models': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'best_top1': best_top1,
            'best_top5': best_top5,
            'epoch': epoch}
        if args.amp:
            save_files["scaler"] = scaler.state_dict()
            
        torch.save(save_files, log_dir)

        if best_top5 < val_top5:
            best_top5 = val_top5

        if best_top1 < val_top1:
            best_top1 = val_top1
            torch.save(save_files, os.path.join(save_dir, "best_model.pth"))

        Loss_train.append(loss)
        Accuracy_train_top1.append(top1)
        Accuracy_train_top5.append(top5)

        Loss_val.append(val_loss)
        Accuracy_val_top1.append(val_top1)
        Accuracy_val_top5.append(val_top5)

        print("Train Epoch: {} \t train loss: {:.4f}, train top1: {:.4f}%, train top5: {:.4f}%".format(epoch, loss, top1, top5))
        print("val_loss: {:.4f}, val_top1: {:.4f}%, val_top5: {:.4f}%".format(val_loss, val_top1, val_top5))
        print("best_val_top1: {:.4f}%, best_val_top5: {:.4f}%, lr: {:.6f}".format(best_top1, best_top5, lr))
        print('Each epoch running time: {:.4f} s'.format(time_end - time_star))
        trainF.write('{},{},{},{},{}\n'.format(epoch, loss, top1, top5, lr))
        valF.write('{},{},{},{},{},{}\n'.format(epoch, val_loss, val_top1, val_top5, best_top1, best_top5))

        trainF.flush()
        valF.flush()

    trainF.close()
    valF.close()

    print('Finished Training!')


if __name__ == "__main__":
    main()
