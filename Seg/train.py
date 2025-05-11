import os
import time
import datetime

import torch
import random
import numpy as np
import torch.backends.cudnn

from torch.utils.data import DataLoader

from utils import train_one_epoch, evaluate, create_lr_scheduler, NEULoad, DCLoad, LOSS
import utils.transforms as T
from models import *
import argparse

import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="pytorch unet train")
parser.add_argument("--seed", default=42, type=int, help='seed for initializing training')
parser.add_argument("--backbone", default=' ', type=str, help='backbone network')
parser.add_argument("--input-size", default=192, type=int)
parser.add_argument("--dataset", default="NEU_Seg", type=str, help="dataset for train")
parser.add_argument("--device", default="cuda:0", help="train device")
parser.add_argument('--num-workers', default=2, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument("--batch-size", default=16, type=int)
parser.add_argument("--epochs", default=100, type=int, metavar="N", help="number of total epochs to train")
parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
parser.add_argument('--resume', default='', help='resume from checkpoint')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')
parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
parser.add_argument("--amp", default=True, type=bool, help="Use torch.cuda.amp for mixed precision train")
args = parser.parse_args()


MODELS = ['MobileNet_V2_075', 'MobileNet_V2_10', 'MobileNet_V2_14', 'MobileNet_V2_20',
          'EfficientFormerV2_S0', 'EfficientFormerV2_S1', 'EfficientFormerV2_S2',
          'EdgeViT_XXS', 'EdgeViT_XS', 'EdgeViT_S',
          'MobileViT_XXS', 'MobileViT_XS', 'MobileViT_S',
          'SSAMNet_Tiny', 'SSAMNet_Small', 'SSAMNet_Base', 'SSAMNet_Large']

backbone_configs = {'MobileNet_V2_075': mobilenet_v2_075(), 'MobileNet_V2_10': mobilenet_v2_10(),
                    'MobileNet_V2_14': mobilenet_v2_14(), 'MobileNet_V2_20': mobilenet_v2_20(),
                    'EfficientFormerV2_S0': efficientformerv2_s0(args.input_size), 'EfficientFormerV2_S1': efficientformerv2_s1(args.input_size),
                    'EfficientFormerV2_S2': efficientformerv2_s2(args.input_size), 'EdgeViT_XXS': edgevit_xxs(),
                    'EdgeViT_XS': edgevit_xs(), 'EdgeViT_S': edgevit_s(), 'MobileViT_XXS': mobilevit_xxs(),
                    'MobileViT_XS': mobilevit_xs(), 'MobileViT_S': mobilevit_s(),
                    'SSAMNet_Tiny': SSAMNet_Tiny(), 'SSAMNet_Small': SSAMNet_Small(),
                    'SSAMNet_Base': SSAMNet_Base(), 'SSAMNet_Large': SSAMNet_Large()}

CONFIGS = {
    **dict.fromkeys(['NEU_Seg'],
                    {'mean': (0.44545, 0.44545, 0.44545),
                     'std': (0.205742, 0.205742, 0.205742)}),
    **dict.fromkeys(['DC-4K'],
                    {'mean': (0.59243, 0.59243, 0.59243),
                     'std': (0.14354, 0.14354, 0.14354)}),
}


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


def get_data():
    if args.dataset == 'NEU_Seg':
        dir_path = './data/NEU_Seg/'
        num_classes = 3
    elif args.dataset == 'DC-4K':
        dir_path = './data/DC-4K/'
        num_classes = 1
    else:
        raise NotImplementedError()

    return dir_path, num_classes


class TrainPreset:
    def __init__(self, input_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        trans = []
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.RandomCrop(input_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class ValPreset:
    def __init__(self, input_size=256, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.CenterCrop(input_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(input_size, train=False, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):

    if train:
        return TrainPreset(input_size, mean=mean, std=std)
    else:
        return ValPreset(input_size, mean=mean, std=std)


def create_model(backbone, num_classes):
    model = MFLiteSeg(backbone, num_classes)
    return model


def get_loss(num_classes=2):
    if args.dataset == 'DC-4K':
        criterion = LOSS(num_classes, loss_type='bce', use_dice=True)
        # criterion = BCE_Soft_clDice()
    else:
        criterion = LOSS(num_classes, loss_type='bce')
    return criterion


def main():
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    dir_path, num_classes = get_data()

    # segmentation nun_classes + background
    num_classes = num_classes + 1

    config = CONFIGS[args.dataset]
    mean = config['mean']
    std = config['std']

    results_dir = os.path.join('./results', args.dataset)
    weight_dir = os.path.join('./save_weights', args.dataset)

    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    results_file = os.path.join(results_dir, args.backbone + "_results.txt")

    if args.dataset == 'NEU_Seg':
        MyLoad = NEULoad
    elif args.dataset == 'DC-4K':
        MyLoad = DCLoad
    else:
        raise NotImplementedError()

    train_dataset = MyLoad(dir_path, train=True, transforms=get_transform(args.input_size, train=True, mean=mean, std=std))
    val_dataset = MyLoad(dir_path, train=False, transforms=get_transform(args.input_size, train=False, mean=mean, std=std))

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              num_workers=args.num_workers,
                              shuffle=True,
                              pin_memory=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            num_workers=args.num_workers,
                            pin_memory=True)

    criterion = get_loss(num_classes)
    model = create_model(backbone_configs[args.backbone], num_classes)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    start_time = time.time()
    best_miou = 0.

    for epoch in range(args.start_epoch, args.epochs):
        loss = train_one_epoch(model, optimizer, criterion, train_loader, device, lr_scheduler=lr_scheduler, scaler=scaler)
        conf_mat, f1, m_iou = evaluate(model, val_loader, device=device, num_classes=num_classes)

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        val_info = str(conf_mat)
        print(val_info)

        pth_dir = os.path.join(weight_dir, args.backbone + '_best_model.pth')

        if best_miou < m_iou:
            best_miou = m_iou
            torch.save(save_file, pth_dir)

        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\ntrain_loss: {loss:.4f}\n"
            best_info = f"f1_score: {f1:.2f}\nbest_mIoU: {best_miou:.2f}\n"
            f.write(train_info + val_info + best_info + "\n\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("train time {}".format(total_time_str))


if __name__ == '__main__':
    
    MODs = ['MobileNet_V2_075', 'MobileNet_V2_10', 'MobileNet_V2_14', 'MobileNet_V2_20']
    for mod in MODs:
        args.backbone = mod
        print(args.backbone)
        main()

    print("Finished Training!")
