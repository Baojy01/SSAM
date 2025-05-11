import torch
from .utils import Metrics, AverageMeter


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    conf_mat = Metrics(num_classes)
    with torch.no_grad():
        for image, target in data_loader:
            image, target = image.to(device), target.to(device)
            output = model(image)

            conf_mat.update(target.flatten(), output.argmax(1).flatten())
            f1, m_iou = conf_mat.get_scores()

    return conf_mat, f1, m_iou


def train_one_epoch(model, optimizer, criterion, data_loader, device, lr_scheduler, scaler=None):
    losses = AverageMeter()
    model.train()

    for image, target in data_loader:
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        losses.update(loss.item(), image.size(0))

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

    return losses.avg


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=5,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
