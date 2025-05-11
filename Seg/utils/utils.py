import torch
import cv2
import numpy as np


class Metrics(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, imgLabel, imgPredict):
        n = self.num_classes
        if self.mat is None:
            # 创建混淆矩阵
            self.mat = torch.zeros((n, n)).to(imgLabel)
        with torch.no_grad():
            # 寻找GT中为目标的像素索引
            k = (imgLabel >= 0) & (imgLabel < n)
            # 统计像素真实类别imgLabel[k]被预测成类别imgPredict[k]的个数
            cont = n * imgLabel[k].to(torch.int64) + imgPredict[k]
            self.mat += torch.bincount(cont, minlength=n ** 2).reshape(n, n)

    def reset(self):
        if self.mat is not None:
            self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        # 计算全局预测准确率(混淆矩阵的对角线为预测正确的个数)
        acc_global = torch.diag(h).sum() / h.sum()  # Pixel Accuracy (PA)
        # 计算每个类别的准确率
        acc = torch.diag(h) / (h.sum(1) + 1e-6)  # Precision (P)
        rec = torch.diag(h) / (h.sum(0) + 1e-6)  # Recall (R)
        # 计算每个类别预测与真实目标的iou
        iou = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, rec, iou

    def get_scores(self):
        _, acc, rec, iou = self.compute()
        f1_score = 2. * acc * rec / (acc + rec + 1e-6)
        f1_score = f1_score.mean().item() * 100
        m_iou = iou.mean().item() * 100
        return f1_score, m_iou

    def __str__(self):
        acc_global, acc, rec, iou = self.compute()
        return (
            'global acc: {:.2f}\n'
            'precision: {}\n'
            'recall: {}\n'
            'IoU: {}\n'
            'mPA: {:.2f}\n'
            'mIoU: {:.2f}\n').format(
            acc_global.item() * 100,
            ['{:.2f}'.format(i) for i in (acc * 100).tolist()],
            ['{:.2f}'.format(i) for i in (rec * 100).tolist()],
            ['{:.2f}'.format(i) for i in (iou * 100).tolist()],
            acc.mean().item() * 100,
            iou.mean().item() * 100)


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


def PLT(img):  # Power-Law Transformation
    img_exp = np.power(img, 3)
    out = cv2.normalize(img_exp, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return out


def Sobel_sharp(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)  # 对x求一阶导
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)  # 对y求一阶导
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    sobel_xy = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    sobel_out = cv2.threshold(sobel_xy + img, 255, 255, cv2.THRESH_TRUNC)[1]
    return sobel_out


def LogT(img):  # Logarithmic Transformation
    img_log = np.log10(np.float32(img) + 1.0)
    img = cv2.normalize(img_log, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return img


def GICA(img):
    T1 = Sobel_sharp(img)
    T2 = LogT(img)
    result = cv2.merge([img, T2, T1])
    return result


def GICA1(img):
    T1 = Sobel_sharp(img)
    T2 = LogT(img)
    result = cv2.merge([img, T1, T2])
    return result


def GIIS(img):
    blur = cv2.blur(img, (3, 3))
    result = cv2.merge([img, blur, blur])
    return result


def Identity(img):
    result = cv2.merge([img, img, img])
    return result


def Input_strategy(img, strategy='Identity'):
    # assert strategy in ['GICA', 'GIIS', 'Identity']

    if strategy == 'GICA':
        im = GICA(img)
    elif strategy == 'GICA1':
        im = GICA1(img)
    elif strategy == 'GIIS':
        im = GIIS(img)
    else:
        im = Identity(img)

    return im
