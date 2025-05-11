import sys
import json
import pickle
import random
from tqdm import tqdm

from models import *


def dataset_path_cla(args):
    if args.dataset == 'SSD-10':
        data_path = './data/SSD-10/'
        num_classes = 10
    elif args.dataset == 'NEU-CLS':
        data_path = './data/NEU-CLS/'
        num_classes = 6
    else:
        raise NotImplementedError()

    return data_path, num_classes


def load_model(arch, num_classes):
    if arch.lower() == 'mobilenet_v2_10':
        model = mobilenet_v2_10(num_classes=num_classes)
    elif arch.lower() == 'mobilenet_v2_20':
        model = mobilenet_v2_20(num_classes=num_classes)
    elif arch.lower() == 'mobilenet_v2_25':
        model = mobilenet_v2_25(num_classes=num_classes)
    elif arch.lower() == 'efficientformerv2_s0':
        model = efficientformerv2_s0(num_classes=num_classes)
    elif arch.lower() == 'efficientformerv2_s1':
        model = efficientformerv2_s1(num_classes=num_classes)
    elif arch.lower() == 'efficientformerv2_s2':
        model = efficientformerv2_s2(num_classes=num_classes)
    elif arch.lower() == 'fasternet_t0':
        model = fasternet_t0(num_classes=num_classes)
    elif arch.lower() == 'fasternet_t1':
        model = fasternet_t1(num_classes=num_classes)
    elif arch.lower() == 'fasternet_t2':
        model = fasternet_t2(num_classes=num_classes)
    elif arch.lower() == 'pvt_v2_b0':
        model = pvt_v2_b0(num_classes=num_classes)
    elif arch.lower() == 'pvt_v2_b1':
        model = pvt_v2_b1(num_classes=num_classes)
    elif arch.lower() == 'efficientvit_m0':
        model = EfficientViT_M0(num_classes=num_classes)
    elif arch.lower() == 'efficientvit_m1':
        model = EfficientViT_M1(num_classes=num_classes)
    elif arch.lower() == 'efficientvit_m2':
        model = EfficientViT_M2(num_classes=num_classes)
    elif arch.lower() == 'efficientvit_m3':
        model = EfficientViT_M3(num_classes=num_classes)
    elif arch.lower() == 'efficientvit_m4':
        model = EfficientViT_M4(num_classes=num_classes)
    elif arch.lower() == 'efficientvit_m5':
        model = EfficientViT_M5(num_classes=num_classes)
    elif arch.lower() == 'edgevit_xxs':
        model = edgevit_xxs(num_classes=num_classes)
    elif arch.lower() == 'edgevit_xs':
        model = edgevit_xs(num_classes=num_classes)
    elif arch.lower() == 'edgevit_s':
        model = edgevit_s(num_classes=num_classes)
    elif arch.lower() == 'edgenext_xxs':
        model = edgenext_xx_small(num_classes=num_classes)
    elif arch.lower() == 'edgenext_xs':
        model = edgenext_x_small(num_classes=num_classes)
    elif arch.lower() == 'edgenext_s':
        model = edgenext_small(num_classes=num_classes)
    elif arch.lower() == 'mobilevit_xxs':
        model = mobilevit_xxs(num_classes=num_classes)
    elif arch.lower() == 'mobilevit_xs':
        model = mobilevit_xs(num_classes=num_classes)
    elif arch.lower() == 'mobilevit_s':
        model = mobilevit_s(num_classes=num_classes)
    elif arch.lower() == 'ssamnet_tiny':
        model = SSAMNet_Tiny(num_classes)
    elif arch.lower() == 'ssamnet_small':
        model = SSAMNet_Small(num_classes)
    elif arch.lower() == 'ssamnet_base':
        model = SSAMNet_Base(num_classes)
    else:
        raise NotImplementedError()

    return model
