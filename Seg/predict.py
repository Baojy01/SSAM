import os
import time

import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from models import *

input_size = 192
backbone_configs = {'MobileNet_V2_075': mobilenet_v2_075(), 'MobileNet_V2_10': mobilenet_v2_10(),
                    'MobileNet_V2_14': mobilenet_v2_14(), 'MobileNet_V2_20': mobilenet_v2_20(),
                    'EfficientFormerV2_S0': efficientformerv2_s0(input_size), 'EfficientFormerV2_S1': efficientformerv2_s1(input_size),
                    'EfficientFormerV2_S2': efficientformerv2_s2(input_size), 'EdgeViT_XXS': edgevit_xxs(),
                    'EdgeViT_XS': edgevit_xs(), 'EdgeViT_S': edgevit_s(), 'MobileViT_XXS': mobilevit_xxs(),
                    'MobileViT_XS': mobilevit_xs(), 'MobileViT_S': mobilevit_s(),
                    'SSAMNet_Tiny': SSAMNet_Tiny(), 'SSAMNet_Small': SSAMNet_Small(),
                    'SSAMNet_Base': SSAMNet_Base(), 'SSAMNet_Large': SSAMNet_Large()}


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def create_model(backbone, num_classes):
    model = MFLiteSeg(backbone=backbone, n_classes=num_classes)
    return model


class_to_color = {
    0: (0, 0, 0),
    1: (107, 142, 35),
    2: (220, 20, 60),
    3: (128, 64, 128),
}


def mask_to_color(mask: np.ndarray):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_index, color in class_to_color.items():
        match = (mask == class_index)
        color_mask[match] = color
    return color_mask


def main(model_index, fig_index):
    classes = 4  # exclude background
    backbones = ['MobileNet_V2_10', 'EfficientFormerV2_S0', 'EdgeViT_XXS', 'SSAMNet_Small']
    imgs = ['000627', '000767', '001332', '003728', '003883', '004006', '004165', '004315', '004354']
    backbone = backbones[model_index]
    img = imgs[fig_index]

    weights_path = "./save_weights/NEU_Seg/%s" % (backbone + '_best_model.pth')
    img_path = "./test/NEU_Seg/%s" % (img + '.jpg')
    mask_path = "./test/NEU_Seg/%s" % (img + '.png')

    mean = (0.44545, 0.44545, 0.44545)
    std = (0.205742, 0.205742, 0.205742)

    if not os.path.exists('test_results'):
        os.makedirs('test_results')

    out_dir = os.path.join('test_results', backbone + '_' + img + "_test_result.png")

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(backbone_configs[backbone], classes)

    # load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)

    # load roi mask
    roi_img = Image.open(mask_path).convert('L')
    roi_img = np.asarray(roi_img)

    # load image
    original_img = Image.open(img_path).convert('RGB')

    # from pil image to tensor and normalize
    data_transform = transforms.Compose([transforms.CenterCrop(192),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])

    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        output = model(img.to(device))
        prediction = output.argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        prediction = mask_to_color(prediction)
        mask = Image.fromarray(prediction)
        mask.save(out_dir)


if __name__ == '__main__':
    for i in range(4):
        for j in range(9):
            main(i, j)
