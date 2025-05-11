import torch
import os
from ptflops import get_model_complexity_info
from models import *
from utils import load_model

models = ['MobileNet_V2_075','MobileNet_V2_10', 'MobileNet_V2_14', 'MobileNet_V2_20',
          'EfficientFormerV2_S0', 'EfficientFormerV2_S1', 'EfficientFormerV2_S2',
          'FasterNet_T0', 'FasterNet_T1', 'FasterNet_T2',
          'EfficientViT_M0', 'EfficientViT_M1', 'EfficientViT_M2',
          'EfficientViT_M3', 'EfficientViT_M4', 'EfficientViT_M5',
          'EdgeViT_XXS', 'EdgeViT_XS', 'EdgeViT_S',
          'EdgeNext_XXS', 'EdgeNext_XS', 'EdgeNext_S',
          'MobileViT_XXS', 'MobileViT_XS', 'MobileViT_S',
          'SSAMNet_Tiny', 'SSAMNet_Small', 'SSAMNet_Base']


def main():
    device = torch.device("cpu")

    if not os.path.exists('./test_outputs'):
        os.makedirs('./test_outputs')

    for arch in models:
        model = load_model(arch, num_classes=10)
        model = model.to(device)
        input_size = 256 if 'MobileViT' in arch else 224
        flops, params = get_model_complexity_info(model, (3, input_size, input_size), as_strings=True, print_per_layer_stat=True)

        flops_params_results = "flops_and_params_results.txt"
        flops_params_results = os.path.join('./test_outputs', flops_params_results)
        with open(flops_params_results, "a") as f:
            info = f"[Model: {arch}]  " \
                   f"FLOPs: {flops}  " \
                   f"Params: {params}   "
            f.write(info + "\n")


if __name__ == "__main__":
    main()
