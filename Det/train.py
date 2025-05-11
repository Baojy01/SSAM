import warnings
import os
from ultralytics import YOLO

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    archs = ['yolo11_mobilenetv2.yaml', 'yolo11_efficientformerv2.yaml', 'yolo11_mobilevit.yaml', 'yolo11_edgevit.yaml']
    dir_paths = ['mb_20', 'ef_s2', 'mbvit_s', 'edgevit_s']

    for i in range(4):
        arch = archs[i]
        dir_path = os.path.join('runs/GC10', dir_paths[i])

    model = YOLO('yolo11_efficientformerv2.yaml', task='detect')  # yolo11_mobilenetv2.yaml
    model.train(data="./mydata/neu_det.yaml",
                imgsz=256,
                epochs=2,
                single_cls=False,
                batch=32,
                close_mosaic=0,
                workers=0,
                cache=False,
                device='0',
                optimizer='AdamW',
                amp=False,
                project='runs/train',
                name='exp',
                seed=42,
                lr0=1e-3,
                cos_lr=True,
                weight_decay=1e-2,
                )
