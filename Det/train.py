import warnings
import os
from ultralytics import YOLO
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    
    data_dir = "./mydata/rdd.yaml"
    
    # arch = 'yolo11_mobilenetv2.yaml' 
    # arch = 'yolo11_efficientformerv2.yaml'
    # arch = 'yolo11_mobilevit.yaml'
    arch = 'yolo11_ssam.yaml'
    
    dir_path = 'runs/RDD/ssam_l'

    model = YOLO(arch, task='detect')
    model.train(data=data_dir,
                imgsz=512,
                epochs=100,
                single_cls=False,
                batch=16,
                close_mosaic=0,
                workers=4,
                cache=False,
                device='0',
                optimizer='AdamW',
                amp=False,
                project=dir_path,
                name='exp',
                seed=42,
                lr0=5e-3,
                cos_lr=True,
                weight_decay=1e-4,
                )
