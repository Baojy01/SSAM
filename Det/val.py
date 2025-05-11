import warnings
from ultralytics import YOLO

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    mods = ['mb_10', 'edgevit_xxs', 'ef_s0', 'ssam_s']
    # mods = ['mb_20', 'edgevit_s', 'ef_s2', 'ssam_l']
    mod = mods[3]
    pt_dir = 'runs/NEU/%s/exp/weights/last.pt' % mod
    s_dir = 'runs/val/NEU/%s' % mod
    model = YOLO(pt_dir, task='detect')
    model.predict(data='rdd.yaml',
                  split='test',
                  imgsz=256,
                  batch=1,
                  source='neu_tt/',
                  # rect=False,
                  # save_json=True, # 这个保存coco精度指标的开关
                  save=True,
                  project=s_dir,
                  name='exp',
                  )
