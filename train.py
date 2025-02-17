import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import RTDETR


if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-Nodule.yaml')
    model.train(data='dataset/data.yaml',
                cache=False,
                imgsz=640,
                epochs=120,
                batch=8,
                workers=4, 
                name='exp',
                )