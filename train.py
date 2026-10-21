import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8s.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='uw.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=16,
                close_mosaic=10,
                workers=8,
                pretrained=False,
                device='0',
                optimizer='SGD',
                project='runs/train',
                name='yolov8s-URPC',
                )