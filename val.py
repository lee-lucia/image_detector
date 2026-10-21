import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# BILIBILI UP 魔傀面具
# 验证参数官方详解链接：https://docs.ultralytics.com/modes/val/#usage-examples:~:text=of%20each%20category-,Arguments%20for%20YOLO%20Model%20Validation,-When%20validating%20YOLO

# 精度小数点保留位数修改问题可看<使用说明.md>下方的<YOLOV8源码常见疑问解答小课堂>第五点

if __name__ == '__main__':
    model = YOLO('runs/train/yolov8s-DUO/weights/best.pt')
    model.val(data='uw.yaml',
              split='test',
              imgsz=640,
              batch=16,
              iou=0.7,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='PMF-URPC',
              )
#yolov8-BIMAFPN-4head-vss+gcsa
#yolov8-BIMAFPN-4head-vss+gcsa-mixup