from ultralytics import YOLO
import torch



model = YOLO('yolov8-change.yaml')
if __name__ == '__main__':
    results = model.train(data=r'C:\Users\violet\Desktop\42\yolov8\datasets\coco.yaml', epochs=100, imgsz=640,device=[0],batch=24,workers=16)