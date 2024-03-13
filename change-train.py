from ultralytics import YOLO

model = YOLO(r'C:\Users\violet\Desktop\42\ultralytics\ultralytics\cfg\models\v8\yolov8-change.yaml')

results = model.train(data=r'C:\Users\violet\Desktop\42\yolov8\datasets\coco.yaml', epochs=100, imgsz=640 , batch_size= -1)