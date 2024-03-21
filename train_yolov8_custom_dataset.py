from ultralytics import YOLO

model = YOLO('yolov9n.pt')

results = model.train(data='config.yaml', epochs=150)