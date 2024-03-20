from realsence_camera_publisher import BGRD_camera_publish
from ultralytics import YOLO
import cv2
import numpy as np

cap = BGRD_camera_publish()
model = YOLO('yolov8n.pt')


while True:
    colour_frame, depth_frame, depth_colourMap_frame = cap.camera_publish()

    results = model.track(colour_frame, persist=True)
    frame_ = results[0].plot()


    boxes = results[0].boxes.cpu().numpy()
    xyxys = boxes.xyxy
    classes = boxes.cls
    print(classes)
    try:
        class_index = np.where(classes==47)
        class_index = list(class_index)
        print(class_index)
        print(type(class_index))
        for index in class_index[0]:
            xyxy = xyxys[index]
            cv2.circle(frame_, (int((xyxy[0]+xyxy[2])/2), int((xyxy[1]+xyxy[3])/2)), 1, (0,0,255), 1)
            distance = depth_frame[int((xyxy[0]+xyxy[2])/2), int((xyxy[1]+xyxy[3])/2)]
            cv2.putText(frame_, f"Distance: {distance}", (int((xyxy[0]+xyxy[2])/2), int((xyxy[1]+xyxy[3])/2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0,0), 2)
    except:
        print("Apple not found")

    #visualise
    cv2.imshow('frame', frame_)
    # cv2.imshow("depth", depth_frame)

    key = cv2.waitKey(1)
    if key == 27:
        cap.stop()
        break