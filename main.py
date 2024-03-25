# Author: Amirbahador Moinedini
# Email : abmoineddini@gmail.com
# Project : Tennis ball tracker
# version : 1

from realsence_camera_publisher import BGRD_camera_publish
from ultralytics import YOLO
import cv2
import numpy as np

cap = BGRD_camera_publish()
model = YOLO('bestV1s.pt')

bz = 5
scaling_factor = 0.5
offset = 40

while True:
    colour_frame, depth_frame, depth_colourMap_frame = cap.camera_publish()

    results = model.track(colour_frame, persist=True)
    frame_ = results[0].plot()


    boxes = results[0].boxes.cpu().numpy()
    xyxys = boxes.xyxy
    classes = boxes.cls
    print(classes)
    try:
        class_index = np.where(classes==0)
        class_index = list(class_index)
        if len(class_index)>0:
            # print(class_index)
            # print(type(class_index))
            for index in class_index[0]:
                xyxy = xyxys[index]
                # print(xyxy)
                cv2.circle(frame_, (int((xyxy[0]+xyxy[2])/2), int((xyxy[1]+xyxy[3])/2)), 1, (0,0,255), 1)
                distance_matrix = [depth_frame[int((xyxy[1]+xyxy[3])/2),    int((xyxy[0]+xyxy[2])/2)],
                                   depth_frame[int((xyxy[1]+xyxy[3])/2)+bz, int((xyxy[0]+xyxy[2])/2)-bz],
                                   depth_frame[int((xyxy[1]+xyxy[3])/2)+bz, int((xyxy[0]+xyxy[2])/2)+bz],
                                   depth_frame[int((xyxy[1]+xyxy[3])/2)-bz, int((xyxy[0]+xyxy[2])/2)+bz],
                                   depth_frame[int((xyxy[1]+xyxy[3])/2)-bz, int((xyxy[0]+xyxy[2])/2)-bz]]
                count = 0
                sum = 0
                # print(distance_matrix)
                for i in distance_matrix:
                    if i != 0:
                        sum = sum + i
                        count = count+1
                
                if sum != 0:
                    distance = int(sum/count)
                cv2.circle(frame_, (int((xyxy[0]+xyxy[2])/2)-bz, int((xyxy[1]+xyxy[3])/2)-bz), 1, (0,0,255), 2)
                cv2.putText(frame_, f"Distance: {distance}", (int((xyxy[0]+xyxy[2])/2), int((xyxy[1]+xyxy[3])/2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0,0), 2)
                # depth_frame = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.5), cv2.COLORMAP_TWILIGHT)
                cv2.circle(depth_colourMap_frame, (int((xyxy[0]+xyxy[2])/2), int((xyxy[1]+xyxy[3])/2)), 2, (255,255,255), 2)

    except Exception as e:
        print(e)
        print("tennis ball not found")
        

    #visualise
    cv2.imshow('colour frame', frame_)
    # print(frame_.shape)
    cv2.imshow("depth frame", depth_colourMap_frame)
    # print(depth_frame.shape)

    key = cv2.waitKey(1)
    if key == 27:
        break