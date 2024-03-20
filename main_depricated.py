from realsence_camera_publisher import BGRD_camera_publish
from ultralytics import YOLO
import cv2
import numpy

cap = BGRD_camera_publish()
model = YOLO('yolov8n.pt')


while True:
    colour_frame, depth_frame, depth_colourMap_frame = cap.camera_publish()

    results = model.track(colour_frame, persist=True)

    # plot results
    for result in results:
        boxes = results[0].boxes.cpu().numpy()
        xyxys = boxes.xyxy
        classes = boxes.cls
        conf = boxes.conf
        ids = boxes.id
        counter =0

        for xyxy in xyxys:
            cv2.rectangle(colour_frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 1)
            cv2.putText(colour_frame, f"class : {classes[counter]}, confidence : {conf[counter]}", (int(xyxy[0]), int(xyxy[1])),
                         cv2.FONT_HERSHEY_SIMPLEX,  0.6, (255, 0,0), 2)
            #cv2.circle(colour_frame, (int((xyxy[0]+xyxy[2])/2), int((xyxy[1]+xyxy[3])/2)), 1, (0,0,255), 1)

            # Finding desired class 47==Apple
            if classes[counter]==47:
                distance = depth_frame[int((xyxy[0]+xyxy[2])/2), int((xyxy[1]+xyxy[3])/2)]
                cv2.putText(colour_frame, f"Distance: {distance}", (int((xyxy[0]+xyxy[2])/2), int((xyxy[1]+xyxy[3])/2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0,0), 2)
            counter = counter+1

    # frame_ = results[0].plot()

    #visualise
    # cv2.imshow('frame', frame_)
    cv2.imshow("colour", colour_frame)
    # cv2.imshow("depth", depth_frame)

    key = cv2.waitKey(1)
    if key == 27:
        cap.stop()
        break