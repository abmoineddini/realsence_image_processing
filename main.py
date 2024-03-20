from realsence_camera_publisher import BGRD_camera_publish
import cv2
import numpy

cap = BGRD_camera_publish()

while True:
    colour_frame, depth_frame = cap.camera_publish()

    cv2.imshow("colour", colour_frame)
    cv2.imshow("depth", depth_frame)

    key = cv2.waitKey(1)
    if key == 27:
        cap.stop()
        break