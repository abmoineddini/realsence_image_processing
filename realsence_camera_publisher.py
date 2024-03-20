import pyrealsense2 as rs
import numpy as np
import cv2


class BGRD_camera_publish():
    def __init__(self):
        # making connection
        self.pipe = rs.pipeline()
        self.cfg = rs.config()

        # enable colour
        self.cfg.enable_stream(rs.stream.color, 640,480, rs.format.bgr8, 30)
        # enable depth
        self.cfg.enable_stream(rs.stream.depth, 640,480, rs.format.z16, 30)

        self.pipe.start(self.cfg)


    def camera_publish(self):
            frame = self.pipe.wait_for_frames()
            depth_frame = frame.get_depth_frame()
            colour_frame = frame.get_color_frame()


            depth_im = np.asanyarray(depth_frame.get_data())
            colour_im = np.asanyarray(colour_frame.get_data())

            depth_colourMap = cv2.applyColorMap(cv2.convertScaleAbs(depth_im, alpha=0.5), cv2.COLORMAP_JET)
            
            return colour_im, depth_im, depth_colourMap
    
    def stop(self):
        self.pipe.stop()
