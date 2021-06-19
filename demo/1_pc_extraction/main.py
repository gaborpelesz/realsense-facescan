import threading

import cv2
import numpy as np
import pyrealsense2 as rs

from realsense import Realsense
import face

import time

fpce = face.FacePointCloud()

# callback
def process_frames(frames, aligned_frames, points3d):
    t0 = time.time()
    color_frame         = frames.get_color_frame()
    color_frame_aligned = aligned_frames.get_color_frame()
    depth_frame         = aligned_frames.get_depth_frame()

    rgb         = np.asanyarray(color_frame.get_data())
    rgb_aligned = np.asanyarray(color_frame_aligned.get_data())
    depth       = np.asanyarray(depth_frame.get_data())

    #np.save("pre_recorded/rgb.npy", rgb)
    #np.save("pre_recorded/rgb_aligned.npy", rgb_aligned)
    #np.save("pre_recorded/depth.npy", depth)

    #rgb         = np.load("pre_recorded/rgb.npy")
    #rgb_aligned = np.load("pre_recorded/rgb_aligned.npy")
    #depth       = np.load("pre_recorded/depth.npy")

    fpce.create_from_rgbd(rgb_aligned, depth)

    print(f"process frame time: {(time.time() - t0) * 1000:.3f} ms")

def main():
    realsense = Realsense()
    try:
        realsense.load_device_config('advanced_config.json')
        realsense.start_pipeline(process_frames, iterations=60)
    except Exception as e:
        print(e)
    finally:
        realsense.stop_pipeline()

if __name__ == "__main__":
    main()