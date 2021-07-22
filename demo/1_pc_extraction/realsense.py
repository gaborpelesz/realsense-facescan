import json
import threading
import time

import numpy as np
import pyrealsense2 as rs

# TODO get these from the config file
d_res = 1280, 720
rgb_res = 1280, 720 # 1920, 1080 when we don't align

class Realsense:
    def __init__(self) -> None:
        self.DS5_product_ids = ["0AD1", "0AD2", "0AD3", "0AD4", "0AD5", "0AF6", "0AFE", "0AFF", "0B00", "0B01", "0B03", "0B07", "0B3A", "0B5C"]
        
        self._stopping = True
        self._config_loaded = False

    def load_device_config(self, path):
        with open(path) as f:
            json_object = json.load(f)

        json_string = str(json_object).replace("'", '"')

        try:
            device = self._find_device()

            advanced_mode = rs.rs400_advanced_mode(device)
            advanced_mode.load_json(json_string)

            self._config_loaded = True
        except Exception as e:
            print(e)
            self._stopping = True

    def stop_pipeline(self):
        self._stopping = True

    def start_pipeline(self, frame_callback, iterations=0):
        self._stopping = False
        self._counter = False if iterations == 0 else True
        counter = 0

        if not self._config_loaded: 
            print("WARNING: No advanced configuration was loaded. Using defaults.")
            print("\tNote: you can load configurations by calling <Realsense_obj>.load_device_config(<json_path>)")

        try:
            pipeline = rs.pipeline()

            config = rs.config()
            config.enable_stream(rs.stream.depth, d_res[0], d_res[1], rs.format.z16, 30)
            config.enable_stream(rs.stream.color, rgb_res[0], rgb_res[1], rs.format.rgb8, 30)

            pipeline.start(config)

            #----------------
            profile = pipeline.get_active_profile()
            depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
            depth_intrinsics = depth_profile.get_intrinsics()
            print("Depth intrinsics:", depth_intrinsics)

            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            print("Depth Scale is: " , depth_scale)
            #----------------

            align = rs.align(rs.stream.color)
            pc    = rs.pointcloud()
        
            while not self._stopping:
                if self._counter:
                    counter += 1
                    if counter == iterations:
                        self._stopping = True

                frames = pipeline.wait_for_frames()

                s = time.time()
                aligned_frames = align.process(frames) # in the end, we won't need alignment

                color_frame         = frames.get_color_frame()
                color_frame_aligned = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                # if either of the frames is invalid, skip
                if not color_frame or not color_frame_aligned or not depth_frame:
                    continue
                
                points3d = pc.calculate(depth_frame)
                pc.map_to(color_frame_aligned)

                cb = threading.Thread(target=frame_callback, args=(frames, aligned_frames, points3d))
                cb.start()
                print(f'frame time: {(time.time() - s)*1000:.3f}ms', end="\r")

            print() # to compensate '\r'

        except Exception as e:
            print("Unexpected error occured while running the Realsense pipeline.")
            print(e)
            self._stopping = True

    def _find_device(self):
        ctx = rs.context()
        ds5_dev = rs.device()
        devices = ctx.query_devices()
        for dev in devices:
            if dev.supports(rs.camera_info.product_id) and str(dev.get_info(rs.camera_info.product_id)) in self.DS5_product_ids:
                if dev.supports(rs.camera_info.name):
                    print("Found device that supports advanced mode:", dev.get_info(rs.camera_info.name))
                return dev
        raise Exception("No D400 product line device that supports advanced mode was found")
