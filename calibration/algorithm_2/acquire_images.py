from ..device_manager import DeviceManager

import cv2

class StereoCamera:
    def __init__(self):
        self._device_manager = DeviceManager()
        

    def save_image_pair(self, l_path, r_path):
        self._l_path = l_path
        self._r_path = r_path
        self._device_manager.start(self._callback)


    def _callback(self, left_rgb, left_d, right_rgb, right_d):
        cv2.imwrite(self._l_path, cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(self._r_path, cv2.cvtColor(right_rgb, cv2.COLOR_RGB2BGR))
        
        self._device_manager.stop()