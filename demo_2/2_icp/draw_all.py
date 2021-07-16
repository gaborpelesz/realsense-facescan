import copy
import time

import open3d as o3d
import numpy as np

import utils

if __name__ == "__main__":
    face_pcds = utils.load_scan_frames()
    print(face_pcds)
    utils.draw_all(face_pcds)