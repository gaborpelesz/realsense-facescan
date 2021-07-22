from calibration.algorithm_2.acquire_images import StereoCamera

import os

if __name__ == "__main__":
    f = './calibration/algorithm_2/frames/'
    
    d = StereoCamera()
    d.save_image_pair(os.path.join(f, 'left.png'), os.path.join(f, 'right.png'))