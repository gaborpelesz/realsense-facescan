from ..device_manager import DeviceManager
from ..device_manager import face
from ..algorithm_1 import icp

import numpy as np

device_manager = DeviceManager()

"""
    Class to calibrate the transformation between the left and right cameras' point clouds.
    The calibration result is a transformation that transforms the right camera's depth into the left's.
"""
class NaiveCalibration:
    def __init__(self, N, K):
        self.device_manager = DeviceManager()
        self.fpce = face.FacePointCloud(manual_adjustments=False)

        self.i = 0
        self.N = N
        self.K = K

        self.S = np.array([])
        self.extrinsics_rtol = np.identity(4)
        self.extrinsics_ltor = np.identity(4)

    def start_calibration(self):
        self.S = np.array([])
        self.device_manager.start(self.update_calibration)

    def update_calibration(self, left_rgb, left_d, right_rgb, right_d):
        left_pcd = self.fpce.create_from_rgbd(left_rgb, left_d)
        right_pcd =self.fpce.create_from_rgbd(right_rgb, right_d)
        # color icp with initial transformation of the last element

        if self.S.size == 0:
            left_mass_center  = np.mean(np.asarray(left_pcd.points), axis=0)
            right_mass_center = np.mean(np.asarray(right_pcd.points), axis=0)
            translation_rtol = left_mass_center - right_mass_center

            # constructing initial transformation with 0 rotation and right to left translation
            self.S = np.row_stack((
                np.column_stack((np.identity(3), translation_rtol.T)),
                np.array([0, 0, 0, 1])
            )).reshape(1,4,4)

            print(f"Initial translation: {self.S[0]}")

        else:
            try:
                if self.S.shape[0] == 1:
                    icp_result = icp.colored_point_cloud_reg(right_pcd, left_pcd, self.S[0])
                else:
                    icp_result = icp.colored_point_cloud_reg(right_pcd, left_pcd, np.mean(self.S[1:], axis=0))

                # TODO
                # append if and only if icp_result.score > threshold
                # basically just appending the new transformation to S
                print(f"Added new calibration! {self.S.shape[0]}/{self.N}")

                self.S = np.vstack((self.S, icp_result.transformation.reshape(1,4,4)))
            except Exception as e:
                pass
                #print(e)
                #print("Warning: colored ICP failed on frame")

            self.i += 1
            if self.i >= self.K or self.S.shape[0] >= self.N:
                #print(self.S)

                if self.S.shape[0] > 1:
                    self.S = self.S[1:] # remove initial estimate
                    self.extrinsics_rtol = self.process_calibration_results(self.S)
                else:
                    self.extrinsics_rtol = self.S[0]
                self.device_manager.stop()

    def process_calibration_results(self, S):
        s = np.mean(S, axis=0)
        #s = S[-1]

        return s

# This function will run when called from: python -m calibration.algorithm_1
def main():
    c = NaiveCalibration(N=40, K=100)
    c.start_calibration()

    print(c.extrinsics_rtol)
    print()
    print(c.extrinsics_rtol.tolist())