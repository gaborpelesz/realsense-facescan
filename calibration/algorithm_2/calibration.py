# for the calibration we are trying to find E (essential matrix)
#
# What have we in possession?
#   - we already know K, K' (camera instrinsics matrices)
#   - first camera matrix P = K [I | 0]
#   - second camera matrix P' = K' [R | t]
#
# R and t are unknown...
#   - but we know  --> E = t x R     (where 'x' is the cross product)
#                  --> thus E has 5 degrees of freedom, with 1 scale ambiguity
#                      (which can be solved by a known sized object on scene)
#
#   - we also know --> (x1'.T) E x1 = 0  (where 'x1' is a point correspondence)
#
# I can estimate E with a variant of the 5-point algorithm
#   - original Nistér: https://www.rose-hulman.edu/class/cs/csse461/handouts/Day37/SINGLES2.pdf
#   - a better more modern approach: http://www.bmva.org/bmvc/2013/Papers/paper0116/paper0116.pdf
#

import cv2
import numpy as np

class Camera:
    def __init__(self, f, c):
        self.fx = f[0]
        self.fy = f[1]
        self.cx = c[0]
        self.cy = c[1]

    def pixel_to_unit(self, pixel_coords):
        return (pixel_coords - [self.cx, self.cy]) / [self.fx, self.fy]

P_left  = Camera([917.873, 915.479], [634.535, 368.434])
P_right = Camera([923.29, 921.4], [631.882, 369.587])
#                       blue        pink       black        green
cube_left1  = np.array([[432,383], [440,419], [433, 534], [425, 511]])
cube_right1 = np.array([[372,501], [451,509], [458, 601], [377, 600]])

#                       blue        pink       black       green     yellow     white
cube_left2  = np.array([[957,540], [996,575], [897,571], [849,535], [1004,476], [967,435]])
cube_right2 = np.array([[808,473], [940,483], [897,498], [778,490], [937, 356], [809,347]])

# points1 = np.array([[199,102], [204,47], [643, 358], [520, 557], [613, 568]])
# points2 = np.array([[528,263], [510,226], [636, 376], [654, 577], [719, 567]])
# points1 = P_left.pixel_to_unit(points1)
# points2 = P_right.pixel_to_unit(points2)

cube_left1 = P_left.pixel_to_unit(cube_left1)
cube_right1 = P_right.pixel_to_unit(cube_right1)
cube_left2 = P_left.pixel_to_unit(cube_left2)
cube_right2 = P_right.pixel_to_unit(cube_right2)

cubes_left = np.vstack((cube_left1, cube_left2))
cubes_right = np.vstack((cube_right1, cube_right2))

I = np.identity(3)
E, m = cv2.findEssentialMat(cubes_left, cubes_right, I, cv2.RANSAC, 0.999, 1.0)
print(E)

points, R, B, mask_pose = cv2.recoverPose(E, cubes_left, cubes_right)
print(R)
B = B.T[0]
print(B) # only a direction in t

white_left = np.array([967,435])
white_right = np.array([809,347])
yellow_left = np.array([1004,476])
yellow_right = np.array([937, 356])

white_left = P_left.pixel_to_unit(white_left)
white_right = P_right.pixel_to_unit(white_right)
yellow_left = P_left.pixel_to_unit(yellow_left)
yellow_right = P_right.pixel_to_unit(yellow_right)

Rt = np.row_stack((np.column_stack((R, B)), [0, 0, 0, 1]))

print(Rt)
print(np.concatenate((white_right, np.ones(2))))

white_right_Rt = Rt @ np.concatenate((white_right, np.ones(2)))

print("white_left")
print(np.concatenate((white_left, np.ones(1))))
print("white_right_Rt")
print(white_right_Rt[:3] - B)

print(np.dot(np.concatenate((white_left, np.ones(2))), np.concatenate((white_right_Rt[:3] - B, np.ones(1)))))

# x1 = np.column_stack((points1, np.ones(5)))
# x2 = np.column_stack((points2, np.ones(5)))

# errors = []

# for i in range(4):
#     err = x1 @ Es[i:i+3] @ x2.T
#     err = np.diagonal(err)
#     print(np.mean(err))
#     errors.append(np.mean(err))

#print(np.dot(np.matmul(points1[0].tolist() + [1], E[9:]), points2[0].tolist() + [1]))

