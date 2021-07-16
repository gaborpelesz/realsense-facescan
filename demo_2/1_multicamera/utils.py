import numpy as np

"""point, origo line distance
    Inputs:
    - oline: ndarray, line through origo, essentially just one 3d point as a vector
    - points: ndarray, array of 3d points in homogeneous form

    ref: https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
"""
def p_ol_dist(oline, points):
    return np.linalg.norm(np.cross(points, points - oline, axis=1), axis=1) / np.linalg.norm(oline)