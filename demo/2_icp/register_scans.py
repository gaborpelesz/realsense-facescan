import copy
import time
import multiprocessing
import itertools

import open3d as o3d
import numpy as np

import utils


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    return pcd_down, pcd_fpfh

def fast_global_reg(source, target, source_fpfh,
                                target_fpfh):
    distance_threshold = 1.0
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        source, target, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result

def point_to_plane_reg(source, target, in_transformation = np.identity(4)):
    distance_threshold = 1

    source.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=20))
    target.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=20))

    # point to plane ICP
    current_transformation = in_transformation
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target, 0.5, current_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

    return result_icp

def colored_point_cloud_reg(source, target, in_transformation = np.identity(4)):
    # colored pointcloud registration
    # This is implementation of following paper
    # J. Park, Q.-Y. Zhou, V. Koltun,
    # Colored Point Cloud Registration Revisited, ICCV 2017
    source.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=20))
    target.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=20))

    voxel_radius = [0.02, 0.01, 0.001]
    max_iter = [1000, 800, 400]
    current_transformation = in_transformation
    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]

        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)

        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source_down, target_down, radius, current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                            relative_rmse=1e-6,
                                                            max_iteration=iter))
        current_transformation = result_icp.transformation

    return result_icp

def registration_process(target, target_down, target_fpfh, source, voxel_size=0.4):
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size=voxel_size)


    #icp = fast_global_reg(source_down, target_down, source_fpfh, target_fpfh)
    icp = colored_point_cloud_reg(source, target) #, icp.transformation)
    #icp = point_to_plane_reg(source, target, icp.transformation)

    return icp.transformation

"""
    Calculates the transformation between every point cloud and the first point cloud.
    For the transformations' computation it first uses a global regitering method than a local refinement.
"""
def full_registration_greedy(pcds: list):
    if pcds == None or len(pcds) == 0:
        return None

    T = [np.identity(4)] # T0 is a transformation with identity

    global_reg_voxel_size = 0.4

    target = pcds[0]
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size=global_reg_voxel_size)

    for i in range(1, len(pcds)):
        #target = pcds[i-1]
        #target_down, target_fpfh = preprocess_point_cloud(target, voxel_size=global_reg_voxel_size)

        print(f"registering: {i+1}/{len(pcds)}{' '*10}", end='\r')
        T.append(registration_process(target, target, target_fpfh, pcds[i], voxel_size=global_reg_voxel_size))
    print()

    return T

if __name__ == "__main__":
    face_pcds = utils.load_scan_frames()

    #face_pcds = face_pcds[:60]

    #face_pcds = face_pcds[58], face_pcds[59]

    t0 = time.time()
    poses = full_registration_greedy(face_pcds)

    pcd_combined = o3d.geometry.PointCloud()
    for i in range(len(face_pcds)):
        #for j in range(i, i - 1, -1):
        #    face_pcds[i].transform(poses[j])
        face_pcds[i].transform(poses[i])
        pcd_combined += face_pcds[i]

    t1 = time.time()

    print(f"Finished registering {len(face_pcds)} scans. ({(t1-t0):.3f} sec)")

    #utils.draw_all([pcd_combined])

    #mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_combined, depth=9)
    radii = [0.005, 0.01, 0.02, 0.04]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd_combined, o3d.utility.DoubleVector(radii))

    print(mesh)
    utils.draw_all([mesh])