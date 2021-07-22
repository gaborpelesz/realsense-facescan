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
    distance_threshold = 0.2
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        source, target, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result

def point_to_plane_reg(source, target, in_transformation = np.identity(4)):
    distance_threshold = 0.2

    source.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=40))
    target.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=40))

    # point to plane ICP
    current_transformation = in_transformation
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target, 1, current_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))

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
    max_iter = len(voxel_radius) * [1000]
    current_transformation = in_transformation
    for scale in range(len(voxel_radius)):
        print(f"COLOR ITER: {scale}")

        iter = max_iter[scale]
        radius = voxel_radius[scale]

        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)

        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source_down, target_down, radius*100, current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                            relative_rmse=1e-6,
                                                            max_iteration=iter))
        current_transformation = result_icp.transformation

    return result_icp

def registration_process(target, target_down, target_fpfh, source, voxel_size=0.4):
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size=voxel_size)

    icp = lambda: None
    icp.transformation = np.identity(4)

    # BEST ICP FOR SCAN_FRAMES_SAVED
    # icp.transformation = np.array([[ 0.70241392,  0.15845887,  0.69390595,  0.49180339],
    #                         [-0.11433526,  0.987363  , -0.10973497, -0.03898585],
    #                         [-0.70252555, -0.00225854,  0.71165494, -0.15097216],
    #                         [ 0.        ,  0.        ,  0.        ,  1.        ]])

    # BEST ICP FOR SCAN_FRAMES_SAVED2
    #   the scan_frames_saved2 folder contains one image pair
    #   but not those with which this extrinsic matrix was achieved
    # icp.transformation = np.array([[ 9.34884337e-01,  7.60460719e-02,  3.46710644e-01,  2.17561697e-01],
    #                                [-7.92610466e-02,  9.96841750e-01, -4.92049612e-03,  1.06019115e-04],
    #                                [-3.45989830e-01, -2.28805538e-02,  9.37959230e-01,  7.34321050e-03],
    #                                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

    # icp.transformation = np.array([[ 1,0,0,  -0.06474042],
    #                                [0, 1, 0,  -0.01945646],
    #                                [0, 0, 1,  0.00924079],
    #                                [0, 0, 0,  1]])
    #icp.transformation = [[0.6028712676826402, 0.35360037760465957, 0.6806900710434488, 0.5245255423253641], [-0.3363973088995225, 0.8859459713340643, -0.14175819894448552, -0.08276327651465193], [-0.6598610847011457, -0.12390776873790263, 0.7067968592318049, -0.1801860094633929], [0.0, 0.0, 0.0, 1.0]]
    icp.transformation = [[0.6340488292210481, 0.3519358903717249, 0.6885660543707393, 0.5321429759538554], [-0.3630413981646129, 0.9216782203534285, -0.13678523803695952, -0.07979600137164221], [-0.6827759701264765, -0.16324946307539462, 0.7121562942384688, -0.17414280600270093], [0.0, 0.0, 0.0, 1.0]]
    
    icp.transformation = np.array(icp.transformation)


    #icp = fast_global_reg(source_down, target_down, source_fpfh, target_fpfh)

    icp = colored_point_cloud_reg(source, target, icp.transformation)
    #icp = point_to_plane_reg(source, target, icp.transformation)
    #print(icp)
    print(icp.transformation)
    print()
    print(icp.transformation.tolist())

    return icp.transformation

"""
    Calculates the transformation between every point cloud and the first point cloud.
    For the transformations' computation it first uses a global regitering method than a local refinement.
"""
def full_registration_greedy(pcds: list):
    if pcds == None or len(pcds) == 0:
        return None

    T = [np.identity(4)] # T0 is a transformation with identity

    global_reg_voxel_size = 0.04

    target = pcds[0]
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size=global_reg_voxel_size)

    for i in range(1, len(pcds)):
        #target = pcds[i-1]
        #target_down, target_fpfh = preprocess_point_cloud(target, voxel_size=global_reg_voxel_size)

        print(f"registering: {i+1}/{len(pcds)}{' '*10}", end='\r')
        T.append(registration_process(target, target, target_fpfh, pcds[i], voxel_size=global_reg_voxel_size))
    print()

    return T

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=60))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=200))
    return pcd_down, pcd_fpfh

def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size# * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result

def simple_registration(pcds: list, in_transformation = np.identity(4)):
    if pcds == None or len(pcds) != 2:
        return None

    T = [np.identity(4)]

    voxel_size = 0.02
    source_down, source_fpfh = preprocess_point_cloud(pcds[0], voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(pcds[1], voxel_size)

    t = np.identity(4)

    result = execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    t = result.transformation

    # result = o3d.pipelines.registration.registration_icp(
    #     source_down, target_down, 0.02, t,
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    #     o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20000))

    # t = result.transformation

    print(result)
    print("Transformation is:")
    print(result.transformation)

    # result = o3d.pipelines.registration.registration_icp(
    #     pcds[0], pcds[1], 0.01, t,
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    #     o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20000))

    # t = result.transformation

    print(result)
    print("Transformation is:")
    print(result.transformation)

    #t = reg_p2p.transformation
    #t = np.identity(4)

    T.append(t)

    return T

if __name__ == "__main__":
    face_pcds = utils.load_scan_frames()

    #face_pcds = face_pcds[:60]

    #face_pcds = face_pcds[58], face_pcds[59]

    t0 = time.time()
    poses = full_registration_greedy(face_pcds)
    #poses = simple_registration(face_pcds)


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
    # radii = [0.005, 0.01, 0.02, 0.04]
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    #     pcd_combined, o3d.utility.DoubleVector(radii))

    # print(mesh)
    utils.draw_all([pcd_combined])