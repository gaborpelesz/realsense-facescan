import numpy as np
import open3d as o3d

def color_icp(pc1, pc2, init_t=np.identity(4), with_global_search=False):
    result = colored_point_cloud_reg(pc1, pc2, init_t)
    return result

def colored_point_cloud_reg(source, target, in_transformation = np.identity(4)):
    # colored pointcloud registration
    # This is implementation of following paper
    # J. Park, Q.-Y. Zhou, V. Koltun,
    # Colored Point Cloud Registration Revisited, ICCV 2017
    source.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=20))
    target.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=20))

    voxel_radius = [0.04, 0.02, 0.01, 0.0001]
    max_iter = [1000, 500, 300, 300]

    current_transformation = in_transformation
    for scale in range(len(voxel_radius)):
        #print(f"COLOR ITER: {scale}")
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

# def preprocess_point_cloud(pcd, voxel_size):
#     pcd_down = pcd.voxel_down_sample(voxel_size)

#     radius_normal = voxel_size * 2
#     pcd_down.estimate_normals(
#         o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=60))

#     radius_feature = voxel_size * 5
#     pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
#         pcd_down,
#         o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=200))
#     return pcd_down, pcd_fpfh