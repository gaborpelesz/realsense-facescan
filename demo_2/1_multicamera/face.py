import time
import os

import numpy as np
import cv2
import open3d as o3d

import utils
import scrfd

import threading

"""FacePointCloud

    An instance of this class can create a point cloud
    from an rgb-d image, while segmenting a face.
    It can track faces from previous frames.
"""
class FacePointCloud:
    def __init__(self, manual_adjustments=False):
        self.depth_intrinsics = o3d.camera.PinholeCameraIntrinsic(width=1280,
                                                             height=720,
                                                             fx=903.545,
                                                             fy=903.545,
                                                             cx=632.27,
                                                             cy=357.557)
        self.color_intrinsics = o3d.camera.PinholeCameraIntrinsic(width=1280,
                                                             height=720,
                                                             fx=923.29,
                                                             fy=921.4,
                                                             cx=631.882,
                                                             cy=369.587)
        self.depth_scale = 1 / 0.0010000000474974513

        self.index = 1
        self.index_lock = threading.Lock()
        self._manual_adjustments = manual_adjustments

    def create_from_rgbd(self, rgb, d, save_to_file=None):
        # masking the face's bounding box in the depth image
        t0 = time.time()

        # scale box by 20% so there will be no 3D points on 
        # the face that might accidentally be masked out
        bbox = scrfd.detect_face(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), scale_box=1.2) # bbox (, kpss)
        
        if self._manual_adjustments:
            bbox = self.manual_face_detection(rgb, initial_bbox=bbox)
        
        d = self.mask_depth(d, bbox)

        # creating PointCloud in open3d
        rgb  = o3d.geometry.Image(rgb)
        d    = o3d.geometry.Image(d)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, d, depth_scale=self.depth_scale, depth_trunc=1000, convert_rgb_to_intensity=False)
        pcd  = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.color_intrinsics)

        # ray in the direction of the face center
        face_center_pix = (bbox[1][0] - (bbox[1][0] - bbox[0][0])//2, bbox[1][1] - (bbox[1][1] - bbox[0][1])//2)
        face_center_ray = np.array([(face_center_pix[0] - self.color_intrinsics.get_principal_point()[0]) / self.color_intrinsics.get_focal_length()[0],
                                    (face_center_pix[1] - self.color_intrinsics.get_principal_point()[1]) / self.color_intrinsics.get_focal_length()[1],
                                    1])

        # remove outliers 
        pcd = self.pc_face_remove_outliers(pcd, face_center_ray, search_radius=0.02) # radius 2cm

        # remaining statistical outliers
        #cl, inlier_idx = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        #outlier_idx = np.setdiff1d(np.arange(np.asarray(pcd.points).shape[0]), inlier_idx)
        #np.asarray(pcd.colors)[outlier_idx] = [1, 0, 0]

        #print(f"Create from rgbd processing time: {(time.time() - t0)*1000:.3f} ms")

        # flip z and y axis for correct visualization
        np_points = np.asarray(pcd.points)
        np_points *= [1, -1, -1]

        # translate center of mass to origo
        #   essentially: center = np.mean(np.asarray(pcd.points), axis=0)
        #                points = points - center
        # pcd.translate(np.array([0.0, 0.0, 0.0]), relative=False)

        # draw 3d point cloud with open3d viewer
        # o3d.visualization.draw_geometries_with_custom_animation([pcd],
        #                                             window_name='Extracted face',
        #                                             width=1920,
        #                                             height=1080,
        #                                             left=50,
        #                                             top=50,
        #                                             optional_view_trajectory_json_file='view.json')

        if not save_to_file is None:
            threading.Thread(target=self.ply_write_indexed, args=(save_to_file, pcd)).start()

    @staticmethod
    def ply_write_indexed(f_path, pcd):
        o3d.io.write_point_cloud(f_path, pcd)

    @staticmethod
    def mask_depth(image, bbox):
        mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
        cv2.rectangle(mask, bbox[0], bbox[1], (1), -1)
        return np.where(mask, image, 0)

    @staticmethod
    def pc_face_remove_outliers(pcd, face_center_ray, search_radius=0.1): # default radius 10 cm
        pcd_points = np.asarray(pcd.points)
        point_distances = utils.p_ol_dist(face_center_ray, pcd_points)

        closest_point_idx = np.where(np.amin(point_distances) == point_distances)[0]

        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[closest_point_idx], 0.2)
        #[k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[closest_point_idx], 200)

        # visualize knn and radius
        p_in_radius = (point_distances < search_radius)
        p_in_radius_3d = np.column_stack((p_in_radius, p_in_radius, p_in_radius))
        colors = np.asarray(pcd.colors)

        #colors[idx[1:], :] += [0, 0.5, 0] # colorize nearest neighbors in green
        #colors[:, :] = np.where(p_in_radius_3d, colors + [1, -0.5, 0], colors) # colorize points in radius to red

        pcd = pcd.select_by_index(idx)

        return pcd

    """Returns the index of the 3D point in 'arr' that is closest to (0,0,0)
        if there multiple points with the same distance it selects the one
        that is closest to the image center (0,0).
    """
    @staticmethod
    def find_closest_to_origo(arr: np.ndarray):
        # find closest points to the origo, by min euclidean distance)
        dists = np.linalg.norm(arr, axis=1)
        closest_points_idx = np.where(dists == np.min(dists))

        if closest_points_idx.shape[0] == 1:
            closest_point_idx = arr[closest_points_idx[0]]

        # if there were more than one points found (very unlikely)
        # find the one which is the closest to the image center (0, 0)
        else:
            closest_points = arr[closest_points_idx]
            closest_points_xy = closest_points[:, :-1] # remove z

            dists_xy = np.linalg.norm(closest_points_xy, axis=1)

            # if still more than one points, just pick the first one
            idx = np.where(dists_xy == np.min(dists_xy))[0][0]
            closest_point_idx = closest_points_idx[idx]

        return closest_points_idx

    def manual_face_detection(self, rgb, initial_bbox=(None, None)):
        window_data = {
            'face_bbox': {'tl': initial_bbox[0], 'br': initial_bbox[1]},
            'image': cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            'canvas': cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            'drawing': False,
            'finished': False
        }

        if not window_data['face_bbox']['tl'] is None and not window_data['face_bbox']['br'] is None:
            window_data['finished'] = True
            FacePointCloud.draw_bbox(window_data['canvas'], window_data['face_bbox']['tl'], 
                                                            window_data['face_bbox']['br'])

        wn = "bbox interactive window. Press ENTER to continue"
        cv2.namedWindow(wn, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(wn, 1920, 1080)
        cv2.setMouseCallback(wn, self._mouse_face_bbox, window_data)
        cv2.startWindowThread()

        while True:
            cv2.imshow(wn, window_data['canvas'])

            key = cv2.waitKey(5)

            if key == ord('r'):
                window_data['face_bbox'] = {'tl': (0, 0), 'br': (0, 0)}
                window_data['image'] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                window_data['canvas'] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                window_data['drawing'] = False
                window_data['finished'] = False

            if key == 13: # ENTER
                if window_data['finished']:
                    break
                else:
                    print('Draw a bounding box by clicking. After that you can exit this program.')
                
            if key == 27: # ESC
                window_data['face_bbox']['tl'] = window_data['face_bbox']['br'] = (0,0)
                break

        cv2.destroyWindow(wn)

        return window_data['face_bbox']['tl'], window_data['face_bbox']['br']
        

    @staticmethod
    def _mouse_face_bbox(event, x, y, flags, window_data):
        if event == cv2.EVENT_LBUTTONDOWN:
            if not window_data['finished']:
                window_data['drawing'] = True
                window_data['face_bbox']['tl'] = x, y
            else:
                # find if we are close enough to a corner to adjust
                tl, br = window_data['face_bbox']['tl'], window_data['face_bbox']['br']
                dist_to_tl = np.linalg.norm([tl[0]-x, tl[1]-y])
                dist_to_br = np.linalg.norm([br[0]-x, br[1]-y])
                
                eps = 30 # 30 pixel radius

                if dist_to_tl < eps or dist_to_br < eps:
                    window_data['finished'] = False
                    window_data['drawing'] = True
                    
                    # if we modify tl, than switch tl and br and act as if we were modifying br
                    if dist_to_tl < eps: 
                        window_data['face_bbox']['tl'] = window_data['face_bbox']['br']
                
                    window_data['face_bbox']['br'] = x, y


        if event == cv2.EVENT_MOUSEMOVE:
            if window_data['drawing']:
                window_data['canvas'] = window_data['image'].copy()
                FacePointCloud.draw_bbox(window_data['canvas'], window_data['face_bbox']['tl'], 
                                                                                        (x, y))

        if event == cv2.EVENT_LBUTTONUP:
            if not window_data['finished']:
                window_data['drawing'] = False
                window_data['finished'] = True
                window_data['canvas'] = window_data['image'].copy()
                
                # make sure top right is actually top right
                tl_x, tl_y = window_data['face_bbox']['tl']
                window_data['face_bbox']['tl'] = min([tl_x, x]), min([tl_y, y])
                window_data['face_bbox']['br'] = max([tl_x, x]), max([tl_y, y])

                FacePointCloud.draw_bbox(window_data['canvas'], window_data['face_bbox']['tl'],
                                                                window_data['face_bbox']['br'])

    @staticmethod
    def draw_bbox(img, p1, p2):
        cv2.rectangle(img, p1, p2, (0,0,255), 3)
        cv2.circle(img, p1, 8, (0,255,0), -1)
        cv2.circle(img, p2, 8, (0,255,0), -1)