import os

import open3d as o3d

def load_scan_frames():
    base_dir = '../scan_frames'

    point_cloud_files = [f for f in os.listdir(base_dir) if f.split('.')[-1].lower() == 'ply']
    scan_frame_files  = filter(lambda x: x.split('.')[0].isdecimal(), point_cloud_files)

    scan_frame_files = sorted(scan_frame_files, key=lambda x: int(x.split('.')[0]))

    scan_frames = [o3d.io.read_point_cloud(f"{os.path.join(base_dir, f'{f}')}")
                    for f in scan_frame_files]

    return scan_frames

def draw_all(pcds):
    o3d.visualization.draw_geometries_with_custom_animation(pcds,
                                            window_name='Extracted face',
                                            width=1920,
                                            height=1080,
                                            left=50,
                                            top=50,
                                            optional_view_trajectory_json_file='view.json')