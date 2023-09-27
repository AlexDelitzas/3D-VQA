import os, sys, re
import argparse
from tqdm import tqdm
import numpy as np
import open3d as o3d
from PIL import Image


def read_pcd(pcd_path):
    source = o3d.io.read_triangle_mesh(pcd_path, enable_post_processing=False, print_progress=False)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.3, origin=[0, 0, 0])
    return source, mesh_frame

def set_up_pcd(vis, pcd_path):
    source, axis_obj = read_pcd(pcd_path)
    vis.add_geometry(source)
    return source

def capture_image(image_path):
    image = vis.capture_screen_float_buffer()
    image = np.asarray(image) * 255
    pil_img = Image.fromarray(image.astype(np.uint8))
    pil_img.save(image_path)
    return False

def capture_image_from_pose(vis, scans, scene, output):
    pcd_path = os.path.join(scans, scene, scene + '_vh_clean_2.ply')
    print(pcd_path)
    source = set_up_pcd(vis, pcd_path)
    vis.update_geometry(source)
    vis.poll_events()
    vis.update_renderer()
    capture_image(os.path.join(output,  scene + '.jpg'))
    vis.update_geometry(source)
    vis.poll_events()
    vis.update_renderer()
    vis.clear_geometries()
    vis.close()                


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, default='scene0000_00', help='scene_id')
    parser.add_argument('--scans', type=str, help='directory having scan_data')
    parser.add_argument('--output', help='directory for saving a top-down image')
    args = parser.parse_args()
    
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=640, height=420)
    vis.get_render_option().background_color = np.asarray([256,256,256])
    ctr = vis.get_view_control()
    ctr.change_field_of_view(step=90)
    
    capture_image_from_pose(vis, args.scans, args.scene, args.output)

    
