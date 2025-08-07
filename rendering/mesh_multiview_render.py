"""
This script renders images of a 3D mesh from multiple viewpoints using Open3D.

Overview:
-----------
1. Mesh Loading & Preprocessing:
   - Loads a mesh (PLY file) from disk.
   - Centers the mesh at the origin and computes vertex normals.
   - Computes the mesh's bounding box to determine an appropriate camera distance.

2. Camera & Rendering Setup:
   - Configures rendering window settings (resolution, field of view).
   - Calculates the camera position using spherical coordinates based on azimuth and elevation values.
   - Sets up the Open3D visualizer for offscreen rendering.

3. Iterative Rendering:
   - Iterates over a range of azimuth angles (0° to 360°) with a defined step.
   - For each azimuth, iterates over specified elevation angles.
   - Captures and saves rendered images.
   - Optionally saves camera intrinsic (K) and extrinsic parameters for each view.

Usage:
        python mesh_multiview_render.py
"""

import math
import pathlib
import numpy as np
import open3d as o3d


ROOT = pathlib.Path(r"C:\Users\natalie.n.griffin4\Documents\DA\squat")
OBJ = ROOT / "squat_meters.ply"
OUT = ROOT / "renders_views"
OUT.mkdir(exist_ok=True, parents=True)

#Rendering settings
RENDER_W, RENDER_H     = 1024, 768
HORIZONTAL_FOV_DEG     = 45
AZIM_STEP_DEG          = 20
ELEV_LIST              = [15, 0, -15]

SAVE_META              = True

###Mesh Loading and Preprocessing###
mesh = o3d.io.read_triangle_mesh(str(OBJ))
#Center the mesh: translate it so its center aligns with the origin.
mesh.translate(-mesh.get_center())
#Compute vertex normals for better rendering and shading.
mesh.compute_vertex_normals()

#Compute the mesh's axis-aligned bounding box.
bbox   = mesh.get_axis_aligned_bounding_box()
#Calculate an approximate radius from the bounding box's extents.
radius = 0.5 * np.linalg.norm(bbox.get_extent())
#Compute half vertical field of view based on window aspect ratio and horizontal FOV.
half_vfov = math.atan((RENDER_H / RENDER_W) * math.tan(math.radians(HORIZONTAL_FOV_DEG / 2)))
# etermine camera distance to ensure the mesh is properly framed.
cam_dist = radius / math.sin(half_vfov)

###Initialize the Visualizer###

#Create an Open3D visualizer window (not visible for offscreen rendering).
vis = o3d.visualization.Visualizer()
vis.create_window(window_name='Open3D', width=RENDER_W, height=RENDER_H, visible=False)
#Add the mesh geometry to the visualizer.
vis.add_geometry(mesh)
#Get the view control object to set camera parameters.
vc = vis.get_view_control()

#Render Loop
#Iterate over azimuth angles (0° to 360°) in steps of AZIM_STEP_DEG.
for az in range(0, 360, AZIM_STEP_DEG):
    az_r = math.radians(az) 
    #Iterate over each specified elevation angle.
    for el in ELEV_LIST:
        el_r = math.radians(el)  
        #Calculate the horizontal component of the camera distance.
        horiz = cam_dist * math.cos(el_r)
        cam_pos = [
            horiz * math.sin(az_r),
            cam_dist * math.sin(el_r),
            horiz * math.cos(az_r)
        ]
        
    
        vc.set_front([-p / cam_dist for p in cam_pos])
        vc.set_lookat([0, 0, 0])
        vc.set_up([0, 1, 0])

        #Retrieve camera parameters (intrinsics and extrinsics).
        cam = vc.convert_to_pinhole_camera_parameters()
        K, extr = cam.intrinsic.intrinsic_matrix, cam.extrinsic
        base = f"camA_{az:03}_elev_{el:+03}"

        #Render and Save Image
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(str(OUT / f"{base}.png"))
        print("saved", base + ".png")

        #Save Camera Metadata
        if SAVE_META:
            META = OUT / "camera_params"
            META.mkdir(exist_ok=True, parents=True)
            np.savetxt(META / f"{base}_K.txt", K)
            np.savetxt(META / f"{base}_extr.txt", extr)
            
vis.destroy_window()
print("render pass complete")