"""
This script converts a mesh file's dimensions from millimetres to metres using Open3D.
It reads an input mesh, scales it by 0.001 (to convert from mm to m), re-centers the mesh
at the origin (0,0,0), and writes the result to a new file.

Usage:
    python convert_to_meters.py
"""

import pathlib
import open3d as o3d

SRC = pathlib.Path(r"C:\Users\natalie.n.griffin4\Documents\DA\Apose\Apose.ply")

# Define the destination (output) file
DST = SRC.with_name("Apose_meters.ply")

# Load the mesh from the source file.
mesh = o3d.io.read_triangle_mesh(str(SRC))

# Convert mesh dimensions from millimetres to metres by scaling.
mesh.scale(0.001, center=mesh.get_center())

# Re-center the mesh at the origin (0, 0, 0) after scaling.
mesh.translate(-mesh.get_center())

# Write the modified mesh to the destination file.
o3d.io.write_triangle_mesh(str(DST), mesh)
print("✓ wrote", DST)