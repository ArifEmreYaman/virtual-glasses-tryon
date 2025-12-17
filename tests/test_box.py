import open3d as o3d
import numpy as np

mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
mesh.compute_vertex_normals()



o3d.visualization.draw_geometries([mesh])
