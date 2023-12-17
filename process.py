import os
import trimesh
import open3d as o3d
import numpy as np
from utils import read_obj, write_obj

def rotation_matrix(angle, axis='x'):
    # get cos and sin from angle
    c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    # get totation matrix
    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 'y':
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    if axis == 'z':
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return R

proj_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "models")
read_path = os.path.join(proj_path, "bunny_hg.obj")
blender_path = os.path.join(proj_path, "bunny_blender.obj")
object = read_obj(read_path)
blender = read_obj(blender_path)
blender_center = blender['vertices'].mean(0)
print(blender_center)

object['vertices'] = np.matmul(rotation_matrix(-90, 'x'), object['vertices'].T).T
scale = np.array([-0.17174, 0.17174, 0.17174])
object['vertices'] = object['vertices'] * scale
for i in range(object['faces'].shape[0]):
    a, b, c = object['faces'][i]
    object['faces'][i] = a, c, b
    
obj_center = object['vertices'].mean(0)
object['vertices'] = object['vertices'] + (blender_center - obj_center)

mesh = trimesh.Trimesh(object['vertices'], object['faces'], process=False)
print(mesh.is_watertight)

# trimesh.repair.broken_faces(mesh)
# trimesh.repair.fill_holes(mesh)
# trimesh.repair.fix_inversion(mesh)
# trimesh.repair.fix_winding(mesh)
# trimesh.repair.fix_normals(mesh)
# print(mesh.is_watertight)

# bunny
out_path = os.path.join(proj_path, "bunny_final.obj")
write_obj(object['vertices'], object['faces'], out_path)

mesh_o3d = o3d.io.read_triangle_mesh(out_path)
print("water tight mesh: ", mesh_o3d.is_watertight())
