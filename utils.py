import os
import subprocess
import numpy as np
import taichi as ti
import json
import open3d as o3d

PROJ_PATH = os.path.dirname(os.path.realpath(__file__))
PARTICLE_dir = "particles"
FOAM_dir = "foam"
RIGID_dir = "rigids"
MESH_dir = "meshes"
FOAM_PCD_dir="foam_pcd"
RENDER_dir = "rendering"
os.makedirs(os.path.join(PROJ_PATH, PARTICLE_dir),exist_ok=True)
os.makedirs(os.path.join(PROJ_PATH, FOAM_dir),exist_ok=True)
os.makedirs(os.path.join(PROJ_PATH, RIGID_dir),exist_ok=True)
os.makedirs(os.path.join(PROJ_PATH, MESH_dir),exist_ok=True)
os.makedirs(os.path.join(PROJ_PATH, FOAM_PCD_dir),exist_ok=True)
os.makedirs(os.path.join(PROJ_PATH, RENDER_dir),exist_ok=True)

def convert_json_to_mesh_command_line(filename, 
                                      particle_radius=0.8,
                                      smoothing_length=2.0,
                                      cube_size=0.5,
                                      surface_threshold=0.6
                                     ):
    # need to install rust tool chain & splashsurf: https://github.com/w1th0utnam3/splashsurf
    filepath_particle = os.path.join(PARTICLE_dir, filename + ".json") 
    filename_mesh = filename + ".obj"
    # todo: splashsurf supports batch processing, but only for .vtk and not for .obj
    bashCommand = "splashsurf reconstruct {} --output-dir={} -o {} --particle-radius={} --smoothing-length={} --cube-size={} --surface-threshold={} --normals=on".format(filepath_particle, MESH_dir, filename_mesh, particle_radius, smoothing_length, cube_size, surface_threshold)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    _, _ = process.communicate() # output, error

# particle to mesh
def convert_particle_info_to_json(input, filename):
    filepath = os.path.join(PROJ_PATH, PARTICLE_dir, filename + ".json")
    input_list = input.tolist()
    with open(filepath, 'w') as outfile:
        json.dump(input_list, outfile)

def convert_foam_info_to_json(input, filename):
    filepath = os.path.join(PROJ_PATH, FOAM_dir, filename + ".json")
    input_list = input.tolist()
    with open(filepath, 'w') as outfile:
        json.dump(input_list, outfile)

def convert_foam_info_to_pcd(input, filename):
    """
    input: foam positions
    filename: frame_00xxx
    """
    filepath = os.path.join(PROJ_PATH, FOAM_PCD_dir, filename + "_foam.ply")
    num_foam = len(input)
    point_cloud = np.zeros((num_foam,3))
    for i in range(num_foam):
        point_cloud[i] = np.array([input[i][0], -input[i][2], input[i][1]])
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    o3d.io.write_point_cloud(filepath, pcd)

def convert_rigid_info_to_json(input, filename):
    filepath = os.path.join(PROJ_PATH, RIGID_dir, filename + ".json")
    with open(filepath, 'w') as outfile:
        json.dump(input, outfile)


def write_obj(vertices, faces, filename):
    with open(filename, 'w') as f:
        for vertex in vertices:
            f.write('v {} {} {}\n'.format(vertex[0], vertex[1], vertex[2]))
        
        for face in faces:
            f.write('f {} {} {}\n'.format(face[0] + 1, face[1] + 1, face[2] + 1))

def read_obj(filename):
    # Read the OBJ file and store vertices and faces in a dictionary
    vertices = []
    normals = []
    faces = []
    
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 0:
                continue
                
            if parts[0] == 'v':
                vertex = tuple(map(float, parts[1:]))
                vertices.append(vertex)
            elif parts[0] == 'vn':
                normal = tuple(map(float, parts[1:]))
                normals.append(normal)
            elif parts[0] == 'f':
                face = tuple(map(int, [p.split('/')[0] for p in parts[1:]]))
                faces.append(face)

    vertices = np.array(vertices)
    normals = np.array(normals)
    faces = np.array(faces) - 1
    obj_data = {'vertices': vertices, 'normals': normals, 'faces': faces}
    return obj_data

# @ti.func
def get_cell(pos, cell_recpr):
    return int(pos * cell_recpr)

@ti.func
def get_cell_ti_v(pos, cell_recpr):
    return int(pos * cell_recpr)

@ti.func
def clamp_ti_v(x, min, max):
    ret = x
    if x < min:
        ret = min
    elif x > max:
        ret = max
    return ret