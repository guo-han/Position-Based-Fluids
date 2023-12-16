import open3d as o3d
import taichi as ti
import numpy as np
import utils
import math

@ti.data_oriented
class StaticRigidBody():
    def __init__(self, config_dict, cell_recpr, grid_size, boundary):
        self.model_name = config_dict["model_name"]
        self.mesh_o3d = o3d.io.read_triangle_mesh(config_dict["model_path"])
        print("water tight mesh: ", self.mesh_o3d.is_watertight())
        print("self intersecting mesh: ", self.mesh_o3d.is_self_intersecting())
        self.n_vertices = len(self.mesh_o3d.vertices)
        self.n_faces = len(self.mesh_o3d.triangles)
        print("Load model {} with {} vertices and {} faces.".format(self.model_name, self.n_vertices, self.n_faces))
        # self.origV = ti.Vector.field(3, ti.f32, self.n_vertices)
        # self.origV.from_numpy(np.asarray(self.mesh_o3d.vertices, dtype = np.float32))
        self.scale = config_dict["model_scale"]
        self.pos = config_dict["model_pos"]
        self.rot = config_dict["model_rotation"]
        self.mesh_o3d.scale(self.scale, center=self.mesh_o3d.get_center())
        R = self.mesh_o3d.get_rotation_matrix_from_xyz(self.rot)
        self.mesh_o3d.rotate(R, center=self.mesh_o3d.get_center())
        self.mesh_o3d.translate(self.pos) 
        
        self.V = ti.Vector.field(3, ti.f32, self.n_vertices)
        self.V.from_numpy(np.asarray(self.mesh_o3d.vertices, dtype = np.float32))
        self.F = ti.field(ti.i32, self.n_faces * 3)
        self.F.from_numpy(np.asarray(self.mesh_o3d.triangles, dtype = np.int32).flatten())

        self.mesh_o3d.compute_vertex_normals()
        self.mesh_o3d.compute_triangle_normals()
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh_o3d)
        self.o3d_scene = o3d.t.geometry.RaycastingScene()
        _ = self.o3d_scene.add_triangles(mesh)

        self.vertexN = ti.Vector.field(3, ti.f32, self.n_vertices)  # per-vertex normals
        self.faceN = ti.Vector.field(3, ti.f32, self.n_faces)       # per-face normals
        self.vertexN.from_numpy(np.asarray(self.mesh_o3d.vertex_normals, dtype = np.float32))
        self.faceN_np = np.asarray(self.mesh_o3d.triangle_normals, dtype = np.float32)
        self.faceN.from_numpy(self.faceN_np)
        # self.C = ti.Vector.field(3, ti.f32, self.n_vertices)   # TODO: per vertex colors
        self.color = config_dict["model_color"] # overall color

        self.V_np = self.V.to_numpy()
        self.min_xyz = np.min(self.V_np, axis = 0)
        self.max_xyz = np.max(self.V_np, axis = 0)

        self.occupied_grid_number = ti.field(int)
        self.aabb_rect_indices = ti.Vector.field(6, int)
        ti.root.place(self.occupied_grid_number, self.aabb_rect_indices)
        self.init_AABB_ti_v(cell_recpr, grid_size)
        # print("minxyz", self.min_xyz, " maxxyz: ", self.max_xyz)
        # print("occupied grid number: ", self.occupied_grid_number[None])
        # print("min max xyz: ", self.aabb_rect_indices[None])
        # print("grid size: ", grid_size)
        self.grid_AABB = ti.Vector.field(n = 3, dtype = ti.int32, shape=(self.occupied_grid_number[None], ))
        self.init_AABB_grid_ti_v()

    def get_sdf_prims_o3d(self, query_points: np.ndarray):
        signed_distances_np = self.o3d_scene.compute_signed_distance(query_points).numpy()
        closest_points = self.o3d_scene.compute_closest_points(query_points)
        geoms_ids = closest_points['primitive_ids'].numpy()
        return signed_distances_np, geoms_ids
    
    @ti.kernel
    def init_AABB_ti_v(self, cell_recpr: ti.float32, grid_size: ti.template()):
        """
        min.xyz + max.xyz, grid indices
        """
        aabb = ti.Vector([self.min_xyz[0], self.min_xyz[1], self.min_xyz[2],
                          self.max_xyz[0], self.max_xyz[1], self.max_xyz[2]])
        ret_indices = ti.Vector([0, 0, 0, 0, 0, 0], ti.i32)

        for i in ti.static(range(3)):
            ret_indices[i] = utils.clamp_ti_v(utils.get_cell_ti_v(aabb[i], cell_recpr), 0, grid_size[i] - 1)
            ret_indices[3 + i] = utils.clamp_ti_v(utils.get_cell_ti_v(aabb[3 + i], cell_recpr) + 1, 1, grid_size[i])
        self.aabb_rect_indices[None] = ret_indices
        self.occupied_grid_number[None] = (ret_indices[3] - ret_indices[0]) * (ret_indices[4] - ret_indices[1]) * (ret_indices[5] - ret_indices[2])

    @ti.kernel
    def init_AABB_grid_ti_v(self):
        """
        min.xyz + max.xyz, grid indices
        """
        aabb = self.aabb_rect_indices[None]
        a = (aabb[3] - aabb[0])
        b = (aabb[4] - aabb[1])
        for grid_idx in ti.grouped(ti.ndrange((aabb[0], aabb[3]), (aabb[1], aabb[4]), (aabb[2], aabb[5]))):
            index = (grid_idx[0] - aabb[0]) + (grid_idx[1] - aabb[1]) * a + (grid_idx[2] - aabb[2]) * a * b
            self.grid_AABB[index] = grid_idx
        