import os
import math
import numpy as np
from sklearn.decomposition import PCA
import taichi as ti
from PBF import Pbf
from utils import PROJ_PATH, MESH_dir, read_obj

@ti.func
def clamp(x, min, max):
    return (min(x, max) - min(x, min)) / (max - min)

@ti.data_oriented
class Foam():
    def __init__(self, fluid) -> None:
        self.h_ = fluid.h_
        self.old_positions = fluid.old_positions
        self.positions = fluid.positions
        self.velocities = fluid.velocities
        self.particle_num_neighbors = fluid.particle_num_neighbors
        self.particle_neighbors = fluid.particle_neighbors

        # potentials
        self.trapped_air = ti.field(float)
        ti.root.dense(ti.i, fluid.num_particles).place(self.trapped_air)

    # def update(self, fluid):
    #     self.old_positions = fluid.old_positions
    #     self.positions = fluid.positions
    #     self.velocities = fluid.velocities
    #     self.particle_num_neighbors = fluid.particle_num_neighbors
    #     self.particle_neighbors = fluid.particle_neighbors

    @ti.func
    def radiallySymmetricWeight(self, x_norm):
        if x_norm > self.h_:
            return 0
        else:
            return 1 - x_norm

    @ti.kernel
    def trappedAir_potential(self,):
        for p_i in self.positions:
            pos_i = self.positions[p_i]
            vel_i = self.velocities[p_i]

            vel_diff = 0
            for j in range(self.particle_num_neighbors[p_i]):
                p_j = self.particle_neighbors[p_i, j]
                if p_j < 0:
                    break
                pos_ji = pos_i - self.positions[p_j]
                vel_ji = vel_i - self.velocities[p_j]

                npos_ji = pos_ji / pos_ji.norm()
                nvel_ji = vel_ji / vel_ji.norm()

                vel_diff += vel_ji.norm() * (1 - nvel_ji.dot(npos_ji)) * self.radiallySymmetricWeight(pos_ji)

            self.trapped_air[p_i] += vel_diff


    def pca_norm(self, points):
        points = [p.to_numpy() for p in points]
        pca = PCA(n_components=3)
        print("############", points)
        # pca.fit(points)


    # @ti.func
    def estimate_norm(self, filename):
        fluid_mesh = read_obj(os.path.join(PROJ_PATH, MESH_dir, filename+".obj"))
        # breakpoint()
        # for p_i in self.positions:
        #     points = ti.Vector.field(3, dtype=ti.f32, shape = self.particle_num_neighbors[p_i]+1) # boarder corners
        #     points[0] = self.positions[p_i]
        #     # for j in range(self.particle_num_neighbors[p_i]):
        #     #     p_j = self.particle_neighbors[p_i, j]
        #     #     if p_j < 0:
        #     #         break
        #     #     point_list.append(self.positions[p_j])
        #     # self.pca_norm(point_list)



    @ti.kernel
    def run(self,):
        pass

    # def estimate_norm(self,):
    #     positions = self.positions.to_numpy()
    #     particle_num_neighbors = self.particle_num_neighbors.to_numpy()
    #     particle_neighbors = self.particle_neighbors.to_numpy()
    #     for p_i in range(positions.shape[0]):
    #         point_list = []
    #         point_list.append(positions[p_i]) 
    #         for j in range(particle_num_neighbors[p_i]):
    #             p_j = particle_neighbors[p_i, j]
    #             if p_j < 0:
    #                 break
    #             point_list.append(positions[p_j])
    #         pca = PCA(n_components=3)
    #         pca.fit(np.array(point_list))