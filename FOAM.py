import os
import math
import numpy as np
from sklearn.decomposition import PCA
import taichi as ti
from PBF import Pbf
from utils import PROJ_PATH, MESH_dir, read_obj

@ti.func
def clamp(x, tmin, tmax):
    return (min(x, tmax) - min(x, tmin)) / (tmax - tmin)

@ti.data_oriented
class Foam():
    def __init__(self, fluid) -> None:
        self.dim = fluid.dim
        self.h_ = fluid.h_
        self.rho0 = fluid.rho0
        self.mass = fluid.mass # 0.8 * math.pi * pow(self.h_,3) 
        num_particles = fluid.num_particles
        self.old_positions = fluid.old_positions
        self.positions = fluid.positions
        self.velocities = fluid.velocities
        self.particle_num_neighbors = fluid.particle_num_neighbors
        self.particle_neighbors = fluid.particle_neighbors
        self.epsilon = 1.0e-9
        
        # foam
        h3 = pow(self.h_, 3)
        self.m_k = 8.0 / (math.pi * h3)
        self.m_l = 48.0 / (math.pi * h3)
        self.foam_scale = 1
        self.timeStepSize = fluid.time_delta
        self.k_ta = 4000
        self.k_wc = 100000
        self.lifetimeMin = 2.0
        self.lifetimeMax = 5.0
        max_num_white_particles = 3000
        self.densities = ti.field(float)
        self.normals = ti.Vector.field(self.dim, float)

        self.foam_positions = ti.Vector.field(self.dim, float)
        self.foam_velocities = ti.Vector.field(self.dim, float)
        self.foam_lifetime = ti.field(float)
        self.foam_type = ti.field(int)
        self.tmp_positions = ti.Vector.field(self.dim, float)
        self.tmp_velocities = ti.Vector.field(self.dim, float)
        self.tmp_lifetime = ti.field(float)
        self.tmp_type = ti.field(int)
        #  ###############
        #  foam_type: 0 -> spray, 1 -> foam, 2 -> bubble
        #  ###############

        # potentials
        self.v_diff = ti.field(float)
        self.curvature = ti.field(float)
        self.energy = ti.field(float)

        ti.root.dynamic(ti.i, max_num_white_particles, chunk_size=32).place(self.foam_positions)
        ti.root.dynamic(ti.i, max_num_white_particles, chunk_size=32).place(self.foam_velocities)
        ti.root.dynamic(ti.i, max_num_white_particles, chunk_size=32).place(self.foam_lifetime)
        ti.root.dynamic(ti.i, max_num_white_particles, chunk_size=32).place(self.foam_type)
        ti.root.dynamic(ti.i, max_num_white_particles, chunk_size=32).place(self.tmp_positions)
        ti.root.dynamic(ti.i, max_num_white_particles, chunk_size=32).place(self.tmp_velocities)
        ti.root.dynamic(ti.i, max_num_white_particles, chunk_size=32).place(self.tmp_lifetime)
        ti.root.dynamic(ti.i, max_num_white_particles, chunk_size=32).place(self.tmp_type)
        ti.root.dense(ti.i, num_particles).place(self.densities, self.normals)
        ti.root.dense(ti.i, num_particles).place(self.v_diff, self.curvature, self.energy)


        # accumulate values
        self.frame_num = ti.field(ti.i32, shape=())
        self.sum_max_vdiff = ti.field(ti.f32, shape=())
        self.sum_max_curvature = ti.field(ti.f32, shape=())
        self.sum_max_energy = ti.field(ti.f32, shape=())
        self.taMax = ti.field(ti.f32, shape=())
        self.taMin = ti.field(ti.f32, shape=())
        self.wcMax = ti.field(ti.f32, shape=())
        self.wcMin = ti.field(ti.f32, shape=())
        self.keMax = ti.field(ti.f32, shape=())
        self.keMin = ti.field(ti.f32, shape=())
        self.foam_counter = ti.field(ti.i32, shape=())

    @ti.func
    def cubic_W(self, r: ti.f32):
        res = 0.0
        q = r / self.h_
        if (q <= 1.0):
            if (q <= 0.5):
                q2= q*q
                q3 = q2*q
                res = self.m_k * (6.0*q3 - 6.0*q2 +1.0)
            else:
                res = self.m_k * 2.0 * pow(1-q, 3)
        return res

    @ti.func
    def cubic_gradW(self, r):
        res = ti.Vector([0., 0., 0.])
        rl = r.norm()
        q = rl / self.h_
        if ((rl > self.epsilon) and (q <= 1.0)):
            gradq = r / rl
            gradq /= self.h_
            if (q <= 0.5):
                res = self.m_l * q * (3.0*q - 2) * gradq
            else:
                factor = 1 - q
                res = self.m_l * (-factor * factor) * gradq
        return res

    @ti.kernel
    def compute_density(self,):
        for p_i in self.positions:
            pos_i = self.positions[p_i]
            density_acc = 0.0

            for j in range(self.particle_num_neighbors[p_i]):
                p_j = self.particle_neighbors[p_i, j]
                if p_j < 0:
                    break
                pos_ji = pos_i - self.positions[p_j]
                density_acc += self.mass * self.cubic_W(pos_ji.norm())

            self.densities[p_i] = density_acc
            
    @ti.kernel
    def compute_normal(self, ):
        for p_i in self.positions:
            pos_i = self.positions[p_i]
            ni = ti.Vector([0., 0., 0.])

            # only interested in surface particles, may need a smaller threshold...
            if self.particle_num_neighbors[p_i] > 8:
                continue

            for j in range(self.particle_num_neighbors[p_i]):
                p_j = self.particle_neighbors[p_i, j]
                if p_j < 0:
                    break
                density_j = self.densities[p_j]
                pos_ji = pos_i - self.positions[p_j]
                ni -= self.mass / density_j * self.cubic_gradW(pos_ji)

            self.normals[p_i] = ni.normalized() if ni.norm() > self.epsilon else ni
    
    @ti.func
    def foam_W(self, r):
        res = 0.0
        q = r / self.h_
        if (q <= 1.0):
            res = 1.0 - q
        return res
        
    @ti.func
    def update_limits(self,):
        max_v, max_c, max_e = -math.inf, -math.inf, -math.inf
        for p_i in self.positions:
            max_v = max_v if max_v > self.v_diff[p_i] else self.v_diff[p_i]
            max_c = max_c if max_c > self.curvature[p_i] else self.curvature[p_i]
            max_e = max_e if max_e > self.energy[p_i] else self.energy[p_i]

        ti.atomic_add(self.sum_max_vdiff[None], max_v)
        ti.atomic_add(self.sum_max_curvature[None], max_c)
        ti.atomic_add(self.sum_max_energy[None], max_e)
        ti.atomic_add(self.frame_num[None], 1)

        # compute limits
        self.taMax[None] = self.sum_max_vdiff[None] / self.frame_num[None]
        self.taMin[None] = 0.1 * self.taMax[None]
        self.wcMax[None] = self.sum_max_curvature[None] / self.frame_num[None]
        self.wcMin[None] = 0.1 * self.wcMax[None]
        self.keMax[None] = self.sum_max_energy[None] / self.frame_num[None]
        self.keMin[None] = 0.1 * self.keMax[None]

    @ti.func
    def getOrthogonalVectors(self, vec):
        tmp = ti.Vector([1., 0., 0.])

        # Check, if v has same direction as vec
        if ((tmp.dot(vec)) > 1-self.epsilon):
            tmp = ti.Vector([0., 1., 0.])

        e1 = (vec.cross(tmp)).normalized() 
        e2 = (vec.cross(e1)).normalized()
        return e1, e2

    @ti.kernel
    def generateFoamParticles(self,):
        for idx in self.positions:
            I_ta = clamp(self.v_diff[idx], self.taMin[None], self.taMax[None])
            I_wc = clamp(self.curvature[idx], self.wcMin[None], self.wcMax[None])
            I_ke = clamp(self.energy[idx], self.keMin[None], self.keMax[None])
            num = int(max(self.foam_scale * I_ke * (self.k_ta*I_ta + self.k_wc*I_wc) * self.timeStepSize + 0.5, 0.0))
            # nt = int(self.foam_scale * I_ke * self.k_ta * I_ta * self.timeStepSize + 0.5)
            # nw = int(self.foam_scale * I_ke * self.k_wc * I_wc * self.timeStepSize + 0.5)

            if num > 300: 
                p = self.positions[idx]
                v = self.velocities[idx]
                vn = v.normalized()
                e1, e2 = self.getOrthogonalVectors(vn)

                e1 *= self.h_
                e2 *= self.h_

                for i in range(10):
                    Xr, Xt, Xh = ti.random(float), ti.random(float), ti.random(float)
                    r = self.h_ * ti.sqrt(Xr)
                    theta = 2*math.pi*Xt
                    h = self.timeStepSize * (Xh - 0.5) * v.norm()

                    xd = p + r*ti.cos(theta)*e1 + r*ti.sin(theta)*e2 + h*vn
                    vd = r*ti.cos(theta)*e1 + r*ti.sin(theta)*e2 + v
                    life = self.lifetimeMin + I_ke / self.keMax[None] * ti.random(float) * (self.lifetimeMax-self.lifetimeMin)

                    self.foam_positions.append(xd)
                    # self.foam_velocities.append(vd)
                    # self.foam_lifetime.append(life)
                    # self.foam_type.append(0)

    @ti.kernel
    def removeParticles(self,):
        # deactivate
        self.tmp_positions.deactivate()
        self.tmp_velocities.deactivate()
        self.tmp_lifetime.deactivate()

        # filter new value 
        for p_i in self.foam_lifetime:
            if (self.foam_lifetime[p_i] > self.epsilon):
                self.tmp_positions.append(self.foam_positions[p_i])
                self.tmp_velocities.append(self.foam_velocities[p_i])
                self.tmp_lifetime.append(self.foam_lifetime[p_i])
                # self.foam_type[p_i-removed] = self.foam_type[p_i]
                # self.foam_counter[None] -= removed
                
        
        # deactivate
        self.foam_positions.deactivate()
        self.foam_velocities.deactivate()
        self.foam_lifetime.deactivate()

        # re-fill values
        for p_i in self.tmp_lifetime:
            self.foam_positions.append(self.tmp_positions[p_i])
            self.foam_velocities.append(self.tmp_velocities[p_i])
            self.foam_lifetime.append(self.tmp_lifetime[p_i])


    @ti.kernel
    def compute_potential(self,):
        for p_i in self.positions:
            pos_i = self.positions[p_i]
            vel_i = self.velocities[p_i]
            ni = self.normals[p_i]

            # init potential terms
            self.v_diff[p_i] = 0.0
            self.curvature[p_i] = 0.0
            self.energy[p_i] = 0.0

            for j in range(self.particle_num_neighbors[p_i]):
                p_j = self.particle_neighbors[p_i, j]
                if p_j < 0:
                    break
                pos_ji = pos_i - self.positions[p_j]
                vel_ji = vel_i - self.velocities[p_j]
                nj = self.normals[p_j]

                mag_pos = pos_ji.norm()
                mag_vel = vel_ji.norm()

                npos_ji = pos_ji.normalized() if mag_pos > self.epsilon else pos_ji
                nvel_ji = vel_ji.normalized() if mag_vel > self.epsilon else vel_ji

                # radially symmetric weight
                Wrs = self.foam_W(mag_pos)

                # Trapped Air Potential
                self.v_diff[p_i] += mag_vel * (1 - nvel_ji.dot(npos_ji)) * Wrs

                # Wave Crest Curvature
                if (-npos_ji.dot(ni) < 0):
                    self.curvature[p_i] += (1 - ni.dot(nj)) * Wrs

            delta = 0.0
            nvel_i = vel_i.normalized()
            if (nvel_i.dot(ni) >= 0.6):
                delta = 1.0
            self.curvature[p_i] *= delta

            # Kninetic Energy
            self.energy[p_i] = 0.5 * self.mass * pow(vel_i.norm(), 2)

        self.update_limits()

        # FOR LOG: compute the sum and max for all vaues
        # sum_vdiff = ti.sum(self.v_diff)
        # sum_curvature = ti.sum(self.curvature)
        # sum_energy = ti.sum(self.energy)
        
        # max_vdiff = ti.max(self.v_diff)
        # max_curvature = ti.max(self.curvature)
        # max_energy = ti.max(self.energy)
     

    def run(self,):
        self.compute_density()
        self.compute_normal()
        self.compute_potential()

        if self.frame_num[None] > 500:
            self.removeParticles()
            self.generateFoamParticles()

# BACKUP: PCA based normal estimation
    # def pca_norm(self, points):
    #     points = [p.to_numpy() for p in points]
    #     pca = PCA(n_components=3)
    #     print("############", points)
    #     pca.fit(points)

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
