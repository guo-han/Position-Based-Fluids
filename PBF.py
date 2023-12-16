import math
import numpy as np
import taichi as ti
from StaticRigidBody import StaticRigidBody
@ti.data_oriented
class Pbf():
    def __init__(self, k) -> None:
        # scale factor
        self.k = k  # control self.boundry, num_particles_xyz and move board velocity strength
        # config fulid grid
        grid_res = (300, 200, 100)
        screen_to_world_ratio = 10.0
        self.boundary = (
            grid_res[0] / screen_to_world_ratio * k,
            grid_res[1] / screen_to_world_ratio * k,
            grid_res[2] / screen_to_world_ratio * k,
        )
        self.cell_size = 2.51
        self.cell_recpr = 1.0 / self.cell_size
        def round_up(f, s):
            return (math.floor(f * self.cell_recpr / s) + 1) * s

        self.grid_size = (round_up(self.boundary[0], 1), round_up(self.boundary[1], 1), round_up(self.boundary[2], 1))

        # config particles
        self.dim = 3 # 3d pbf
        self.num_particles_x = 15 * self.k
        self.num_particles_y = 10 * self.k
        self.num_particles_z = 10 * self.k
        self.num_particles = self.num_particles_x * self.num_particles_y * self.num_particles_z
        self.max_num_particles_per_cell = 100 # 3d
        self.max_num_neighbors = 100 # 3d
        self.time_delta = 1.0 / 60.0
        self.epsilon = 1e-5
        particle_radius = 3.0
        self.particle_radius_in_world = particle_radius / screen_to_world_ratio

        # PBF params
        self.h_ = 1.1
        self.mass = 1.0
        self.rho0 = 1.0
        self.lambda_epsilon = 100.0
        self.pbf_num_iters = 5
        self.corr_deltaQ_coeff = 0.2
        self.corrK = 0.01
        self.g = ti.Vector([0.0, -9.8, 0.0])
        # Need ti.pow()
        # corrN = 4.0
        self.neighbor_radius = self.h_ * 1.05

        self.poly6_factor = 315.0 / 64.0 / math.pi
        self.spiky_grad_factor = -45.0 / math.pi

        self.old_positions = ti.Vector.field(self.dim, float)
        self.positions = ti.Vector.field(self.dim, float)
        self.velocities = ti.Vector.field(self.dim, float)
        self.omegas = ti.Vector.field(self.dim, float)
        self.vorticity_forces = ti.Vector.field(self.dim, float)
        self.velocities_deltas = ti.Vector.field(self.dim, float)
        self.density = ti.field(float)
        self.grid_num_particles = ti.field(int)
        self.grid2particles = ti.field(int)
        self.particle_num_neighbors = ti.field(int)
        self.particle_neighbors = ti.field(int)
        self.lambdas = ti.field(float)
        self.position_deltas = ti.Vector.field(self.dim, float)
        # 0: x-pos, 1: timestep in sin()
        self.board_states = ti.Vector.field(2, float)

        self.rb_fp_collision_stiffness = 500    # TODO: check this setting
        # self.rb_fp_collision_sdf_lower_bound = -0.1
        self.forces = ti.Vector.field(self.dim, float)
        self.rb = None
        self.rb_particle_collision_set = ti.Vector.field(self.dim, float)
        self.rb_particle_collision_idx_set = ti.field(int)
        self.rb_particle_collision_num = ti.field(int)
        self.confirmed_rb_particle_collision_num = ti.field(int)
        self.sdf_negative_indices = ti.field(int)
        self.sdf_negatives = ti.field(float)
        self.primitive_indices = ti.field(int)
        self.particle_colors = ti.Vector.field(3, float)

        ti.root.dense(ti.i, self.num_particles).place(self.old_positions, self.positions, self.velocities, self.omegas, self.vorticity_forces, \
                                                      self.density, self.forces, self.rb_particle_collision_set, self.rb_particle_collision_idx_set, self.particle_colors,\
                                                      self.sdf_negative_indices, self.sdf_negatives, self.primitive_indices)
        grid_snode = ti.root.dense(ti.ijk, self.grid_size)
        grid_snode.place(self.grid_num_particles)
        grid_snode.dense(ti.l, self.max_num_particles_per_cell).place(self.grid2particles)
        nb_node = ti.root.dense(ti.i, self.num_particles)
        nb_node.place(self.particle_num_neighbors)
        nb_node.dense(ti.j, self.max_num_neighbors).place(self.particle_neighbors)
        ti.root.dense(ti.i, self.num_particles).place(self.lambdas, self.position_deltas, self.velocities_deltas)
        ti.root.place(self.board_states, self.rb_particle_collision_num, self.confirmed_rb_particle_collision_num)

        self.rb_particle_collision_num[None] = 0
        self.confirmed_rb_particle_collision_num[None] = 0

        self.colors = np.tile(
            np.array([6.0/255.0,133.0/255.0,135.0/255.0], dtype=np.float32), (self.num_particles, 1)
        )

        self.reset_color()

    def set_rigid_body(self, rb):
        self.rb = rb

    def reset_color(self):
        self.particle_colors.from_numpy(self.colors)

    @ti.kernel
    def init_particles(self):
        for i in range(self.num_particles):
            x = i % self.num_particles_x
            y = i // (self.num_particles_x*self.num_particles_z)
            z = (i % (self.num_particles_x*self.num_particles_z)) // self.num_particles_x
            delta = self.h_ * 0.8
            offs = ti.Vector([(self.boundary[0] - delta * self.num_particles_x) * 0.95, 0, self.boundary[2] * 0.02]) # self.boundary[1] * 0.02
            self.positions[i] = ti.Vector([x, z, y]) * delta + offs
            for c in ti.static(range(self.dim)):
                self.velocities[i][c] = (ti.random() - 0.5) * 4
        self.board_states[None] = ti.Vector([self.boundary[0] - self.epsilon, -0.0])


    @ti.kernel
    def move_board(self):
        # probably more accurate to exert force on particles according to hooke's law.
        b = self.board_states[None]
        b[1] += 1.0
        period = 250
        vel_strength = 8.0 + 1*self.k
        if b[1] >= 2 * period:
            b[1] = 0
        b[0] += -ti.sin(b[1] * np.pi / period) * vel_strength * self.time_delta
        self.board_states[None] = b

    def run_pbf(self):
        self.prologue()
        for _ in range(self.pbf_num_iters):
            self.substep()
        self.epilogue()

    @ti.func
    def poly6_value(self, s, h):
        result = 0.0
        if 0 < s and s < h:
            x = (h * h - s * s) / (h * h * h)
            result = self.poly6_factor * x * x * x
        return result


    @ti.func
    def spiky_gradient(self, r, h):
        result = ti.Vector([0.0, 0.0, 0.0])
        r_len = r.norm()
        if 0 < r_len and r_len < h:
            x = (h - r_len) / (h * h * h)
            g_factor = self.spiky_grad_factor * x * x
            result = r * g_factor / r_len
        return result


    @ti.func
    def compute_scorr(self, pos_ji):
        # Eq (13)
        x = self.poly6_value(pos_ji.norm(), self.h_) / self.poly6_value(self.corr_deltaQ_coeff * self.h_, self.h_)
        # pow(x, 4)
        x = x * x
        x = x * x
        return (-self.corrK) * x

    @ti.func
    def get_cell(self, pos):
        return int(pos * self.cell_recpr)

    @ti.func
    def is_in_grid(self, c):
        # @c: Vector(i32)
        return 0 <= c[0] and c[0] < self.grid_size[0] and 0 <= c[1] and c[1] < self.grid_size[1] and 0 <= c[2] and c[2] < self.grid_size[2]

    @ti.func
    def confine_position_to_boundary(self, p):
        bmin = self.particle_radius_in_world
        bmax = ti.Vector([self.board_states[None][0], self.boundary[1], self.boundary[2]]) - self.particle_radius_in_world
        for i in ti.static(range(self.dim)):
            # Use randomness to prevent particles from sticking into each other after clamping
            if p[i] <= bmin:
                p[i] = bmin + self.epsilon * ti.random()
            elif bmax[i] <= p[i]:
                p[i] = bmax[i] - self.epsilon * ti.random()
        return p
    
    @ti.func
    def compute_density(self):
        for p_i in self.positions:
            pos_i = self.positions[p_i]
            density_constraint = 0.0

            for j in range(self.particle_num_neighbors[p_i]):
                p_j = self.particle_neighbors[p_i, j]
                if p_j < 0:
                    break
                pos_ji = pos_i - self.positions[p_j]
                # Eq(2)
                density_constraint += self.poly6_value(pos_ji.norm(), self.h_)  # mass in Eq(2) is moved to Eq(1)

            # Eq(1)
            density_constraint += self.poly6_value(0, self.h_)  # self contribution
            self.density[p_i] = density_constraint * self.mass

    @ti.func
    def clear_forces(self):
        for i in self.forces:
            self.forces[i] *= 0.0
    
    @ti.func
    def collect_set_of_potential_collided_particles_ti_v(self, grid_idx: ti.template(), accumulated_count: ti.int32):
        for idx in range(self.grid_num_particles[grid_idx]):
            p_idx = self.grid2particles[grid_idx[0], grid_idx[1], grid_idx[2], idx]
            p_pos = self.positions[p_idx]
            self.rb_particle_collision_set[idx + accumulated_count] = p_pos
            self.rb_particle_collision_idx_set[idx + accumulated_count] = p_idx

    @ti.kernel
    def update_grid(self):
        # update grid
        for I in ti.grouped(self.grid_num_particles):
            self.grid_num_particles[I] = 0
        for p_i in self.positions:
            cell = self.get_cell(self.positions[p_i])
            # ti.Vector doesn't seem to support unpacking yet
            # but we can directly use int Vectors as indices
            offs = ti.atomic_add(self.grid_num_particles[cell], 1)
            self.grid2particles[cell, offs] = p_i

    @ti.kernel
    def collect_set_of_potential_collided_particles(self):
        self.rb_particle_collision_num[None] = 0
        counter = 0
        for _ in range(1):    # To serialize loop 
            for I in range(self.rb.grid_AABB.shape[0]): 
                grid_idx = self.rb.grid_AABB[I]
                self.collect_set_of_potential_collided_particles_ti_v(grid_idx, counter)
                grid_num = self.grid_num_particles[grid_idx[0], grid_idx[1], grid_idx[2]]
                counter += grid_num
        self.rb_particle_collision_num[None] = counter
    
    @ti.kernel
    def apply_colision_forces_after_collision_detect(self):  
        for i in range(self.confirmed_rb_particle_collision_num[None]):
            idx = self.sdf_negative_indices[i]
            p_idx = self.rb_particle_collision_idx_set[idx]
            self.particle_colors[p_idx] = [0.0, 1.0, 0.0]   # TODO: change the specification format latter
            dis_values = self.sdf_negatives[i]
            # if dis_values < self.rb_fp_collision_sdf_lower_bound:
                # dis_values = self.rb_fp_collision_sdf_lower_bound
            collision_force = self.rb_fp_collision_stiffness * (- dis_values) * self.rb.faceN[self.primitive_indices[i]]
            self.forces[p_idx] += collision_force # TODO: pos or collision point???????
    
    @ti.kernel
    def color_potential_particles(self):
        for i in range(self.rb_particle_collision_num[None]):
            p_idx = self.rb_particle_collision_idx_set[i]
            self.particle_colors[p_idx] = [0.0, 0.0, 0.0]

    def add_fluid_rb_collision_forces(self):
        self.collect_set_of_potential_collided_particles()
        # Visualization, TODO: comment later
        self.reset_color()
        self.color_potential_particles()
        if self.rb_particle_collision_num[None] == 0:
            return
        potential_positions = self.rb_particle_collision_set.to_numpy()[:self.rb_particle_collision_num[None]]
        sdfs, primitive_indices = self.rb.get_sdf_prims_o3d(potential_positions)
        self.confirmed_rb_particle_collision_num[None] = np.count_nonzero(sdfs < 0)
        if self.confirmed_rb_particle_collision_num[None] == 0:
            return
        
        sdf_negative_indices_np = np.zeros(shape = (self.num_particles,), dtype = int)
        sdf_negative_indices_np[:self.confirmed_rb_particle_collision_num[None]] = np.where(sdfs < 0)[0]
        self.sdf_negative_indices.from_numpy(sdf_negative_indices_np)

        sdf_negative_np = np.zeros(shape = (self.num_particles, ), dtype = np.float32)
        sdf_negative_np[:self.confirmed_rb_particle_collision_num[None]] = sdfs[np.where(sdfs < 0)]
        # np.clip(sdf_negative_np, self.rb_fp_collision_sdf_lower_bound, 0, out=sdf_negative_np)
        self.sdf_negatives.from_numpy(sdf_negative_np)

        primitive_indices_np = np.zeros(shape = (self.num_particles, ), dtype = int)
        primitive_indices_np[:self.confirmed_rb_particle_collision_num[None]] = primitive_indices[np.where(sdfs < 0)]
        self.primitive_indices.from_numpy(primitive_indices_np)
        
        # print(np.min(sdf_negative_np[:self.confirmed_rb_particle_collision_num[None]]), np.max(sdf_negative_np[:self.confirmed_rb_particle_collision_num[None]]))
        self.apply_colision_forces_after_collision_detect()

    @ti.kernel
    def prologue_part1(self):
        # save old positions
        for i in self.positions:
            self.old_positions[i] = self.positions[i]
        # apply external forces to fluid
        self.clear_forces()
        # apply gravity within boundary
        G = self.mass * self.g
        for i in self.positions:
            self.forces[i] += G

    @ti.kernel
    def prologue_part2(self):
        # apply external forces, update velocities and positions
        for i in self.velocities:
            self.velocities[i] += self.forces[i] / self.mass * self.time_delta
            self.positions[i] += self.velocities[i] * self.time_delta
            self.positions[i] = self.confine_position_to_boundary(self.positions[i])
        # TODO: if consider dynamiics 跳掉了add boundary collision impulses，需不需要confine rigid body positions to boundary
        # clear neighbor lookup table
        for I in ti.grouped(self.grid_num_particles):
            self.grid_num_particles[I] = 0
        for I in ti.grouped(self.particle_neighbors):
            self.particle_neighbors[I] = -1

        # update grid
        for p_i in self.positions:
            cell = self.get_cell(self.positions[p_i])
            # ti.Vector doesn't seem to support unpacking yet
            # but we can directly use int Vectors as indices
            offs = ti.atomic_add(self.grid_num_particles[cell], 1)
            self.grid2particles[cell, offs] = p_i
        # find particle neighbors
        for p_i in self.positions:
            pos_i = self.positions[p_i]
            cell = self.get_cell(pos_i)
            nb_i = 0
            for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
                cell_to_check = cell + offs
                if self.is_in_grid(cell_to_check):
                    for j in range(self.grid_num_particles[cell_to_check]):
                        p_j = self.grid2particles[cell_to_check, j]
                        if nb_i < self.max_num_neighbors and p_j != p_i and (pos_i - self.positions[p_j]).norm() < self.neighbor_radius:
                            self.particle_neighbors[p_i, nb_i] = p_j
                            nb_i += 1
            self.particle_num_neighbors[p_i] = nb_i

    ## add visualization for 
    def prologue(self):
        self.prologue_part1()
        # TODO: add update grid here?
        if self.rb != None:
            self.update_grid()
            self.add_fluid_rb_collision_forces()
        self.prologue_part2()

    @ti.kernel
    def substep(self,):
        # compute lambdas
        # Eq (8) ~ (11)
        for p_i in self.positions:
            pos_i = self.positions[p_i]

            grad_i = ti.Vector([0.0, 0.0, 0.0])
            sum_gradient_sqr = 0.0
            density_constraint = 0.0

            for j in range(self.particle_num_neighbors[p_i]):
                p_j = self.particle_neighbors[p_i, j]
                if p_j < 0:
                    break
                pos_ji = pos_i - self.positions[p_j]
                grad_j = self.spiky_gradient(pos_ji, self.h_)
                grad_i += grad_j
                sum_gradient_sqr += grad_j.dot(grad_j)
                # Eq(2)
                density_constraint += self.poly6_value(pos_ji.norm(), self.h_)

            # Eq(1)
            density_constraint = (self.mass * density_constraint / self.rho0) - 1.0

            sum_gradient_sqr += grad_i.dot(grad_i)
            self.lambdas[p_i] = (-density_constraint) / (sum_gradient_sqr + self.lambda_epsilon)
        # compute position deltas
        # Eq(12), (14)
        for p_i in self.positions:
            pos_i = self.positions[p_i]
            lambda_i = self.lambdas[p_i]

            pos_delta_i = ti.Vector([0.0, 0.0, 0.0])
            for j in range(self.particle_num_neighbors[p_i]):
                p_j = self.particle_neighbors[p_i, j]
                if p_j < 0:
                    break
                lambda_j = self.lambdas[p_j]
                pos_ji = pos_i - self.positions[p_j]
                scorr_ij = self.compute_scorr(pos_ji)
                pos_delta_i += (lambda_i + lambda_j + scorr_ij) * self.spiky_gradient(pos_ji, self.h_)

            pos_delta_i /= self.rho0
            self.position_deltas[p_i] = pos_delta_i
        # apply position deltas
        for i in self.positions:
            self.positions[i] += self.position_deltas[i]


    @ti.kernel
    def epilogue(self):
        # confine to boundary
        for i in self.positions:
            pos = self.positions[i]
            self.positions[i] = self.confine_position_to_boundary(pos)
        # update velocities
        for i in self.positions:
            self.velocities[i] = (self.positions[i] - self.old_positions[i]) / self.time_delta

        # calculate density first
        self.compute_density()

        # calculate vorticity
        for i in self.positions:
            pos_i = self.positions[i]
            self.omegas[i] = 0.0
            for j in range(self.particle_num_neighbors[i]):
                p_j = self.particle_neighbors[i, j]
                if p_j < 0:
                    break
                pos_ji = pos_i - self.positions[p_j]
                self.omegas[i] += self.mass * (self.velocities[p_j] - self.velocities[i]).cross(self.spiky_gradient(pos_ji, self.h_)) / (self.epsilon + self.density[p_j])
        # calculate vorticity force
        for i in self.positions:
            pos_i = self.positions[i]
            eta = pos_i * 0.0
            self.vorticity_forces[i] = 0.0
            for j in range(self.particle_num_neighbors[i]):
                p_j = self.particle_neighbors[i, j]
                if p_j < 0:
                    break
                pos_ji = pos_i - self.positions[p_j]
                eta += self.mass * self.omegas[j].norm() * self.spiky_gradient(pos_ji, self.h_) / (self.epsilon + self.density[p_j])
            location_vector = eta / (self.epsilon + eta.norm())
            self.vorticity_forces[i] += 0.5 * (location_vector.cross(self.omegas[i]))

        # apply vorticity force
        for i in self.positions:
            self.velocities[i] += (self.vorticity_forces[i] / self.mass) * self.time_delta

        # add viscosity
        for i in self.positions:
            pos_i = self.positions[i]
            self.velocities_deltas[i] = 0.0
            for j in range(self.particle_num_neighbors[i]):
                p_j = self.particle_neighbors[i, j]
                if p_j < 0:
                    break
                pos_ji = pos_i - self.positions[p_j]
                self.velocities_deltas[i] += self.mass * (self.velocities[p_j] - self.velocities[i]) * self.poly6_value(pos_ji.norm(), self.h_) / (self.epsilon + self.density[p_j])
        
        for i in self.positions:
            self.velocities[i] += 0.1 * self.velocities_deltas[i]   # note: 0.1 is xsph constant, 0.01 in the paper