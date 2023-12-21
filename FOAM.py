import math
import taichi as ti

@ti.func
def clamp(x, tmin, tmax):
    return (min(x, tmax) - min(x, tmin)) / (tmax - tmin)

## ATTENTION: only allow dimension equal to 3 situation
vec3 = ti.math.vec3
@ti.dataclass
class FoamParticle:
    position: vec3
    velocity: vec3
    lifetime: float
    typei: int

    @ti.func
    def set_position(self, pos: vec3):
        self.position = pos
        return
    
    @ti.func
    def set_velocity(self, vel: vec3):
        self.velocity = vel
        return

    @ti.func
    def set_lifetime(self, lt: ti.f32):
        self.lifetime = lt
        return
    
    @ti.func
    def set_type(self, typei = ti.i32):
        self.typei = typei
        return

    @ti.func
    def set_pv(self, pos: vec3, vel: vec3):
        self.set_position(pos)
        self.set_velocity(vel)
        return
    
    @ti.func
    def set_pvl(self, pos: vec3, vel: vec3, lt: float):
        self.set_position(pos)
        self.set_velocity(vel)
        self.set_lifetime(lt)
        return
    
    @ti.func
    def set_pvlt(self, pos: vec3, vel: vec3, lt: float, typei: int):
        self.set_position(pos)
        self.set_velocity(vel)
        self.set_lifetime(lt)
        self.set_type(typei)
        return

    @ti.func
    def get_position(self) -> vec3:
        return self.position
    
    @ti.func
    def get_velocity(self) -> vec3:
        return self.velocity
    
    @ti.func
    def get_lifetime(self) -> ti.f32:
        return self.lifetime
    
    @ti.func
    def get_type(self) -> ti.i32:
        return self.typei
    
    @ti.func
    def get_pvlt(self):
        return self.get_position(), self.get_velocity(), self.get_lifetime(), self.get_type()
    
    @ti.func
    def clear(self):
        self.position *= 0
        self.velocity *= 0
        self.lifetime *= 0
        return
    
    @ti.func
    def init(self):
        self.position = vec3(0, 0, 0)
        self.velocity = vec3(0, 0, 0)
        self.lifetime = 0
        self.typei = -1
        return

@ti.data_oriented
class Foam():
    def __init__(self, fluid) -> None:
        # inherit from fluid
        self.fluid = fluid
        self.dim = fluid.dim # should always be 3
        self.r_ = fluid.particle_radius_in_world
        self.h_ = fluid.h_
        self.h3 = pow(self.h_, 3)
        self.invPI = 1 / math.pi
        self.rho0 = fluid.rho0
        self.mass = 0.8
        num_particles = fluid.num_particles
        self.old_positions = fluid.old_positions
        self.positions = fluid.positions
        self.velocities = fluid.velocities
        self.particle_num_neighbors = fluid.particle_num_neighbors
        self.particle_neighbors = fluid.particle_neighbors
        self.epsilon = 1.0e-9
        self.g = fluid.g
        self.timeStepSize = fluid.time_delta
        
        # foam
        self.m_k = 8.0 / (math.pi * self.h3)
        self.m_l = 48.0 / (math.pi * self.h3)
        self.inertia = 2.0
        self.foam_scale = 1
        self.k_ta = 1
        self.k_wc = 1
        self.k_vo = 1
        self.n = 1000 # controlling the maximum num of foams that can be generated each frame
        self.k_buoyancy = 2.0
        self.k_drag = 0.8
        self.lifetimeMin = 2.0
        self.lifetimeMax = 5.0
        self.densities = ti.field(float)
        self.omegas = ti.Vector.field(self.dim, float)
        self.normals = ti.Vector.field(self.dim, float)
        ti.root.dense(ti.i, num_particles).place(self.densities, self.omegas, self.normals)
        self.max_foam_per_particle = 3
        self.particle_to_foam_grid = FoamParticle.field(shape = (num_particles, self.max_foam_per_particle))
        self.all_foam_pos = ti.Vector.field(self.dim, float, shape = (num_particles * self.max_foam_per_particle))
        self.particle_to_foam_counter = ti.field(int, shape = (num_particles, ))
        # set all initial values to zero, taichi does not support default values of dataclass, doubt whether it is reasonable
        self.init()   # TODO: check whether it is reasonable
        self.particle_to_foam_to_neighbor_grid = ti.field(int, shape = (num_particles, self.max_foam_per_particle, self.fluid.max_num_neighbors))
        self.particle_to_foam_to_neighbor_count = ti.field(int, shape = (num_particles, self.max_foam_per_particle, ))      
        #  #############################################
        #  foam_type: spray -> 0, foam -> 1, bubbles -> 2
        #  #############################################

        # potentials
        self.v_diff = ti.field(float)
        self.curvature = ti.field(float)
        self.omega_diff = ti.field(float)
        self.energy = ti.field(float)
        self.white_particles = ti.Vector.field(self.dim, float, shape = (num_particles * self.max_foam_per_particle))
        self.red_particles = ti.Vector.field(self.dim, float, shape = (num_particles * self.max_foam_per_particle))
        self.green_particles = ti.Vector.field(self.dim, float, shape = (num_particles * self.max_foam_per_particle))
        self.yellow_particles = ti.Vector.field(self.dim, float, shape = (num_particles * self.max_foam_per_particle))
        ti.root.dense(ti.i, num_particles).place(self.v_diff, self.curvature, self.omega_diff, self.energy)

        # neighbors
        self.neighbor_radius = fluid.neighbor_radius
        self.max_num_neighbors = fluid.max_num_neighbors
        self.grid_num_particles = fluid.grid_num_particles
        self.grid2particles = fluid.grid2particles

        # accumulate values
        self.frame_num = ti.field(ti.i32, shape=())
        self.sum_max_vdiff = ti.field(ti.f32, shape=())
        self.sum_max_curvature = ti.field(ti.f32, shape=())
        self.sum_max_omega = ti.field(ti.f32, shape=())
        self.sum_max_energy = ti.field(ti.f32, shape=())
        self.taMax = ti.field(ti.f32, shape=())
        self.taMin = ti.field(ti.f32, shape=())
        self.wcMax = ti.field(ti.f32, shape=())
        self.wcMin = ti.field(ti.f32, shape=())
        self.voMax = ti.field(ti.f32, shape=())
        self.voMin = ti.field(ti.f32, shape=())
        self.keMax = ti.field(ti.f32, shape=())
        self.keMin = ti.field(ti.f32, shape=())
        self.foam_counter = ti.field(ti.i32, shape=()) # total foam

    @ti.kernel
    def init(self):
        for pi in ti.grouped(self.particle_to_foam_grid):
            self.particle_to_foam_grid[pi].init()

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
                density_acc += self.cubic_W(pos_ji.norm())

            self.densities[p_i] = self.mass * density_acc
            
    @ti.kernel
    def compute_normal(self, ):
        for p_i in self.positions:
            pos_i = self.positions[p_i]
            ni = ti.Vector([0., 0., 0.])
            self.yellow_particles[p_i] = ti.Vector([0., 0., 0.])

            # only interested in surface particles
            if self.particle_num_neighbors[p_i] > 16:
                continue

            self.yellow_particles[p_i] = pos_i

            for j in range(self.particle_num_neighbors[p_i]):
                p_j = self.particle_neighbors[p_i, j]
                if p_j < 0:
                    break
                density_j = self.densities[p_j]
                pos_ji = pos_i - self.positions[p_j]
                ni -= self.mass / density_j * self.cubic_gradW(pos_ji)

            self.normals[p_i] = ni.normalized() if ni.norm() > self.epsilon else ni
    
    @ti.kernel
    def compute_omega(self,):
        for p_i in self.positions:
            pos_i = self.positions[p_i]
            vel_i = self.velocities[p_i]
            di = self.densities[p_i]
            omega_i = ti.Vector([0., 0., 0.])

            for j in range(self.particle_num_neighbors[p_i]):
                p_j = self.particle_neighbors[p_i, j]
                if p_j < 0:
                    break
                
                pos_ji = pos_i - self.positions[p_j]
                vel_ji = vel_i - self.velocities[p_j]

                omega_i -= self.mass / di * vel_ji.cross(self.cubic_gradW(pos_ji))

            self.omegas[p_i] = omega_i


    @ti.func
    def foam_W(self, r):
        res = 0.0
        q = r / self.h_
        if (q <= 1.0):
            res = 1.0 - q
        return res * 3 * self.invPI / self.h3
        
    @ti.func
    def confine_position_to_boundary(self, p, v):
        bmin = self.r_
        bmax = ti.Vector([self.fluid.board_states[None][0], self.fluid.boundary[1], self.fluid.boundary[2]]) - self.r_
        for i in ti.static(range(self.dim)):  
            if p[i] <= bmin:
                p[i] = bmin + self.epsilon * ti.random()
                # v[i] *= -1
            elif bmax[i] <= p[i]:
                # p[i] = bmax[i] - self.epsilon * ti.random()
                p = ti.Vector([0., 0., 0.])
                v = ti.Vector([0., 0., 0.])
        return p, v

    @ti.func
    def update_limits(self,):
        # followed paper: Bender et al., "Turbulent Micropolar SPH Fluids with Foam", 2018 
        max_v, max_c, max_o, max_e = -math.inf, -math.inf, -math.inf, -math.inf
        for p_i in self.positions:
            ti.atomic_max(max_v, self.v_diff[p_i])
            ti.atomic_max(max_c, self.curvature[p_i])
            ti.atomic_max(max_o, self.omega_diff[p_i])
            ti.atomic_max(max_e, self.energy[p_i])

        ti.atomic_add(self.sum_max_vdiff[None], max_v)
        ti.atomic_add(self.sum_max_curvature[None], max_c)
        ti.atomic_add(self.sum_max_omega[None], max_o)
        ti.atomic_add(self.sum_max_energy[None], max_e)
        ti.atomic_add(self.frame_num[None], 1)

        # compute limits
        # NOTE: Here we fixed the limit values for a stable result in our bunny scene.
        self.taMax[None] = 28 # self.sum_max_vdiff[None] / self.frame_num[None]
        self.taMin[None] = 0.1 * self.taMax[None]
        self.wcMax[None] = 3.6 # self.sum_max_curvature[None] / self.frame_num[None]
        self.wcMin[None] = 0.1 * self.wcMax[None]
        self.voMax[None] = 130 # self.sum_max_omega[None] / self.frame_num[None]
        self.voMin[None] = 0.1 * self.voMax[None]
        self.keMax[None] = 2900 # self.sum_max_energy[None] / self.frame_num[None]
        self.keMin[None] = 0.1 * self.keMax[None]

        # DEBUG log:
        # if (self.frame_num[None] % 100 == 99):
        #     print(f"Trapped Air: min({self.taMin[None]}) max({self.taMax[None]})")
        #     print(f"Wave Crest: min({self.wcMin[None]}) max({self.wcMax[None]})")
        #     print(f"Vorticity: min({self.voMin[None]}) max({self.voMax[None]})")
        #     print(f"Kinematic Energy: min({self.keMin[None]}) max({self.keMax[None]})")

    @ti.func
    def getOrthogonalVectors(self, vec):
        tmp = ti.Vector([1., 0., 0.])

        # Check, if v has same direction as vec
        if ((tmp.dot(vec)) > 1-self.epsilon):
            tmp = ti.Vector([0., 1., 0.])

        e1 = vec.cross(tmp)
        e2 = vec.cross(e1)
        e1 = e1.normalized() if e1.norm() > self.epsilon else e1
        e2 = e2.normalized() if e2.norm() > self.epsilon else e2
        return e1, e2

    @ti.kernel
    def generateFoam(self,):
        for idx in self.positions:
            I_ta = clamp(self.v_diff[idx], self.taMin[None], self.taMax[None])
            I_wc = clamp(self.curvature[idx], self.wcMin[None], self.wcMax[None])
            I_vo = clamp(self.omega_diff[idx], self.voMin[None], self.voMax[None])
            I_ke = clamp(self.energy[idx], self.keMin[None], self.keMax[None])
            num = int(max(self.foam_scale * I_ke * (self.k_ta*I_ta + self.k_wc*I_wc + self.k_vo*I_vo) * self.timeStepSize * self.n + 0.5, 0.0))
            # nt = int(self.foam_scale * I_ke * self.k_ta * I_ta * self.timeStepSize + 0.5)
            # nw = int(self.foam_scale * I_ke * self.k_wc * I_wc * self.timeStepSize + 0.5)
            p = self.positions[idx]
            v = self.velocities[idx]
            vn = v.normalized() if v.norm() > self.epsilon else v
            e1, e2 = self.getOrthogonalVectors(vn)

            e1 *= self.r_
            e2 *= self.r_
            
            for i in range(num):
                if self.particle_to_foam_counter[idx] >= self.max_foam_per_particle:
                    break
                
                Xr, Xt, Xh = ti.random(float), ti.random(float), ti.random(float)
                r = self.r_ * ti.sqrt(Xr)
                theta = 2*math.pi*Xt
                h = self.timeStepSize * (Xh - 0.5) * v.norm()

                xd = p + r*ti.cos(theta)*e1 + r*ti.sin(theta)*e2 + h*vn
                vd = r*ti.cos(theta)*e1 + r*ti.sin(theta)*e2 + v
                life = self.lifetimeMin + I_ke / self.keMax[None] * ti.random(float) * (self.lifetimeMax-self.lifetimeMin)
                # life = 0
                xd, vd = self.confine_position_to_boundary(xd, vd)
                self.particle_to_foam_grid[idx, self.particle_to_foam_counter[idx]].set_pvlt(xd, vd, life, 1) 
                ti.atomic_add(self.particle_to_foam_counter[idx], 1)

    @ti.kernel
    def removeFoam(self,):
        # filter lifetime value
        for iidx in self.positions:
            local_counter = 0
            for jidx in range(self.particle_to_foam_counter[iidx]):
                local_particle = self.particle_to_foam_grid[iidx, jidx]
                if local_particle.get_type() == 1:  # check whether it is foam/bubble
                    self.particle_to_foam_grid[iidx, jidx].lifetime -= self.timeStepSize
                # if local_particle.get_type() == 2:  # check whether it is foam/bubble
                #     self.particle_to_foam_grid[iidx, jidx].lifetime -= 0.5*self.timeStepSize
                if self.particle_to_foam_grid[iidx, jidx].lifetime > self.epsilon:
                    localp, localv, locall, localt = self.particle_to_foam_grid[iidx, jidx].get_pvlt()
                    self.particle_to_foam_grid[iidx, local_counter].set_pvlt(localp, localv, locall, localt)
                    if jidx > local_counter:
                        self.particle_to_foam_grid[iidx, jidx].init()
                    ti.atomic_add(local_counter, 1)
                else:
                    self.particle_to_foam_grid[iidx, jidx].init()

            self.particle_to_foam_counter[iidx] = local_counter

    @ti.kernel
    def compute_potential(self,):
        for p_i in self.positions:
            pos_i = self.positions[p_i]
            vel_i = self.velocities[p_i]
            ni = self.normals[p_i]

            # init potential terms
            self.v_diff[p_i] = 0.0
            self.curvature[p_i] = 0.0
            self.omega_diff[p_i] = 0.0
            self.energy[p_i] = 0.0
            factor = self.mass / self.densities[p_i]

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
                self.v_diff[p_i] += mag_vel * (1 - nvel_ji.dot(npos_ji)) * Wrs * factor

                # Wave Crest Curvature
                if (-npos_ji.dot(ni) < 0):
                    self.curvature[p_i] += (1 - ni.dot(nj)) * Wrs * factor

                # vorticity
                self.omega_diff[p_i] += (self.omegas[p_i] - self.omegas[p_j]).norm() * Wrs * factor

            delta = 0.0
            nvel_i = vel_i.normalized() if vel_i.norm() > self.epsilon else vel_i
            if (nvel_i.dot(ni) >= 0.6):
                delta = 1.0
            self.curvature[p_i] *= delta

            # Kninetic Energy
            self.energy[p_i] = 0.5 * self.mass * pow(vel_i.norm(), 2) + 0.5*self.inertia*pow(self.omegas[p_i].norm(),2)

        self.update_limits()

        # DEBUG log: compute the sum and max for all vaues
        # sum_vdiff = ti.sum(self.v_diff)
        # sum_curvature = ti.sum(self.curvature)
        # sum_energy = ti.sum(self.energy)
        
        # max_vdiff = ti.max(self.v_diff)
        # max_curvature = ti.max(self.curvature)
        # max_energy = ti.max(self.energy)
     
    @ti.func
    def find_neighbors(self,):
        # clear neighbor lookup table
        for I in ti.grouped(self.particle_to_foam_to_neighbor_grid):
            self.particle_to_foam_to_neighbor_grid[I] = -1

        for iidx in self.positions:
            for jidx in range(self.particle_to_foam_counter[iidx]):
                pos_ij = self.particle_to_foam_grid[iidx, jidx].get_position()
                cell = self.fluid.get_cell(pos_ij)
                nb_i = 0
                for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
                    cell_to_check = cell + offs
                    if self.fluid.is_in_grid(cell_to_check):
                        for j in range(self.grid_num_particles[cell_to_check]):
                            p_j = self.grid2particles[cell_to_check, j]
                            if nb_i < self.max_num_neighbors and (pos_ij - self.positions[p_j]).norm() < self.neighbor_radius:
                                self.particle_to_foam_to_neighbor_grid[iidx, jidx, nb_i] = p_j
                                nb_i += 1
                self.particle_to_foam_to_neighbor_count[iidx, jidx] = nb_i
                # foam
                self.particle_to_foam_grid[iidx, jidx].set_type(1)
                # spray
                if nb_i < 6: self.particle_to_foam_grid[iidx, jidx].set_type(0)
                # bubble
                if nb_i > 16: self.particle_to_foam_grid[iidx, jidx].set_type(2)

    @ti.kernel
    def advectFoam(self,):
        self.find_neighbors()
        for iidx in self.positions:
            for jidx in range(self.particle_to_foam_counter[iidx]):
                # self.find_
                type = int(self.particle_to_foam_grid[iidx, jidx].get_type())
                pos_ij = self.particle_to_foam_grid[iidx, jidx].get_position()
                vel_ij = self.particle_to_foam_grid[iidx, jidx].get_velocity()
                
                if (type == 0): # spray
                    vel_ij += self.timeStepSize * self.g
                    pos_ij += self.timeStepSize * vel_ij
                elif (type == 1) or (type == 2): # foam / bubbles
                    vf = ti.Vector([0.0, 0.0, 0.0])
                    sumK = 0.0
                    for j in range(self.particle_to_foam_to_neighbor_count[iidx, jidx]):
                        p_j = self.particle_to_foam_to_neighbor_grid[iidx, jidx, j]
                        if p_j < 0:
                            break
                        pos_ji = pos_ij - self.positions[p_j]
                        vel_j = self.velocities[p_j]
                        K = self.cubic_W(pos_ji.norm())

                        vf += vel_j * K
                        sumK += K
                    vf = vf / sumK

                    if (type == 1):
                        pos_ij += self.timeStepSize * vf
                    elif (type == 2):
                        vel_ij += self.k_drag*(vf-vel_ij) - self.timeStepSize*self.k_buoyancy*self.g + ti.Vector([0., 2., 0.])
                        pos_ij += self.timeStepSize * vel_ij
                
                pos_ij, vel_ij = self.confine_position_to_boundary(pos_ij, vel_ij)
                self.particle_to_foam_grid[iidx, jidx].set_pv(pos_ij, vel_ij)

    @ti.kernel
    def draw_classifiedFrom(self,):
        for iidx in self.positions:
            for jidx in range(self.max_foam_per_particle):    # self.particle_to_foam_counter[iidx]
                local_idx = iidx * self.max_foam_per_particle + jidx
                self.all_foam_pos[local_idx] = self.particle_to_foam_grid[iidx, jidx].position
                local_type = self.particle_to_foam_grid[iidx, jidx].get_type()
                self.white_particles[local_idx] = ti.Vector([0., 0., 0.])
                self.green_particles[local_idx] = ti.Vector([0., 0., 0.])
                self.red_particles[local_idx] = ti.Vector([0., 0., 0.])
                if local_type == 0: self.green_particles[local_idx] = self.all_foam_pos[local_idx]
                if local_type == 1: self.white_particles[local_idx] = self.all_foam_pos[local_idx]
                if local_type == 2: self.red_particles[local_idx] = self.all_foam_pos[local_idx]

    def run(self,):
        self.compute_density()
        self.compute_normal()
        self.compute_omega()
        self.compute_potential()
        
        if self.frame_num[None] > -1:
            self.removeFoam()
            self.advectFoam()
            self.generateFoam()

        self.draw_classifiedFrom()