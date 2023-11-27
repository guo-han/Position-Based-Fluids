import math

import numpy as np

import taichi as ti

ti.init(arch=ti.gpu)
k = 3
screen_res = (800, 400)
grid_res = (300, 200, 100)
screen_to_world_ratio = 10.0
boundary = (
    grid_res[0] / screen_to_world_ratio * k,
    grid_res[1] / screen_to_world_ratio * k,
    grid_res[2] / screen_to_world_ratio * k,
)
cell_size = 2.51
cell_recpr = 1.0 / cell_size


def round_up(f, s):
    return (math.floor(f * cell_recpr / s) + 1) * s


grid_size = (round_up(boundary[0], 1), round_up(boundary[1], 1), round_up(boundary[2], 1))

dim = 3 # 3d
bg_color = (1/255,47/255,65/255)
particle_color = (6/255,133/255,135/255)
boundary_color = 0xEBACA2
num_particles_x = 10 * k
num_particles_y = 10 * k
num_particles_z = 10 * k
num_particles = num_particles_x * num_particles_y * num_particles_z
max_num_particles_per_cell = 100 # 3d
max_num_neighbors = 100 # 3d
time_delta = 1.0 / 60.0
epsilon = 1e-5
particle_radius = 3.0
particle_radius_in_world = particle_radius / screen_to_world_ratio

# PBF params
h_ = 1.1
mass = 1.0
rho0 = 1.0
lambda_epsilon = 100.0
pbf_num_iters = 5
corr_deltaQ_coeff = 0.2
corrK = 0.01
# Need ti.pow()
# corrN = 4.0
neighbor_radius = h_ * 1.05

poly6_factor = 315.0 / 64.0 / math.pi
spiky_grad_factor = -45.0 / math.pi

old_positions = ti.Vector.field(dim, float)
positions = ti.Vector.field(dim, float)
velocities = ti.Vector.field(dim, float)
omegas = ti.Vector.field(dim, float)
vorticity_forces = ti.Vector.field(dim, float)
velocities_deltas = ti.Vector.field(dim, float)
density = ti.field(float)
grid_num_particles = ti.field(int)
grid2particles = ti.field(int)
particle_num_neighbors = ti.field(int)
particle_neighbors = ti.field(int)
lambdas = ti.field(float)
position_deltas = ti.Vector.field(dim, float)
# 0: x-pos, 1: timestep in sin()
board_states = ti.Vector.field(2, float)

ti.root.dense(ti.i, num_particles).place(old_positions, positions, velocities, omegas, vorticity_forces, density)
grid_snode = ti.root.dense(ti.ijk, grid_size)
grid_snode.place(grid_num_particles)
grid_snode.dense(ti.l, max_num_particles_per_cell).place(grid2particles)
nb_node = ti.root.dense(ti.i, num_particles)
nb_node.place(particle_num_neighbors)
nb_node.dense(ti.j, max_num_neighbors).place(particle_neighbors)
ti.root.dense(ti.i, num_particles).place(lambdas, position_deltas, velocities_deltas)
ti.root.place(board_states)

# boarder corners
points_pos = ti.Vector.field(3, dtype=ti.f32, shape = 8)

@ti.func
def poly6_value(s, h):
    result = 0.0
    if 0 < s and s < h:
        x = (h * h - s * s) / (h * h * h)
        result = poly6_factor * x * x * x
    return result


@ti.func
def spiky_gradient(r, h):
    result = ti.Vector([0.0, 0.0, 0.0])
    r_len = r.norm()
    if 0 < r_len and r_len < h:
        x = (h - r_len) / (h * h * h)
        g_factor = spiky_grad_factor * x * x
        result = r * g_factor / r_len
    return result


@ti.func
def compute_scorr(pos_ji):
    # Eq (13)
    x = poly6_value(pos_ji.norm(), h_) / poly6_value(corr_deltaQ_coeff * h_, h_)
    # pow(x, 4)
    x = x * x
    x = x * x
    return (-corrK) * x


@ti.func
def get_cell(pos):
    return int(pos * cell_recpr)


@ti.func
def is_in_grid(c):
    # @c: Vector(i32)
    return 0 <= c[0] and c[0] < grid_size[0] and 0 <= c[1] and c[1] < grid_size[1] and 0 <= c[2] and c[2] < grid_size[2]


@ti.func
def confine_position_to_boundary(p):
    bmin = particle_radius_in_world
    bmax = ti.Vector([board_states[None][0], boundary[1], boundary[2]]) - particle_radius_in_world
    for i in ti.static(range(dim)):
        # Use randomness to prevent particles from sticking into each other after clamping
        if p[i] <= bmin:
            p[i] = bmin + epsilon * ti.random()
        elif bmax[i] <= p[i]:
            p[i] = bmax[i] - epsilon * ti.random()
    return p

@ti.func
def compute_density():
    for p_i in positions:
        pos_i = positions[p_i]
        density_constraint = 0.0

        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break
            pos_ji = pos_i - positions[p_j]
            # Eq(2)
            density_constraint += poly6_value(pos_ji.norm(), h_)  # mass in Eq(2) is moved to Eq(1)

        # Eq(1)
        density_constraint += poly6_value(0, h_)  # self contribution
        density[p_i] = density_constraint * mass


@ti.kernel
def move_board():
    # probably more accurate to exert force on particles according to hooke's law.
    b = board_states[None]
    b[1] += 1.0
    period = 250
    vel_strength = 8.0 + 2*k
    if b[1] >= 2 * period:
        b[1] = 0
    b[0] += -ti.sin(b[1] * np.pi / period) * vel_strength * time_delta
    board_states[None] = b


@ti.kernel
def prologue():
    # save old positions
    for i in positions:
        old_positions[i] = positions[i]
    # apply gravity within boundary
    for i in positions:
        g = ti.Vector([0.0, -9.8, 0.0])
        pos, vel = positions[i], velocities[i]
        vel += g * time_delta
        pos += vel * time_delta
        positions[i] = confine_position_to_boundary(pos)

    # clear neighbor lookup table
    for I in ti.grouped(grid_num_particles):
        grid_num_particles[I] = 0
    for I in ti.grouped(particle_neighbors):
        particle_neighbors[I] = -1

    # update grid
    for p_i in positions:
        cell = get_cell(positions[p_i])
        # ti.Vector doesn't seem to support unpacking yet
        # but we can directly use int Vectors as indices
        offs = ti.atomic_add(grid_num_particles[cell], 1)
        grid2particles[cell, offs] = p_i
    # find particle neighbors
    for p_i in positions:
        pos_i = positions[p_i]
        cell = get_cell(pos_i)
        nb_i = 0
        for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
            cell_to_check = cell + offs
            if is_in_grid(cell_to_check):
                for j in range(grid_num_particles[cell_to_check]):
                    p_j = grid2particles[cell_to_check, j]
                    if nb_i < max_num_neighbors and p_j != p_i and (pos_i - positions[p_j]).norm() < neighbor_radius:
                        particle_neighbors[p_i, nb_i] = p_j
                        nb_i += 1
        particle_num_neighbors[p_i] = nb_i


@ti.kernel
def substep():
    # compute lambdas
    # Eq (8) ~ (11)
    for p_i in positions:
        pos_i = positions[p_i]

        grad_i = ti.Vector([0.0, 0.0, 0.0])
        sum_gradient_sqr = 0.0
        density_constraint = 0.0

        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break
            pos_ji = pos_i - positions[p_j]
            grad_j = spiky_gradient(pos_ji, h_)
            grad_i += grad_j
            sum_gradient_sqr += grad_j.dot(grad_j)
            # Eq(2)
            density_constraint += poly6_value(pos_ji.norm(), h_)

        # Eq(1)
        density_constraint = (mass * density_constraint / rho0) - 1.0

        sum_gradient_sqr += grad_i.dot(grad_i)
        lambdas[p_i] = (-density_constraint) / (sum_gradient_sqr + lambda_epsilon)
    # compute position deltas
    # Eq(12), (14)
    for p_i in positions:
        pos_i = positions[p_i]
        lambda_i = lambdas[p_i]

        pos_delta_i = ti.Vector([0.0, 0.0, 0.0])
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break
            lambda_j = lambdas[p_j]
            pos_ji = pos_i - positions[p_j]
            scorr_ij = compute_scorr(pos_ji)
            pos_delta_i += (lambda_i + lambda_j + scorr_ij) * spiky_gradient(pos_ji, h_)

        pos_delta_i /= rho0
        position_deltas[p_i] = pos_delta_i
    # apply position deltas
    for i in positions:
        positions[i] += position_deltas[i]


@ti.kernel
def epilogue():
    # confine to boundary
    for i in positions:
        pos = positions[i]
        positions[i] = confine_position_to_boundary(pos)
    # update velocities
    for i in positions:
        velocities[i] = (positions[i] - old_positions[i]) / time_delta

    # calculate density first
    compute_density()

    # calculate vorticity
    for i in positions:
        pos_i = positions[i]
        omegas[i] = 0.0
        for j in range(particle_num_neighbors[i]):
            p_j = particle_neighbors[i, j]
            if p_j < 0:
                break
            pos_ji = pos_i - positions[p_j]
            omegas[i] += mass * (velocities[p_j] - velocities[i]).cross(spiky_gradient(pos_ji, h_)) / (epsilon + density[p_j])
    # calculate vorticity force
    for i in positions:
        pos_i = positions[i]
        eta = pos_i * 0.0
        vorticity_forces[i] = 0.0
        for j in range(particle_num_neighbors[i]):
            p_j = particle_neighbors[i, j]
            if p_j < 0:
                break
            pos_ji = pos_i - positions[p_j]
            eta += mass * omegas[j].norm() * spiky_gradient(pos_ji, h_) / (epsilon + density[p_j])
        location_vector = eta / (epsilon + eta.norm())
        vorticity_forces[i] += 0.5 * (location_vector.cross(omegas[i]))

    # apply vorticity force
    for i in positions:
        velocities[i] += (vorticity_forces[i] / mass) * time_delta

    # add viscosity
    for i in positions:
        pos_i = positions[i]
        velocities_deltas[i] = 0.0
        for j in range(particle_num_neighbors[i]):
            p_j = particle_neighbors[i, j]
            if p_j < 0:
                break
            pos_ji = pos_i - positions[p_j]
            velocities_deltas[i] += mass * (velocities[p_j] - velocities[i]) * poly6_value(pos_ji.norm(), h_) / (epsilon + density[p_j])
    
    for i in positions:
        velocities[i] += 0.1 * velocities_deltas[i]


def run_pbf():
    prologue()
    for _ in range(pbf_num_iters):
        substep()
    epilogue()

# y  z 
# | / -> num_particles_z
# |/
# 0--x
#  num_particles_x
@ti.kernel
def init_particles():
    for i in range(num_particles):
        x = i % num_particles_x
        y = i // (num_particles_x*num_particles_z)
        z = (i % (num_particles_x*num_particles_z)) // num_particles_x
        delta = h_ * 0.8
        offs = ti.Vector([(boundary[0] - delta * num_particles_x) * 0.5, boundary[1] * 0.02, boundary[2] * 0.02])
        positions[i] = ti.Vector([x, z, y]) * delta + offs
        for c in ti.static(range(dim)):
            velocities[i][c] = (ti.random() - 0.5) * 4
    board_states[None] = ti.Vector([boundary[0] - epsilon, -0.0])


def print_stats():
    print("PBF stats:")
    num = grid_num_particles.to_numpy()
    avg, max_ = np.mean(num), np.max(num)
    print(f"  #particles per cell: avg={avg:.2f} max={max_}")
    num = particle_num_neighbors.to_numpy()
    avg, max_ = np.mean(num), np.max(num)
    print(f"  #neighbors per particle: avg={avg:.2f} max={max_}")
    print("Vorticity force of particle 0: {}".format(vorticity_forces[0]))

def render(window, scene, canvas, camera):
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)

    b = board_states[None]
    board_len = 10 * k
    points_pos[0] = [b[0], -1, 0]
    points_pos[1] = [b[0], -1, board_len]

    points_pos[2] = [b[0], -1, 0]
    points_pos[3] = [b[0], 18, 0]

    points_pos[4] = [b[0], 18, 0]
    points_pos[5] = [b[0], 18, board_len]

    points_pos[6] = [b[0], 18, board_len]
    points_pos[7] = [b[0], -1, board_len]
    
    scene.lines(points_pos, color = (0.28, 0.68, 0.99), width = 10.0)
    scene.particles(positions, color = particle_color, radius = 0.1)
    scene.set_camera(camera)

    canvas.scene(scene)
    window.show()

def main():
    init_particles()
    window = ti.ui.Window("PBF3D", screen_res)
    canvas = window.get_canvas()
    canvas.set_background_color(bg_color)
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()

    scene.point_light(pos=(1, 2, 3), color=(1,1,1))
    scene.ambient_light((0.8, 0.8, 0.8))
    camera.position(boundary[0]/2, boundary[1]/2, 40 * k)
    camera.lookat(boundary[0]/2, boundary[1]/4, 0)
    camera.up(0, 1, 0)
    camera.projection_mode(ti.ui.ProjectionMode.Perspective)

    print(f"boundary={boundary} grid={grid_size} cell_size={cell_size}")

    frame = 0
    start = False
    while window.running:
        if window.get_event(ti.ui.PRESS):
            if window.event.key in [ti.ui.ESCAPE]: break
            if window.event.key in [ti.ui.SPACE]: start = not start
        if start:
            move_board()
            run_pbf()
            print_stats()
        # rendering
        render(window, scene, canvas, camera)
        frame += 1


if __name__ == "__main__":
    main()


