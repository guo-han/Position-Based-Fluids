import math
import numpy as np
import taichi as ti
from PBF import Pbf

# scale factor
k = 3 
# config rendering
ti.init(arch=ti.gpu)
screen_res = (800, 400)
bg_color = (1/255,47/255,65/255)
particle_color = (6/255,133/255,135/255)
boundary_color = 0xEBACA2
points_pos = ti.Vector.field(3, dtype=ti.f32, shape = 8) # boarder corners

# init objects
fluid = Pbf(k)
# foam = Foam()
# rigid = RigidBody()

def render(window, scene, canvas, camera):
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)

    b = fluid.board_states[None]
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
    scene.particles(fluid.positions, color = particle_color, radius = 0.1)
    scene.set_camera(camera)

    canvas.scene(scene)
    window.show()

def print_stats():
    print("PBF stats:")
    num = fluid.grid_num_particles.to_numpy()
    avg, max_ = np.mean(num), np.max(num)
    print(f"  #particles per cell: avg={avg:.2f} max={max_}")
    num = fluid.particle_num_neighbors.to_numpy()
    avg, max_ = np.mean(num), np.max(num)
    print(f"  #neighbors per particle: avg={avg:.2f} max={max_}")
    print("Vorticity force of particle 0: {}".format(fluid.vorticity_forces[0]))

def run():
    fluid.move_board()
    fluid.run_pbf()
    print_stats()

def main():
    fluid.init_particles()
    window = ti.ui.Window("PBF3D", screen_res)
    canvas = window.get_canvas()
    canvas.set_background_color(bg_color)
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()

    scene.point_light(pos=(1, 2, 3), color=(1,1,1))
    scene.ambient_light((0.8, 0.8, 0.8))
    camera.position(fluid.boundary[0]/2, fluid.boundary[1]/2, 40 * k)
    camera.lookat(fluid.boundary[0]/2, fluid.boundary[1]/4, 0)
    camera.up(0, 1, 0)
    camera.projection_mode(ti.ui.ProjectionMode.Perspective)

    print(f"boundary={fluid.boundary} grid={fluid.grid_size} cell_size={fluid.cell_size}")

    frame = 0
    start = True
    while window.running:
        if window.get_event(ti.ui.PRESS):
            if window.event.key in [ti.ui.ESCAPE]: break
            if window.event.key in [ti.ui.SPACE]: start = not start
        if start:
            run()
        # rendering
        render(window, scene, canvas, camera)
        frame += 1


if __name__ == "__main__":
    main()
