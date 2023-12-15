import os
import math
import numpy as np
import taichi as ti
from PBF import Pbf
from FOAM import Foam
from utils import PROJ_PATH, convert_particle_info_to_json, convert_json_to_mesh_command_line, convert_foam_info_to_json
from utils import convert_rigid_info_to_json, convert_foam_info_to_pcd
from StaticRigidBody import StaticRigidBody
from rb_config import *

# scale factor
k = 3
# config rendering
# ti.init(arch=ti.gpu) # , debug=True
ti.init()
screen_res = (800, 400)
bg_color = (1/255,47/255,65/255)
particle_color = (6/255,133/255,135/255)
foam_color = (1,1,1)
red_color = (1., 0., 0.)
green_color = (0., 1., 0.)
yellow_color = (255/255, 175/255, 0.)
points_pos = ti.Vector.field(3, dtype=ti.f32, shape = 8) # boarder corners

# init objects
fluid = Pbf(k)
foam = Foam(fluid)
rock_rb = StaticRigidBody(sample_rock_config_dict, fluid.cell_recpr, fluid.grid_size)
fluid.set_rigid_body(rock_rb)

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
    scene.particles(fluid.positions, color = particle_color, radius = 0.1, per_vertex_color = fluid.particle_colors)
    # scene.particles(foam.foam_positions, color = foam_color, radius = 0.1)
    scene.particles(foam.white_particles, color = foam_color, radius = 0.1)
    scene.particles(foam.red_particles, color = red_color, radius = 0.1)
    scene.particles(foam.green_particles, color = green_color, radius = 0.1)
    # scene.particles(foam.yellow_particles, color = yellow_color, radius = 0.1)
    scene.mesh(rock_rb.V, rock_rb.F, rock_rb.vertexN, rock_rb.color) 
    scene.set_camera(camera)

    canvas.scene(scene)
    window.show()

def print_stats():
    print("PBF stats:")
    num = fluid.grid_num_particles.to_numpy()
    print(f" #Total particles: {fluid.positions.shape[0]}")
    avg, max_ = np.mean(num), np.max(num)
    print(f"  #particles per cell: avg={avg:.2f} max={max_}")
    num = fluid.particle_num_neighbors.to_numpy()
    avg, max_ = np.mean(num), np.max(num)
    print(f"  #neighbors per particle: avg={avg:.2f} max={max_}")
    print("Vorticity force of particle 0: {}".format(fluid.vorticity_forces[0]))

def bake(frame, bake_foam = False,start=150, end=160):
    if frame >= start and frame < end:
        print(f"Baking frame {frame-start+1}/{end-start}")
        filename = f"frame_{frame:05d}"
        pos_np = fluid.positions.to_numpy()
        # pos_np = pos_np[:, (0, 2, 1)] # why???
        foam_np = foam.foam_positions.to_numpy()
        convert_particle_info_to_json(pos_np, filename)
        convert_foam_info_to_json(foam_np, filename)
        convert_json_to_mesh_command_line(filename)
        convert_foam_info_to_pcd(foam_np, filename)

def run():
    fluid.move_board()
    fluid.run_pbf()
    foam.run()
    # print_stats()

def main():
    fluid.init_particles()

    window = ti.ui.Window("PBF3D", screen_res)
    canvas = window.get_canvas()
    canvas.set_background_color(bg_color)
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()

    scene.point_light(pos=(1, 2, 3), color=(1,1,1))
    scene.ambient_light((0.8, 0.8, 0.8))

    # top-down view
    # camera.position(fluid.boundary[0]/2, 100, 20)
    # camera.lookat(fluid.boundary[0]/2, 0, 0)

    # side view
    camera.position(fluid.boundary[0]/2, fluid.boundary[1]/2, 40 * k)
    camera.lookat(fluid.boundary[0]/2, fluid.boundary[1]/4, 0)
    # camera.position(-fluid.boundary[0]/2, fluid.boundary[1]/4, fluid.boundary[2]/2)
    # camera.lookat(0, fluid.boundary[1]/4, fluid.boundary[2]/2)
    camera.up(0, 1, 0)
    camera.projection_mode(ti.ui.ProjectionMode.Perspective)

    print(f"boundary={fluid.boundary} grid={fluid.grid_size} cell_size={fluid.cell_size}")

    export_rigid_info = False
    print(rock_rb.center)
    if export_rigid_info:
        rigid_dict_json = {
            "scalings": rock_rb.scale,
            "pos": rock_rb.pos,
            "center": rock_rb.center.tolist(),
        }
        convert_rigid_info_to_json(rigid_dict_json, 'rock')

    frame = 0
    start = True
    bake_mesh = True
    while window.running:
        if window.get_event(ti.ui.PRESS):
            if window.event.key in [ti.ui.ESCAPE]: break
            if window.event.key in [ti.ui.SPACE]: start = not start
        if start:
            run()
            if bake_mesh:
                bake(frame, bake_foam=True)
            frame += 1
        # rendering
        render(window, scene, canvas, camera)
        

if __name__ == "__main__":
    main()
