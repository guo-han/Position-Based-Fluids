import os
import math
import numpy as np
import taichi as ti
from PBF import Pbf
from FOAM import Foam
from utils import PROJ_PATH, convert_particle_info_to_json, convert_json_to_mesh_command_line
from utils import convert_rigid_info_to_json, convert_foam_info_to_pcd, convert_spary_info_to_pcd
from StaticRigidBody import StaticRigidBody
from rb_config import *

# scale factor
k = 3
# config rendering
ti.init(arch=ti.gpu)  # , debug=True
screen_res = (1600, 800)
bg_color = (1/255,47/255,65/255)
particle_color = (6/255,133/255,135/255)
foam_color = (1., 1., 1.)
red_color = (1., 0., 0.)
green_color = (0., 1., 0.)
points_pos = ti.Vector.field(3, dtype=ti.f32, shape = 8) # boarder corners

# init objects
fluid = Pbf(k)
foam = Foam(fluid)
rabbit_rb = StaticRigidBody(rabbit_config_dict, fluid.cell_recpr, fluid.grid_size, fluid.boundary)
fluid.set_rigid_body(rabbit_rb)

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

    # draw fluid
    scene.particles(fluid.positions, color = particle_color, radius = 0.1, per_vertex_color = fluid.particle_colors)

    # draw all diffuse particles
    scene.particles(foam.all_foam_pos, color = foam_color, radius = 0.1)
    
    # draw classisified particles: foam, spray and bubbles
    scene.particles(foam.white_particles, color = foam_color, radius = 0.1)
    scene.particles(foam.red_particles, color = red_color, radius = 0.1)
    scene.particles(foam.green_particles, color = green_color, radius = 0.1)

    # draw rigid body
    scene.mesh(rabbit_rb.V, rabbit_rb.F, rabbit_rb.vertexN, rabbit_rb.color) 
    scene.set_camera(camera)

    canvas.scene(scene)
    window.show()

def bake(frame, start=150, end=160):
    if frame >= start and frame < end:
        print(f"Baking frame {frame-start+1}/{end-start}")
        filename = f"frame_{frame:05d}"
        pos_np = fluid.positions.to_numpy()
        foam_np = foam.white_particles.to_numpy()
        bubble_np = foam.red_particles.to_numpy()
        spray_np = foam.green_particles.to_numpy()
        foam_np = np.concatenate((foam_np, bubble_np), axis=0)
        convert_particle_info_to_json(pos_np, filename)
        convert_json_to_mesh_command_line(filename)
        convert_foam_info_to_pcd(foam_np, filename)
        convert_spary_info_to_pcd(spray_np, filename)

def run():
    fluid.move_board()
    fluid.run_pbf()
    foam.run()

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
    # camera.position(-fluid.boundary[0]/2, fluid.boundary[1]/4, fluid.boundary[2]/2)
    # camera.lookat(0, fluid.boundary[1]/4, fluid.boundary[2]/2)

    # front view
    camera.position(fluid.boundary[0]/2, fluid.boundary[1]/2, 40 * k)
    camera.lookat(fluid.boundary[0]/2, fluid.boundary[1]/4, 0)

    camera.up(0, 1, 0)
    camera.projection_mode(ti.ui.ProjectionMode.Perspective)

    print(f"boundary={fluid.boundary} grid={fluid.grid_size} cell_size={fluid.cell_size}")

    frame = 0
    start = True
    bake_mesh = False
    while window.running:
        if window.get_event(ti.ui.PRESS):
            if window.event.key in [ti.ui.ESCAPE]: break
            if window.event.key in [ti.ui.SPACE]: start = not start
        if start:
            run()
            if bake_mesh:
                bake(frame, True, 240, 250)
            frame += 1
        # rendering
        render(window, scene, canvas, camera)
        

if __name__ == "__main__":
    main()
