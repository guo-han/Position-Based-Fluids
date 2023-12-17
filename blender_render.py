import bpy
from mathutils import Vector, Euler
import math
import os
import time
import logging
import json
import colorsys
from math import sin, cos, pi
from tqdm import tqdm
from rb_config import *
TAU = 2*pi

def main():
    dir_mesh = './meshes/'
    dir_rendering = './rendering/'
    dir_particles = './particles/'
    # dir_foam = './foam/'
    dir_foam_pcd = './foam_pcd/'
    # dir_rigid = './rigids/'
    path_envmap= './textures/Skies-001.jpg'
    dir_bunny = './data/models/bunny_final.obj'

    path_video = "./rendering/test.avi"
    obj_name_list = sorted(os.listdir(dir_mesh))
    # foam_name_list = sorted(os.listdir(dir_foam))
    # rigid_name_list = sorted(os.listdir(dir_rigid))
    foam_pcd_name_list = sorted(os.listdir(dir_foam_pcd))
    obj_list_num = len(obj_name_list)
    # obj_path = dir_path+obj_name+".obj"
    obj_path_list = [dir_mesh+obj_name for obj_name in obj_name_list]
    # foam_path_list = [dir_foam + foam_name for foam_name in foam_name_list]
    # rigid_path_list = [dir_rigid + rigid_name for rigid_name in rigid_name_list]
    foam_pcd_path_list = [dir_foam_pcd + foam_pcd_name for foam_pcd_name in foam_pcd_name_list]
    # render_img_path = dir_path+obj_name+"_render_total.png"
    render_img_path = []
    obj_name_list_wo_ext = []
    foam_pcd_name_list_wo_ext = []
    for idx, obj_name in enumerate(obj_name_list):
        # extract fluid mesh name without extension
        fileName = os.path.splitext(obj_name)[0]
        obj_name_list_wo_ext.append(fileName)
        # extract foam ply name without extension
        pcdName = os.path.splitext(foam_pcd_name_list[idx])[0]
        foam_pcd_name_list_wo_ext.append(pcdName)
        render_img_path.append(dir_rendering+fileName+"_render.png")


    # Remove all elements
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    scale_ratio = [0.12, 0.12, 0.12]

    # # Set cursor to (0, 0, 0)
    # bpy.context.scene.cursor.location = (0, 0, 0)

    # Create camera
    bpy.ops.object.add(type='CAMERA')
    camera = bpy.data.objects['Camera']
    # camera.location = (16.68, -14.07, 10.20)
    # camera.rotation_euler = (math.radians(62), math.radians(-2.91), math.radians(45.6))
    camera.location = (5.967, -17.686, 7.3802)
    camera.rotation_euler = (math.radians(71.1632), math.radians(2.54867), math.radians(4.63553))
    
    # Make this the current camera
    bpy.context.scene.camera = camera

    # create water material
    mat = bpy.data.materials.new(name="WaterMaterial")
    mat.use_nodes = True
    mat.node_tree.nodes.new(type='ShaderNodeBsdfGlass')
    mat.node_tree.nodes['Glass BSDF'].inputs['Roughness'].default_value = 0.0
    mat.node_tree.nodes['Glass BSDF'].inputs['IOR'].default_value = 1.33
    inp = mat.node_tree.nodes['Material Output'].inputs['Surface']
    outp = mat.node_tree.nodes['Glass BSDF'].outputs['BSDF']
    mat.node_tree.links.new(inp,outp)

    # create diffuse white material
    mat = bpy.data.materials.new(name="FoamMaterial")
    mat.use_nodes = True
    mat.node_tree.nodes.new(type='ShaderNodeBsdfDiffuse')
    mat.node_tree.nodes['Diffuse BSDF'].inputs['Roughness'].default_value = 0.0
    mat.node_tree.nodes['Diffuse BSDF'].inputs['Color'].default_value = (1, 0., 0., 1)
    # mat.node_tree.nodes['Glass BSDF'].inputs['IOR'].default_value = 1.33
    inp = mat.node_tree.nodes['Material Output'].inputs['Surface']
    outp = mat.node_tree.nodes['Diffuse BSDF'].outputs['BSDF']
    mat.node_tree.links.new(inp,outp)

    # # create foam node group
    # geo = bpy.data.node_groups.new(type = "GeometryNodeTree", name="foam_geo")
    # geo.nodes.new(type = "GeometryNodeMeshToPoints")
    # geo.nodes.new(type = "NodeGroupInput")
    # geo.nodes.new(type = "NodeGroupOutput")
    # links = geo.links
    # group_input = geo.nodes["Group Input"].outputs[0]
    # group_output = geo.nodes["Group Output"].inputs[0]
    # print(group_input)
    # print(group_output)
    # mtp_out = geo.nodes["Mesh to Points"].outputs[0]
    # print(mtp_out)
    # mtp_in = geo.nodes["Mesh to Points"].inputs[0]
    # print(mtp_in)
    # links.new(group_input, mtp_in)
    # links.new(group_output, mtp_out)

    # create diffuse white material
    mat = bpy.data.materials.new(name="RigidMaterial")
    mat.use_nodes = True
    mat.node_tree.nodes.new(type='ShaderNodeBsdfDiffuse')
    mat.node_tree.nodes['Diffuse BSDF'].inputs['Roughness'].default_value = 0.0
    mat.node_tree.nodes['Diffuse BSDF'].inputs['Color'].default_value = (1., 1., 1., 1)
    # mat.node_tree.nodes['Glass BSDF'].inputs['IOR'].default_value = 1.33
    inp = mat.node_tree.nodes['Material Output'].inputs['Surface']
    outp = mat.node_tree.nodes['Diffuse BSDF'].outputs['BSDF']
    mat.node_tree.links.new(inp,outp)

    scene = bpy.context.scene
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    bpy.context.scene.cycles.samples = 128

    # set background image
    scene.world.use_nodes = True
    tree_nodes = scene.world.node_tree.nodes
    tree_nodes.clear()
    node_background = tree_nodes.new(type='ShaderNodeBackground')
    node_environment = tree_nodes.new('ShaderNodeTexEnvironment')
    node_environment.image = bpy.data.images.load(path_envmap)
    # node_environment.location = -300,0
    node_output = tree_nodes.new(type='ShaderNodeOutputWorld')   
    # node_output.location = 200,0
    # Link all nodes
    links = scene.world.node_tree.links
    link = links.new(node_environment.outputs["Color"], node_background.inputs["Color"])
    link = links.new(node_background.outputs["Background"], node_output.inputs["Surface"])

    bpy.ops.object.light_add(type='SUN')
    light_ob = bpy.context.object
    light = light_ob.data
    light.energy = 1
    light_ob.location = (3.644, 15.456, 12.611)
    light_ob.rotation_euler = (math.radians(40.5), math.radians(46), math.radians(143))


    #scene.render.engine = 'BLENDER_EEVEE'

    # bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "METAL"
    # scene.cycles.device = 'GPU'
    # scene.render.engine = 'CYCLES' # 'BLENDER_EEVEE'
    # scene.render.film_transparent = True
    # # (1920 1080) (1280 720) (960 540) (640 360)
    # scene.render.resolution_x = 1920
    # scene.render.resolution_y = 1080
    # scene.render.image_settings.file_format = 'PNG'
    
    # for i in range()

    for i in range(1):
        # load foam point cloud
        bpy.ops.import_mesh.ply(filepath = foam_pcd_path_list[i])
        foam_name = foam_pcd_name_list_wo_ext[i]
        print(foam_pcd_path_list[i])
        foam = bpy.data.objects[foam_name]
        foam.select_set(True)
        bpy.ops.transform.resize(value=(scale_ratio[0], scale_ratio[1], scale_ratio[2]))
        # change pcd to geometry
        bpy.ops.node.new_geometry_nodes_modifier()
        print(bpy.context.object.modifiers[0].node_group.nodes[0])
        geo = bpy.context.object.modifiers[0].node_group
        geo.nodes.new(type = "GeometryNodeMeshToPoints")
        geo.nodes.new(type = "GeometryNodeSetMaterial")
        group_input = geo.nodes["Group Input"].outputs[0]
        group_output = geo.nodes["Group Output"].inputs[0]
        print(group_input)
        print(group_output)
        set_material = geo.nodes["Set Material"]
        set_material.inputs[2].default_value = bpy.data.materials["FoamMaterial"]
        mtp_out = geo.nodes["Mesh to Points"].outputs[0]
        geo.nodes["Mesh to Points"].inputs[3].default_value = 1
        print(mtp_out)
        mtp_in = geo.nodes["Mesh to Points"].inputs[0]
        print(mtp_in)
        links = geo.links
        links.new(group_input, mtp_in)
        links.new(set_material.inputs["Geometry"], mtp_out)
        links.new(set_material.outputs["Geometry"], group_output)    

        # load water
        obj_name = obj_name_list_wo_ext[i]
        bpy.ops.import_scene.obj(filepath=obj_path_list[i]) # import water
        water = bpy.data.objects[obj_name]
        water.select_set(True)
        bpy.ops.transform.resize(value=(scale_ratio[0], scale_ratio[1], scale_ratio[2]))
        water.data.materials.clear()
        water.data.materials.append(bpy.data.materials["WaterMaterial"])     
        scene.render.filepath = render_img_path[i]

        rigid_all = []
        if i==0:
            # load rigid
            # with open(rigid_path_list[0], 'r') as load_rigid:
            #     rigid_dict = json.load(load_rigid)
            # scalings = rigid_dict['scalings']
            # rigid_pos = rigid_dict['pos']
            # rigid_center = rigid_dict['center']
            bpy.ops.import_scene.obj(filepath=rabbit_config_dict['model_path'])
            rigid = bpy.data.objects[rabbit_config_dict['model_name']]
            rigid_all.append(rigid)
            scalings = rabbit_config_dict['model_scale']
            bpy.ops.transform.resize(value=(scale_ratio[0]*scalings, scale_ratio[1]*scalings, scale_ratio[2]*scalings))
            # rigid.location = ((rigid_pos[0]  - rigid_center[0]*scalings) * scale_ratio[0], 
            #                 (-rigid_pos[2] - rigid_center[2]*scalings) * scale_ratio[2],
            #                 (rigid_pos[1] - rigid_center[1]*scalings * scale_ratio[1]))
            
            rigid.data.materials.clear()
            rigid.data.materials.append(bpy.data.materials["RigidMaterial"])


        bpy.ops.object.select_all(action="DESELECT")
        # render and remove object
        bpy.ops.render.render(write_still=True, use_viewport=False)
        water.select_set(True)
        bpy.ops.object.delete()
        foam.select_set(True)
        bpy.ops.object.delete()
        # for j in range(1000):
        #     foam = foam_all[j]
        #     foam.select_set(True)
        #     bpy.ops.object.delete()
        # for j in range(len(rigid_all)):
        #     rigid = rigid_all[j]
        #     rigid.select_set(True)
        #     bpy.ops.object.delete()

    create_video = False
    if create_video:
        scene.render.use_sequencer = True
        scene.sequence_editor_create()

        for i in range (obj_list_num):
            scene.sequence_editor.sequences.new_image(
                name=obj_name_list_wo_ext[i],
                filepath=render_img_path[i],
                channel=1, frame_start=i)

        scene.frame_end = obj_list_num
        scene.render.image_settings.file_format = 'AVI_JPEG' 
        scene.render.filepath = path_video
        bpy.ops.render.render( animation=True )

if __name__ == '__main__':
    main()