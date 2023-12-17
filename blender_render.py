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
import glob
from rb_config import *
TAU = 2*pi

def main():
    dir_mesh = './meshes/'
    dir_rendering = './rendering/'
    dir_particles = './particles/'
    dir_foam_pcd = './foam_pcd/'
    dir_spray_pcd = './spray_pcd/'
    path_envmap= './textures/Skies-001.jpg'
    dir_bunny = './data/models/bunny_final.obj'
    path_box = './Assets/box.obj'
    path_curve = './Assets/plane.obj'
    path_lights = './Assets/lights.obj'

    path_video = "./rendering/test.avi"
    obj_name_list = sorted(glob.glob(os.path.join(dir_mesh, "*.obj")))
    foam_pcd_name_list = sorted(glob.glob(os.path.join(dir_foam_pcd, "*.ply")))
    spray_pcd_name_list = sorted(glob.glob(os.path.join(dir_spray_pcd, "*.ply")))
    obj_list_num = len(obj_name_list)
    # obj_path = dir_path+obj_name+".obj"
    # obj_path_list = [dir_mesh+obj_name for obj_name in obj_name_list]
    # foam_pcd_path_list = [dir_foam_pcd + foam_pcd_name for foam_pcd_name in foam_pcd_name_list]
    # spray_pcd_path_list = [dir_spray_pcd + spray_pcd_name for spray_pcd_name in spray_pcd_name_list]
    obj_path_list = obj_name_list
    foam_pcd_path_list = foam_pcd_name_list
    spray_pcd_path_list = spray_pcd_name_list
    # render_img_path = dir_path+obj_name+"_render_total.png"
    render_img_path = []
    obj_name_list_wo_ext = []
    foam_pcd_name_list_wo_ext = []
    spray_pcd_name_list_wo_ext = []
    for idx, obj_name in enumerate(obj_name_list):
        # extract fluid mesh name without extension
        fileName = os.path.splitext(obj_name)[0]
        fileName = fileName.split("/")[-1]
        obj_name_list_wo_ext.append(fileName)
        # extract foam ply name without extension
        pcdName = os.path.splitext(foam_pcd_name_list[idx])[0]
        pcdName = pcdName.split("/")[-1]
        foam_pcd_name_list_wo_ext.append(pcdName)
        # extract spray ply name without extension
        pcdName = os.path.splitext(spray_pcd_name_list[idx])[0]
        pcdName = pcdName.split("/")[-1]
        spray_pcd_name_list_wo_ext.append(pcdName)
        render_img_path.append(dir_rendering+fileName+"_render.png")


    # Remove all elements
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # scale_ratio = [0.12, 0.12, 0.12]
    scale_ratio = [1.0, 1.0, 1.0]

    # # Set cursor to (0, 0, 0)
    # bpy.context.scene.cursor.location = (0, 0, 0)

    # Create camera
    bpy.ops.object.add(type='CAMERA')
    camera = bpy.data.objects['Camera']
    # camera.location = (16.68, -14.07, 10.20)
    # camera.rotation_euler = (math.radians(62), math.radians(-2.91), math.radians(45.6))
    # camera.location = (13.2098, -2.61762, 11.2616)
    # camera.rotation_euler = (math.radians(43.962), math.radians(0), math.radians(83.495))
    camera.location = (132.151, -139.624, 10.7156)
    camera.rotation_euler = (math.radians(95.5347), math.radians(-0.000094), math.radians(33.8674))
    
    # Make this the current camera
    bpy.context.scene.camera = camera

    # create water material
    mat = bpy.data.materials.new(name="WaterMaterial")
    mat.use_nodes = True
    mat.node_tree.nodes.new(type='ShaderNodeBsdfGlass')
    mat.node_tree.nodes['Glass BSDF'].inputs['Roughness'].default_value = 0.0
    mat.node_tree.nodes['Glass BSDF'].inputs['IOR'].default_value = 1.33
    mat.node_tree.nodes['Glass BSDF'].inputs['Color'].default_value = (0.76, 0.906, 1, 1)
    inp = mat.node_tree.nodes['Material Output'].inputs['Surface']
    outp = mat.node_tree.nodes['Glass BSDF'].outputs['BSDF']
    mat.node_tree.links.new(inp,outp)

    # create foam material
    mat = bpy.data.materials.new(name="FoamMaterial")
    mat.use_nodes = True
    foam_mat_links = mat.node_tree.links
    # add mix shader and connect mix shader with material output
    mat.node_tree.nodes.new(type="ShaderNodeMixShader")
    foam_mat_links.new(mat.node_tree.nodes['Mix Shader'].outputs[0], mat.node_tree.nodes['Material Output'].inputs['Surface'])
    # add transparent bsdf and connect to the first shader of mix shader
    mat.node_tree.nodes.new(type="ShaderNodeBsdfTransparent")
    foam_mat_links.new(mat.node_tree.nodes['Transparent BSDF'].outputs[0], mat.node_tree.nodes['Mix Shader'].inputs[1])
    # add principled bsdf and connect to the second shader of mix shader
    mat.node_tree.nodes.new(type="ShaderNodeBsdfPrincipled")
    mat.node_tree.nodes['Principled BSDF'].inputs[19].default_value = (1.0, 1., 1., 1.)
    mat.node_tree.nodes['Principled BSDF'].inputs[20].default_value = 0.1
    foam_mat_links.new(mat.node_tree.nodes['Principled BSDF'].outputs[0], mat.node_tree.nodes['Mix Shader'].inputs[2])
    # add ColorRamp, change it and add to mix shader fac
    mat.node_tree.nodes.new(type="ShaderNodeValToRGB")
    mat.node_tree.nodes["Color Ramp"].color_ramp.elements[0].position = 0
    mat.node_tree.nodes["Color Ramp"].color_ramp.elements[1].position = 0.139
    foam_mat_links.new(mat.node_tree.nodes['Color Ramp'].outputs[0], mat.node_tree.nodes['Mix Shader'].inputs[0])
    # add musgrave texture, change values and link to the fac input of colorRamp
    mat.node_tree.nodes.new(type="ShaderNodeTexMusgrave")
    mat.node_tree.nodes["Musgrave Texture"].inputs[2].default_value = 2
    mat.node_tree.nodes["Musgrave Texture"].inputs[3].default_value = 1
    mat.node_tree.nodes["Musgrave Texture"].inputs[4].default_value = 2
    mat.node_tree.nodes["Musgrave Texture"].inputs[5].default_value = 2
    foam_mat_links.new(mat.node_tree.nodes['Musgrave Texture'].outputs[0], mat.node_tree.nodes['Color Ramp'].inputs[0])

    # create spray material
    mat = bpy.data.materials.new(name="SprayMaterial")
    mat.use_nodes = True
    spray_mat_links = mat.node_tree.links
    # add mix shader and connect mix shader with material output
    mat.node_tree.nodes.new(type="ShaderNodeMixShader")
    spray_mat_links.new(mat.node_tree.nodes['Mix Shader'].outputs[0], mat.node_tree.nodes['Material Output'].inputs['Surface'])
    # add transparent bsdf and connect to the first shader of mix shader
    mat.node_tree.nodes.new(type="ShaderNodeBsdfTransparent")
    spray_mat_links.new(mat.node_tree.nodes['Transparent BSDF'].outputs[0], mat.node_tree.nodes['Mix Shader'].inputs[1])
    # add principled bsdf and connect to the second shader of mix shader
    mat.node_tree.nodes.new(type="ShaderNodeBsdfPrincipled")
    mat.node_tree.nodes['Principled BSDF'].inputs[16].default_value = 1.2
    mat.node_tree.nodes['Principled BSDF'].inputs[17].default_value = 1.0
    mat.node_tree.nodes['Principled BSDF'].inputs[19].default_value = (1.0, 1., 1., 1.)
    mat.node_tree.nodes['Principled BSDF'].inputs[20].default_value = 0.25
    spray_mat_links.new(mat.node_tree.nodes['Principled BSDF'].outputs[0], mat.node_tree.nodes['Mix Shader'].inputs[2])
    # add ColorRamp, change it and add to mix shader fac
    mat.node_tree.nodes.new(type="ShaderNodeValToRGB")
    mat.node_tree.nodes["Color Ramp"].color_ramp.elements[0].position = 0
    mat.node_tree.nodes["Color Ramp"].color_ramp.elements[1].position = 0.139
    spray_mat_links.new(mat.node_tree.nodes['Color Ramp'].outputs[0], mat.node_tree.nodes['Mix Shader'].inputs[0])
    # add musgrave texture, change values and link to the fac input of colorRamp
    mat.node_tree.nodes.new(type="ShaderNodeTexMusgrave")
    mat.node_tree.nodes["Musgrave Texture"].inputs[2].default_value = 2
    mat.node_tree.nodes["Musgrave Texture"].inputs[3].default_value = 1
    mat.node_tree.nodes["Musgrave Texture"].inputs[4].default_value = 2
    mat.node_tree.nodes["Musgrave Texture"].inputs[5].default_value = 2
    spray_mat_links.new(mat.node_tree.nodes['Musgrave Texture'].outputs[0], mat.node_tree.nodes['Color Ramp'].inputs[0])


    # create box material
    mat = bpy.data.materials.new(name="boxMaterial")
    mat.use_nodes = True
    mat.node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
    inp = mat.node_tree.nodes['Material Output'].inputs['Surface']
    outp = mat.node_tree.nodes['Principled BSDF'].outputs['BSDF']
    mat.node_tree.links.new(inp,outp)

    # create curve material
    mat = bpy.data.materials.new(name="curveMaterial")
    mat.use_nodes = True
    mat.node_tree.nodes.new(type='ShaderNodeBsdfDiffuse')
    mat.node_tree.nodes['Diffuse BSDF'].inputs[1].default_value = 1
    mat.node_tree.nodes.new(type="ShaderNodeTexChecker")
    mat.node_tree.nodes["Checker Texture"].inputs[3].default_value = 15
    mat.node_tree.nodes["Checker Texture"].inputs[1].default_value = (0.165, 0.165, 0.165, 1)
    mat.node_tree.nodes["Checker Texture"].inputs[2].default_value = (0.051, 0.051, 0.051, 1)
    mat.node_tree.links.new( mat.node_tree.nodes["Checker Texture"].outputs[0], mat.node_tree.nodes['Diffuse BSDF'].inputs[0])
    inp = mat.node_tree.nodes['Material Output'].inputs['Surface']
    outp = mat.node_tree.nodes['Diffuse BSDF'].outputs['BSDF']
    mat.node_tree.links.new(inp,outp)

    # create light material
    mat = bpy.data.materials.new(name="lightMaterial")
    mat.use_nodes = True
    mat.node_tree.nodes.new(type='ShaderNodeEmission')
    mat.node_tree.nodes["Emission"].inputs[1].default_value = 5
    inp = mat.node_tree.nodes['Material Output'].inputs['Surface']
    outp = mat.node_tree.nodes['Emission'].outputs[0]
    mat.node_tree.links.new(inp,outp)

    # create diffuse white material
    mat = bpy.data.materials.new(name="RigidMaterial")
    mat.use_nodes = True
    mat.node_tree.nodes.new(type='ShaderNodeBsdfGlass')
    mat.node_tree.nodes['Glass BSDF'].inputs['Color'].default_value = (1, 0.072, 0.214, 1)
    mat.node_tree.nodes['Glass BSDF'].inputs['IOR'].default_value = 8.15
    inp = mat.node_tree.nodes['Material Output'].inputs['Surface']
    outp = mat.node_tree.nodes['Glass BSDF'].outputs['BSDF']
    mat.node_tree.links.new(inp,outp)

    scene = bpy.context.scene
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 50
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

    # bpy.ops.object.light_add(type='SUN')
    # light_ob = bpy.context.object
    # light = light_ob.data
    # light.energy = 1
    # light_ob.location = (3.644, 15.456, 12.611)
    # light_ob.rotation_euler = (math.radians(40.5), math.radians(46), math.radians(143))
    # bpy.ops.object.light_add(type='AREA')
    # light_ob = bpy.context.object
    # light = light_ob.data
    # light.energy = 100
    # light.size = 6
    # light_ob.location = (4.5389, -1.18225, 4.27092)


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
        geo = bpy.context.object.modifiers[0].node_group
        geo.nodes.new(type = "GeometryNodeMeshToPoints")
        geo.nodes.new(type = "GeometryNodeSetMaterial")
        geo.nodes.new(type = "FunctionNodeRandomValue")
        group_input = geo.nodes["Group Input"].outputs[0]
        group_output = geo.nodes["Group Output"].inputs[0]
        set_material = geo.nodes["Set Material"]
        set_material.inputs[2].default_value = bpy.data.materials["FoamMaterial"]
        mtp_out = geo.nodes["Mesh to Points"].outputs[0]
        geo.nodes["Random Value"].inputs[2].default_value = 0.1
        geo.nodes["Random Value"].inputs[3].default_value = 0.4
        links = geo.links
        links.new(geo.nodes["Random Value"].outputs[1], geo.nodes["Mesh to Points"].inputs[3])
        mtp_in = geo.nodes["Mesh to Points"].inputs[0]
        links.new(group_input, mtp_in)
        links.new(set_material.inputs["Geometry"], mtp_out)
        links.new(set_material.outputs["Geometry"], group_output)    

        # load spray point cloud
        bpy.ops.import_mesh.ply(filepath = spray_pcd_path_list[i])
        spray_name = spray_pcd_name_list_wo_ext[i]
        print(foam_pcd_path_list[i])
        spray = bpy.data.objects[spray_name]
        spray.select_set(True)
        bpy.ops.transform.resize(value=(scale_ratio[0], scale_ratio[1], scale_ratio[2]))
        # change pcd to geometry
        bpy.ops.node.new_geometry_nodes_modifier()
        geo = bpy.context.object.modifiers[0].node_group
        geo.nodes.new(type = "GeometryNodeMeshToPoints")
        geo.nodes.new(type = "GeometryNodeSetMaterial")
        geo.nodes.new(type = "FunctionNodeRandomValue")
        group_input = geo.nodes["Group Input"].outputs[0]
        group_output = geo.nodes["Group Output"].inputs[0]
        set_material = geo.nodes["Set Material"]
        set_material.inputs[2].default_value = bpy.data.materials["SprayMaterial"]
        mtp_out = geo.nodes["Mesh to Points"].outputs[0]
        geo.nodes["Random Value"].inputs[2].default_value = 0.1
        geo.nodes["Random Value"].inputs[3].default_value = 0.4
        links = geo.links
        links.new(geo.nodes["Random Value"].outputs[1], geo.nodes["Mesh to Points"].inputs[3])
        mtp_in = geo.nodes["Mesh to Points"].inputs[0]
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
            bpy.ops.import_scene.obj(filepath=rabbit_config_dict['model_path'])
            rigid = bpy.data.objects[rabbit_config_dict['model_name']]
            rigid_all.append(rigid)
            scalings = rabbit_config_dict['model_scale']
            bpy.ops.transform.resize(value=(scale_ratio[0]*scalings, scale_ratio[1]*scalings, scale_ratio[2]*scalings))
            rigid.data.materials.clear()
            rigid.data.materials.append(bpy.data.materials["RigidMaterial"])
            
            # load box
            bpy.ops.import_scene.obj(filepath=path_box)
            box = bpy.data.objects['Cube.001']
            box.data.materials.clear()
            box.data.materials.append(bpy.data.materials["boxMaterial"])

            # load lights
            bpy.ops.import_scene.obj(filepath=path_lights)
            for j in range(1, 11):
                light_name = 'top{}'.format(j)
                light = bpy.data.objects[light_name]
                light.data.materials.clear()
                light.data.materials.append(bpy.data.materials["lightMaterial"])

            # load curve
            bpy.ops.import_scene.obj(filepath=path_curve)
            curve = bpy.data.objects['curvePlane']
            curve.data.materials.clear()
            curve.data.materials.append(bpy.data.materials["curveMaterial"])

        bpy.ops.object.select_all(action="DESELECT")
        # render and remove object
        bpy.ops.render.render(write_still=True, use_viewport=False)
        water.select_set(True)
        bpy.ops.object.delete()
        foam.select_set(True)
        bpy.ops.object.delete()
        spray.select_set(True)
        bpy.ops.object.delete()


    create_video = True
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