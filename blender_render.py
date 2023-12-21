import bpy
from mathutils import Vector, Euler
import math
import os
import time
import logging
import json
import colorsys
from math import sin, cos, pi
import glob
from rb_config import rabbit_render_config_dict as rabbit_config_dict
TAU = 2*pi
from utils import PROJ_PATH

def main():
    dir_mesh = os.path.join(PROJ_PATH, 'meshes')
    dir_rendering = os.path.join(PROJ_PATH, 'rendering')
    dir_foam_pcd = os.path.join(PROJ_PATH, 'foam_pcd')
    dir_spray_pcd = os.path.join(PROJ_PATH, 'spray_pcd')
    path_envmap= os.path.join(PROJ_PATH, 'textures', 'Skies-001.jpg')
    dir_bunny = os.path.join(PROJ_PATH, 'data', 'models', 'bunny_final.obj')
    path_box = os.path.join(PROJ_PATH, 'Assets', 'box.obj')
    path_curve = os.path.join(PROJ_PATH, 'Assets', 'plane.obj')
    path_lights = os.path.join(PROJ_PATH, 'Assets', 'lights.obj')

    path_video = os.path.join(PROJ_PATH, 'rendering', 'test.avi')
    obj_name_list = sorted(glob.glob(os.path.join(dir_mesh, "*.obj")))
    foam_pcd_name_list = sorted(glob.glob(os.path.join(dir_foam_pcd, "*.ply")))
    spray_pcd_name_list = sorted(glob.glob(os.path.join(dir_spray_pcd, "*.ply")))
    obj_list_num = len(obj_name_list)
    obj_path_list = obj_name_list
    foam_pcd_path_list = foam_pcd_name_list
    spray_pcd_path_list = spray_pcd_name_list
    render_img_path = []
    obj_name_list_wo_ext = []
    foam_pcd_name_list_wo_ext = []
    spray_pcd_name_list_wo_ext = []
    for idx, obj_name in enumerate(obj_name_list):
        # extract fluid mesh name without extension
        fileName = os.path.splitext(obj_name)[0]
        fileName = fileName.split("/")[-1] if "/" in fileName else fileName.split("\\")[-1]
        obj_name_list_wo_ext.append(fileName)
        # extract foam ply name without extension
        pcdName = os.path.splitext(foam_pcd_name_list[idx])[0]
        pcdName = pcdName.split("/")[-1] if "/" in pcdName else pcdName.split("\\")[-1]
        foam_pcd_name_list_wo_ext.append(pcdName)
        # extract spray ply name without extension
        pcdName = os.path.splitext(spray_pcd_name_list[idx])[0]
        pcdName = pcdName.split("/")[-1] if "/" in pcdName else pcdName.split("\\")[-1]
        spray_pcd_name_list_wo_ext.append(pcdName)
        render_img_path.append(os.path.join(dir_rendering, fileName+"_render.png"))


    # Remove all elements
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    scale_ratio = [1.0, 1.0, 1.0]

    # Create camera
    bpy.ops.object.add(type='CAMERA')
    camera = bpy.data.objects['Camera']
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
    # If you get error on this, you might need to change "Color Ramp" to "ColorRamp"
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
    # If you get error on this, you might need to change "Color Ramp" to "ColorRamp"
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
    scene.cycles.device = 'GPU'
    bpy.context.scene.cycles.samples = 64

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

    # importing objects
    # load bunny
    bpy.ops.import_scene.obj(filepath=rabbit_config_dict['model_path'])
    rigid = bpy.data.objects[rabbit_config_dict['model_name']]
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

    for i in range(obj_list_num):
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