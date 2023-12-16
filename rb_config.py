import math
### Very Important Note: The loaded mesh must be water tight for correct signed distance calculation. You can check it by function is_watertight from open3d
rabbit_config_dict = {
    "model_name": "bunny",
    "model_path": './data/models/bunny_final.obj',
    'model_color': (0.9, 0.9, 0.9),
    'model_scale': 1,
    'model_pos': [0, 0, 0],
    'model_rotation': [0, 0, 0],
}

# sample_rock_config_dict = {
#     "model_name": "sample_rock",
#     "model_path": './data/models/Rock_9.obj',
#     'model_color': (0.78, 0.66, 0.082),
#     'model_scale': 0.08,
#     'model_pos': [3.0, 0.0, 3.0],
#     'model_rotation': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
# }