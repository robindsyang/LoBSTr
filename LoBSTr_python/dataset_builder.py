from os import walk
import numpy as np
from Animation import Animation as Anim
import time

scaling_parameter = 0.0594

files = []
path = './data_test/'
for (dirpath, dirnames, filenames) in walk(path):
    files.extend(filenames)

name_list = []
local_t_list = []
world_t_list = []
reflocal_t_list = []
body_v_list = []
ref_v_list = []

for file in files:
    start = time.time()
    data = Anim.load_bvh(path + file, 'left', scaling_parameter)
    data.delete_joints(['LHipJoint', 'RHipJoint',
                        'LowerBack', 'Neck',
                        'LeftShoulder', 'LeftFingerBase', 'LeftHandIndex1', 'LThumb',
                        'RightShoulder', 'RightFingerBase', 'RightHandIndex1', 'RThumb'])
    data.compute_reflocal_transform()
    data.compute_velocities()

    name_list.append(data.name)
    local_t_list.append(data.local_transformations)
    world_t_list.append(data.world_transformations)
    reflocal_t_list.append(data.reflocal_transformations)
    end = time.time()
    print(file + ' done / elapsed_time = ' + str(end - start))

dataset_name = "dataset_EG2021_120fps"
np.savez_compressed(dataset_name,
                    name=name_list,
                    local=local_t_list,
                    world=world_t_list,
                    reflocal=reflocal_t_list,
                    bodyvel=body_v_list,
                    refvel=ref_v_list)