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
contact_list = []

for file in files:
    """ Original """
    start = time.time()
    data = Anim.load_bvh(path + file, 'left', scaling_parameter)
    data.downsample_half()
    data.delete_joints(['LHipJoint', 'RHipJoint',
                        'LowerBack', 'Spine', 'Spine1', 'Neck', 'Neck1',
                        'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftFingerBase', 'LeftHandIndex1', 'LThumb',
                        'RightShoulder', 'RightArm', 'RightForeArm', 'RightFingerBase', 'RightHandIndex1', 'RThumb'])
    data.add_noise(0.01, 1.5)
    data.compute_reflocal_transform()
    data.compute_velocities()

    name_list.append(data.name)
    local_t_list.append(data.local_transformations)
    world_t_list.append(data.world_transformations)
    reflocal_t_list.append(data.reflocal_transformations)
    body_v_list.append(data.body_velocities)
    ref_v_list.append(data.ref_velocities)
    contact_list.append(data.contact)
    end = time.time()
    print(data.name + ' done / elapsed_time = ' + str(end - start))

    """ MIRRORED """
    start = time.time()
    data = data.mirror_sequence()
    data.downsample_half()
    data.delete_joints(['LHipJoint', 'RHipJoint',
                        'LowerBack', 'Spine', 'Spine1', 'Neck', 'Neck1',
                        'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftFingerBase', 'LeftHandIndex1', 'LThumb',
                        'RightShoulder', 'RightArm', 'RightForeArm', 'RightFingerBase', 'RightHandIndex1', 'RThumb'])
    data.add_noise(0.01, 1.5)
    data.compute_reflocal_transform()
    data.compute_velocities()

    name_list.append(data.name)
    local_t_list.append(data.local_transformations)
    world_t_list.append(data.world_transformations)
    reflocal_t_list.append(data.reflocal_transformations)
    body_v_list.append(data.body_velocities)
    ref_v_list.append(data.ref_velocities)
    contact_list.append(data.contact)
    end = time.time()
    print(data.name + ' done / elapsed_time = ' + str(end - start))

dataset_name = "dataset_EG2021_60fps"
np.savez_compressed(dataset_name,
                    name=name_list,
                    local=local_t_list,
                    world=world_t_list,
                    reflocal=reflocal_t_list,
                    bodyvel=body_v_list,
                    refvel=ref_v_list,
                    contact=contact_list)
