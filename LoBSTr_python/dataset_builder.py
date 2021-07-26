from os import walk
import numpy as np
from Animation import Animation as Anim
import time

scaling_parameter = 0.0594

files = []
path = './data_old/all/'
dataset_name = "dataset_EG2021_60fps"

# path = './data_old/train/'
# dataset_name = "dataset_EG2021_60fps_train"

# path = './data_old/valid/'
# dataset_name = "dataset_EG2021_60fps_valid"

for (dirpath, dirnames, filenames) in walk(path):
    files.extend(filenames)

name_list = []
local_t_list = []
world_t_list = []
reflocal_t_list = []
refworld_t_list = []
body_v_list = []
ref_v_list = []
contact_list = []

print(path + " dataset builder started.")
for file in files:
    start = time.time()
    data = Anim.load_bvh(path + file, 'left', scaling_parameter)
    data.downsample_half()
    # never delete lowebody joints
    data.delete_joints(['LowerBack', 'Spine', 'Spine1', 'Neck', 'Neck1',
                        'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftFingerBase', 'LeftHandIndex1', 'LThumb',
                        'RightShoulder', 'RightArm', 'RightForeArm', 'RightFingerBase', 'RightHandIndex1', 'RThumb'])
    m_data = data.mirror()

    data.compute_world_transform()
    m_data.compute_world_transform()
    # data.add_noise_world(['Hips', 'Head', 'LeftHand', 'RightHand'], 0.01, 1.5)
    # m_data.add_noise_world(['Hips', 'Head', 'LeftHand', 'RightHand'], 0.01, 1.5)
    data.compute_ref_transform()
    m_data.compute_ref_transform()
    data.compute_velocities()
    m_data.compute_velocities()

    name_list.append(data.name)
    name_list.append(m_data.name)
    local_t_list.append(data.local_transformations)
    local_t_list.append(m_data.local_transformations)
    world_t_list.append(data.world_transformations)
    world_t_list.append(m_data.world_transformations)
    reflocal_t_list.append(data.reflocal_transformations)
    reflocal_t_list.append(m_data.reflocal_transformations)
    refworld_t_list.append(data.refworld_transformations)
    refworld_t_list.append(m_data.refworld_transformations)
    body_v_list.append(data.body_velocities)
    body_v_list.append(m_data.body_velocities)
    ref_v_list.append(data.ref_velocities)
    ref_v_list.append(m_data.ref_velocities)
    end = time.time()
    print(data.name + ' & ' + m_data.name + ' done / elapsed_time = ' + str(end - start))

name_list = np.array(name_list)
local_t_list = np.array(local_t_list)
world_t_list = np.array(world_t_list)
reflocal_t_list = np.array(reflocal_t_list)
refworld_t_list = np.array(refworld_t_list)
body_v_list = np.array(body_v_list)
ref_v_list = np.array(ref_v_list)

np.savez_compressed(dataset_name,
                    name=name_list,
                    local=local_t_list,
                    world=world_t_list,
                    reflocal=reflocal_t_list,
                    refworld=refworld_t_list,
                    bodyvel=body_v_list,
                    refvel=ref_v_list,
                    contact=contact_list)
