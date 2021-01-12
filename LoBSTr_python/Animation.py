import os
import re
import numpy as np
from copy import copy, deepcopy
from scipy.spatial.transform import Rotation as R


class Animation:
    def __init__(self, name, coordinate_system, fps, length, joints, parents, local_transformations):
        self.name = name
        self.coordinate_system = coordinate_system
        self.fps = fps
        self.length = length
        self.joints = joints
        self.parents = parents
        self.local_transformations = local_transformations
        self.world_transformations = None
        self.reflocal_transformations = None
        self.refworld_transformations = None
        self.body_velocities = None
        self.ref_velocities = None

    def copy(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def deepcopy(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def load_bvh(filepath, coordinate_system, scaling_parameter):
        base = os.path.basename(filepath)
        name = os.path.splitext(base)[0]
        bvh = open(filepath, 'r')

        joints = []
        parents = []
        offsets = []
        length = 0
        fps = 0

        data = []

        current_joint = 0
        end_site = False

        for line in bvh:
            joint_line = re.match("ROOT\s+(\w+)", line)
            if joint_line == None:
                joint_line = re.match("\s*JOINT\s+(\w+)", line)

            if joint_line:
                joints.append(joint_line.group(1))
                parents.append(current_joint)
                current_joint = len(parents) - 1
                continue

            endsite_line = re.match("\s*End\sSite", line)
            if endsite_line:
                end_site = True
                continue

            offset_line = re.match("\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line)
            if offset_line:
                if not end_site:
                    offsets.append(np.array([offset_line.group(1), offset_line.group(2), offset_line.group(3)]))
                continue

            if "}" in line:
                if end_site:
                    end_site = False
                else:
                    current_joint = parents[current_joint]
                continue

            if "Frames" in line:
                length = int(line.split(' ')[-1])
                continue

            if "Frame Time:" in line:
                fps = int(1 / float(line.split(' ')[-1]))
                continue

            if "HIERARCHY" in line or "{" in line or "CHANNELS" in line or "MOTION" in line:
                continue

            data.append(np.array(line.strip().split(' ')))

        joints = np.asarray(joints, dtype=str)
        offsets = np.asarray(offsets, dtype=np.float32)
        parents = np.asarray(parents, dtype=np.int8)
        data = np.asarray(data, dtype=np.float32)
        data = data.reshape(length, joints.shape[0] + 1, 3)

        # scaler for positions
        offsets *= scaling_parameter
        data[:, 0] *= scaling_parameter

        # to left-handed coordinate system
        if coordinate_system == 'left':
            # x -> -x for positions
            offsets[:, 0] *= -1
            data[:, 0, 0] *= -1
            # z y x -> -z -y x for rotation (right -> left)
            data[:, 1:, :2] *= -1

        # rotation order to 'xyz'
        data[:, 1:] = data[:, 1:, [2, 1, 0]]

        local_transformations = np.zeros((length, joints.shape[0], 4, 4))

        for f in range(length):
            for j in range(1, joints.shape[0] + 1):
                local_t_mat = np.eye(4)
                r = R.from_euler('xyz', data[f, j], degrees=True)
                p = data[f, 0] if j == 1 else offsets[j - 1]
                local_t_mat[:3, 3] = p
                local_t_mat[:3, :3] = r.as_matrix()
                local_transformations[f, j - 1] = local_t_mat

        return Animation(name, coordinate_system, fps, length, joints, parents, local_transformations)

    def mirror(self):
        mirrored = self.deepcopy({})
        mirrored.name = "m_" + mirrored.name

        left_indices = [i for i, joint in enumerate(mirrored.joints) if
                        "LHip" in joint or "Left" in joint or "LThumb" in joint]
        right_indices = [i for i, joint in enumerate(mirrored.joints) if
                         "RHip" in joint or "Right" in joint or "RThumb" in joint]

        temp = mirrored.local_transformations.copy()

        # trajectory x -> -x
        mirrored.local_transformations[:, 0, 0, 3] *= -1

        # switch
        mirrored.local_transformations[:, left_indices, :3, :3] = temp[:, right_indices, :3, :3]
        mirrored.local_transformations[:, right_indices, :3, :3] = temp[:, left_indices, :3, :3]

        # flip
        rot_mat = mirrored.local_transformations[:, :, :3, :3].reshape(-1, 3, 3)
        quat = (R.from_matrix(rot_mat)).as_quat()
        quat[:, [1, 2]] *= -1
        mirrored.local_transformations[:, :, :3, :3] \
            = R.as_matrix(R.from_quat(quat)).reshape(mirrored.length, len(mirrored.joints), 3, 3)

        return mirrored

    def delete_joints(self, joint_names):
        # check local offset
        for joint in joint_names:
            idx = np.where(self.joints == joint)[0][0]
            parent_idx = self.parents[idx]
            children_indices = np.where(self.parents == idx)[0]

            for child in children_indices:
                self.parents[child] = parent_idx
                for f in range(self.length):
                    self.local_transformations[f, child] = np.matmul(self.local_transformations[f, idx],
                                                                     self.local_transformations[f, child])

            indices = np.where(self.parents > idx)
            for id in indices:
                self.parents[id] = self.parents[id] - 1

            self.joints = np.delete(self.joints, idx)
            self.parents = np.delete(self.parents, idx)
            self.local_transformations = np.delete(self.local_transformations, idx, axis=1)

    def downsample_half(self):
        self.local_transformations = self.local_transformations[::2]
        self.length = self.local_transformations.shape[0]

    def compute_world_transform(self):
        world_transformations = np.zeros((self.length, self.joints.shape[0], 4, 4))
        for f in range(self.length):
            world_transformations[f, 0] = np.eye(4)
            for j in range(0, self.joints.shape[0]):
                local_t_mat = self.local_transformations[f, j]
                world_transformations[f, j] = np.matmul(world_transformations[f, self.parents[j]], local_t_mat)

        self.world_transformations = world_transformations

    def add_noise_world(self, noised_joints, sigma, degree):
        for f in range(self.length):
            for joint in noised_joints:
                j = np.where(self.joints == joint)[0][0]
                noise_transformation = np.eye(4)

                # random angle
                phi = np.random.uniform(0, np.pi * 2)
                cos_theta = np.random.uniform(-1, 1)

                theta = np.arccos(cos_theta)
                x = np.sin(theta) * np.cos(phi)
                y = np.sin(theta) * np.sin(phi)
                z = np.cos(theta)

                axis = np.array([x, y, z])

                radian = np.deg2rad(degree)
                rot = R.from_rotvec(radian * axis).as_matrix()

                # random distance
                scale = np.random.normal(0, sigma)
                v = np.random.uniform(-1, 1, size=3)
                v = v / np.linalg.norm(v) * scale

                noise_transformation[:3, :3] = rot
                noise_transformation[:3, 3] = np.transpose(v)
                self.world_transformations[f, j] = np.matmul(self.world_transformations[f, j], noise_transformation)

    def compute_ref_transform(self):
        reflocal_transformations = self.local_transformations.copy()
        refworld_transformations = self.world_transformations.copy()

        for f in range(self.length):
            basis_matrix = np.transpose(reflocal_transformations[f, 0, :3, :3])
            basis_matrix[2, 1] = 0
            basis_matrix[2] = basis_matrix[2] / np.linalg.norm(basis_matrix[2])
            basis_matrix[1] = np.array([0, 1, 0])
            basis_matrix[0] = np.cross(basis_matrix[1], basis_matrix[2])
            reflocal_transformations[f, 0, :3, :3] = np.transpose(basis_matrix)
            refworld_transformations[f, 0, :3, :3] = np.transpose(basis_matrix)

            direct_joints = np.where(self.parents == 0)
            direct_joints = np.delete(direct_joints, 0)

            ref_world_t = reflocal_transformations[f, 0].reshape(4, 4)
            for j in range(1, self.joints.shape[0]):
                d_joint_world_t = self.world_transformations[f, j].reshape(4, 4)
                d_joint_reflocal_t = np.matmul(np.linalg.inv(ref_world_t), d_joint_world_t)
                refworld_transformations[f, j] = d_joint_reflocal_t

                if j in direct_joints:
                    reflocal_transformations[f, j] = d_joint_reflocal_t

        self.reflocal_transformations = reflocal_transformations
        self.refworld_transformations = refworld_transformations

    def compute_velocities(self):
        self.body_velocities = np.delete(self.world_transformations.copy(), 0, 0)
        self.ref_velocities = np.delete(self.world_transformations.copy(), 0, 0)

        for f in range(self.length - 1):
            ref_transform = self.world_transformations[f + 1, 0]
            for j in range(len(self.joints)):
                self.body_velocities[f, j] = np.matmul(np.linalg.inv(self.world_transformations[f, j]),
                                                       self.world_transformations[f + 1, j])
                self.ref_velocities[f, j] = np.matmul(np.linalg.inv(ref_transform), self.body_velocities[f, j])

        # delete first frames to match the length
        self.local_transformations = np.delete(self.local_transformations, 0, 0)
        self.world_transformations = np.delete(self.world_transformations, 0, 0)
        self.reflocal_transformations = np.delete(self.reflocal_transformations, 0, 0)
        self.refworld_transformations = np.delete(self.refworld_transformations, 0, 0)
        self.length -= 1

    def write_csv(self, representation, precision):
        if representation == 'local':
            data = self.local_transformations.copy()
        elif representation == 'reflocal':
            data = self.reflocal_transformations.copy()
        elif representation == 'world':
            data = self.world_transformations.copy()
        elif representation == 'refworld':
            data = self.refworld_transformations.copy()
        elif representation == 'body_vel':
            data = self.body_velocities.copy()
        elif representation == 'ref_vel':
            data = self.ref_velocities.copy()
        else:
            print("wrong representation")
            exit()

        data = data.reshape(self.length, self.joints.shape[0], 4, 4)

        # 9d transformation
        data = data[:, :, :3, 1:].reshape(self.length, -1)
        np.savetxt(str(self.name) + '_' + representation + '.csv', data, delimiter=',', fmt=precision)


if __name__ == '__main__':
    filename = 'LocomotionFlat01_000'
    a = Animation.load_bvh('./data/PFNN/' + filename + '.bvh', 'left', 0.0594)
    a.downsample_half()
    a.delete_joints(['LowerBack', 'Spine', 'Spine1', 'Neck', 'Neck1',
                     'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftFingerBase', 'LeftHandIndex1', 'LThumb',
                     'RightShoulder', 'RightArm', 'RightForeArm', 'RightFingerBase', 'RightHandIndex1', 'RThumb'])

    m_a = a.mirror()

    a.write_csv('local', '%1.6f')
    m_a.write_csv('local', '%1.6f')
    a.compute_world_transform()
    m_a.compute_world_transform()
    # a.add_noise_world(['Hips', 'Head', 'LeftHand', 'RightHand'], 0.01, 1.5)
    # m_a.add_noise_world(['Hips', 'Head', 'LeftHand', 'RightHand'], 0.01, 1.5)
    a.write_csv('world', '%1.6f')
    m_a.write_csv('world', '%1.6f')
    a.compute_ref_transform()
    m_a.compute_ref_transform()

    fk = np.eye(4)
    fk2 = np.eye(4)
    for i in range(1, 5):
        fk = np.matmul(fk, a.reflocal_transformations[0, i])
        fk2 = np.matmul(fk2, a.reflocal_transformations[0, i+5])

    print(fk)
    print(fk2)
    print(a.refworld_transformations[0, 4])
    print(a.refworld_transformations[0, 9])

    a.write_csv('reflocal', '%1.6f')
    m_a.write_csv('reflocal', '%1.6f')
    a.write_csv('refworld', '%1.6f')
    m_a.write_csv('refworld', '%1.6f')
    a.compute_velocities()
    m_a.compute_velocities()
