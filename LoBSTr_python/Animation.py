import os
import re
import numpy as np
from scipy.spatial.transform import Rotation as R


class Animation:
    def __init__(self, name, coordinate_system, fps, length, joints, parents,
                 local_transformations,
                 world_transformations):
        self.name = name
        self.coordinate_system = coordinate_system
        self.fps = fps
        self.length = length
        self.joints = joints
        self.parents = parents
        self.local_transformations = local_transformations
        self.world_transformations = world_transformations
        self.reflocal_transformations = None
        self.body_velocities = None
        self.ref_velocities = None

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
        world_transformations = np.zeros((length, joints.shape[0], 4, 4))

        # contact = []
        # toebase_indices = [i for i, joint in enumerate(joints) if "ToeBase" in joint]

        for f in range(length):
            world_transformations[f, 0] = np.eye(4)
            for j in range(1, joints.shape[0] + 1):
                local_t_mat = np.eye(4)
                r = R.from_euler('xyz', data[f, j], degrees=True)
                p = data[f, 0] if j == 1 else offsets[j - 1]
                local_t_mat[:3, 3] = p
                local_t_mat[:3, :3] = r.as_matrix()

                local_transformations[f, j - 1] = local_t_mat
                world_transformations[f, j - 1] = np.matmul(world_transformations[f, parents[j - 1]], local_t_mat)

        #     for j in toebase_indices:
        #         if world_transformations[f, j, 1, 3] < 0.05:
        #             contact.append(1)
        #         else:
        #             contact.append(0)
        #
        # contact = np.array(contact).reshape(length, 2)

        # 9d
        # local_transformations = local_transformations[:, :, :3, 1:].reshape(length, joints.shape[0], -1)
        # world_transformations = world_transformations[:, :, :3, 1:].reshape(length, joints.shape[0], -1)

        # 16d
        # local_transformations = local_transformations.reshape(length, joints.shape[0], -1)
        # world_transformations = world_transformations.reshape(length, joints.shape[0], -1)

        return Animation(name, coordinate_system, fps, length, joints, parents, local_transformations,
                         world_transformations)

    def delete_joints(self, joint_names):
        for joint in joint_names:
            idx = np.where(self.joints == joint)[0][0]
            parent_idx = self.parents[idx]
            children_indices = np.where(self.parents == idx)[0]

            for child in children_indices:
                self.parents[child] = parent_idx
                for f in range(self.length):
                    parent_world_t = self.world_transformations[f, parent_idx].reshape(4, 4)
                    child_world_t = self.world_transformations[f, child].reshape(4, 4)
                    child_local_t = np.matmul(np.linalg.inv(parent_world_t), child_world_t)
                    self.local_transformations[f, child] = child_local_t

            indices = np.where(self.parents > idx)
            for id in indices:
                self.parents[id] = self.parents[id] - 1

            self.joints = np.delete(self.joints, idx)
            self.parents = np.delete(self.parents, idx)
            self.local_transformations = np.delete(self.local_transformations, idx, axis=1)
            self.world_transformations = np.delete(self.world_transformations, idx, axis=1)

    def write_csv(self, representation, precision):
        if representation == 'local':
            data = self.local_transformations.copy()
        elif representation == 'reflocal':
            data = self.reflocal_transformations.copy()
        elif representation == 'world':
            data = self.world_transformations.copy()
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

    def compute_reflocal_transform(self):
        reflocal_transformations = self.local_transformations.copy()

        for f in range(self.length):
            basis_matrix = np.transpose(reflocal_transformations[f, 0, :3, :3])
            basis_matrix[2, 1] = 0
            basis_matrix[2] = basis_matrix[2] / np.linalg.norm(basis_matrix[2])
            basis_matrix[1] = np.array([0, 1, 0])
            basis_matrix[0] = np.cross(basis_matrix[1], basis_matrix[2])
            reflocal_transformations[f, 0, :3, :3] = np.transpose(basis_matrix)

            direct_joints = np.where(self.parents == 0)
            direct_joints = np.delete(direct_joints, 0)

            ref_world_t = reflocal_transformations[f, 0].reshape(4, 4)
            for j in direct_joints:
                d_joint_world_t = self.world_transformations[f, j].reshape(4, 4)
                d_joint_reflocal_t = np.matmul(np.linalg.inv(ref_world_t), d_joint_world_t)
                reflocal_transformations[f, j] = d_joint_reflocal_t

        # self.reflocal_transformations = reflocal_transformations.reshape(self.length, self.joints.shape[0], -1)
        self.reflocal_transformations = reflocal_transformations

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
        self.length -= 1

    def downsample_half(self):
        self.local_transformations = self.local_transformations[::2]
        self.world_transformations = self.world_transformations[::2]
        self.length = self.local_transformations.shape[0]

    def add_noise(self, sigma, degree):
        for f in range(self.length):
            for j in range(len(self.joints)):
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


if __name__ == '__main__':
    # a = animation.load_bvh('./data/bvh_test_PFNN.bvh', 'left', 1.0)
    filename = 'LocomotionFlat01_000'
    a, m_a = Animation.load_bvh('./data/PFNN/' + filename + '.bvh', 'left', 0.0594)

    a.downsample_half()
    m_a.downsample_half()

    a.delete_joints(['LHipJoint', 'RHipJoint',
                     'LowerBack', 'Spine', 'Spine1', 'Neck', 'Neck1',
                     'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftFingerBase', 'LeftHandIndex1', 'LThumb',
                     'RightShoulder', 'RightArm', 'RightForeArm', 'RightFingerBase', 'RightHandIndex1', 'RThumb'])
    m_a.delete_joints(['LHipJoint', 'RHipJoint',
                       'LowerBack', 'Spine', 'Spine1', 'Neck', 'Neck1',
                       'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftFingerBase', 'LeftHandIndex1', 'LThumb',
                       'RightShoulder', 'RightArm', 'RightForeArm', 'RightFingerBase', 'RightHandIndex1', 'RThumb'])

    a.add_noise(0.01, 1)
    m_a.add_noise(0.01, 1)

    # a.delete_joints(['LHipJoint', 'LeftUpLeg', 'LeftLeg', 'LeftFoot',
    #                  'RHipJoint', 'RightUpLeg', 'RightLeg', 'RightFoot',
    #                  'LowerBack', 'Spine', 'Spine1', 'Neck', 'Neck1',
    #                  'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftFingerBase', 'LeftHandIndex1', 'LThumb',
    #                  'RightShoulder', 'RightArm', 'RightForeArm', 'RightFingerBase', 'RightHandIndex1', 'RThumb'])

    a.compute_reflocal_transform()
    m_a.compute_reflocal_transform()

    a.compute_velocities()
    m_a.compute_velocities()

    a.write_csv('local', '%1.6f')
    a.write_csv('world', '%1.6f')
    a.write_csv('reflocal', '%1.6f')
    m_a.write_csv('local', '%1.6f')
    m_a.write_csv('world', '%1.6f')
    m_a.write_csv('reflocal', '%1.6f')
    # a.write_csv('body_vel', '%1.6f')
    # a.write_csv('ref_vel', '%1.6f')
