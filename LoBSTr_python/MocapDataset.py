import torch
import numpy as np
import time
from torch.utils.data import Dataset, DataLoader

# upper indices
upper_indices = [0, 11, 12, 13]
lower_indices = [1, 2, 3, 4, 6, 7, 8, 9]
toebase_indices = [5, 10]

class MocapDataest(Dataset):
    """Motion Capture dataset"""

    def __init__(self, npz_path, window_size, batch_size):
        dataset_npz = np.load(npz_path, 'r', allow_pickle=True)

        self.data_count = len(dataset_npz['name'])
        self.batch_size = batch_size

        self.name = dataset_npz['name']
        self.local = dataset_npz['local']
        self.world = dataset_npz['world']
        self.reflocal = dataset_npz['reflocal']
        self.refworld = dataset_npz['refworld']
        self.bodyvel = dataset_npz['bodyvel']
        self.refvel = dataset_npz['refvel']
        self.window_size = window_size
        self.input = dataset_npz['refvel']
        self.output = dataset_npz['reflocal']

        # LoBSTr input/output
        for i in range(self.data_count):
            frame = self.input[i].shape[0]
            self.input[i] = self.input[i][:, upper_indices, :3, 1:].reshape(frame, -1)
            self.input[i] = np.concatenate((self.input[i], self.world[i][:, 0, 1, 3].reshape(frame, 1)), axis=1)
            self.output[i] = self.output[i][:, lower_indices, :3, 1:3].reshape(frame, -1)

            # temp = self.refworld[i].copy()
            # temp = temp[:, upper_indices, :3, 1:].reshape(frame, -1)
            #
            # test = np.concatenate((temp[:, :36], self.output[0]), axis=1)
            # test_vel = np.concatenate((self.input[i][:, :36], self.output[i]), axis=1)
            # np.savetxt('LocomotionFlat01_000_LoBSTr_worldoutput.csv', test, delimiter=',')
            # np.savetxt('LocomotionFlat01_000_LoBSTr_inputoutput.csv', test_vel, delimiter=',')
            # exit()

        input_cat = np.vstack(self.input)
        output_cat = np.vstack(self.output)

        self.input_mean = np.mean(input_cat, axis=0, keepdims=True)
        self.input_std = np.std(input_cat, axis=0, keepdims=True)
        self.output_mean = np.mean(output_cat, axis=0, keepdims=True)
        self.output_std = np.std(output_cat, axis=0, keepdims=True)

    def __len__(self):
        return max(self.name.shape[0], self.batch_size)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx = idx % self.data_count

        world = self.world[idx]
        input = self.input[idx];
        output = self.output[idx];

        np.random.seed(int((time.time() * 100000) % 10000))
        frame = np.random.randint(input.shape[0] - self.window_size + 1)

        toebase_transformation = world[frame + self.window_size - 1, toebase_indices, :3, 3]
        contact = np.zeros(2)

        if toebase_transformation[0, 1] < 0.05:
            contact[0] = 1
        if toebase_transformation[1, 1] < 0.05:
            contact[1] = 1

        sample = {'input': torch.tensor(input[frame:frame + self.window_size]).float(),
                  'gt_pose': torch.tensor(output[frame + self.window_size - 1]).float(),
                  'gt_prev_pose': torch.tensor(output[frame + self.window_size - 2]).float(),
                  'gt_contact': torch.tensor(contact).long()}

        return sample

    def getinput_byname(self, name):
        for i in range(self.data_count):
            if self.name[i] == name:
                idx = i;

        return self.input[idx].copy(), self.refworld[idx].copy()

if __name__ == '__main__':
    mocap_dataset = MocapDataest('dataset_EG2021_60fps_train.npz', 60, 256)
    dataloader = DataLoader(mocap_dataset, batch_size=256, shuffle=True, num_workers=1)

    for i_batch, sample_batched in enumerate(dataloader):
        print("batch num: " + str(i_batch), "batch size: " + str(len(sample_batched['input'])))
        print(sample_batched['input'].shape)
        print(sample_batched['gt_pose'].shape)
        print(sample_batched['gt_prev_pose'].shape)
        print(sample_batched['gt_contact'].shape)

