import torch
import numpy as np
import time
from torch.utils.data import Dataset, DataLoader

# upper indices
upper_indices = [0, 9, 10, 11]
lower_indices = [1, 2, 3, 5, 6, 7]
toebase_indices = [4, 8]


class MocapDataest(Dataset):
    """Motion Capture dataset"""

    def __init__(self, npz_path, window_size):
        dataset_npz = np.load(npz_path, 'r', allow_pickle=True)

        self.name = dataset_npz['name']
        self.local = dataset_npz['local']
        self.world = dataset_npz['world']
        self.reflocal = dataset_npz['reflocal']
        self.bodyvel = dataset_npz['bodyvel']
        self.refvel = dataset_npz['refvel']
        self.window_size = window_size

        for i in range(len(self)):
            if i == 0:
                input = self.refvel[i]
                output = self.reflocal[i]
            else:
                input = np.concatenate((input, self.refvel[i]), axis=0)
                output = np.concatenate((output, self.reflocal[i]), axis=0)

        input = input[:, upper_indices].reshape(input.shape[0], -1)
        output = output[:, lower_indices].reshape(output.shape[0], -1)

        self.input_mean = np.mean(input, axis=0)
        self.input_std = np.std(input, axis=0)
        self.output_mean = np.mean(output, axis=0)
        self.output_std = np.std(output, axis=0)

        eps = 1e-16

        for i in range(len(self)):
            # self.refvel[i] = (self.refvel[i][:, upper_indices].reshape(self.refvel[i].shape[0], -1) - self.input_mean)\
            #                  /(self.input_std + eps)
            # self.reflocal[i] = (self.reflocal[i][:, lower_indices].reshape(self.reflocal[i].shape[0], -1) - self.output_mean)\
            #                    / (self.output_std + eps)
            self.refvel[i] = self.refvel[i][:, upper_indices].reshape(self.refvel[i].shape[0], -1)
            self.reflocal[i] = self.reflocal[i][:, lower_indices].reshape(self.reflocal[i].shape[0], -1)

    def __len__(self):
        return self.name.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        name = self.name[idx]
        local = self.local[idx]
        world = self.world[idx]
        reflocal = self.reflocal[idx]
        bodyvel = self.bodyvel[idx]
        refvel = self.refvel[idx]

        np.random.seed(int((time.time() * 10000) % 1000))
        frame = np.random.randint(local.shape[0] - self.window_size + 1)

        toebase_transformations = world[frame + self.window_size - 2:frame + self.window_size, toebase_indices, :3, 3]
        contact = np.zeros(2)

        if toebase_transformations[1, 0, 1] < 0.05:
            contact[0] = 1
        if toebase_transformations[1, 1, 1] < 0.05:
            contact[1] = 1

        sample = {'input': torch.tensor(refvel[frame:frame + self.window_size]).float(),
                  'gt_pose': torch.tensor(reflocal[frame + self.window_size - 1]).float(),
                  'gt_prev_pose': torch.tensor(reflocal[frame + self.window_size - 2]).float(),
                  'gt_contact': torch.tensor(contact).long()}
                  # 'gt_poss': torch.tensor(toebase_transformations).float()}

        return sample

if __name__ == '__main__':
    mocap_dataset = MocapDataest('dataset_EG2021_60fps.npz', 60)
    dataloader = DataLoader(mocap_dataset, batch_size=2, shuffle=True, num_workers=1)

    for i_batch, sample_batched in enumerate(dataloader):
        print("batch num: " + str(i_batch), "batch size: " + str(len(sample_batched['input'])))
        print(sample_batched['input'].shape)
        print(sample_batched['gt_pose'].shape)
        print(sample_batched['gt_contact'].shape)
        print(sample_batched['gt_poss'].shape)
