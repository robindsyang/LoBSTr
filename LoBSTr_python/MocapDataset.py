import torch
import numpy as np
import time
from torch.utils.data import Dataset, DataLoader


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
        self.contact = dataset_npz['contact']

        self.window_size = window_size

    def __len__(self):
        return len(self.name)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        name = self.name[idx]
        local = self.local[idx]
        world = self.world[idx]
        reflocal = self.reflocal[idx]
        bodyvel = self.bodyvel[idx]
        refvel = self.refvel[idx]
        contact = self.contact[idx]

        np.random.seed(int((time.time()*10000)%1000))
        frame = np.random.randint(local.shape[0] - self.window_size + 1)

        # sample = {'name': name,
        #           'local': torch.tensor(local[frame:frame + self.window_size]),
        #           'world': torch.tensor(world[frame:frame + self.window_size]),
        #           'reflocal': torch.tensor(reflocal[frame:frame + self.window_size]),
        #           'bodyvel': torch.tensor(bodyvel[frame:frame + self.window_size]),
        #           'refvel': torch.tensor(refvel[frame:frame + self.window_size])}

        sample = {'input': torch.tensor(refvel[frame:frame + self.window_size, 6:].reshape(60, -1)).float(),
                  'output_pose': torch.tensor(reflocal[frame + self.window_size - 1, :6].flatten()).float(),
                  'output_contact': torch.tensor(contact[frame + self.window_size - 1]).float()}

        return sample


if __name__ == '__main__':
    mocap_dataset = MocapDataest('dataset_EG2021_60fps.npz', 60)
    dataloader = DataLoader(mocap_dataset, batch_size=2, shuffle=True, num_workers=8)

    for i_batch, sample_batched in enumerate(dataloader):
        print("batch num: " + str(i_batch), "batch size: " + str(len(sample_batched['input'])))
        print(sample_batched['input'].shape)
        print(sample_batched['output_pose'].shape)
        print(sample_batched['output_contact'].shape)