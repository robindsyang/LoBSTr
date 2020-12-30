import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class MocapDataest(Dataset):
    """Motion Capture dataset"""

    def __init__(self, npz_path, window_size, transform=None):
        dataset_npz = np.load(npz_path, 'r', allow_pickle=True)

        self.name = dataset_npz['name']
        self.local = dataset_npz['local']
        self.world = dataset_npz['world']
        self.reflocal = dataset_npz['reflocal']

        self.window_size = window_size
        self.transform = transform

    def __len__(self):
        return len(self.name)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        name = self.name[idx]
        local = self.local[idx]
        world = self.world[idx]
        reflocal = self.reflocal[idx]

        frame = np.random.randint(local.shape[0] - self.window_size + 1)

        sample = {'name': name, 'local': local[frame:frame + self.window_size],
                  'world': world[frame:frame + self.window_size], 'reflocal': reflocal[frame:frame + self.window_size]}

        if self.transform:
            sample = self.transform(sample)

        return sample



if __name__ == '__main__':
    mocap_dataset = MocapDataest('dataset_EG2021_120fps.npz', 120)
    dataloader = DataLoader(mocap_dataset, batch_size=1, shuffle=True, num_workers=8)

    for i_batch, sample_batched in enumerate(dataloader):
        print("batch num: " + str(i_batch), "batch size: " + str(len(sample_batched['name'])))
