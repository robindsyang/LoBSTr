import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from MocapDataset import MocapDataest
import network_architecture as netarch
import custom_loss as c_loss

import datetime
import yaml
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(r'config.yaml') as file:
    configs = yaml.load(file, Loader=yaml.FullLoader)['EG2021']

training_set = MocapDataest(configs['training_set'], configs['window_size'])
valid_set = MocapDataest(configs['valid_set'], configs['window_size'])

input_dim, output_dim, hidden_dim, latent_dim = configs['net_params']
batch_size = configs['batch_size']

model = netarch.LoBSTr_GRU(input_dim, output_dim, hidden_dim, latent_dim).to(device)

alpha, beta, gamma, delta = configs['loss_hparams']

criterion_L1 = nn.L1Loss()
criterion_FKV = c_loss.FK_Velocity_Loss()
criterion_CrossEnt = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=float(configs['lr']))

now = datetime.datetime.now()
nowDatetime = now.strftime('%m-%d_%H:%M')

summary = SummaryWriter()

best_v_loss = 0

try:
    for epoch in range(configs['num_epochs']):
        model.train()
        training_dataloader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=8)

        batch_loss = 0
        v_batch_loss = 0

        for training_batch_idx, training_batch in enumerate(training_dataloader):
            gt_pose, gt_contact = training_batch['gt_pose'].to(device), training_batch['gt_contact'].to(device)
            output_pose, output_contact = model(training_batch['input'].to(device))
            loss = alpha * criterion_L1(output_pose, gt_pose) \
                   + delta * (criterion_CrossEnt(output_contact[:, :2], gt_contact[:, 0])
                              + criterion_CrossEnt( output_contact[:, :2], gt_contact[:, 0]))
            batch_loss += loss.item() / len(training_batch['input'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        validation_dataloader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=8)
        for valid_batch_idx, valid_batch in enumerate(validation_dataloader):
            gt_pose, gt_contact = valid_batch['gt_pose'].to(device), valid_batch['gt_contact'].to(device)
            output_pose, output_contact = model(valid_batch['input'].to(device))
            loss = alpha * criterion_L1(output_pose, gt_pose) \
                   + delta * (criterion_CrossEnt(output_contact[:, :2], gt_contact[:, 0])
                              + criterion_CrossEnt( output_contact[:, :2], gt_contact[:, 0]))
            v_batch_loss += loss.item() / len(valid_batch['input'])

        print('|Epoch ' + str(epoch) + '| loss:' + str(batch_loss) + ' / v_loss: ' + str(v_batch_loss))

        for param_group in optimizer.param_groups:
            param_group['lr'] *= configs['lr_decay']

except KeyboardInterrupt:
    print('Training aborted.')
