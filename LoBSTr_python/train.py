import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import datetime
from MocapDataset import MocapDataest
import network_architecture as netarch
import custom_loss as c_loss

from tensorboardX import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

training_set = MocapDataest('dataset_EG2021_60fps.npz', 60)
valid_set = MocapDataest('dataset_EG2021_60fps.npz', 60)

window_size = 60
input_dim = 4 * 4 * 4
output_dim = 6 * 4 * 4
hidden_dim = 1024
latent_dim = 256
batch_size = 1

model = netarch.LoBSTr_GRU(input_dim, output_dim, hidden_dim, latent_dim).to(device)

num_epochs = 1000
lr = 1e-3
lr_decay = 0.999

alpha = 1
beta = 0.1
gamma = 0.1
delta = 1e-6

criterion_L1 = nn.L1Loss()
criterion_FKV = c_loss.FK_Velocity_Loss()
criterion_CrossEnt = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

now = datetime.datetime.now()
nowDatetime = now.strftime('%m-%d_%H:%M')
print('Training for %d epochs' % num_epochs)
summary = SummaryWriter()

best_v_loss = 0

try:
    for epoch in range(num_epochs):
        model.train()
        training_dataloader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=8)

        batch_loss = 0
        v_batch_loss = 0

        for training_batch_idx, training_batch in enumerate(training_dataloader):
            output_pose, output_contact = model(training_batch['input'].to(device))
            loss = criterion_L1(output_pose, training_batch['output_pose'].to(device))
            batch_loss += loss.item() / len(training_batch['input'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        validation_dataloader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=8)
        for valid_batch_idx, valid_batch in enumerate(validation_dataloader):
            v_output_pose, v_output_contact = model(valid_batch['input'].to(device))
            v_loss = criterion_L1(v_output_pose, valid_batch['output_pose'].to(device))
            v_batch_loss += v_loss.item() / len(valid_batch['input'])

        print('|Epoch ' + str(epoch) + '| loss:' + str(batch_loss) + ' / v_loss: ' + str(v_batch_loss))

except KeyboardInterrupt:
    print('Training aborted.')
