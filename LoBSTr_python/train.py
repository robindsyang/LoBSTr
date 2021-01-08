import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from load_configs import config_name, device, training_set, valid_set, window_size, architecture, model, \
    optimizer, num_epochs, lr_decay, batch_size, alpha, beta, gamma, delta
import custom_loss as c_loss

import datetime

torch.autograd.set_detect_anomaly(True)

criterion_L1 = nn.L1Loss()
criterion_FKV = c_loss.FK_Velocity_Loss()
criterion_CrossEnt = nn.CrossEntropyLoss()

now = datetime.datetime.now()
nowDatetime = now.strftime('%m%d_%H%M%S')

summary = SummaryWriter()

best_v_loss = 0

try:
    for epoch in range(num_epochs):
        batch_loss = 0
        v_batch_loss = 0

        model.train()
        training_dataloader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=8)
        for training_batch_idx, training_batch in enumerate(training_dataloader):
            # gt_pose, gt_contact, gt_poss = training_batch['gt_pose'].to(device), \
            #                                training_batch['gt_contact'].to(device), \
            #                                training_batch['gt_poss'].to(device)
            gt_prev_pose, gt_pose, gt_contact = training_batch['gt_prev_pose'].to(device), \
                                                training_batch['gt_pose'].to(device), \
                                                training_batch['gt_contact'].to(device)
            output_pose, output_contact = model(training_batch['input'].to(device))

            pose_loss = criterion_L1(output_pose, gt_pose)
            fk_loss, vel_loss = criterion_FKV(output_pose, gt_pose, gt_prev_pose)
            contact_loss = criterion_CrossEnt(output_contact[:, :2], gt_contact[:, 0]) \
                           + criterion_CrossEnt(output_contact[:, 2:], gt_contact[:, 1])
            loss = alpha * pose_loss + beta * fk_loss + gamma * vel_loss + delta * contact_loss

            batch_loss += loss.item() / len(training_batch['input'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        validation_dataloader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=8)
        for valid_batch_idx, valid_batch in enumerate(validation_dataloader):
            # gt_pose, gt_contact, gt_poss = valid_batch['gt_pose'].to(device), \
            #                                valid_batch['gt_contact'].to(device), \
            #                                valid_batch['gt_poss'].to(device)
            gt_prev_pose, gt_pose, gt_contact = valid_batch['gt_prev_pose'].to(device), \
                                                valid_batch['gt_pose'].to(device), \
                                                valid_batch['gt_contact'].to(device)
            output_pose, output_contact = model(valid_batch['input'].to(device))

            pose_loss = criterion_L1(output_pose, gt_pose)
            fk_loss, vel_loss = criterion_FKV(output_pose, gt_pose, gt_prev_pose)
            contact_loss = criterion_CrossEnt(output_contact[:, :2], gt_contact[:, 0]) \
                           + criterion_CrossEnt(output_contact[:, 2:], gt_contact[:, 1])
            loss = alpha * pose_loss + beta * fk_loss + gamma * vel_loss + delta * contact_loss

            v_batch_loss += loss.item() / len(valid_batch['input'])

        batch_loss /= (training_batch_idx + 1)
        v_batch_loss /= (valid_batch_idx + 1)

        print('|Epoch ' + str(epoch)
              + '| avg_loss_anim:' + "%.6f" % batch_loss + ' / avg_v_loss_anim: ' + "%.6f" % batch_loss)

        if epoch == 0:
            best_v_loss = v_batch_loss
            os.mkdir('models/' + config_name + '_' + nowDatetime)
        else:
            if v_batch_loss < best_v_loss:
                best_v_loss = v_batch_loss
                torch.save(model, 'models/' + config_name + '_' + nowDatetime + '/' + str(architecture) + '_' + str(
                    window_size))
                print('model saved at ' + str(epoch))

        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay

except KeyboardInterrupt:
    print('Training aborted.')
