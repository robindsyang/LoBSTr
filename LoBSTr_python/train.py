import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from load_configs import config_name, device, training_set, valid_set, window_size, architecture, model, \
    optimizer, num_epochs, lr_decay, batch_size, alpha, beta, gamma, delta
import custom_loss as c_loss

import datetime

def standardize(sequence, mean, std):
    eps = 1e-15
    n_sequence = (sequence - mean) / (std+eps)
    return n_sequence

def destandardize(sequence, mean, std):
    dn_sequence = sequence * std + mean
    return dn_sequence

torch.autograd.set_detect_anomaly(True)

criterion_L1 = nn.L1Loss()
criterion_FKV = c_loss.FK_Velocity_Loss()
criterion_CrossEnt = nn.CrossEntropyLoss()

now = datetime.datetime.now()
nowDatetime = now.strftime('%m%d_%H%M%S')

summary = SummaryWriter()

best_v_loss = 0
eps = 1e-15
input_mean = torch.tensor(training_set.input_mean).float().to(device)
input_std = torch.tensor(training_set.input_std).float().to(device)
output_mean = torch.tensor(training_set.output_mean).float().to(device)
output_std = torch.tensor(training_set.output_std).float().to(device)

try:
    for epoch in range(num_epochs):
        batch_loss = 0
        v_batch_loss = 0

        batch_pose_loss = 0
        v_batch_pose_loss = 0
        batch_p_loss = 0
        v_batch_p_loss = 0
        batch_v_loss = 0
        v_batch_v_loss = 0
        batch_contact_loss = 0
        v_batch_contact_loss = 0

        model.train()
        training_dataloader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=0)
        for training_batch_idx, training_batch in enumerate(training_dataloader):
            gt_prev_pose, gt_pose, gt_contact = training_batch['gt_prev_pose'].to(device), \
                                                training_batch['gt_pose'].to(device), \
                                                training_batch['gt_contact'].to(device)

            s_input = standardize(training_batch['input'].to(device), input_mean, input_std)
            s_gt_pose = standardize(gt_pose, output_mean, output_std)

            output_pose, output_contact = model(s_input)

            pose_loss = criterion_L1(output_pose, s_gt_pose)

            ds_output_pose = destandardize(output_pose, output_mean, output_std)
            fk_loss, vel_loss = criterion_FKV(ds_output_pose, gt_pose, gt_prev_pose)

            contact_loss = criterion_CrossEnt(output_contact[:, :2], gt_contact[:, 0]) \
                          + criterion_CrossEnt(output_contact[:, 2:], gt_contact[:, 1])

            loss = alpha * pose_loss + beta * fk_loss + gamma * vel_loss + delta * contact_loss

            batch_pose_loss += pose_loss.item()
            batch_p_loss += fk_loss.item()
            batch_v_loss = vel_loss.item()
            batch_contact_loss = contact_loss.item()

            batch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        validation_dataloader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0)
        for valid_batch_idx, valid_batch in enumerate(validation_dataloader):
            gt_prev_pose, gt_pose, gt_contact = valid_batch['gt_prev_pose'].to(device), \
                                                valid_batch['gt_pose'].to(device), \
                                                valid_batch['gt_contact'].to(device)

            s_input = standardize(valid_batch['input'].to(device), input_mean, input_std)
            s_gt_pose = standardize(gt_pose, output_mean, output_std)

            output_pose, output_contact = model(s_input)

            pose_loss = criterion_L1(output_pose, s_gt_pose)

            ds_output_pose = destandardize(output_pose, output_mean, output_std)
            fk_loss, vel_loss = criterion_FKV(ds_output_pose, gt_pose, gt_prev_pose)

            contact_loss = criterion_CrossEnt(output_contact[:, :2], gt_contact[:, 0]) \
                         + criterion_CrossEnt(output_contact[:, 2:], gt_contact[:, 1])
            loss = alpha * pose_loss + beta * fk_loss + gamma * vel_loss + delta * contact_loss

            v_batch_pose_loss += pose_loss.item()
            v_batch_p_loss += fk_loss.item()
            v_batch_v_loss = vel_loss.item()
            v_batch_contact_loss = contact_loss.item()

            v_batch_loss += loss.item()

        summary.add_scalar('training_pose_loss', batch_pose_loss, epoch)
        summary.add_scalar('training_fk_loss', batch_p_loss, epoch)
        summary.add_scalar('training_velocity_loss', batch_v_loss, epoch)
        summary.add_scalar('training_contact_loss', batch_contact_loss, epoch)
        summary.add_scalar('validation_pose_loss', v_batch_pose_loss, epoch)
        summary.add_scalar('validation_fk_loss', batch_p_loss, epoch)
        summary.add_scalar('validation_velocity_loss', batch_v_loss, epoch)
        summary.add_scalar('validation_contact_loss', v_batch_contact_loss, epoch)

        print('|Epoch ' + str(epoch + 1)
              + '| avg_loss_anim:' + "%.6f" % batch_loss
              + ' / avg_v_loss_anim: ' + "%.6f" % v_batch_loss)

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
