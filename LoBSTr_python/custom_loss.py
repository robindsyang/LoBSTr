import torch
import torch.nn as nn

torch.autograd.set_detect_anomaly(True)

nn.Module.dump_patches = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion_MSE = nn.MSELoss()


class FK_Velocity_Loss(nn.Module):
    """
    validity checked
    """

    def __init__(self):
        super(FK_Velocity_Loss, self).__init__()
        torch.set_printoptions(precision=6)

    def forward(self, output_pose, gt_pose, gt_prev_pose, gt_pos):
        gt_pose = gt_pose.reshape(-1, 8, 4, 4)

        gt_prev_pose = gt_prev_pose.view(-1, 2, 4, 4, 4)
        gt_pose = gt_pose.view(-1, 2, 4, 4, 4)
        output_pose = output_pose.view(-1, 2, 4, 4, 4)

        gt_prev_fk = torch.eye(4).to(device)
        gt_prev_fk = gt_prev_fk.repeat(2, 1, 1)
        gt_prev_fk = gt_prev_fk.repeat(gt_prev_pose.shape[0], 1, 1, 1)

        gt_fk = torch.eye(4).to(device)
        gt_fk = gt_fk.repeat(2, 1, 1)
        gt_fk = gt_fk.repeat(gt_pose.shape[0], 1, 1, 1)

        output_fk = torch.eye(4).to(device)
        output_fk = output_fk.repeat(2, 1, 1)
        output_fk = output_fk.repeat(output_pose.shape[0], 1, 1, 1)

        for i in range(2):
            for j in range(4):
                gt_prev_fk[:, i] = torch.bmm(gt_prev_fk[:, i], gt_prev_pose[:, i, j])
                gt_fk[:, i] = torch.bmm(gt_fk[:, i], gt_pose[:, i, j])
                output_fk[:, i] = torch.bmm(output_fk[:, i].clone(), output_pose[:, i, j])

        gt_prev_fk = gt_prev_fk[:, :, :3, 3]
        gt_fk = gt_fk[:, :, :3, 3]
        output_fk = output_fk[:, :, :3, 3]

        gt_vel = gt_fk - gt_prev_fk
        output_vel = output_fk - gt_prev_fk

        pos_loss = criterion_MSE(output_fk, gt_fk)
        vel_loss = criterion_MSE(output_vel, gt_vel)

        return pos_loss, vel_loss
