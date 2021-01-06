import torch
import torch.nn as nn

nn.Module.dump_patches = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FK_Velocity_Loss(nn.Module):

    def __init__(self):
        super(FK_Velocity_Loss, self).__init__()
        torch.set_printoptions(precision=6)

    def forward(self):
        return