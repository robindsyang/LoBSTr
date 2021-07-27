import torch
import torch.nn as nn

nn.Module.dump_patches = True

class LoBSTr_GRU(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, latent_dim):
        super(LoBSTr_GRU, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(self.input_dim, self.hidden_dim, bias=True, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, output_dim)
        self.contact1 = nn.Linear(latent_dim, 4)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.contact1.weight)

    def forward(self, input_tensor):
        hidden = torch.zeros(1, input_tensor.shape[0], self.hidden_dim).to(self.device)
        out, hidden = self.gru(input_tensor, hidden)
        gru_out = out[:, -1]
        out = self.fc1(gru_out)
        latent_vector = self.relu(out)

        lower_pose = self.fc2(latent_vector)
        contact = self.contact1(latent_vector)

        return lower_pose, contact


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LoBSTr_GRU(4 * 4 * 4, 6 * 4 * 4, 1024, 256).to(device)
    input = torch.rand((2, 60, 4 * 4 * 4)).to(device)
    output_pose, output_contact = model(input)
    print(output_pose.shape)
    print(output_contact.shape)
