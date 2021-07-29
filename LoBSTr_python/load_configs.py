import torch
from MocapDataset import MocapDataest
import network_architecture as net_arch
import yaml

print("loading configs")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# config name
config_name = 'LoBSTr'

with open(r'config.yaml') as file:
    configs = yaml.load(file, Loader=yaml.FullLoader)[config_name]

training_set = MocapDataest(configs['training_set'], configs['window_size'], configs['batch_size'])
valid_set = MocapDataest(configs['valid_set'], configs['window_size'], configs['batch_size'])

window_size = configs['window_size']
input_dim, output_dim, hidden_dim, latent_dim = configs['net_params']
architecture = configs['model']
if architecture == 'LoBSTr_GRU':
    model = net_arch.LoBSTr_GRU(input_dim, output_dim, hidden_dim, latent_dim).to(device)
alpha, beta, gamma, delta = configs['loss_hparams']

lr = configs['lr']
lr_decay = configs['lr_decay']
if configs['optimizer'] == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

num_epochs = configs['num_epochs']
batch_size = configs['batch_size']

input_mean = torch.tensor(training_set.input_mean).float().to(device)
input_std = torch.tensor(training_set.input_std).float().to(device)
output_mean = torch.tensor(training_set.output_mean).float().to(device)
output_std = torch.tensor(training_set.output_std).float().to(device)

print("configs loaded")

