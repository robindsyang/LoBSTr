import numpy as np
import torch
import zmq
from load_configs import input_mean, input_std, output_mean, output_std, window_size

import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)

def standardize(sequence, mean, std):
    eps = 1e-15
    n_sequence = (sequence - mean) / (std + eps)
    return n_sequence

def destandardize(sequence, mean, std):
    dn_sequence = sequence * std + mean
    return dn_sequence

device = torch.device('cuda')
model = torch.load('models/LoBSTr_0727_181224/LoBSTr_GRU_60', map_location=torch.device('cuda'))
# device = torch.device('cpu')
# model = torch.load('models/LoBSTr_0727_181224/LoBSTr_GRU_60', map_location=torch.device('cpu'))
model.eval()

context = zmq.Context()
replysocket = context.socket(zmq.REP)
replysocket.bind('tcp://*:3550')
print("server initiated")

while True:
    msg = replysocket.recv()
    message = np.fromstring(msg, dtype=np.float32, sep=',')
    message = np.reshape(message, (window_size, -1))

    input = torch.from_numpy(message).float().to(device)
    input = torch.unsqueeze(input, 0)
    input = standardize(input, input_mean, input_std)

    predicted, contact = model(input)

    predicted = destandardize(predicted, output_mean, output_std)
    predicted = torch.squeeze(predicted)
    predicted = predicted.detach().cpu().numpy()

    contact = torch.squeeze(contact)
    out_contact = torch.zeros(2)

    p_layer = torch.nn.Softmax()
    contact[:2] = p_layer(contact[:2])
    contact[2:] = p_layer(contact[2:])

    if (contact[1] > 0.5):
        out_contact[0] = 1

    if (contact[3] > 0.5):
        out_contact[1] = 1

    out_contact = out_contact.detach().cpu().numpy()

    output = np.concatenate((predicted, out_contact))

    output = ' '.join(str(x) for x in output)
    replysocket.send_string(output)


