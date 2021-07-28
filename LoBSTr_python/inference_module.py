import time
import numpy as np
import torch
import zmq
from load_configs import input_mean, input_std, output_mean, output_std, window_size


def standardize(sequence, mean, std):
    eps = 1e-15
    n_sequence = (sequence - mean) / (std + eps)
    return n_sequence


def destandardize(sequence, mean, std):
    dn_sequence = sequence * std + mean
    return dn_sequence


TIMEOUT = 10000

device = torch.device('cuda')
model = torch.load('models/LoBSTr_0727_181224', map_location=torch.device('cuda'))
# device = torch.device('cpu')
# model = torch.load('models/LoBSTr_0727_181224', map_location=torch.device('cpu'))
model.eval()

context = zmq.Context()
pullsocket = context.socket(zmq.PULL)
pushsocket = context.socket(zmq.PUSH)
pullsocket.bind('tcp://*:12345')
pushsocket.bind('tcp://*:12346')
print("server initiated")

while True:
    # start_time = time.time()
    msg = pullsocket.recv()
    message = np.fromstring(msg, dtype=np.float32, sep=',')
    message = np.reshape(message, (window_size, -1))
    input = torch.from_numpy(message).float().to(device)
    input = standardize(input, input_mean, input_std)
    input = torch.unsqueeze(input, 0)

    lowerpose, contact = model(input)

    lowerpose = torch.squeeze(lowerpose)
    lowerpose = destandardize(lowerpose, output_mean, output_std)
    lowerpose = lowerpose.detach().cpu().numpy()

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
    output = np.concatenate((lowerpose, out_contact))
    output = ' '.join(str(x) for x in output)
    pushsocket.send_string(output)

    # end_time = time.time()
    # print(end_time - start_time)

