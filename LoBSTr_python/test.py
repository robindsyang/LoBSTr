import time
import numpy as np
import torch
from load_configs import input_mean, input_std, output_mean, output_std, valid_set, window_size, device

def standardize(sequence, mean, std):
    eps = 1e-15
    n_sequence = (sequence - mean) / (std+eps)
    return n_sequence

def destandardize(sequence, mean, std):
    dn_sequence = sequence * std + mean
    return dn_sequence

output_dim = output_mean.shape[0]

start = time.time()
model = torch.load("models/LoBSTr_0727_181224/LoBSTr_GRU_60").to(device)
model.eval()

files = ['LocomotionFlat06_001']

for filename in files:
    input_sequence, world_sequence, local_sequence = valid_set.getinput_byname(filename)
    world_sequence = world_sequence[59:, [0, 11, 12, 13]]
    world_sequence = world_sequence[:, :, :3, 1:]
    world_sequence = world_sequence.reshape(world_sequence.shape[0], -1)

    predictions = []
    contacts = []

    for i in range(window_size, input_sequence.shape[0] + 1):
        input_np = input_sequence[i - window_size:i]
        input = torch.from_numpy(input_np).float()
        input = torch.unsqueeze(input, 0).to(device)
        s_input = standardize(input, input_mean, input_std)

        lowerpose, contact = model(s_input)

        predicted = torch.squeeze(lowerpose)
        predicted = destandardize(predicted, output_mean, output_std).squeeze()
        predictions.append(predicted.detach().cpu().numpy())
        contact = torch.squeeze(contact)
        out_contact = torch.zeros(2)

        if (contact[1] > contact[0]):
            out_contact[0] = 1

        if (contact[3] > contact[2]):
            out_contact[1] = 1

        contacts.append(out_contact.detach().cpu().numpy())

    predictions = np.asarray(predictions)
    contacts = np.asarray(contacts)

    predictions = np.concatenate((world_sequence, predictions), axis=1)
    predictions = np.concatenate((predictions, contacts), axis=1)

    np.savetxt(filename + '_LoBSTr_output.csv', predictions, delimiter=',')