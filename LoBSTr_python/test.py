import time
import numpy as np
import torch
from load_configs import training_set, valid_set, window_size, device

def normalize_np(sequence, mean, std):
    eps = 1e-15
    n_animation = np.divide((sequence - mean), std + eps)
    return n_animation

def denormalize_np(sequence, mean, std):
    dn_animation = (np.multiply(sequence, std) + mean).astype(np.float32)
    return dn_animation

output_mean = training_set.output_mean
output_std = training_set.output_std
output_dim = output_mean.shape[0]

start = time.time()
model = torch.load("models/LoBSTr_0416_135009/LoBSTr_GRU_60").to(device)
model.eval()

files = ['LocomotionFlat06_001']

for filename in files:
    input_sequence, world_sequence = valid_set.getinput_byname(filename)
    world_sequence = world_sequence[59:, [0, 11, 12, 13]]
    world_sequence = world_sequence[:, :, :3, 1:]
    world_sequence = world_sequence.reshape(world_sequence.shape[0], -1)

    predictions = []
    contacts = []

    for i in range(window_size, input_sequence.shape[0] + 1):
        input_np = input_sequence[i - window_size:i]
        input = torch.from_numpy(input_np).float()
        input = torch.unsqueeze(input, 0).to(device)

        lowerpose, contact = model(input)

        predicted = torch.squeeze(lowerpose)
        predictions.append(predicted.detach().cpu().numpy())
        contact = torch.squeeze(contact)
        out_contact = torch.zeros(2)

        if (contact[1] > contact[0]):
            out_contact[0] = 1

        if (contact[3] > contact[2]):
            out_contact[1] = 1

        contacts.append(out_contact.detach().cpu().numpy())

    print(world_sequence.shape)

    predictions = np.asarray(predictions)
    print(predictions.shape)

    contacts = np.asarray(contacts)
    print(contacts.shape)

    predictions = np.concatenate((world_sequence, predictions), axis=1)
    predictions = np.concatenate((predictions, contacts), axis=1)

    print(predictions.shape)

    np.savetxt(filename + '_LoBSTr_inputoutput.csv', predictions, delimiter=',')