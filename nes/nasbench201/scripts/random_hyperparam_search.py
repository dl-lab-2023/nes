import os
import torch
import torch.nn as nn

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision

import numpy as np

import random

from nes.ensemble_selection.create_baselearners import load_nn_module

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)

learning_rates = [0.001, 0.01, 0.1]
optimizers = ['adam', 'sgd']
weight_decays = [0.0001, 0.001, 0.01]

num_experiments = 10
train_epochs = 200
batch_size = 64
train_portion = 0.8

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = torchvision.datasets.CIFAR10(
    root='./data', train=True, transform=transform)
num_train = len(train_data)
indices = list(range(num_train))
split = int(np.floor(train_portion * num_train))

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size,
    sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
    pin_memory=True, num_workers=0, worker_init_fn=np.random.seed(seed_value))

val_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size,
    sampler=torch.utils.data.sampler.SubsetRandomSampler(
        indices[split:num_train]),
    pin_memory=True, num_workers=0, worker_init_fn=np.random.seed(seed_value))


arch_id = 1

# TODO use TAB models + datasets!!!
model_ckpt = f'./NAS-Bench-201-v1_0-e61699.pth'

device = torch.device(f'cuda:0')
dest_dir = os.path.join('experiments', 'run_'+str(arch_id))
scheme = 'nes_re'
dataset = 'cifar10'

print("Loading module...")
# Should be the best architecture obtained from NES
model = load_nn_module(model_ckpt, None, arch_seed=arch_id,
                       init_seed=seed_value,
                       scheme=scheme,
                       dataset=dataset,
                       device=device,
                       save_dir=dest_dir,
                       n_datapoints=40000,
                       nb201=True,
                       oneshot=False)
print("done")

criterion = nn.CrossEntropyLoss()

for experiment in range(num_experiments):

    learning_rate = random.choice(learning_rates)
    optimizer = random.choice(optimizers)
    weight_decay = random.choice(weight_decays)

    print("Random hyperparamter optimization")
    print(f"Learning Rate: {learning_rate}")
    print(f"Optimizer: {optimizer}")
    print(f"Weight Decay: {weight_decay}")
    print("Training...\n")

    if optimizer == 'adam':
        optim = torch.optim.Adam(model.parameters(), learning_rate)
    elif optimizer == 'sgd':
        optim = torch.optim.SGD(model.parameters(), learning_rate)

    for epoch in range(train_epochs):
        print(f"Starting epoch {epoch}...")
        for data, labels in (data_loader):
            optimizer.zero_grad()

            outputs = model(data)
            loss = criterion(outputs, labels)

            for param in model.parameters():
                loss += weight_decay * torch.norm(param)

            loss.backward()
            optimizer.step()

    print(f"Experiment {experiment} completed\n")

print("Max iterations for search reached")
