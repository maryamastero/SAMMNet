import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from torch_geometric.loader import DataLoader
import os

from vanilla_models import *
from utils import *
from dataset import *

from datetime import datetime
current_time = datetime.now()

torch.manual_seed(0)
np.random.seed(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


model_name = 'SAGE'
in_channels = 111  
out_channels = 512  
n_layers = 3
lr = 0.0001
step_size = 5
gamma = 0.1
num_epochs = 500
best_valid_loss = float('inf')
patience = 20
current_patience = 0
cat = True
max_norm = 1

model_path = f'pretrain_results/{model_name}/21_10'

path = f'finetune_results/{model_name}/13_12'



if not os.path.exists(path):
    os.makedirs(path)

print('Loading data...')

train_set = MoleculeDataset(root="../data/", filename="train.csv")
validation_set = MoleculeDataset(root="../data/", filename="valid.csv")

train_loader = DataLoader(train_set[:1], batch_size=1, shuffle=True, follow_batch=['x_r', 'x_p'])
validation_loader = DataLoader(validation_set[:1], batch_size=1, shuffle=False, follow_batch=['x_r', 'x_p'])

# Define model
print(f'Loading model...')
if model_name == 'SAGE':
    gnn = SAGEGNN(in_channels, out_channels, n_layers=n_layers)
elif model_name == 'GCN':
    gnn = GCNGNN(in_channels, out_channels, n_layers=n_layers)
elif model_name == 'GIN':
    gnn = GINGNN(in_channels, out_channels, n_layers=n_layers)
else:
    raise NotImplementedError

model_filename = f'{model_path}/best_model.pth'
gnn.load_state_dict(torch.load(model_filename))
gnn = gnn.to(device)

# Define optimizer and loss

optimizer = Adam(gnn.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


train_losses = []
valid_losses = []
train_accuracies = []
valid_accuracies = []
train_symmetry_aware_accuracies = []

print('Training...')

for epoch in range(num_epochs):
    print('Epoch:', epoch+1)
    gnn.train()
    total_train_loss = 0.0
    total_train_accuracies = 0.0
    total_symmetry_aware_accuracies = 0.0

    for step, data in enumerate(train_loader):  

        optimizer.zero_grad()

        data = data.to(device)
        h_r = gnn(data.x_r, data.edge_index_r)
        h_p = gnn(data.x_p, data.edge_index_p)

        # Perform matching
        soft_matching = match_nodes(h_p, h_r)

        ground_trouth = data.p2r_mapper
        valid_mask = ~data.y_r != -1  # Mask for actual nodes (we padded product molecule graph to have same size graph as reactant)

        loss = F.nll_loss(F.log_softmax(soft_matching[valid_mask], dim=-1), ground_trouth[valid_mask])
        accuracy = calculate_accuracy(soft_matching, data)
        total_train_accuracies += accuracy
        
        symmetry_aware_mapping = get_symmetry_aware_atom_mapping(soft_matching, data)
        symmetry_aware_accuracy = get_symmetry_aware_accuracy(symmetry_aware_mapping, data)
        total_symmetry_aware_accuracies += symmetry_aware_accuracy

        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        nn.utils.clip_grad_norm_(gnn.parameters(), max_norm)

    # Calculate average training loss for the epoch
    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    average_train_accuracy = total_train_accuracies / len(train_loader)
    train_accuracies.append(average_train_accuracy)
    
    average_symmetry_aware_accuracy = total_symmetry_aware_accuracies / len(train_loader)
    train_symmetry_aware_accuracies.append(average_symmetry_aware_accuracy)


    
    # Validation phase
    gnn.eval()
    total_valid_loss = 0.0
    total_valid_accuracies = 0.0

    with torch.no_grad():
        for data in validation_loader:

            data = data.to(device)

            h_r = gnn(data.x_r, data.edge_index_r)
            h_p = gnn(data.x_p, data.edge_index_p)

            soft_matching = match_nodes(h_p, h_r)

            ground_trouth = data.p2r_mapper
            valid_mask = ~data.y_r != -1  # Mask for valid nodes

            loss = F.nll_loss(F.log_softmax(soft_matching[valid_mask], dim=-1), ground_trouth[valid_mask])
            total_valid_loss += loss.item()

            accuracy = calculate_accuracy(soft_matching, data)
            total_valid_accuracies += accuracy


    avg_valid_loss = total_valid_loss / len(validation_loader)
    valid_losses.append(avg_valid_loss)
    ave_valid_accuracy = total_valid_accuracies / len(validation_loader)
    valid_accuracies.append(ave_valid_accuracy)

    scheduler.step()

    if avg_valid_loss < best_valid_loss:
        best_valid_loss = avg_valid_loss
        current_patience = 0
        torch.save(gnn.state_dict(), f'{path}/best_model.pth')
    else:
        current_patience += 1
        if current_patience >= patience:
            print("Early stopping at epoch:", epoch+1)
            break

print(f"Model name: {model_name}, n_layers: {n_layers}, out_channels: {out_channels}")
print(f'Train Loss: {train_losses[-1]:.4f}, Validation Loss: {valid_losses[-1]:.4f}')
print(f'Training Acc: {average_train_accuracy:.4f} , Symmetry_aware Acc: {average_symmetry_aware_accuracy: .4f}, Validation Acc: {ave_valid_accuracy:.4f}')
print(100*'-')

write_data(f'{path}/losses_train.txt', train_losses)
write_data(f'{path}/accuracies_train.txt', train_accuracies)
write_data(f'{path}/accuracies_valid.txt', valid_accuracies)
write_data(f'{path}/losses_valid.txt', valid_losses)
write_data(f'{path}/train_symmetry_aware_accuracies.txt', train_symmetry_aware_accuracies)



import json

params = {
    "model_name": model_name,
    "in_channels": in_channels,
    "out_channels": out_channels,
    "n_layers": n_layers,
    "learning_rate": lr,
    "step_size": step_size,
    "gamma": gamma,
    "patience": patience,
    "num_epochs":epoch+1,
    "train_loss": avg_train_loss,
    "valid_loss": avg_valid_loss,
    "train_accuracy": average_train_accuracy,
    "valid_accuracy": ave_valid_accuracy,
    f'Symmetry_aware Acc: {average_symmetry_aware_accuracy: .4f}'
    "path": path 
}

with open(f'{path}/hyperparameters.json', 'w') as f:
    json.dump(params, f)

# Get the current time
current_time1 = datetime.now()

# Print the current time
print('time',current_time1-current_time)