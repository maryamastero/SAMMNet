import torch
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from vanilla_models import *
from utils import *
from dataset import *
from mask_atoms import *
import os
torch.manual_seed(0)
np.random.seed(0)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_name = 'SAGE'
in_channels = 111  
out_channels = 512  
n_layers = 3
lr=0.0001
step_size = 20
gamma=0.1
num_epochs = 500
best_valid_loss = float('inf')
patience = 20
current_patience = 0
cat = True
mask_ratio=0.15
batch_size = 1

path = f'pretrain_results/{model_name}/13_12'

if not os.path.exists(path):
    os.makedirs(path)

print('Loading data...')

train_set = MoleculeDataset(root="../data/", filename="train.csv")
validation_set = MoleculeDataset(root="../data/", filename="valid.csv")

train_loader = DataLoader(train_set[:1], batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_set[:1], batch_size=batch_size, shuffle=False)


unique_atom_types = np.load('unique_atom_types.npy')
n_classes = len(unique_atom_types)

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
gnn = gnn.to(device)



out_channels = in_channels + out_channels * n_layers
calssifier = nn.Linear(out_channels, n_classes)
calssifier = calssifier.to(device)

gnn_optimizer = Adam(gnn.parameters(), lr=lr, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(gnn_optimizer, step_size=step_size, gamma=gamma)
classifier_optimizer = Adam(calssifier.parameters(), lr=lr, weight_decay=5e-4)

criterion = nn.CrossEntropyLoss()

print('Starting training...')

train_losses = []
train_accuracies = []
valid_losses = []
valid_accuracies = []
predicteds = []
actuals = []

for epoch in range(num_epochs):
    print('Epoch:', epoch+1)
    gnn.train()
    total_train_loss = 0.0
    total_correct = 0.0
    total_nodes = 0.0
    total_train_accuracies = 0.0

    for step, data in enumerate(train_loader):  
        
        data = data.to(device)
        data = mask_product_atom_labels(data, mask_ratio=mask_ratio)
        
        gnn_optimizer.zero_grad()
        classifier_optimizer.zero_grad()

        node_pred = calssifier(gnn(data.x_p[:data.act_n_p], data.edge_index_p[:data.act_n_p]))

        loss = criterion(node_pred[data.mask], data.mapped_labels)
        loss.backward()
        gnn_optimizer.step()
        classifier_optimizer.step()

        total_train_loss += loss.item() 

        _, predicted_index = node_pred[data.mask].max(dim=1)
        predicted = [index_to_atom_type[p.item()] for p in predicted_index]
        predicted = torch.tensor(predicted)
        
        total_correct += (predicted == data.masked_node_labels).sum().item()
        total_nodes += data.masked_node_labels.size(0)
        train_accuracy = total_correct / total_nodes
        total_train_accuracies += train_accuracy

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    average_train_accuracy = total_train_accuracies / len(train_loader)
    train_accuracies.append(train_accuracy)

    gnn.eval()
    total_valid_loss = 0.0
    total_valid_accuracies = 0.0
    total_correct = 0.0
    total_nodes = 0.0
    total_valid_accuracies = 0.0

    with torch.no_grad():
        for data in validation_loader:

            data = data.to(device)
            data = mask_product_atom_labels(data, mask_ratio=mask_ratio)
            
            node_pred = calssifier(gnn(data.x_p[:data.act_n_p], data.edge_index_p[:data.act_n_p]))

            loss = criterion(node_pred[data.mask], data.mapped_labels)

            total_valid_loss += loss.item() 

            _, predicted_index = node_pred[data.mask].max(dim=1)
            predicted = [index_to_atom_type[p.item()] for p in predicted_index]
            predicted = torch.tensor(predicted)

            total_correct += (predicted == data.masked_node_labels).sum().item()
            total_nodes += data.masked_node_labels.size(0)
            valid_accuracy = total_correct / total_nodes
            total_valid_accuracies = valid_accuracy


    avg_valid_loss = total_valid_loss / len(validation_loader)
    valid_losses.append(avg_valid_loss)
    ave_valid_accuracy = total_valid_accuracies / len(validation_loader)
    valid_accuracies.append(valid_accuracy)

    scheduler.step()

    if avg_valid_loss < best_valid_loss:
        best_valid_loss = avg_valid_loss
        current_patience = 0
        torch.save(gnn.state_dict(), f'{path}/best_model.pth')
    else:
        current_patience += 1
        if current_patience >= patience:
            print("Early stopping at epoch:", epoch)
            break
   
print(100*'-')


print(f"  Model name: {model_name}, n_layers: {n_layers}, out_channels: {out_channels}, lr: {lr}")
print("Train loss: ", train_losses[-1])
print("Valid loss: ", valid_losses[-1])


write_data(f'{path}/train_losses.txt',train_losses)
write_data(f'{path}/valid_losses.txt', valid_losses)

write_data(f'{path}/train_accuracies.txt', train_accuracies)
write_data(f'{path}/valid_accuracies.txt', valid_accuracies)



import json

params = {
    "model_name": model_name,
    "in_channels": in_channels,
    "out_channels": out_channels,
    "n_classes": n_classes,
    "n_layers": n_layers,
    "learning_rate": lr,
    "step_size": step_size,
    "gamma": gamma,
    "patience": 20,
    "num_epochs":epoch+1,
    "train_loss": round(np.mean(train_losses),4),
    "valid_loss": round(np.mean(valid_losses),4),
    "train_accuracy": round(np.mean(train_accuracies),4),
    "valid_accuracy": round(np.mean(valid_accuracies),4),
    "path": path 
}

final_path = os.path.join(path, 'hyperparameters.json')
with open(final_path, 'w') as f:
    json.dump(params, f)

