import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GCNConv, SAGEConv

    
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class GINGNN(nn.Module):
    """
    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        n_layers (int, optional): The number of layers in the model. Defaults to 3.
        cat (bool, optional): Whether to concatenate the output of all layers. Defaults to True.
    """
    def __init__(self, in_channels, out_channels, n_layers = 3, cat = True):
        super(GINGNN, self).__init__()
        self.apply(xavier_init) 

        self.in_channels = in_channels
        self.n_layers = n_layers
        self.cat = cat

        self.gnn_layers = torch.nn.ModuleList()
        for _ in range(n_layers):
            self.gnn_layers.append(GINConv(nn.Sequential(
                                           nn.Linear(in_channels, out_channels),
                                           nn.ReLU(),
                                           nn.Linear(out_channels, out_channels))))
            in_channels = out_channels

    def forward(self, x, edge_index, *args): 
        xs = [x]

        for conv in self.gnn_layers:
            xs += [conv(xs[-1], edge_index)]

        x = torch.cat(xs, dim=-1) if self.cat else xs[-1]

        return x



class GCNGNN(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers = 3, cat = True):
        super(GCNGNN, self).__init__()
        self.n_layers = n_layers
        self.cat = cat

        self.gnn_layers = torch.nn.ModuleList()
        for _ in range(n_layers):
            self.gnn_layers.append(GCNConv(in_channels, out_channels))
            in_channels = out_channels

    def forward(self, x, edge_index, *args):
        xs = [x]

        for conv in self.gnn_layers:
            xs += [conv(xs[-1], edge_index)]

        x = torch.cat(xs, dim=-1) if self.cat else xs[-1]
        return x

class SAGEGNN(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers = 3, cat = True):
        super(SAGEGNN, self).__init__()
        self.n_layers = n_layers
        self.cat = cat
        self.gnn_layers = torch.nn.ModuleList()
        for _ in range(n_layers):
            self.gnn_layers.append(SAGEConv(in_channels, out_channels))
            in_channels = out_channels
       
    def forward(self, x, edge_index, *args):
        xs = [x]

        for conv in self.gnn_layers:
            xs += [conv(xs[-1], edge_index)]

        x = torch.cat(xs, dim=-1) if self.cat else xs[-1]
        return x



def xavier_init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
if __name__ == "__main__":
    torch.manual_seed(42)
    in_channels = 16  # Dimension of node features
    out_channels = 64 # Dimension of node embeddings
    x = torch.randn(100, 16)
    edge_index = torch.randint(100, (2, 400), dtype=torch.long)
    gnn = GINGNN(in_channels, out_channels)
    print(gnn)
    out = gnn(x, edge_index)
    print(out.size())
   
