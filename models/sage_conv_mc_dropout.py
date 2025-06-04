import torch
from torch.nn import Linear, Dropout, ModuleList
import torch.nn.functional as F
from layers import SAGEConv, MLPReadout
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool


class SAGEMCDropout(torch.nn.Module):
    def __init__(self, in_dim: int, n_classes: int, net_params):
        super().__init__()
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        aggregator_type = net_params['sage_aggregator']
        n_layers = net_params['L']
        batch_norm = net_params['batch_norm']
        residual = net_params['residual']
        self.readout = net_params['readout']

        self.embedding_h = Linear(in_dim, hidden_dim)
        self.in_feat_dropout = Dropout(in_feat_dropout)
        self.dropout_fc = Dropout(in_feat_dropout)

        self.layers = ModuleList([SAGEConv(hidden_dim, hidden_dim, activation=F.relu,
                                           aggr=aggregator_type, batch_norm=batch_norm, residual=residual) for _ in
                                  range(n_layers - 1)])
        self.layers.append(SAGEConv(hidden_dim, out_dim, activation=F.relu,
                                    aggr=aggregator_type, batch_norm=batch_norm, residual=residual))
        self.MLP_layer = MLPReadout(out_dim, n_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain representaion from nodes
        x = self.embedding_h(x)
        x = self.in_feat_dropout(x)
        # 2. Obtain node embeddings
        for conv in self.layers:
            x = conv(x, edge_index)

        # 3. Readout layer
        if self.readout == "mean":
            x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        elif self.readout == "max":
            x = global_max_pool(x, batch)  # [batch_size, hidden_channels]
        elif self.readout == "sum":
            x = global_add_pool(x, batch)  # [batch_size, hidden_channels]
        else:
            x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 4. Apply dropout
        x = self.dropout_fc(x)

        # 5. Apply a final classifier
        x = self.MLP_layer(x)

        return x
