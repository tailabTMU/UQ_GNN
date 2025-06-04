import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm
from torch_geometric.nn import global_mean_pool


class GraphClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers, num_classifier_layers):
        super(GraphClassifier, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        in_hidden_dim = input_dim
        for i in range(num_layers):
            conv = SAGEConv(in_hidden_dim, hidden_dim)
            in_hidden_dim = hidden_dim
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(in_hidden_dim))

        fc_modules = []
        num_in_layers = in_hidden_dim
        for i in range(num_classifier_layers):
            num_output_layers = (num_in_layers // 2)
            if i == num_classifier_layers - 1:
                num_output_layers = num_classes
            fc = Linear(num_in_layers, num_output_layers)
            num_in_layers = num_output_layers
            fc_modules.append(fc)
            if i != num_classifier_layers - 1:
                fc_modules.append(torch.nn.ReLU())

        self.fcs = torch.nn.Sequential(*fc_modules)

    def forward(self, x, edge_index, batch):

        # 1. Obtain node embeddings
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = batch_norm(x)
            x = F.relu(x)

        # 2. Apply readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = self.fcs(x)

        return x


class GraphClassifierSelfDistillation(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers, num_classifier_layers):
        super(GraphClassifierSelfDistillation, self).__init__()

        self.fcs = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        in_hidden_dim = input_dim
        for i in range(num_layers):
            conv = SAGEConv(in_hidden_dim, hidden_dim)
            in_hidden_dim = hidden_dim
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(in_hidden_dim))

        for i in range(num_layers):
            fc_modules = []
            num_in_layers = in_hidden_dim
            for i in range(num_classifier_layers):
                num_output_layers = (num_in_layers // 2)
                if i == num_classifier_layers - 1:
                    num_output_layers = num_classes
                fc = Linear(num_in_layers, num_output_layers)
                num_in_layers = num_output_layers
                fc_modules.append(fc)
                if i != num_classifier_layers - 1:
                    fc_modules.append(torch.nn.ReLU())
            self.fcs.append(torch.nn.Sequential(*fc_modules))

    def forward(self, x, edge_index, batch):
        feature_list = []
        output_list = []
        # 1. Obtain node embeddings and apply readout to make the features of every classifier
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = batch_norm(x)
            x = F.relu(x)
            feature_list.append(global_mean_pool(x, batch))  # [batch_size, hidden_channels]

        # 2. Apply a final classifier
        for idx, fc in enumerate(self.fcs):
            output_list.append(fc(feature_list[idx]))

        return output_list, feature_list
