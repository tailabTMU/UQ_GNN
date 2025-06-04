import torch
from torch.nn import Linear, Dropout
from torch_geometric.nn import BatchNorm
import torch.nn.functional as F
from torch_geometric import nn
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class GraphConvGNN(torch.nn.Module):
    def __init__(self, metadata, num_classes, device, hidden_channels, num_layers=3, dropout=0.5):
        super(GraphConvGNN, self).__init__()
        torch.manual_seed(12345)

        batch = torch.tensor([], dtype=torch.long)
        batch = batch.to(device)
        x = torch.tensor([])
        x = x.to(device)
        self.device = device
        self.initial_batch = batch
        self.initial_x = x

        self.dropout = 0.0
        if 0.0 < dropout < 1.0:
            self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.linear_hidden_ratio = 2

        node_types = []
        for layer_num in range(1, num_layers + 1):
            hetero_conv_dict = {}
            for edge_type in metadata['edges']:
                source = edge_type[0]
                dest = edge_type[2]
                source_size = metadata['num_features'][source]
                dest_size = metadata['num_features'][dest]
                node_types.append(source)
                node_types.append(dest)
                if layer_num > 1:
                    source_size = hidden_channels
                    dest_size = hidden_channels
                if edge_type[0] == edge_type[2]:
                    hetero_conv_dict[edge_type] = nn.GraphConv(source_size, hidden_channels)
                else:
                    hetero_conv_dict[edge_type] = nn.GraphConv((source_size, dest_size), hidden_channels)
            convs = nn.HeteroConv(hetero_conv_dict)
            self.convs.append(convs)

        node_types = list(set(node_types))
        self.batch_norm = torch.nn.ModuleDict({node_type: BatchNorm(hidden_channels) for node_type in node_types})
        linear_hidden = hidden_channels * self.linear_hidden_ratio
        self.lin = Linear(linear_hidden, num_classes)

    def forward(self, x_dict, edge_index_dict, batch_dict):
        layer_pooling_results = []
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            # Apply Relu
            x_dict = {key: F.relu(x_tensor) for key, x_tensor in x_dict.items()}

            x_dict = {key: self.batch_norm[key](x_tensor) for key, x_tensor in x_dict.items()}

            x_visit_mean = self.initial_x
            x_visit_max = self.initial_x
            x_service_mean = self.initial_x
            x_service_max = self.initial_x
            for key in ['visit', 'service']:
                x_tensor = x_dict[key]
                average_pooling = gap(x_tensor, batch_dict[key])
                max_pooling = gmp(x_tensor, batch_dict[key])
                if key == 'visit':
                    x_visit_mean = average_pooling
                    x_visit_max = max_pooling
                elif key == 'service':
                    x_service_mean = average_pooling
                    x_service_max = max_pooling
            layer_mean_pooling = x_visit_mean + x_service_mean
            layer_max_pooling = x_visit_max + x_service_max
            layer_pooling = torch.cat([layer_mean_pooling, layer_max_pooling], dim=1)
            layer_pooling_results.append(layer_pooling)

        final_readout = torch.zeros(layer_pooling_results[0].shape).to(self.device)
        for layer_pooling_result in layer_pooling_results:
            final_readout = final_readout + layer_pooling_result

        if self.dropout > 0.0:
            final_readout = F.dropout(final_readout, p=self.dropout, training=self.training)

        # 3. Apply a final classifier
        res = self.lin(final_readout)
        return res

    def __repr__(self) -> str:
        return 'GraphConvGNN'


class SelfDistillationGraphConvGNN(torch.nn.Module):
    def __init__(self, metadata, num_classes, device, hidden_channels, num_layers=3, dropout=0.5):
        super(SelfDistillationGraphConvGNN, self).__init__()
        torch.manual_seed(12345)

        batch = torch.tensor([], dtype=torch.long)
        batch = batch.to(device)
        x = torch.tensor([])
        x = x.to(device)
        self.device = device
        self.initial_batch = batch
        self.initial_x = x

        self.dropout = 0.0
        if 0.0 < dropout < 1.0:
            self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.linear_hidden_ratio = 2

        node_types = []
        for layer_num in range(1, num_layers + 1):
            hetero_conv_dict = {}
            for edge_type in metadata['edges']:
                source = edge_type[0]
                dest = edge_type[2]
                source_size = metadata['num_features'][source]
                dest_size = metadata['num_features'][dest]
                node_types.append(source)
                node_types.append(dest)
                if layer_num > 1:
                    source_size = hidden_channels
                    dest_size = hidden_channels
                if edge_type[0] == edge_type[2]:
                    hetero_conv_dict[edge_type] = nn.GraphConv(source_size, hidden_channels)
                else:
                    hetero_conv_dict[edge_type] = nn.GraphConv((source_size, dest_size), hidden_channels)
            convs = nn.HeteroConv(hetero_conv_dict)
            self.convs.append(convs)

        node_types = list(set(node_types))
        self.batch_norm = torch.nn.ModuleDict({node_type: BatchNorm(hidden_channels) for node_type in node_types})

        linear_hidden = hidden_channels * self.linear_hidden_ratio

        self.fcs = torch.nn.ModuleList([Linear(linear_hidden, num_classes) for _ in range(num_layers)])

    def forward(self, x_dict, edge_index_dict, batch_dict):
        feature_list = []
        output_list = []

        layer_pooling_results = []
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            # Apply Relu
            x_dict = {key: F.relu(x_tensor) for key, x_tensor in x_dict.items()}

            x_dict = {key: self.batch_norm[key](x_tensor) for key, x_tensor in x_dict.items()}

            x_visit_mean = self.initial_x
            x_visit_max = self.initial_x
            x_service_mean = self.initial_x
            x_service_max = self.initial_x
            for key in ['visit', 'service']:
                x_tensor = x_dict[key]
                average_pooling = gap(x_tensor, batch_dict[key])
                max_pooling = gmp(x_tensor, batch_dict[key])
                if key == 'visit':
                    x_visit_mean = average_pooling
                    x_visit_max = max_pooling
                elif key == 'service':
                    x_service_mean = average_pooling
                    x_service_max = max_pooling
            layer_mean_pooling = x_visit_mean + x_service_mean
            layer_max_pooling = x_visit_max + x_service_max
            layer_pooling = torch.cat([layer_mean_pooling, layer_max_pooling], dim=1)
            layer_pooling_results.append(layer_pooling)

            classifier_readout = torch.zeros(layer_pooling_results[0].shape).to(self.device)
            for layer_pooling_result in layer_pooling_results:
                classifier_readout = classifier_readout + layer_pooling_result

            feature_list.append(classifier_readout)

        for idx, fc in enumerate(self.fcs):
            final_readout = feature_list[idx]
            if self.dropout > 0.0:
                final_readout = F.dropout(final_readout, p=self.dropout, training=self.training)
            output_list.append(fc(final_readout))

        return output_list, feature_list

    def __repr__(self) -> str:
        return 'SelfDistillationGraphConvGNN'

class MCDropoutGraphConvGNN(torch.nn.Module):
    def __init__(self, metadata, num_classes, device, hidden_channels, num_layers=3, dropout=0.5):
        super(MCDropoutGraphConvGNN, self).__init__()
        torch.manual_seed(12345)

        batch = torch.tensor([], dtype=torch.long)
        batch = batch.to(device)
        x = torch.tensor([])
        x = x.to(device)
        self.device = device
        self.initial_batch = batch
        self.initial_x = x

        if dropout <= 0 or dropout >= 1.0:
            raise Exception('Dropout must be between 0 and 1')

        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.linear_hidden_ratio = 2

        node_types = []
        for layer_num in range(1, num_layers + 1):
            hetero_conv_dict = {}
            for edge_type in metadata['edges']:
                source = edge_type[0]
                dest = edge_type[2]
                source_size = metadata['num_features'][source]
                dest_size = metadata['num_features'][dest]
                node_types.append(source)
                node_types.append(dest)
                if layer_num > 1:
                    source_size = hidden_channels
                    dest_size = hidden_channels
                if edge_type[0] == edge_type[2]:
                    hetero_conv_dict[edge_type] = nn.GraphConv(source_size, hidden_channels)
                else:
                    hetero_conv_dict[edge_type] = nn.GraphConv((source_size, dest_size), hidden_channels)
            convs = nn.HeteroConv(hetero_conv_dict)
            self.convs.append(convs)

        node_types = list(set(node_types))
        self.batch_norm = torch.nn.ModuleDict({node_type: BatchNorm(hidden_channels) for node_type in node_types})
        self.dropout_fc = Dropout(dropout)
        self.dropout = dropout

        linear_hidden = hidden_channels * self.linear_hidden_ratio


        self.lin = Linear(linear_hidden, num_classes)

    def forward(self, x_dict, edge_index_dict, batch_dict):
        layer_pooling_results = []
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            # Apply Relu
            x_dict = {key: F.relu(x_tensor) for key, x_tensor in x_dict.items()}

            x_dict = {key: self.batch_norm[key](x_tensor) for key, x_tensor in x_dict.items()}

            x_visit_mean = self.initial_x
            x_visit_max = self.initial_x
            x_service_mean = self.initial_x
            x_service_max = self.initial_x
            for key in ['visit', 'service']:
                x_tensor = x_dict[key]
                average_pooling = gap(x_tensor, batch_dict[key])
                max_pooling = gmp(x_tensor, batch_dict[key])
                if key == 'visit':
                    x_visit_mean = average_pooling
                    x_visit_max = max_pooling
                elif key == 'service':
                    x_service_mean = average_pooling
                    x_service_max = max_pooling
            layer_mean_pooling = x_visit_mean + x_service_mean
            layer_max_pooling = x_visit_max + x_service_max
            layer_pooling = torch.cat([layer_mean_pooling, layer_max_pooling], dim=1)
            layer_pooling_results.append(layer_pooling)

        final_readout = torch.zeros(layer_pooling_results[0].shape).to(self.device)
        for layer_pooling_result in layer_pooling_results:
            final_readout = final_readout + layer_pooling_result

        # 3. Apply dropout
        final_readout = self.dropout_fc(final_readout)

        # 4. Apply a final classifier
        res = self.lin(final_readout)
        return res

    def __repr__(self) -> str:
        return 'MCDropoutGraphConvGNN'
