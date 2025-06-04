import torch
from torch_geometric.nn import SAGEConv as Base, BatchNorm
from typing import List, Optional, Tuple, Union
from torch import Tensor
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.typing import Adj, OptPairTensor, Size


class SAGEConv(torch.nn.Module):
    def __init__(self,
                 in_channels: Union[int, Tuple[int, int]],
                 out_channels: int,
                 aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
                 normalize: bool = False,
                 root_weight: bool = True,
                 project: bool = False,
                 bias: bool = True,
                 batch_norm: bool = False,
                 residual: bool = False,
                 activation=None):
        super().__init__()
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.batch_norm_h = BatchNorm(out_channels)
        self.residual = residual
        self.activation = activation
        self.sage_conv = Base(in_channels, out_channels, aggr, normalize=normalize, root_weight=root_weight,
                              project=project, bias=bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, size: Size = None) -> Tensor:
        x_in = x
        x = self.sage_conv(x, edge_index, size)
        # activation
        if self.activation is not None:
            x = self.activation(x)
        # normalization
        if self.batch_norm is not None:
            x = self.batch_norm_h(x)
        # residual
        if self.residual:
            x = x_in + x
        return x
