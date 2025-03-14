from typing import Optional, Tuple, Union

import random

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
import torch.nn as nn
import math
import copy

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Size,
)
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)
# from torch_geometric.utils.sparse import set_sparse_value

class NavieGAT(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int = 0,
        heads: int = 2,
        concat: bool = False,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = False,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.fill_value = fill_value

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:

        self.lin = Linear(in_channels, heads * out_channels,
                                bias=False, weight_initializer='glorot')
        
        self.Wq = Linear(heads*(out_channels), heads * out_channels,
                        bias=False, weight_initializer='glorot')
        self.Wk = Linear(heads*(out_channels),heads * out_channels,
                         bias=False, weight_initializer='glorot')

        self.Proj = Linear(heads*(out_channels) + edge_dim ,heads * out_channels)


        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # super().reset_parameters()
        self.lin.reset_parameters()

        zeros(self.bias)

    def forward(self, x: Tensor,
                edge_index: Tensor,
                edge_attr: Tensor = None,
                return_attention_weights=None):

        H, C = self.heads, self.out_channels

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:

        assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
        num_nodes = x.shape[0]
        x = self.lin(x).view(-1, H, C) #[N,H,C]

        alpha = self.edge_updater(edge_index, x=x, num_nodes=num_nodes)

        # alpha融合
        out = self.propagate(edge_index,alpha=alpha,x=x,edge_attr=edge_attr)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
            # alpha = alpha.mean(dim = 1)

        if self.bias is not None:
            out = out + self.bias

        if return_attention_weights:
            return out,(edge_index,alpha)
        else:
            return out

    def edge_update(self,edge_index, x_i,x_j,num_nodes,index) -> Tensor:
        """
        这个函数和edge_updater对应，把点的信息 node[N,H] -> node_i(目标点),node_j(源点) [E,H] ,不加i,j的参数保持不变。
        """
        E,H,D = x_i.shape
        source_features = x_j.view(E,-1)
        target_features = x_i.view(E,-1)

        source_features = self.Wq(source_features).view(E,H,-1)
        target_features = self.Wk(target_features).view(E,H,-1)

        alpha = (source_features * target_features).sum(dim = -1) / math.sqrt(D + 1e-6)

        alpha = softmax(alpha,index=edge_index[0],num_nodes=num_nodes)

        return alpha

    def message(self,alpha,x_j,edge_attr) -> Tensor:
        """
        这个函数和propagate对应，msg融合注意力
        """
        E,H = alpha.shape
        raw_msg = x_j.view(E,-1)
        raw_msg = torch.cat((raw_msg,edge_attr),dim=-1)
        msg = torch.relu(self.Proj(raw_msg)).view(E,H,-1)
        return alpha.unsqueeze(-1) * msg

if __name__ == '__main__':
    
    embed_dim = 200
    type_embed_dim = 100

    edge_index = torch.randint(0,99,size=(2,5000)).cuda(1)
    x = torch.randn(300,embed_dim).cuda(1)

    gat = NavieGAT(in_channels=embed_dim,out_channels=embed_dim,concat=False).cuda(1)
    out = gat(x,edge_index)
    print()