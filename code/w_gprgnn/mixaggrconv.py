from typing import List, Optional, Set

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter
from torch_scatter import gather_csr, scatter, segment_csr
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops, is_undirected

from torch_geometric.nn.inits import glorot, zeros


class MixAggrConvLayer(MessagePassing):
    r"""GNN layer with a mixture of aggregation ops
    """

    def __init__(self, in_channels, out_channels, cand_aggrs, bias=True, improved=False,
                 cached=False, normalize_adj=False, self_msg='none', **kwargs):
                 

        self._cand_aggrs = cand_aggrs

        super(MixAggrConvLayer, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize_adj
        self.self_msg = self_msg

        self.weights = nn.ParameterList()
        for _ in range(len(self._cand_aggrs)):
            self.weights.append(Parameter(torch.Tensor(in_channels, out_channels)))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        for i in range(len(self.weights)):
            glorot(self.weights[i])
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        if is_undirected(edge_index, num_nodes=num_nodes):
            deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        else:
            raise NotImplementedError
            #if cfg.gnn.flow == 'source_to_target':
            #    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
            #else:
            #    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
            #deg_inv_sqrt = deg.pow(-1.0)
            #deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            #norm = (deg_inv_sqrt[row] if cfg.gnn.flow == 'source_to_target' else deg_inv_sqrt[col]) * edge_weight

        return edge_index, norm

    def forward(self, x, edge_index, edge_weight=None, edge_feature=None, aggr_weight=None):
        """"""
        # for `aggregate()`
        if aggr_weight is not None:
            self._aggr_weight = aggr_weight
        else:
            self._aggr_weight = len(self._cand_aggrs) * [0]
            self._aggr_weight[0] = 1.0
            self._aggr_weight = torch.Tensor(self._aggr_weight).to(x.device).view(1, -1).repeat(x.size(0), 1)

        if self._cand_aggrs[-1] == 'self':
            x_self = torch.matmul(x, self.weights[-1])
            x = torch.cat([torch.matmul(x, self.weights[i]) for i in range(len(self.weights)-1)], -1)
        else:
            x = torch.cat([torch.matmul(x, self.weights[i]) for i in range(len(self.weights))], -1)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if self.normalize:
                edge_index, norm = self.norm(edge_index, x.size(self.node_dim),
                                             edge_weight, self.improved,
                                             x.dtype)
            else:
                if self.self_msg == 'none':
                    # add self-loop to ensure the final output has considered h_{v}^{l-1}
                    edge_index, edge_weight = add_remaining_self_loops(
                        edge_index, edge_weight, 2 if self.improved else 1, x.size(self.node_dim))
                norm = edge_weight
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        x_msg = self.propagate(edge_index, x=x, norm=norm,
                               edge_feature=edge_feature)
        if self._cand_aggrs[-1] == 'self':
            return x_msg + torch.reshape(self._aggr_weight[:,-1], (-1, 1)) * x_self
        else:
            return x_msg

    def message(self, x_j, norm, edge_feature):
        if edge_feature is None:
            return norm.view(-1, 1) * x_j if norm is not None else x_j
        else:
            return norm.view(-1, 1) * (
                    x_j + edge_feature) if norm is not None else (
                    x_j + edge_feature)

    def aggregate(self, inputs: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """
        if ptr is not None:
            ptr = expand_left(ptr, dim=self.node_dim, dims=inputs.dim())
            return segment_csr(inputs, ptr, reduce=self.aggr)
        else:
            results = 0
            for i, aggr in enumerate(self._cand_aggrs):
                if (i + 1) == len(self._cand_aggrs) and self._cand_aggrs[-1] == 'self':
                    break
                results += self._aggr_weight[:,i:i+1] * scatter(inputs[:,i*self.out_channels:(i+1)*self.out_channels], index, dim=self.node_dim, dim_size=dim_size,
                           reduce=self._cand_aggrs[i])
            return results

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
