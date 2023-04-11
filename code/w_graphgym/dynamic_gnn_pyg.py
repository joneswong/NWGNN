import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

from graphgym.config import cfg
from graphgym.models.head_pyg import head_dict
from graphgym.models.gnn_pyg import FeatureEncoder, GNNPreMP
from graphgym.models.layer_pyg import (GeneralLayer, GeneralMultiLayer,
                                       BatchNorm1dNode, BatchNorm1dEdge)
from graphgym.init import init_weights
from graphgym.register import register_network


class DynamicGNN(torch.nn.Module):

    def __init__(self, dim_in, dim_out, **kwargs):
        super(DynamicGNN, self).__init__()

        GNNHead = head_dict[cfg.dataset.task]

        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        assert cfg.gnn.layers_pre_mp > 0, "layers_pre_mp = {}".format(cfg.gnn.layers_pre_mp)
        self.pre_mp = GNNPreMP(dim_in, cfg.gnn.dim_inner)
        dim_in = cfg.gnn.dim_inner

        # meta-controller
        self.temperature = nn.Parameter(torch.Tensor([cfg.dynamic_gnn.init_temp]),
            requires_grad=False)
        self.depth_controller = GeneralMultiLayer('linear', cfg.dynamic_gnn.layers_ctrl,
            dim_in, 2, dim_inner=cfg.gnn.dim_inner, final_act=True)

        #self._cand_aggrs = ['mean', 'max', 'self'] if cfg.dynamic_gnn.use_selfagg else ['mean', 'max']
        self._cand_aggrs = cfg.dynamic_gnn.cand_aggrs
        self._aggr_prior = torch.zeros((len(self._cand_aggrs),), dtype=torch.float32)
        if self._cand_aggrs[-1] == 'self' and cfg.dynamic_gnn.compensate:
            compensation = len(self._cand_aggrs) * [.0]
            compensation[-1] = 1.0
            self._aggr_prior = self._aggr_prior - torch.Tensor(compensation)
        self.aggr_controller = GeneralMultiLayer('linear', cfg.dynamic_gnn.layers_ctrl,
            dim_in, len(self._cand_aggrs), dim_inner=cfg.gnn.dim_inner, final_act=True)

        assert cfg.gnn.layers_mp > 0, "layers_mp = {}".format(cfg.gnn.layers_mp)
        assert (cfg.dynamic_gnn.adaptive_aggr and cfg.gnn.layer_type == 'mixaggrconv') or ((not cfg.dynamic_gnn.adaptive_aggr))# and cfg.gnn.layer_type != 'mixaggrconv')
        self.convs = nn.ModuleList()
        kwargs = dict(cand_aggrs=self._cand_aggrs) if cfg.gnn.layer_type == 'mixaggrconv' else dict()
        for i in range(cfg.gnn.layers_mp):
            d_in = dim_in if i == 0 else cfg.gnn.dim_inner
            layer = GeneralLayer(cfg.gnn.layer_type, d_in, cfg.gnn.dim_inner, has_act=True, **kwargs)
            self.convs.append(layer)
        #self._depth_prior = torch.stack([1.0 - torch.arange(cfg.gnn.layers_mp) / float(cfg.gnn.layers_mp), torch.zeros((cfg.gnn.layers_mp, ))]).T
        self._depth_prior = torch.zeros((cfg.gnn.layers_mp, 2), dtype=torch.float32)

        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

        self.apply(init_weights)

    def _apply(self, fn):
        super(DynamicGNN, self)._apply(fn)
        self._aggr_prior = fn(self._aggr_prior)
        self._depth_prior = fn(self._depth_prior)
        return self

    def forward(self, batch):
        batch = self.pre_mp(batch)
        continue_pr = 1
        selected_x = 0

        for i in range(len(self.convs)):
            last_x = batch.x

            if cfg.dynamic_gnn.adaptive_depth:
                # predict the depth by the depth-controllers
                batch.x = last_x.detach() if cfg.dynamic_gnn.stackelberg else last_x
                stop_logits = self.depth_controller(batch).x
                batch.x = last_x
                if cfg.dynamic_gnn.use_gumbel:
                    stop_prob = F.gumbel_softmax(stop_logits + self._depth_prior[i],
                        hard=cfg.dynamic_gnn.is_hard,
                        tau=self.temperature if self.training else 0.02)
                else:
                    stop_prob = F.softmax((stop_logits + self._depth_prior[i]) / self.temperature, dim=-1)
            else:
                stop_prob = torch.Tensor([[.0, 1.0]]).to(last_x.device).repeat(last_x.size(0), 1) \
                    if i == len(self.convs)-1 else torch.Tensor([[1.0, .0]]).to(last_x.device).repeat(last_x.size(0), 1)

            # predict the aggregation op by the aggr controller
            if cfg.dynamic_gnn.adaptive_aggr:
                batch.x = last_x.detach() if cfg.dynamic_gnn.stackelberg else last_x
                aggr_logits = self.aggr_controller(batch).x
                batch.x = last_x
                if cfg.dynamic_gnn.use_gumbel:
                    aggr_prob = F.gumbel_softmax(aggr_logits + self._aggr_prior,
                        hard=cfg.dynamic_gnn.is_hard,
                        tau=self.temperature if self.training else 0.02)
                else:
                    aggr_prob = F.softmax((aggr_logits + self._aggr_prior) / self.temperature, dim=-1)
                #logging.info("{}\t{}".format(i, torch.mean(aggr_prob, 0).detach().cpu().numpy()))
                batch = self.convs[i](batch, aggr_weight=aggr_prob)
            else:
                batch = self.convs[i](batch)
            
            selected_x = selected_x + (continue_pr * stop_prob[:,1]).view(-1, 1) * batch.x
            continue_pr = continue_pr * stop_prob[:,0]

        batch.x = selected_x
        batch = self.post_mp(batch)

        return batch


register_network('dynamic_gnn', DynamicGNN)
