import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

from mixaggrconv import MixAggrConvLayer


class InstancewiseGNN(torch.nn.Module):

    def __init__(self, dataset, args):
        super(InstancewiseGNN, self).__init__()

        self.use_bn = args.use_bn
        self.dropout = args.dropout
        self.adaptive_depth = not args.ablate_depth
        self.adaptive_aggr = not args.ablate_aggr
        self.init_temp = args.init_temp

        # pre-GNN layer
        self.pre_mp = nn.Linear(dataset.num_features, args.hidden, bias=not self.use_bn)
        dim_in = args.hidden
        self.pre_post_layer = []
        if self.use_bn:
            self.pre_post_layer.append(nn.BatchNorm1d(dim_in))
        self.pre_post_layer.append(nn.Dropout(p=self.dropout))
        self.pre_post_layer.append(nn.ReLU())
        self.pre_post_layer = nn.Sequential(*self.pre_post_layer)

        # meta-controller
        self.temperature = nn.Parameter(torch.Tensor([self.init_temp]),
            requires_grad=False)
        self.depth_controller = nn.Linear(dim_in, 2)
        self._cand_aggrs = ['mean', 'max', 'self'] if args.use_selfagg else ['mean', 'max']
        self.aggr_controller = nn.Linear(dim_in, len(self._cand_aggrs))

        # backbone
        self.convs = nn.ModuleList()
        self.convs_post_layer = nn.ModuleList()
        for i in range(args.max_depth):
            layer = MixAggrConvLayer(args.hidden, args.hidden, self._cand_aggrs, bias=not self.use_bn)
            self.convs.append(layer)
            layer_wrapper = []
            if self.use_bn:
                layer_wrapper.append(nn.BatchNorm1d(dim_in))
            layer_wrapper.append(nn.Dropout(p=self.dropout))
            layer_wrapper.append(nn.ReLU())
            self.convs_post_layer.append(nn.Sequential(*layer_wrapper))

        # classifier
        self.post_mp = nn.Linear(dim_in, dataset.num_classes)

    def reset_parameters(self):
        self.pre_mp.reset_parameters()
        self.temperature.data = torch.Tensor([self.init_temp])
        self.depth_controller.reset_parameters()
        self.aggr_controller.reset_parameters()
        for m, post_m in zip(self.convs, self.convs_post_layer):
            m.reset_parameters()
            post_m.reset_parameters()
        self.post_mp.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        continue_pr = 1
        selected_x = 0

        x = self.pre_post_layer(self.pre_mp(x))

        for i in range(len(self.convs)):
            # determine the depth
            if self.adaptive_depth:
                stop_logits = self.depth_controller(x.detach())
                stop_prob = F.softmax(stop_logits / self.temperature, dim=-1)
            else:
                stop_prob = torch.Tensor([[.0, 1.0]]).to(x.device).repeat(x.size(0), 1) \
                    if i == len(self.convs)-1 else torch.Tensor([[1.0, .0]]).to(x.device).repeat(x.size(0), 1)

            # determine and apply the aggr
            if self.adaptive_aggr:
                aggr_logits = self.aggr_controller(x.detach())
                aggr_prob = F.softmax(aggr_logits / self.temperature, dim=-1)
                if i == 0:
                    np.save('aggr_prob.npy', aggr_prob.detach().cpu().numpy())
                x = self.convs[i](x, edge_index, aggr_weight=aggr_prob)
            else:
                x = self.convs[i](x, edge_index)
            x = F.dropout(F.relu(x), p=self.dropout, training=self.training)

            selected_x = selected_x + (continue_pr * stop_prob[:,1]).view(-1, 1) * x
            continue_pr = continue_pr * stop_prob[:,0]

        x = self.post_mp(selected_x)

        return F.log_softmax(x, dim=1)
