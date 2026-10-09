import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.distributions.binomial import Binomial
import torch.utils.data as dataloader
import torch.utils.data as data


class Attention(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(Attention, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)

        self.softmax = nn.Softmax()
        if attn_drop > 0:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, embeds):

        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            sp = self.tanh(self.fc(embed)).mean(dim=0)

            beta.append(attn_curr.matmul(sp.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)

        z_mp = 0
        for i in range(len(embeds)):
            z_mp += embeds[i] * beta[i]
        return z_mp, list(beta.data.cpu().numpy())


class Gat_layer(nn.Module):
    src_nodes_dim = 0
    trg_nodes_dim = 1

    nodes_dim = 0
    head_dim = 1

    def __init__(self, hidden_dim, attn_drop, head_num = 4, bias=True):
        super(Gat_layer, self).__init__()
        
        self.head_num = head_num
        self.num_out_features = hidden_dim // head_num

        self.linear_proj = nn.Linear(hidden_dim, head_num * hidden_dim // head_num, bias=True)
        nn.init.xavier_normal_(self.linear_proj.weight, gain=1.414)


        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, head_num, self.num_out_features), requires_grad=True)
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, head_num, self.num_out_features), requires_grad=True)

        nn.init.xavier_normal_(self.scoring_fn_target.data, gain=1.414)
        nn.init.xavier_normal_(self.scoring_fn_source.data, gain=1.414)

        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(hidden_dim))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        self.leakyReLU = nn.LeakyReLU()#
        self.activation = nn.PReLU()

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):

        src_nodes_index = edge_index[0]
        trg_nodes_index = edge_index[1]


        scores_source = scores_source.index_select(0, src_nodes_index)
        scores_target = scores_target.index_select(0, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(0, src_nodes_index)

        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    def explicit_broadcast(self, this, other):

        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        # Explicitly expand so that shapes are the same
        return this.expand_as(other)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):

        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)


        size = list(exp_scores_per_edge.shape)
        size[0] = num_of_nodes

        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)


        neighborhood_sums.scatter_add_(0, trg_index_broadcasted, exp_scores_per_edge)


        return neighborhood_sums.index_select(0, trg_index)

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):

        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()

        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index, num_of_nodes)

        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)

        return attentions_per_edge.unsqueeze(-1)

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):

        size = list(nodes_features_proj_lifted_weighted.shape)
        size[0] = num_of_nodes
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)

        trg_index_broadcasted = self.explicit_broadcast(edge_index[1], nodes_features_proj_lifted_weighted)

        out_nodes_features.scatter_add_(0, trg_index_broadcasted, nodes_features_proj_lifted_weighted)

        return out_nodes_features

    def forward(self, in_nodes_features, edge_index):#d.h  整体特征, edge_idx  剩余边索引

        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.head_num, self.num_out_features)

        scoring_fn_source_curr = self.attn_drop(self.scoring_fn_source)
        scoring_fn_target_curr = self.attn_drop(self.scoring_fn_target)

        scores_source = (nodes_features_proj * scoring_fn_source_curr).sum(dim=-1)# (N, head_num)
        scores_target = (nodes_features_proj * scoring_fn_target_curr).sum(dim=-1)# (N, head_num)


        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source, scores_target, nodes_features_proj, edge_index)

        scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)

        num_of_nodes = in_nodes_features.shape[0]# 节点总数

        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[1], num_of_nodes)


        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge# (E, head_num, num_out_features)

        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes)


        out_nodes_features = out_nodes_features.view(-1, self.head_num * self.num_out_features)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)


class Mp_attn_encoder(nn.Module):
    def __init__(self, h):
        super(Mp_attn_encoder, self).__init__()

        self.node_att = nn.ModuleDict({mp: Gat_layer(h.hidden_dim, h.attn_drop) for mp in h.mp_name})#mp_name = "mam","mdm"
        self.assist_node_att = nn.ModuleDict({mp: Gat_layer(h.hidden_dim, h.attn_drop) for mp in h.assist_mp_name})
        self.mp_att = Attention(h.hidden_dim, h.attn_drop)
        self.assist_mp_att = Attention(h.hidden_dim, h.attn_drop)

        self.mp_name = h.mp_name
        self.assist_mp_name = h.assist_mp_name
        
        self.nei_mask = h.nei_mask#True

        self.nei_rate = h.nei_rate

    def forward(self, d, full=False):
        embeds = []
        assist_embeds = []
        mp_list = self.mp_name[:]
        assist_mp_list = self.assist_mp_name[:]


        for mp in mp_list:

            edge_idx = d.mp_dict[mp]._indices()
            edge_weight = d.mp_dict[mp]._values()

            if self.training and self.nei_mask and not full:
                edge_num = edge_idx.shape[1]
                if edge_num == 0:
                    continue

                remove_num = int(edge_num * self.nei_rate)

                keep_num = max(edge_num - remove_num, 1)

                _, top_indices = torch.topk(edge_weight, k=keep_num, largest=True)

                edge_idx = edge_idx.index_select(1, top_indices)

                edge_weight = edge_weight.index_select(0, top_indices)

            attn_embed = self.node_att[mp](d.h, edge_idx)
            embeds.append(attn_embed)

        z_mp, mp_weight = self.mp_att(embeds)

        for mp in assist_mp_list:
            assist_edge_idx = d.assist_mp_dict[mp]._indices()
            assist_edge_weight = d.assist_mp_dict[mp]._values()


            if self.training and self.nei_mask and not full:
                assist_edge_num = assist_edge_idx.shape[1]
                if assist_edge_num == 0:
                    continue
                assist_remove_num = int(assist_edge_num * self.nei_rate)
                assist_keep_num = max(assist_edge_num - assist_remove_num, 1)
                _, assist_top_indices = torch.topk(assist_edge_weight, k=assist_keep_num, largest=True)

                assist_edge_idx = assist_edge_idx.index_select(1, assist_top_indices)

                assist_edge_weight = assist_edge_weight.index_select(0, assist_top_indices)

            attn_embed = self.assist_node_att[mp](d.h, assist_edge_idx)
            assist_embeds.append(attn_embed)

        assist_z_mp, assist_mp_weight = self.assist_mp_att(assist_embeds)

        if not self.training:
            for mp, w in zip(mp_list, mp_weight):
                print("{} {:.3f}".format(mp, w), end=" ")
            print()

        return z_mp, assist_z_mp
    def diffusion_forward(self, d, full=False):
        embeds = []  #
        mp_list = self.mp_name[:]
        for mp in mp_list:

            edge_idx = d.rebuild_mp[mp]._indices()
            edge_weight = d.rebuild_mp[mp]._values()
            attn_embed = self.node_att[mp](d.h, edge_idx)
            embeds.append(attn_embed)

        z_mp, mp_weight = self.mp_att(embeds)

        return z_mp

    def forward2(self, d, full=False):
        embeds = []
        mp_list = self.mp_name[:]

        for mp in mp_list:
            edge_idx = d.mp_dict[mp]._indices()
            edge_weight = d.mp_dict[mp]._values()

            attn_embed = self.node_att[mp](d.h, edge_idx)
            embeds.append(attn_embed)

        z_mp, mp_weight = self.mp_att(embeds)

        return z_mp


class DiffusionData(data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        item = self.data[index]
        return item, index

    def __len__(self):
        return len(self.data)