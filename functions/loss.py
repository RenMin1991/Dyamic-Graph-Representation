# -*- coding: utf-8 -*-
"""
graph loss for GraphNet
RenMin20181220
"""

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def combine_pair_sim(gf_1, flo_1, gf_2, flo_2):
    """
    combine the graph feature and float feature togather
    """
    gf_1 = F.normalize(gf_1, dim=1)
    gf_2 = F.normalize(gf_2, dim=1)
    flo_1 = F.normalize(flo_1, dim=0)
    flo_2 = F.normalize(flo_2, dim=0)

    graph_1 = gf_1.view(-1)
    graph_2 = gf_2.view(-1)
    
    com_feat_1 = t.cat([graph_1,flo_1],dim=0)
    com_feat_2 = t.cat([graph_2,flo_2],dim=0)

    sim = (com_feat_1*com_feat_2).sum()
    return sim    


def graph_pair_sim(gf_1, cd_1, gf_2, cd_2, weight_loc, weight_feat):
    """
    get graph similarity
    """
    gf_1 = F.normalize(gf_1, dim=1)
    gf_2 = F.normalize(gf_2, dim=1)
    
    
    sim_mat = t.mm(gf_1, gf_2.t())
    sim_mat[sim_mat>0.99] = 0.

    feat_vector = sim_mat.diagonal()
    sim_feat = feat_vector.mean()
    
    weight_mat = (feat_vector.unsqueeze(0).expand(-1, gf_1.size(0))-feat_vector.min()) * (feat_vector.unsqueeze(1).expand(gf_1.size(0),-1)-feat_vector.min())
    weight_mat = weight_mat.contiguous().view(1, -1)
    weight_mat = F.normalize(weight_mat, dim=1)

    loc_mat_1 = cd_1.unsqueeze(1).expand(-1,gf_1.size(0),-1) - cd_1.unsqueeze(0).expand(gf_1.size(0),-1,-1)
    loc_mat_2 = cd_2.unsqueeze(1).expand(-1,gf_1.size(0),-1) - cd_2.unsqueeze(0).expand(gf_1.size(0),-1,-1)
    loc_dis_mat = (((loc_mat_1 - loc_mat_2)**2).sum(2))
    loc_dis_mat = loc_dis_mat.contiguous().view(1, -1)
    dis_loc = (loc_dis_mat * weight_mat).sum()
    
    return weight_feat*sim_feat - weight_loc*dis_loc


class GraphLoss(nn.Module):
    """
    graph loss for GraphNet
    """
    def __init__(self, weight_loc, weight_feat, margin):
        super(GraphLoss, self).__init__()
        self.weight_loc = weight_loc
        self.weight_feat = weight_feat
        self.margin = margin
        
    def forward(self, graph_feat_a, coord_a, graph_feat_p, coord_p, graph_feat_n, coord_n, float_a, float_p, float_n):
        """
        graph_feat: (Batch, Num_nodes, Feature)
        coord: (Batch, Num_nodes, 2)
        """
        loss_b = np.zeros(1)
        loss_b = Variable(t.from_numpy(loss_b))
        loss_b = (loss_b.cuda()).float()
        
        hard_sample = []
        
        for i, gf_a in enumerate(graph_feat_a, 0): # Batch
            cd_a = coord_a[i]
            gf_p = graph_feat_p[i]
            cd_p = coord_p[i]
            gf_n = graph_feat_n[i]
            cd_n = coord_n[i]

            flo_a = float_a[i]
            flo_p = float_p[i]
            flo_n = float_n[i]

            p_sim = graph_pair_sim(gf_a, cd_a, gf_p, cd_p, self.weight_loc, self.weight_feat)
            n_sim_1 = graph_pair_sim(gf_a, cd_a, gf_n, cd_n, self.weight_loc, self.weight_feat)
            n_sim_2 = graph_pair_sim(gf_p, cd_p, gf_n, cd_n, self.weight_loc, self.weight_feat)


            z = np.zeros(1)
            z = t.from_numpy(z)
            z = z.cuda()
            loss_1 = t.max((n_sim_1-p_sim+self.margin).float(), (Variable(z).float()))
            loss_2 = t.max((n_sim_2-p_sim+self.margin).float(), (Variable(z).float()))

            if loss_1+loss_2 > 2.*self.margin:
                hard_sample.append(i)
            
            loss_b = loss_b + loss_1 + loss_2
        loss_b = 0.5 * loss_b/(i+1.)
        return loss_b, hard_sample
