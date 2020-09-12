# -*- coding: utf-8 -*-
"""
open test loss for graph net
RenMin 20190104
"""

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def graph_pair_sim(gf_1, cd_1, gf_2, cd_2, weight_loc, weight_feat):
    """
    get graph similarity
    """

    gf_1 = F.normalize(gf_1, dim=1)
    gf_2 = F.normalize(gf_2, dim=1)
    
    
    sim_mat = t.mm(gf_1, gf_2.t())

    gate = sim_mat.trace() / gf_1.size(0)

    feat_vector = sim_mat.diagonal()

    #pdb.set_trace()
    feat_vector, sort_index = feat_vector.sort(descending=True)
    #sim_feat = feat_vector.mean()

    sim_feat = feat_vector[:16].mean()


    weight_mat = (feat_vector.unsqueeze(0).expand(-1, gf_1.size(0))-feat_vector.min()) * (feat_vector.unsqueeze(1).expand(gf_1.size(0),-1)-feat_vector.min())
    weight_mat = weight_mat.contiguous().view(1, -1)
    weight_mat = F.normalize(weight_mat, dim=1)
    weight_mat_np = np.zeros(weight_mat.size())
    weight_mat_np[0:,0:] = weight_mat[0:,0:].cpu().detach().numpy()

    loc_mat_1 = cd_1.unsqueeze(1).expand(-1,gf_1.size(0),-1) - cd_1.unsqueeze(0).expand(gf_1.size(0),-1,-1)
    loc_mat_2 = cd_2.unsqueeze(1).expand(-1,gf_1.size(0),-1) - cd_2.unsqueeze(0).expand(gf_1.size(0),-1,-1)
    loc_dis_mat = (((loc_mat_1 - loc_mat_2)**2).sum(2))
    loc_dis_mat = loc_dis_mat.contiguous().view(1, -1)
    dis_loc = (loc_dis_mat * weight_mat).mean()

    return weight_feat*sim_feat - weight_loc*dis_loc
    



class GraphSim(nn.Module):
    """
    graph loss for GraphNet
    """
    def __init__(self, weight_loc=1., weight_feat=1.):
        super(GraphSim, self).__init__()
        self.weight_loc = weight_loc
        self.weight_feat = weight_feat
        
    def forward(self, graph_feat_a, coord_a, graph_feat_b, coord_b):
        """
        graph_feat: (N_rot, Num_nodes, Feature)
        coord: (N_rot, Num_nodes, 2)
        """
        sims = []
        for i in range(graph_feat_a.size(0)): 
            gf_a = graph_feat_a[i]
            cd_a = coord_a[i]
            for j in range(graph_feat_b.size(0)):
                gf_b = graph_feat_b[j]
                cd_b = coord_b[j]
            
                sim = graph_pair_sim(gf_a, cd_a, gf_b, cd_b, self.weight_loc, self.weight_feat)

                sims.append(sim.item())

        sim_r = max(sims)

        return sim_r

def feature_sim(feature_a, feature_b):
    """ cosine similarity"""
    sims = []

    for i in range(feature_a.size(0)):
        f_a = feature_a[i].unsqueeze(0)
        for j in range(feature_b.size(0)):
            f_b = feature_b[j].unsqueeze(0)

            sim = t.mm(f_a,f_b.t()) / (t.mm(f_a,f_a.t())*t.mm(f_b,f_b.t()))**0.5

            sims.append(sim.item())

    sim_r = max(sims)

    return sim_r