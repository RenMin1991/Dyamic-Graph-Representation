import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, cut_off, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.cut_off = cut_off
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, coord):
        #pdb.set_trace()
        h = torch.matmul(input, self.W)
        B = h.size(0)
        N = h.size(1)

        a_input = torch.cat([h.repeat(1, 1, N).view(B, N * N, -1), h.repeat(1, N, 1)], dim=2).view(B, N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        loc_mat = coord.unsqueeze(2).expand(-1,-1,N,-1) - coord.unsqueeze(1).expand(-1,N,-1,-1)
        loc_mat = (loc_mat**2).sum(3)
        loc_mat = torch.exp(-self.cut_off*loc_mat)
        zero_vec = -9e15*torch.ones_like(e)
        e_loc = e * loc_mat

        attention = torch.where(loc_mat>1e-2, e_loc, zero_vec)


        #adj = torch.eye(N).cuda()
        #attention = torch.where(adj>0., e, zero_vec)

        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



