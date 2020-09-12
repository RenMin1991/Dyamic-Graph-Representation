# -*- coding: utf-8 -*-
'''
graph net model
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from model.graph_function import th_batch_map_coordinates, get_coord, n_Sigmoid
from model.layer_graph import GraphAttentionLayer



class GraphGen(nn.Module):
    def __init__(self, num_node):
        super(GraphGen, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(96, 48, 5, 1, 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(48, 24, 5, 1, 2)
        self.fc1 = nn.Linear(4*8*24, 128)
        self.fc2 = nn.Linear(128, 2*num_node)
        self._init_params()

    def _init_params(self):
        self.fc2.weight.data.normal_(0,0.4)
        self.fc2.bias.data.normal_(0,0.4)

    def forward(self, x):
        x = self.pool1(x)
        x = F.relu(self.conv1(x))
        x = self.pool2(x)
        x = F.relu(self.conv2(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = n_Sigmoid(x, 3)
        # to: Batch, Num_Points, 2
        x = x.view(x.size(0), -1, 2)
        return x


class GraphFeat(nn.Module):
    def __init__(self, num_node):
        super(GraphFeat, self).__init__()
        self.graph_gen = GraphGen(num_node=num_node)

    def forward(self, x):
        #pdb.set_trace()
        x_shape = x.size()
        coord = self.graph_gen(x)
        coord = get_coord(coord, int(x_shape[2]), int(x_shape[3])) # Batch, Num_Points, 2

        coords = coord.unsqueeze(1).expand(-1, int(x_shape[1]), -1, -1)
        coords = coords.contiguous().view(-1, coords.size(2), 2)
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))

        graph_feat = th_batch_map_coordinates(x, coords)
        # to: Batch, Num_Points, Channel
        graph_feat = graph_feat.contiguous().view(-1, int(x_shape[1]), coord.size(1)).permute(0,2,1)

        return graph_feat, coord
        
class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channel, channel//reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channel//reduction, channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #pdb.set_trace()
        b, _, f = x.size()
        y = self.avg_pool(x.permute(0,2,1)).view(b,f)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b,1,f)
        return x * y.expand_as(x)


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout, alpha, cut_off, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.se1 = SELayer(nfeat)
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, cut_off=cut_off, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.se2 = SELayer(nhid * nheads)
        self.out_att = GraphAttentionLayer(nhid * nheads, nout, dropout=dropout, alpha=alpha, cut_off=cut_off, concat=False)

    def forward(self, x, coord):
        #pdb.set_trace()
        x = self.se1(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, coord) for att in self.attentions], dim=2)
        x = self.se2(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, coord)
        x = F.elu(x)
        return x


    
class GraphNet(nn.Module):
    def __init__(self, num_classes=2000):
        super(GraphNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 24, (5,9), 1, (2,4))
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(24, 48, (5,7), 1, (2,3))
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(48, 64, 5, 1, 2)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = nn.Conv2d(64, 96, 5, 1, 2)

        self.graph_feat = GraphFeat(num_node=32)
        self.gat = GAT(nfeat=96,nhid=16,nout=96,dropout=0.6,alpha=0.2,cut_off=2.,nheads=8)
        self.conv_redu = nn.Conv2d(96, 32, 1, 1, 0)        

        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.conv5 = nn.Conv2d(96, 96, 5, 1, 2)
        self.pool5 = nn.MaxPool2d(kernel_size=2)
        
        self.fc1 = nn.Linear(4*8*96, 256)
        self.dropout = nn.Dropout(0.7)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        #pdb.set_trace()
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        
        graph_feat_res, coord = self.graph_feat(x)
        graph_feat = self.gat(graph_feat_res, coord)
        graph_feat = graph_feat + graph_feat_res
        graph_feat = F.elu(self.conv_redu(graph_feat.permute(0,2,1).unsqueeze(-1)))
        graph_feat = graph_feat.squeeze(3).permute(0,2,1)

        x = self.pool4(x)
        x = F.relu(self.conv5(x))
        x = self.pool5(x)
        
        x = x.contiguous().view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        out = self.fc2(x)

        return graph_feat, coord, x, out
        



