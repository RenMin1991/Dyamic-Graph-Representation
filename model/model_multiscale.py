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

class FeatExtrac(nn.Module):
    def __init__(self, num_class_th, num_class_la, num_class_cs):
        super(FeatExtrac, self).__init__()
        self.conv1 = nn.Conv2d(1, 24, (5,9), 1, (2,4))
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(24, 48, (5,7), 1, (2,3))
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(48, 64, 5, 1, 2)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = nn.Conv2d(64, 96, 5, 1, 2)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.conv5 = nn.Conv2d(96, 96, 5, 1, 2)
        self.pool5 = nn.MaxPool2d(kernel_size=2)
        #self.conv6 = nn.Conv2d(128, 128, 3, 1, 1)
        #self.pool6 = nn.MaxPool2d(kernel_size=2)
        
        self.fc1 = nn.Linear(4*8*96, 256)
        self.dropout = nn.Dropout(0.7)
        self.fc2_th = nn.Linear(256, num_class_th)
        self.fc2_la = nn.Linear(256, num_class_la)
        self.fc2_cs = nn.Linear(256, num_class_cs)
        
    def forward(self, x):
        #pdb.set_trace()
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = F.relu(self.conv5(x))
        x = self.pool5(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        out_th = self.fc2_th(x)
        out_la = self.fc2_la(x)
        out_cs = self.fc2_cs(x)
        
        return x, out_th, out_la, out_cs


class GraphGen3(nn.Module):
    def __init__(self, num_node):
        super(GraphGen3, self).__init__()
        self.conv1_1 = nn.Conv2d(64, 32, 1, 1, 0)
        self.conv1_2 = nn.Conv2d(32, 16, 3, 1, 1)
        self.conv2_1 = nn.Conv2d(16, 8, 1, 1, 0)
        self.conv2_2 = nn.Conv2d(8, 4, 3, 1, 1)
        self.fc = nn.Linear(32*64*4, 2*num_node)
        self._init_params()

    def _init_params(self):
        self.fc.weight.data.normal_(0,0.05)
        self.fc.bias.data.normal_(0,0.05)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = n_Sigmoid(x, 3)
        # to: Batch, Num_Points, 2
        x = x.view(x.size(0), -1, 2)
        return x

class GraphGen4(nn.Module):
    def __init__(self, num_node):
        super(GraphGen4, self).__init__()
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

class GraphGen5(nn.Module):
    def __init__(self, num_node):
        super(GraphGen5, self).__init__()
        self.conv1_1 = nn.Conv2d(96, 48, 1, 1, 0)
        self.conv1_2 = nn.Conv2d(48, 24, 3, 1, 1)
        self.conv2_1 = nn.Conv2d(24, 12, 1, 1, 0)
        self.conv2_2 = nn.Conv2d(12, 6, 3, 1, 1)
        self.fc = nn.Linear(8*16*6, 2*num_node)
        self._init_params()

    def _init_params(self):
        self.fc.weight.data.normal_(0,0.1)
        self.fc.bias.data.normal_(0,0.1)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = n_Sigmoid(x, 3)
        # to: Batch, Num_Points, 2
        x = x.view(x.size(0), -1, 2)
        return x

class GraphGen9(nn.Module):
    def __init__(self, num_node):
        super(GraphGen9, self).__init__()
        self.conv1_1 = nn.Conv2d(96, 48, 1, 1, 0)
        self.conv1_2 = nn.Conv2d(48, 24, 3, 1, 1)
        self.conv2_1 = nn.Conv2d(24, 12, 1, 1, 0)
        self.conv2_2 = nn.Conv2d(12, 6, 3, 1, 1)
        self.fc = nn.Linear(16*32*6, 2*num_node)
        self._init_params()

    def _init_params(self):
        self.fc.weight.data.normal_(0,0.1)
        self.fc.bias.data.normal_(0,0.1)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = n_Sigmoid(x, 3)
        # to: Batch, Num_Points, 2
        x = x.view(x.size(0), -1, 2)
        return x

class GraphFeat(nn.Module):
    def __init__(self, layer, num_node):
        super(GraphFeat, self).__init__()
        if layer==3:
            self.graph_gen = GraphGen3(num_node=num_node)
        if layer==4:
            self.graph_gen = GraphGen4(num_node=num_node)
        if layer==5:
            self.graph_gen = GraphGen5(num_node=num_node)
        if layer==9:
            self.graph_gen = GraphGen9(num_node=num_node)

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
        
        ###
        self.graph_feat3 = GraphFeat(layer=3, num_node=64)
        self.gat3 = GAT(nfeat=64,nhid=16,nout=64,dropout=0.6,alpha=0.2,cut_off=2.,nheads=8)
        self.conv_redu3 = nn.Conv2d(64, 32, 1, 1, 0)        
        ###
        
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = nn.Conv2d(64, 96, 5, 1, 2)

        ###
        self.graph_feat = GraphFeat(layer=4, num_node=32)
        self.gat = GAT(nfeat=96,nhid=16,nout=96,dropout=0.6,alpha=0.2,cut_off=2.,nheads=8)
        self.conv_redu = nn.Conv2d(96, 32, 1, 1, 0)        
              
        ###
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.conv5 = nn.Conv2d(96, 96, 5, 1, 2)

        ###
        self.graph_feat5 = GraphFeat(layer=5, num_node=16)
        self.gat5 = GAT(nfeat=96,nhid=16,nout=96,dropout=0.6,alpha=0.2,cut_off=2.,nheads=8)
        self.conv_redu5 = nn.Conv2d(96, 32, 1, 1, 0)        
        ###

        self.pool5 = nn.MaxPool2d(kernel_size=2)
        
        self.fc1 = nn.Linear(4*8*96, 256)
        self.dropout = nn.Dropout(0.7)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))

        graph_feat_res3, coord3 = self.graph_feat3(x)
        graph_feat3 = self.gat3(graph_feat_res3, coord3)
        graph_feat3 = graph_feat3 + graph_feat_res3
        graph_feat3 = F.elu(self.conv_redu3(graph_feat3.permute(0,2,1).unsqueeze(-1)))
        graph_feat3 = graph_feat3.squeeze(3).permute(0,2,1)

        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        
        graph_feat_res4, coord4 = self.graph_feat(x)
        graph_feat4 = self.gat(graph_feat_res4, coord4)
        graph_feat4 = graph_feat4 + graph_feat_res4
        graph_feat4 = F.elu(self.conv_redu(graph_feat4.permute(0,2,1).unsqueeze(-1)))
        graph_feat4 = graph_feat4.squeeze(3).permute(0,2,1)

        x = self.pool4(x)
        x = F.relu(self.conv5(x))

        graph_feat_res5, coord5 = self.graph_feat5(x)
        graph_feat5 = self.gat5(graph_feat_res5, coord5)
        graph_feat5 = graph_feat5 + graph_feat_res5
        graph_feat5 = F.elu(self.conv_redu5(graph_feat5.permute(0,2,1).unsqueeze(-1)))
        graph_feat5 = graph_feat5.squeeze(3).permute(0,2,1)

        x = self.pool5(x)
        
        x = x.contiguous().view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        outputs = self.fc2(x)

        return graph_feat3, coord3, graph_feat4, coord4, graph_feat5, coord5, x, outputs
        



