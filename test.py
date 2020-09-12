# -*- coding: utf-8 -*-
"""
open test for graph_net
RenMin
"""

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import csv

from configs.config_test import Config
from data.txt_dataset import TxtDataset
from model.model_singlescale import GraphNet
from functions.match_fn import GraphSim, feature_sim
from functions.iris_pre import iris_rotation
import pdb



# parameters
pdb.set_trace()
config = Config()

data_folder_temp = config.data_folder_tempGet()
txt_path_temp = config.txt_path_tempGet()
data_folder_ver = config.data_folder_verGet()
txt_path_ver = config.txt_path_verGet()

rot_list = config.rot_listGet()
weight_loc = config.weight_locGet()
weight_feat = config.weight_featGet()

sim_path = config.sim_pathGet()

pretrained_path = config.pretrained_pathGet()


# define network
pre_data = torch.load(pretrained_path, map_location=lambda storage, loc:storage)
pre_dict = pre_data['model']

model = GraphNet()
model_dict = model.state_dict()

pre_dict = {k:v for k,v in pre_dict.items() if k in model_dict}
model_dict.update(pre_dict)
model.load_state_dict(model_dict)
model = model.cuda()
model.eval()


# pre-process
transform = transforms.Compose([
        transforms.Resize(size=[128,256]),
        transforms.ToTensor(),
        transforms.Normalize((0.4376,),(0.3479,))
        ])


# loss function
graph_sim = GraphSim(weight_loc, weight_feat)
graph_sim = graph_sim.cuda()


# get data
temp_set = TxtDataset(txt=txt_path_temp, data_folder=data_folder_temp, transform=transform)
temp_loader = DataLoader(temp_set, batch_size=1, shuffle=False)

ver_set = TxtDataset(txt=txt_path_ver, data_folder=data_folder_ver, transform=transform)
ver_loader = DataLoader(ver_set, batch_size=1, shuffle=False)

# matching
def matching():
    sims = []
    labels = []
    for i, data_ver in enumerate(ver_loader, 0):
        inputs_v, label_v = data_ver
        inputs_v = iris_rotation(inputs_v, rot_list)
        inputs_v = inputs_v.cuda()
        inputs_v = Variable(inputs_v)
        
        for j, data_temp in enumerate(temp_loader, 0):
            inputs_t, label_t = data_temp
            inputs_t = iris_rotation(inputs_t, rot_list)
            inputs_t = inputs_t.cuda()
            inputs_t = Variable(inputs_t)
            
            # forward
            graph_feat_v, coord_v, feature_v, _ = model(inputs_v)
            graph_feat_t, coord_t, feature_t, _ = model(inputs_t)
            
            sim_graph = graph_sim(graph_feat_v, coord_v, graph_feat_t, coord_t)
            sim_f = feature_sim(feature_v, feature_t)
            sims.append(sim_graph*sim_f)
            
            if label_v == label_t:
                labels.append(1)
            else:
                labels.append(0)
    
    f = open(sim_path, 'w')
    writer = csv.writer(f)
    writer.writerow(sims)
    writer.writerow(labels)
    f.close()
            
if __name__ == '__main__':
    matching()
            
            























