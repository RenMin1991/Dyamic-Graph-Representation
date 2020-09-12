# -*- coding: utf-8 -*-
"""
train triplet for GraphNet
RenMin
"""


import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torchvision.transforms as transforms

from configs.config_train_singlescale import Config
from data.txt_dataset import TxtDataset
from data.batch_data import TripletBatch
from model.model_singlescale import GraphNet
from functions.loss import GraphLoss
import pdb



# parameters
config = Config()
lr = config.lrGet()
steps = config.stepsGet()
weight_loc = config.weight_locGet()
weight_feat = config.weight_featGet()
weight_graph = config.weight_graphGet()
weight_class = config.weight_classGet()
margin_triplet = config.margin_tripletGet()
batch = config.batchGet()
momentum = config.momentumGet()
weight_decay = config.weight_decayGet()
num_classes = config.num_classesGet()
data_folder = config.data_folderGet()
txt_path = config.txt_pathGet()
save_file = config.save_fileGet()
log_step = config.log_stepGet()
save_step = config.save_stepGet()


# define network
model = GraphNet(num_classes)
model = model.cuda()



# optimizer
optimizer = optim.SGD(model.parameters(),
                            lr,
                            momentum=momentum,
                            weight_decay=weight_decay)


# loss function
graph_loss = GraphLoss(weight_loc, weight_feat,  margin_triplet)
graph_loss = graph_loss.cuda()

softmax_loss = nn.CrossEntropyLoss()
softmax_loss = softmax_loss.cuda()

# pre-process
transform = transforms.Compose([
        transforms.Resize(size=[128,256]),
        transforms.ToTensor(),
        transforms.Normalize((0.4376,),(0.3479,))
        ])


# get data
trainset = TxtDataset(txt=txt_path, data_folder=data_folder, transform=transform)

#train
def train():
    running_loss = 0.

    for step in range(steps):
   
        model.train()
    
        inputs_a,label_a,inputs_p,label_p,inputs_n,label_n,inds = TripletBatch(dataset=trainset, batch=batch, hard_samples=[])
        
        inputs_a,label_a = inputs_a.cuda(),label_a.cuda()
        inputs_p,label_p = inputs_p.cuda(),label_p.cuda()
        inputs_n,label_n = inputs_n.cuda(),label_n.cuda()
        inputs_a, inputs_p, inputs_n = Variable(inputs_a), Variable(inputs_p), Variable(inputs_n)
        label_a, label_p, label_n = Variable(label_a), Variable(label_p), Variable(label_n)
    
        # zero grad
        optimizer.zero_grad()
    
        # forward
        graph_feat_a, coord_a, float_a, outputs_a = model(inputs_a)
        graph_feat_p, coord_p, float_p, outputs_p = model(inputs_p)
        graph_feat_n, coord_n, float_n, outputs_n = model(inputs_n)

        #pdb.set_trace()
    
        # loss
        loss_graph, _ = graph_loss(graph_feat_a, coord_a, graph_feat_p, coord_p, graph_feat_n, coord_n, float_a, float_p, float_n)
    
        loss_class = (softmax_loss(outputs_a, label_a) + softmax_loss(outputs_p, label_p) + softmax_loss(outputs_n, label_n))/3.
    
        loss = weight_graph*loss_graph + weight_class*loss_class
    
        # backward
        loss.backward()
        optimizer.step()
        running_loss = running_loss + float(loss.item())
    
        if step%log_step==log_step-1:
            print ('step', step+1, 't_loss', running_loss/(step+1.))
    
        # save optimizer and model
        if step%save_step==save_step-1:
            all_data = dict(
                optimizer = optimizer.state_dict(),
                model = model.state_dict(),
                steps = step + 1,
                )
            file_name = save_file + str(step+1)+'.pth'
            torch.save(all_data, file_name)

   
if __name__ == '__main__':
    train()



