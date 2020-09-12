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

from configs.config_train_multiscale import Config
from data.txt_dataset import TxtDataset
from data.batch_data import TripletBatch
from model.model_multiscale import GraphNet
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
pretrained_path = config.pretrained_pathGet()


# define network
pre_data = torch.load(pretrained_path, map_location=lambda storage, loc:storage)
pre_dict = pre_data['model']

model = GraphNet(num_classes)
model_dict = model.state_dict()

pre_dict = {k:v for k,v in pre_dict.items() if k in model_dict}
model_dict.update(pre_dict)
model.load_state_dict(model_dict)

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
        graph_feat_a3, coord_a3, graph_feat_a4, coord_a4, graph_feat_a5, coord_a5, float_a, outputs_a = model(inputs_a)
        graph_feat_p3, coord_p3, graph_feat_p4, coord_p4, graph_feat_p5, coord_p5, float_p, outputs_p = model(inputs_p)
        graph_feat_n3, coord_n3, graph_feat_n4, coord_n4, graph_feat_n5, coord_n5, float_n, outputs_n = model(inputs_n)

        #pdb.set_trace()
    
        # loss
        loss_graph3, _ = graph_loss(graph_feat_a3, coord_a3, graph_feat_p3, coord_p3, graph_feat_n3, coord_n3, float_a, float_p, float_n)
        loss_graph4, _ = graph_loss(graph_feat_a4, coord_a4, graph_feat_p4, coord_p4, graph_feat_n4, coord_n4, float_a, float_p, float_n)
        loss_graph5, _ = graph_loss(graph_feat_a5, coord_a5, graph_feat_p5, coord_p5, graph_feat_n5, coord_n5, float_a, float_p, float_n)
        loss_graph = (loss_graph3 + loss_graph4 + loss_graph5)/3. 
    
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



