# -*- coding: utf-8 -*-
"""
feature extraction by GraphNet
RenMin
"""


import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from configs.config_FE import Config
from data.txt_dataset import TxtDataset
from model.model_multiscale import GraphNet



# parameters
config = Config()

num_samples = config.num_samplesGet()
data_folder = config.data_folderGet()
txt_path = config.txt_pathGet()

feature_path = config.feature_pathGet()

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


# get data
testset = TxtDataset(txt=txt_path, data_folder=data_folder, transform=transform)
test_loader = DataLoader(testset, batch_size=1, shuffle=False)

# feature extraction
def FeatExtract():
    feat_g3 = torch.zeros(num_samples, 64, 32)
    feat_g4 = torch.zeros(num_samples, 32, 32)
    feat_g5 = torch.zeros(num_samples, 16, 32)
    feat_f = torch.zeros(num_samples, 256)
    labels = torch.zeros(num_samples)
    

    for i, data in enumerate(test_loader, 0):
        
        inputs, label = data
        inputs = inputs.cuda()
        inputs = Variable(inputs)
        
        graph_feat3, _, graph_feat4, _, graph_feat5, _, flo, _ = model(inputs)
        
        feat_g3[i, :, :] = graph_feat3[0].data
        feat_g4[i, :, :] = graph_feat4[0].data
        feat_g5[i, :, :] = graph_feat5[0].data
        feat_f[i, :] = flo[0].data
        labels[i] = label
        
    features = dict(
                feat_g3 = feat_g3,
                feat_g4 = feat_g4,
                feat_g5 = feat_g5,
                feat_f = feat_f,
                labels = labels
                )
    torch.save(features, feature_path)

   
if __name__ == '__main__':
    FeatExtract()



