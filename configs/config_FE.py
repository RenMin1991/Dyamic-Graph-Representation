# -*- coding: utf-8 -*-
"""
configuration of GraphNet
RenMin
"""

class Config(object):
    def __init__(self):
        # data
        self.num_samples = 20000                                              # number of samples
        self.data_folder = '../data/iris/CASIA-Iris-Thousand/'                # data folder of the iris images
        self.txt_path = '../data/iris/CASIA-Iris-Thousand/Thousand_train.txt' # path to the .txt file of labels                                             
        
        # others
        self.feature_path = 'features/feature.pth'                            # storage location of features

        self.pretrained_path = 'checkpoint/graph_multiscale_ND.pth'           # pretrained model


    def num_samplesGet(self):
        return self.num_samples
    def data_folderGet(self):
        return self.data_folder
    def txt_pathGet(self):
        return self.txt_path
    def num_classesGet(self):
        return self.num_classes
    
    def feature_pathGet(self):
        return self.feature_path
    def pretrained_pathGet(self):
        return self.pretrained_path
    
