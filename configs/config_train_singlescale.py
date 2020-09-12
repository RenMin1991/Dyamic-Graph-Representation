# -*- coding: utf-8 -*-
"""
configuration of GraphNet
RenMin
"""

class Config(object):
    def __init__(self):
        
        # optimizer
        self.lr = 1e-4                                                        # learning rate
        self.steps = 10000                                                    # steps of training
        self.batch = 64                                                       # mini-batch
        self.momentum = 0.9                                                   # momentum of optimizer
        self.weight_decay = 1e-4                                              # weight decay of optimizr
        
        # data
        self.data_folder = 'path/to/data/'                                    # data folder of the iris images
        self.txt_path = 'path/to/label.txt'                                   # path to the .txt file of labels 
        self.num_classes = 2000                                               # number of classes
        
        # loss
        self.weight_loc = 1.                                                  # weight of structure similarity
        self.weight_feat = 1.                                                 # weight of node similarity
        self.weight_graph = 1.                                                # weight of graph loss
        self.weight_class = 0.5                                               # weight of classification loss
        self.margin_triplet = 0.5                                             # margin of graph triplet loss
        
        # others
        self.save_file = 'path/to/checkpoint_'                      # storage of checkpoint
        self.log_step = 100                                                   # frequency of printing log
        self.save_step = 10000                                                # frequency of saving checkpoint


      
    def lrGet(self):
        return self.lr
    def stepsGet(self):
        return self.steps
    def batchGet(self):
        return self.batch
    def momentumGet(self):
        return self.momentum
    def weight_decayGet(self):
        return self.weight_decay
    
    def data_folderGet(self):
        return self.data_folder
    def txt_pathGet(self):
        return self.txt_path
    def num_classesGet(self):
        return self.num_classes
    
    def weight_locGet(self):
        return self.weight_loc
    def weight_featGet(self):
        return self.weight_feat
    def weight_graphGet(self):
        return self.weight_graph
    def weight_classGet(self):
        return self.weight_class
    def margin_tripletGet(self):
        return self.margin_triplet
    
    def save_fileGet(self):
        return self.save_file
    def log_stepGet(self):
        return self.log_step
    def save_stepGet(self):
        return self.save_step
    def pretrained_pathGet(self):
        return self.pretrained_path
    
