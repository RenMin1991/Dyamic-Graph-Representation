# -*- coding: utf-8 -*-
"""
configuration of GraphNet
RenMin
"""

class Config(object):
    def __init__(self):
        self.data_folder_temp = '../data/iris/CASIA-Iris-Thousand/'                # data folder of the template images
        self.txt_path_temp = '../data/iris/CASIA-Iris-Thousand/Thousand_train.txt' # path to the .txt file of template images
        self.data_folder_ver = '../data/iris/CASIA-Iris-Thousand/'                 # data folder of the verified images
        self.txt_path_ver = '../data/iris/CASIA-Iris-Thousand/Thousand_train.txt'  # path to the .txt file of verified images
        self.sim_path = 'sim.csv'                                                  # storage location of similarities
        self.pretrained_path = 'checkpoint/graph_singlescale_ND.pth'               # pretrained model
        
        self.rot_list = [-4, -2, 0, 2, 4]
        self.weight_loc = 0.
        self.weight_feat = 1.      


    def data_folder_tempGet(self):
        return self.data_folder_temp
    def txt_path_tempGet(self):
        return self.txt_path_temp
    def data_folder_verGet(self):
        return self.data_folder_ver
    def txt_path_verGet(self):
        return self.txt_path_ver
    def sim_pathGet(self):
        return self.sim_path
    def rot_listGet(self):
        return self.rot_list
    def weight_locGet(self):
        return self.weight_loc
    def weight_featGet(self):
        return self.weight_feat
    def pretrained_pathGet(self):
        return self.pretrained_path
    
