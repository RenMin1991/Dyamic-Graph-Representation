# -*- coding: utf-8 -*-
"""
configuration of GraphNet
RenMin
"""

class Config(object):
    def __init__(self):
        self.data_folder_temp = 'path/to/template/'                                # data folder of the template images
        self.txt_path_temp = 'path/to/template/label.txt'                          # path to the .txt file of template images
        self.data_folder_ver = 'path/to/verified/images/'                          # data folder of the verified images
        self.txt_path_ver = 'path/to/verified/label.txt'                           # path to the .txt file of verified images
        self.sim_path = 'path/to/similarities.csv'                                 # storage location of similarities
        self.pretrained_path = 'path/to/pretrained.pth'                            # pretrained model
        
        self.rot_list = [-4, -2, 0, 2, 4]                                          # location list of Bit-shift
        self.weight_loc = 0.                                                       # weight of structure similarity
        self.weight_feat = 1.                                                      # weight of node similarity


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
    
