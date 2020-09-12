'''
    class of dataset
    Ren Min
    2018.06.07
'''

from torch.utils.data import Dataset
import torch as t
from PIL import Image
import pdb

def default_loader(path, rgb):
    if rgb:
        return Image.open(path).convert('RGB')
    else:
        return Image.open(path)#.convert('RGB')

class TxtDataset(Dataset):
    def __init__ (self, txt, data_folder, transform=None, target_transform=None, rgb=False):
        super(TxtDataset, self).__init__()
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
        fh.close()
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = default_loader
        self.rgb = rgb
        self.data_folder = data_folder

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        fn = self.data_folder + fn
        img = self.loader(fn, self.rgb)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


class MultiDataset(Dataset):
    def __init__ (self, list_dataset, list_class):
        super(MultiDataset, self).__init__()
        #pdb.set_trace()
        self.list_dataset = list_dataset

        list_len = []
        for i in range(len(list_dataset)):
            list_len.append(len(list_dataset[i]))
        len_pro_set = list_len
        for i in range(len(list_len)-1):
            len_pro_set[i+1] = len_pro_set[i] + list_len[i+1]
        self.len_pro_set = len_pro_set

        self.list_class = list_class
             
    def __getitem__(self, index):
        #pdb.set_trace()
        for index_set in range(len(self.len_pro_set)):
            if index < self.len_pro_set[index_set]:
                break
        if index_set==0:
            index_sample = index
        else:
            index_sample = index - self.len_pro_set[index_set-1]
        dataset = self.list_dataset[index_set]
        img, label = dataset[index_sample]
        
        if index_set > 0:
            label = label + self.list_class[index_set-1]
        return img, label
    
    def __len__(self):
        return self.len_pro_set[-1]




