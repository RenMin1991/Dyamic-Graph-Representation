# -*- coding: utf-8 -*-
"""
get triplet data and pairwise data

Ren Min 20180831
"""

import torch as t

import numpy as np

import pdb


def get_positive_pair(dataset):
    pair_signal = 0
    len_data = len(dataset)
    index_a = np.random.randint(0, len_data)
    sample_a, label_a = dataset[index_a]
    p_signal = 0
    p_count = 0
    while p_signal==0:
        p_low = max(0, index_a-5)
        p_high = min(len_data-1, index_a+5)
        index_p = np.random.randint(p_low, p_high)

        if index_p==index_a:
            index_p = index_a + 1

        sample_p, label_p = dataset[index_p]

        if label_a == label_p:
            p_signal = 1
            pair_signal = 1
        p_count += 1
        if p_count > 500:
            print ('The same class should be close!')
            p_signal = 1
        return sample_a, sample_p, pair_signal, label_a, label_p, index_a, index_p



def TripletBatch(dataset, batch, hard_samples):
    
    # get triplet batch data
    
    len_data = len(dataset)
    len_hard = len(hard_samples)

    labels_a = t.zeros([batch])
    labels_p = t.zeros([batch])
    labels_n = t.zeros([batch])

    sample, _ = dataset[0]
    batch_a = t.zeros([batch, sample.size()[0], sample.size()[1], sample.size()[2]])
    batch_p = t.zeros([batch, sample.size()[0], sample.size()[1], sample.size()[2]])
    batch_n = t.zeros([batch, sample.size()[0], sample.size()[1], sample.size()[2]])
    
    batch_index = np.zeros((batch, 3))
    
    for i in range(batch):

        if i < len_hard:
            index_a = int(hard_samples[i][0])
            index_p = int(hard_samples[i][1])
            index_n = int(hard_samples[i][2])

            sample_a, label_a = dataset[index_a]
            sample_p, label_p = dataset[index_p]
            sample_n, label_n = dataset[index_n]

        else:
            # get positive pair
            pair_signal = 0
            while pair_signal==0:
                sample_a, sample_p, p, label_a, label_p, index_a, index_p = get_positive_pair(dataset=dataset)
                pair_signal = p
        
            # get negtive sample
            n_signal = 0
            while n_signal==0:
                index_n = np.random.randint(0, len_data)
                sample_n, label_n = dataset[index_n]
                if label_n != label_a:
                    n_signal = 1
	
        labels_a[i] = label_a
        labels_p[i] = label_p
        labels_n[i] = label_n
    
        batch_a[i,:,:,:] = sample_a
        batch_p[i,:,:,:] = sample_p
        batch_n[i,:,:,:] = sample_n
        
        batch_index[i,0] = index_a
        batch_index[i,1] = index_p
        batch_index[i,2] = index_n
        
    return batch_a, labels_a.long(), batch_p, labels_p.long(), batch_n, labels_n.long(), batch_index
                

        
