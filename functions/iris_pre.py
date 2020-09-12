"""
iris image rotation functions for graph net
RenMin
"""

import torch
from random import randint


def translation(x, trans_col, trans_row):
    # trans_col>0: move towards right
    # trans_row>0: move towards down
    if trans_col != 0:
        x_a = x[:,:,:,-trans_col:]
        x_b = x[:,:,:,:-trans_col]
        x_col = torch.cat((x_a,x_b),dim=3)
    else:
        x_col = x
    if trans_row != 0:
        x_col_a = x_col[:, :, -trans_col:, :]
        x_col_b = x_col[:, :, :-trans_col, :]
        x_col_row = torch.cat((x_col_a,x_col_b),dim=2)
    else:
        x_col_row = x_col
    return x_col_row

def iris_rotation(x, rot_list):
    N = len(rot_list)
    x_rot = torch.zeros([N, x.size(1), x.size(2), x.size(3)])
    for i in range(N):
        x_rot[i,:,:,:] = translation(x, trans_col=rot_list[i], trans_row=0)
    return x_rot

def iris_occlusion_1(x, rate_occ, mask):
    B, C, H, W = x.size()
    W_occ = int(W * rate_occ)

    if mask==0:
        mask = torch.randn(B, C, H, W_occ)
    elif mask==1:
        mask = torch.zeros(B, C, H, W_occ)
    #mask = torch.ones(B, C, H, W_occ)
    
    x[:,:,:,:W_occ] = mask

    return x

def iris_occlusion_2(x, rate_main, rate_list, mask):
    B, C, H, W = x.size()
    x = iris_occlusion_1(x, rate_main)
    for rate in rate_list:
        L_occ = int((W*H*rate)**0.5)

        if mask==0:
            mask = torch.randn(B, C, L_occ, L_occ)
        elif mask==1:
            mask = torch.zeros(B, C, L_occ, L_occ)
        #mask = torch.ones(B, C, L_occ, L_occ)

        row = randint(0, H-L_occ)
        col = randint(0, W-L_occ)
        
        x[:, :, row:row+L_occ, col:col+L_occ] = mask

    return x

def iris_occlusion_3(x, rate_left, rate_right, mask):
    B, C, H, W = x.size()
    L_left = int((W*H*rate_left)**0.5)
    if mask==0:
        mask_left = torch.randn(B, C, L_left, L_left)
    elif mask==1:
        mask_left = torch.zeros(B, C, L_left, L_left)
    x[:,:,-L_left:,int(W*0.25-L_left*0.5):int(W*0.25-L_left*0.5)+L_left] = mask_left

    H_right = int((W*H*rate_right*0.5)**0.5)
    W_right = int((W*H*rate_right*0.5)**0.5*2)
    if mask==0:
        mask_right = torch.randn(B, C, H_right, W_right)
    elif mask==1:
        mask_right = torch.zeros(B, C, H_right, W_right)
    x[:,:,-H_right:,int(W*0.75-W_right*0.5):int(W*0.75-W_right*0.5)+W_right] = mask_right

    return x


















