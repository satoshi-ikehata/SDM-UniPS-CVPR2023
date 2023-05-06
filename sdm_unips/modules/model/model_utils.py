"""
Scalable, Detailed and Mask-free Universal Photometric Stereo Network (CVPR2023)
# Copyright (c) 2023 Satoshi Ikehata
# All rights reserved.
"""

import os
import torch
import numpy as np

def loadmodel(model, filename, strict=True):
    if os.path.exists(filename):
        params = torch.load('%s' % filename)
        model.load_state_dict(params,strict=strict)
        print('Loading pretrained model... %s ' % filename)
    else:
        print('Pretrained model not Found')
    return model

def mode_change(net, Training):
    if Training == True:
        for param in net.parameters():
            param.requires_grad = True
        net.train()
    if Training == False:
        for param in net.parameters():
            param.requires_grad = False
        net.eval()

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def masking(img, mask):
    # img [B, C, H, W]
    # mask [B, 1, H, W] [0,1]
    img_masked = img * mask.expand((-1, img.shape[1], -1, -1))
    return img_masked

def print_model_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('# parameters: %d' % params)

def angular_error(x1, x2, mask = None): # tensor [B, 3, H, W]
    if mask is not None:
        dot = torch.sum(x1 * x2 * mask, dim=1, keepdim=True)
        dot = torch.max(torch.min(dot, torch.Tensor([1.0-1.0e-12])), torch.Tensor([-1.0+1.0e-12]))
        emap = torch.abs(180 * torch.acos(dot)/np.pi) * mask
        mae = torch.sum(emap) / torch.sum(mask)
        return mae, emap
    if mask is None:
        dot = torch.sum(x1 * x2, dim=1, keepdim=True)
        dot = torch.max(torch.min(dot, torch.Tensor([1.0-1.0e-12])), torch.Tensor([-1.0+1.0e-12]))
        error = torch.abs(180 * torch.acos(dot)/np.pi)
        return error

def make_index_list(maxNumImages, numImageList):
    index = np.zeros((len(numImageList) * maxNumImages), np.int32)
    for k in range(len(numImageList)):
        index[maxNumImages*k:maxNumImages*k+numImageList[k]] = 1
    return index
    
