"""
Scalable, Detailed and Mask-free Universal Photometric Stereo Network (CVPR2023)
# Copyright (c) 2023 Satoshi Ikehata
# All rights reserved.
"""

import torch
import torch.nn.functional as F
import numpy as np
import math

def render(n, l, base, rough, metallic, emit = 1.0, SPECULAR = 1, device = None):
    # n [B, 3, h, w]
    # v [B, 3, 1]
    # l [B, 3, 1] # for aspecific lighting
    # bc [B, 3, h, w]
    # mt [B, 1, h, w]
    # rg [B, 1, h, w]
    bc = base
    rg = rough
    emit = torch.Tensor([emit]).to(device)
    metallic = metallic.to(device)
    v = torch.Tensor(np.zeros((l.shape[0], l.shape[1]))).to(device)
    v[:,2] = 1
    h = n.shape[2]
    w = n.shape[3]
    bcChannels = bc.shape[1]
    n = F.normalize(n, p=2, dim=1)
    n = n.view(-1, 3, h*w)
    bc = bc.view(-1, bcChannels, h*w)
    rg = rg.view(-1, 1, h*w)
    metallic = metallic.view(-1, 1, h*w)
    v = v.view(-1, 3, 1).expand(-1, -1, h*w)
    l = l.view(-1, 3, 1).expand(-1, -1, h*w)

    ################### Experimental the angle between l and v should always be fixed
    hf = 0.5 * (l + v)
    hf = F.normalize(hf, p=2, dim = 1)# [B, 3, h*w]

    nl = torch.max(torch.sum(n * l, 1), torch.Tensor([0]).to(device)) # [B, h*W]
    nv = torch.max(torch.sum(n * v, 1), torch.Tensor([0]).to(device)) # [B, h*w]
    nh = torch.sum(n * hf, 1) # [B, h*w]
    lh = torch.sum(l * hf, 1) # [B, numLight]

    # Diffuse
    nl = nl.view(-1, 1, h*w)
    nv = nv.view(-1, 1, h*w)
    nh = nh.view(-1, 1, h*w)
    lh = lh.view(-1, 1, h*w)
    FD90 = 0.5 + 2 * (lh * rg)

    FD = ( 1 + (FD90 - 1) * (1 - nl) * (1 - nl) * (1 - nl) * (1 - nl) * (1 - nl) ) * ( 1 + (FD90 - 1) * (1 - nv) * (1 - nv) * (1 - nv) * (1 - nv) * (1 - nv) )
    # fd [B, numLight] bc [B, 1]


    # GGX SPECULAR
    # specular Fs
    Cspec0 = 0.08 * SPECULAR * (1 - metallic) + bc * metallic
    Fs = Cspec0 + (1 - Cspec0) * (1 - lh) * (1 - lh) * (1 - lh) * (1 - lh) * (1 - lh)


    # specular Gs
    a = torch.min(torch.max(torch.Tensor([1.0e-6]).to(device), rg * rg), torch.Tensor([1.0]).to(device)) # roughness 0.0...1 <= roughness <= 1
    ah = a
    Gs_L = 1 / ( nl + torch.sqrt(ah * ah + (1 - ah * ah) * nl * nl) + 1.0e-12)
    Gs_V = 1 / ( nv + torch.sqrt(ah * ah + (1 - ah * ah) * nv * nv) + 1.0e-12)
    Gs = Gs_L * Gs_V

    # specular Ds
    Ds = a * a / (math.pi * (1 + (a * a - 1) * nh * nh) * (1 + (a * a - 1) * nh * nh) + 1.0e-12)

    fd = FD * bc * (1 - metallic) / math.pi
    fr = (Gs * Fs * Ds)

    nl = nl.view(-1, 1, h, w)
    fd = fd.view(-1, bcChannels, h, w)
    fr = fr.view(-1, bcChannels, h, w)

    return nl, emit * fd, emit * fr
