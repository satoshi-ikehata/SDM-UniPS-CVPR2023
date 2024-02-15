
"""
Scalable, Detailed and Mask-free Universal Photometric Stereo Network (CVPR2023)
# Copyright (c) 2023 Satoshi Ikehata
# All rights reserved.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, trunc_normal_

from .model_utils import *
from . import transformer
from . import convnext
from . import uper
from ..utils import gauss_filter
from ..utils.ind2sub import *
from .decompose_tensors import *

class ImageFeatureExtractor(nn.Module):
    def __init__(self, input_nc):
        super(ImageFeatureExtractor, self).__init__()
        back = []

        ### ConvNexT backbone (from sctratch)
        out_channels = (96, 192, 384, 768)
        back.append(convnext.ConvNeXt(in_chans=input_nc, use_checkpoint=False))      
        self.backbone = nn.Sequential(*back)
        self.out_channels = out_channels

    def forward(self, x):
        feats = self.backbone(x) # glc[batch][scale]     
        return feats

class ImageFeatureFusion(nn.Module):
    def __init__(self, in_channels, use_efficient_attention=False):
        super(ImageFeatureFusion, self).__init__()
        self.fusion =  uper.UPerHead(in_channels = in_channels)
        
        attn = []
        self.num_comm_enc = [0,1,2,4]
   
        for i in range(len(in_channels)):
            if self.num_comm_enc[i] > 0:
                attn.append(transformer.CommunicationBlock(in_channels[i], num_enc_sab = self.num_comm_enc[i], dim_hidden=in_channels[i], ln=True, dim_feedforward = in_channels[i], use_efficient_attention=use_efficient_attention))
        self.comm = nn.Sequential(*attn)  
    
    def forward(self, glc, nImgArray):
        batch_size = len(nImgArray)
        sum_nimg = torch.sum(nImgArray)
        
        out_fuse = []
        attn_cnt = 0
        for k in range(len(glc)):
            if self.num_comm_enc[k] > 0:
                in_fuse = glc[k]
                _, C, H, W = in_fuse.shape # ((K+1) * sum(nImg), C, H, W)
                in_fuse = in_fuse.reshape(-1, sum_nimg, C, H, W).permute(0, 3, 4, 1, 2) # ((K+1), H, W, sum(nImg), C)
                K = in_fuse.shape[0] - 1
                in_fuse = in_fuse.reshape(-1, sum_nimg, C)
                feats = []
                ids = 0
                for b in range(batch_size):
                    feat = in_fuse[:, ids:ids+nImgArray[b], :]       
                    feat = self.comm[attn_cnt](feat)
                    feats.append(feat)
                    ids = ids + nImgArray[b]
                feats = torch.cat(feats, dim=1) # ((K+1)*H*W, sum(nImg), C)
                feats = feats.reshape(K+1, H*W, sum_nimg, C).permute(0, 2, 3, 1) # ((K+1), sum_nimg, C, H*W)
                feats = feats.reshape((K+1)*sum_nimg, C, H, W)
                out_fuse.append(feats)
                attn_cnt += 1
            else:
                out_fuse.append(glc[k])            
        out = self.fusion(out_fuse) 
        return out

class ScaleInvariantSpatialLightImageEncoder(nn.Module): # image feature encoder at canonical resolution
    def __init__(self, input_nc, use_efficient_attention=False):
        super(ScaleInvariantSpatialLightImageEncoder, self).__init__()
        self.backbone = ImageFeatureExtractor(input_nc)
        self.fusion = ImageFeatureFusion(self.backbone.out_channels, use_efficient_attention=use_efficient_attention)
        self.feat_dim = 256

    def forward(self, x, nImgArray, canonical_resolution):
        N, C, H, W = x.shape        
        mosaic_scale = H // canonical_resolution
        K = mosaic_scale * mosaic_scale

        """ (1a) resizing x to (Hc, Wc)"""
        x_resized = F.interpolate(x, size= (canonical_resolution, canonical_resolution), mode='bilinear', align_corners=True)

        """ (1b) decomposing x into K x K of (Hc, Wc) non-overlapped blocks (stride)"""           
        x_grid = divide_tensor_spatial(x, block_size=canonical_resolution, method='tile_stride') # (B, K, C, canonical_resolution, canonical_resolution)
        x_grid = x_grid.permute(1,0,2,3,4).reshape(-1, C, canonical_resolution, canonical_resolution) # (K*B, C, canonical_resolutioin, canonical_resolution)
  
        """(2a) feature extraction """
        x = self.fusion(self.backbone(x_resized), nImgArray)
        f_resized = x.reshape(1, N, self.feat_dim, canonical_resolution//4 * canonical_resolution//4) # (1, N, C, canonical_resolution//4 * canonical_resolution//4)
        del x_resized

        """(2b) feature extraction """
        x = self.fusion(self.backbone(x_grid), nImgArray) # (K * N, C, canonical_resolution//4, canonical_resolution//4)
        x = x.reshape(K, N, x.shape[1], canonical_resolution//4, canonical_resolution//4)
        glc_grid = merge_tensor_spatial(x, method='tile_stride')
        del x_grid
       
        """ (3) upsample """
        glc_resized = F.interpolate(f_resized.reshape(N, self.feat_dim, canonical_resolution//4, canonical_resolution//4) , size= (H//4, W//4), mode='bilinear', align_corners=True)
        del f_resized

        glc = glc_resized + glc_grid
        return glc

 
class GLC_Upsample(nn.Module):
    def __init__(self, input_nc, num_enc_sab=1, dim_hidden=256, dim_feedforward=1024, use_efficient_attention=False):
        super(GLC_Upsample, self).__init__()       
        self.comm = transformer.CommunicationBlock(input_nc, num_enc_sab = num_enc_sab, dim_hidden=dim_hidden, ln=True, dim_feedforward = dim_feedforward,use_efficient_attention=False)
       
    def forward(self, x):
        x = self.comm(x)        
        return x

class GLC_Aggregation(nn.Module):
    def __init__(self, input_nc, num_agg_transformer=2, dim_aggout=384, dim_feedforward=1024, use_efficient_attention=False):
        super(GLC_Aggregation, self).__init__()              
        self.aggregation = transformer.AggregationBlock(dim_input = input_nc, num_enc_sab = num_agg_transformer, num_outputs = 1, dim_hidden=dim_aggout, dim_feedforward = dim_feedforward, num_heads=8, ln=True, attention_dropout=0.1, use_efficient_attention=use_efficient_attention)

    def forward(self, x):
        x = self.aggregation(x)      
        return x

class Regressor(nn.Module):
    def __init__(self, input_nc, num_enc_sab=1, use_efficient_attention=False, dim_feedforward=256, output='normal'):
        super(Regressor, self).__init__()     
        # Communication among different samples (Pixel-Sampling Transformer)
        self.comm = transformer.CommunicationBlock(input_nc, num_enc_sab = num_enc_sab, dim_hidden=input_nc, ln=True, dim_feedforward = dim_feedforward, use_efficient_attention=use_efficient_attention)   
        self.prediction_normal = PredictionHead(input_nc, 3)
        self.target = output
        if output == 'brdf':   
            self.prediction_base = PredictionHead(input_nc, 3) # No urcainty
            self.prediction_rough = PredictionHead(input_nc, 1)
            self.prediction_metal = PredictionHead(input_nc, 1)

    def forward(self, x, num_sample_set):
        """Standard forward
        INPUT: img [Num_Pix, F]
        OUTPUT: [Num_Pix, 3]"""  
        if x.shape[0] % num_sample_set == 0:
            x_ = x.reshape(-1, num_sample_set, x.shape[1])
            x_ = self.comm(x_)            
            x = x_.reshape(-1, x.shape[1])
        else:
            ids = list(range(x.shape[0]))
            num_split = len(ids) // num_sample_set
            x_1 = x[:(num_split)*num_sample_set, :].reshape(-1, num_sample_set, x.shape[1])
            x_1 = self.comm(x_1).reshape(-1, x.shape[1])
            x_2 = x[(num_split)*num_sample_set:,:].reshape(1, -1, x.shape[1])
            x_2 = self.comm(x_2).reshape(-1, x.shape[1])
            x = torch.cat([x_1, x_2], dim=0)

        x_n = self.prediction_normal(x)        
        if self.target == 'brdf':
            x_brdf = (self.prediction_base(x), self.prediction_rough(x), self.prediction_metal(x))
        else:
            x_brdf = []
        return x_n, x_brdf
    
class PredictionHead(nn.Module):
    def __init__(self, dim_input, dim_output):
        super(PredictionHead, self).__init__()
        modules_regression = []
        modules_regression.append(nn.Linear(dim_input, dim_input//2))
        modules_regression.append(nn.ReLU())
        modules_regression.append(nn.Linear(dim_input//2, dim_output))
        self.regression = nn.Sequential(*modules_regression)

    def forward(self, x):
        return self.regression(x)

class Net(nn.Module):
    def __init__(self, pixel_samples, output, device):
        super().__init__()
        self.device = device
        self.target = output
        self.pixel_samples = pixel_samples
        self.glc_smoothing = True

   
        self.input_dim = 4 # RGB + mask   
        self.image_encoder = ScaleInvariantSpatialLightImageEncoder(self.input_dim, use_efficient_attention=False).to(self.device)

        self.input_dim = 3 # RGB only
        self.glc_upsample = GLC_Upsample(256+self.input_dim, num_enc_sab=1, dim_hidden=256, dim_feedforward=1024, use_efficient_attention=True).to(self.device)
        self.glc_aggregation = GLC_Aggregation(256+self.input_dim, num_agg_transformer=2, dim_aggout=384, dim_feedforward=1024, use_efficient_attention=False).to(self.device)
        

        self.regressor = Regressor(384, num_enc_sab=1, use_efficient_attention=True, dim_feedforward=1024, output=self.target).to(self.device) 
        
    def no_grad(self):
        mode_change(self.image_encoder, False)
        mode_change(self.glc_upsample, False)
        mode_change(self.glc_aggregation, False)
        mode_change(self.regressor, False)
    

    def forward(self, I, M, nImgArray, decoder_resolution, canonical_resolution):     
        
        decoder_resolution = decoder_resolution[0,0].cpu().numpy().astype(np.int32).item()
        canonical_resolution = canonical_resolution[0,0].cpu().numpy().astype(np.int32).item()

        """init"""
        B, C, H, W, Nmax = I.shape

        """ Image Encoder at Canonical Resolution """
        I_enc = I.permute(0, 4, 1, 2, 3)# B Nmax C H W       
        M_enc = M # B 1 H W               
        img_index = make_index_list(Nmax, nImgArray) # Extract objects > 0
        I_enc = I_enc.reshape(-1, I_enc.shape[2], I_enc.shape[3], I_enc.shape[4]) 
        M_enc = M_enc.unsqueeze(1).expand(-1, Nmax, -1, -1, -1).reshape(-1, 1, H, W)
        data = torch.cat([I_enc * M_enc, M_enc], dim=1)     
        data = data[img_index==1,:,:,:] # torch.size([B, N, 4, H, W])d
        glc = self.image_encoder(data, nImgArray, canonical_resolution) # torch.Size([B, N, 256, H/4, W/4]) [img, mask]

        """ Sample Decoder at Original Resokution"""
        I_dec = []
        M_dec = []
        N_dec = []

        img = I.permute(0, 4, 1, 2, 3).to(self.device)
        mask = M 
         
        decoder_imgsize = (decoder_resolution, decoder_resolution)
        img = img.reshape(-1, img.shape[2], img.shape[3], img.shape[4])
        img = img[img_index==1, :, :, :]
        I_dec = F.interpolate(img, size=decoder_imgsize, mode='bilinear', align_corners=False)  
        M_dec = F.interpolate(mask, size=decoder_imgsize, mode='nearest')  
       
        C = img.shape[1]
        H = decoder_imgsize[0]
        W = decoder_imgsize[1]            
    
        nout = torch.zeros(B, H * W, 3).to(self.device)
        bout = torch.zeros(B, H * W, 3).to(self.device)
        rout = torch.zeros(B, H * W, 1).to(self.device)
        mout = torch.zeros(B, H * W, 1).to(self.device)

        if self.glc_smoothing:  
            f_scale = decoder_resolution//canonical_resolution # (2048/256)
            smoothing = gauss_filter.gauss_filter(glc.shape[1], 10 * f_scale+1, 1).to(glc.device) # channels, kernel_size, sigma
            glc = smoothing(glc)
        p = 0
        for b in range(B):                
            target = range(p, p+nImgArray[b])
            p = p+nImgArray[b]
            m_ = M_dec[b, :, :, :].reshape(-1, H * W).permute(1,0)        
            ids = np.nonzero(m_>0)[:,0]  
            ids = ids[np.random.permutation(len(ids))]                               
            if len(ids) > self.pixel_samples:
                num_split = len(ids) // self.pixel_samples + 1
                idset = np.array_split(ids, num_split)
            else:
                idset = [ids]     

            o_ = I_dec[target, :, :, :].reshape(nImgArray[b], C, H * W).permute(2,0,1)  # [N, c, h, w]]
            for ids in idset:
                o_ids = o_[ids, :, :]
                coords = ind2coords(np.array((H, W)), ids).expand(nImgArray[b],-1,-1,-1)
                glc_ids = F.grid_sample(glc[target, :, :, :], coords.to(self.device), mode='bilinear', align_corners=False).reshape(len(target), -1, len(ids)).permute(2,0,1) # [m, N, f]                   

                """ glc_ids """
                x = torch.cat([o_ids, glc_ids], dim=2) # [len(ids), N, 256+3]
                glc_ids = self.glc_upsample(x)            
                x = torch.cat([o_ids, glc_ids], dim=2) # [len(ids), N, 256+3]

                x = self.glc_aggregation(x)  #[len(ids), 384]       
                x_n, x_brdf = self.regressor(x, len(ids)) # [len(ids), 3]       
                X_n = F.normalize(x_n, dim=1, p=2)
                if self.target == 'normal':
                    nout[b, ids, :] = X_n.detach()  
                if self.target == 'brdf':
                    bout[b, ids, :] = torch.relu(x_brdf[0]).detach()  
                    rout[b, ids, :] = torch.relu(x_brdf[1]).detach()  
                    mout[b, ids, :] = torch.relu(x_brdf[2]).detach()  

        nout = nout.permute(0, 2, 1).reshape(B, 3, H, W)
        bout = bout.permute(0, 2, 1).reshape(B, 3, H, W)
        rout = rout.permute(0, 2, 1).reshape(B, 1, H, W)
        mout = mout.permute(0, 2, 1).reshape(B, 1, H, W)


  
        return nout, bout, rout, mout


