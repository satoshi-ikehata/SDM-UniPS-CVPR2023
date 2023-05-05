"""
Scalable, Detailed and Mask-free Universal Photometric Stereo Network (CVPR2023)
# Copyright (c) 2023 Satoshi Ikehata
# All rights reserved.
"""

import glob
import torch.utils.data as data
from .dataloader import realdata
import numpy as np

class dataio(data.Dataset):
    def __init__(self, mode, args):
        self.mode = mode
        data_root = args.test_dir
        extension = args.test_ext
        self.numberOfImageBuffer = args.max_image_num
        self.prefix= args.test_prefix
        self.mask_margin = args.mask_margin
        self.outdir = args.session_name
        self.data_root = data_root
        self.extension = extension
        self.data_name = []
        self.set_id = []
        self.valid = []
        self.sample_id = []
        self.dataCount = 0
        self.dataLength = -1
        self.mode = mode
        self.max_image_resolution = None
        
        print('Exploring %s' % (data_root))
        objlist = glob.glob(f"{data_root}/*{extension}")
        objlist = sorted(objlist)
        self.objlist = objlist
        print(f"Found {len(self.objlist)} objects!\n")
        self.data = realdata.dataloader(self.numberOfImageBuffer, mask_margin=self.mask_margin, outdir=self.outdir)

    def __getitem__(self, index_):

        objid = index_
        objdir = self.objlist[objid]
        self.data.load(objdir, prefix = self.prefix, max_image_resolution = self.max_image_resolution)
        img = self.data.I.transpose(2,0,1,3) # c, h, w, N
        numberOfImages = self.data.I.shape[3]           
        nml = self.data.N.transpose(2,0,1) # 3, h, w
        mask = np.transpose(self.data.mask, (2,0,1)) # 1, h, w
        roi = self.data.roi
        return img, nml, mask, numberOfImages, roi

    def __len__(self):
        return len(self.objlist)
