# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 21:17:13 2020

@author: Administrator
"""


import torch.nn as nn


class DepthNormalizer(nn.Module):
    def __init__(self, opt):
        super(DepthNormalizer, self).__init__()
        self.opt = opt

    def forward(self, xyz):
        '''
        规格化深度值z
        args:
            xyz: [B, 3, N] depth value
        '''
        z_feat = xyz[:,2:3,:] * (self.opt.loadSize // 2) / self.opt.z_size

        return z_feat