# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 22:31:10 2020

@author: Administrator
"""

import torch
import torch.nn as nn

def index(feat, uv):
    '''
    grid_sample：通过双线性插值的方式变换图像，提取图像特征 从feat提出uv范围内的图像特征
    args:
        feat: [B, C, H, W] image features
        uv: [B, 2, N] normalized image coordinates ranged in [-1, 1]
    return:
        [B, C, N] sampled pixel values
    '''
    uv = uv.transpose(1, 2) #1，2维转置
    uv = uv.unsqueeze(2) #在第3位置后插入一维度
    samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)
    return samples[:, :, :, 0]

def orthogonal(points, calib, transform=None):
    '''
    使用正交投影将点投影到屏幕空间
    args:
        points: [B, 3, N] 3d points in world coordinates
        calib: [B, 3, 4] projection matrix
        transform: [B, 2, 3] screen space transformation
    return:
        [B, 3, N] 3d coordinates in screen space
    '''
    rot = calib[:, :3, :3]  #[B, 3, 3]
    trans = calib[:, :3, 3:4]  #[B, 3, 1]
    #print(points.shape, calib.shape)
    pts = torch.baddbmm(trans, rot, points) #[B, 3, N]
    if transform is not None:
        scale = transform[:2, :2]
        shift = transform[:2, 2:3]
        pts[:, :2, :] = torch.baddbmm(shift, scale, pts[:, :2, :])
    return pts

def perspective(points, calib, transform=None):
    '''
    使用透视投影将点投影到屏幕空间
    args:
        points: [B, 3, N] 3d points in world coordinates
        calib: [B, 3, 4] projection matrix
        transform: [B, 2, 3] screen space trasnformation
    return:
        [B, 3, N] 3d coordinates in screen space
    '''
    rot = calib[:, :3, :3]
    trans = calib[:, :3, 3:4]
    homo = torch.baddbmm(trans, rot, points)
    xy = homo[:, :2, :] / homo[:, 2:3, :]
    if transform is not None:
        scale = transform[:2, :2]
        shift = transform[:2, 2:3]
        xy = torch.baddbmm(shift, scale, xy)
    
    xyz = torch.cat([xy, homo[:, 2:3, :]], 1)
    return xyz

class BasePIFuNet(nn.Module):
    def __init__(self, projection_mode='orthogonal', criteria={'occ', nn.MSELoss()}):
        '''
        args:
            projection_mode: orthonal 正交 / perspective 透视
        '''
        super(BasePIFuNet, self).__init__()
        self.name = 'base'

        self.criteria = criteria

        self.index = index
        self.projection = orthogonal if projection_mode == 'orthogonal' else perspective

        self.preds = None
        self.labels = None
        self.nmls = None
        self.labels_nml = None
        self.preds_surface = None # with normal loss only
        
    def forward(self, points, images, calibs, transforms=None):
        '''
        args:
            points: [B, 3, N] 3d points in world space
            images: [B, C, H, W] input images
            calibs: [B, 3, 4] calibration matrices for each image
            transforms: [B, 2, 3] image space coordinate transforms
        return:
            [B, C, N] prediction corresponding to the given points
        '''
        self.filter(images)
        self.query(points, calibs, transforms)
        return self.get_preds()

    def filter(self, images):
        '''
        对图像应用全卷积网络。
        生成的特征将被存储。
        args:
            images: [B, C, H, W]
        '''
        None
    
    def query(self, points, calibs, trasnforms=None, labels=None):
        '''
        给定三维点，我们得到给定摄像机矩阵的二维投影。
        过滤器需要预先调用。
        预测存储到self.preds
        args:
            points: [B, 3, N] 3d points in world space
            calibs: [B, 3, 4] calibration matrices for each image 校准矩阵
            transforms: [B, 2, 3] image space coordinate transforms 空间坐标变换矩阵
            labels: [B, C, N] ground truth labels (for supervision only)
        return:
            [B, C, N] prediction
        '''
        None

    def calc_normal(self, points, calibs, transforms=None, delta=0.1):
        '''
        返回曲面法线。
        args:
            points: [B, 3, N] 3d points in world space
            calibs: [B, 3, 4] calibration matrices for each image
            transforms: [B, 2, 3] image space coordinate transforms
            delta: perturbation for finite difference 扰动
        '''
        None

    def get_preds(self):
        '''
        返回当前预测值
        return:
            [B, C, N] prediction
        '''
        return self.preds

    def get_error(self, gamma=None):
        '''
        返回loss
        '''
        return self.error_term(self.preds, self.labels, gamma)
