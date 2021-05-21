# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 21:01:54 2020

@author: Administrator
"""
import torch
from BasePIFuNet import BasePIFuNet
from Filter import Filter
from MLP import MLP
import torch.nn as nn
from DepthNormalizer import DepthNormalizer
from net_util import init_net
from networks import define_G
import torch.nn.functional as F 

class PIFuNetwNML(BasePIFuNet):
    
    def __init__(self,
                 opt,
                 projection_mode='orthogonal',
                 criteria={'occ': nn.MSELoss()}
                 ):
        super(PIFuNetwNML,self).__init__(
                projection_mode=projection_mode,
                 criteria=criteria)
        
        self.name = 'hg_pifu'
        self.opt = opt
        
        in_channels = 3
        try:
            if opt.use_front_normal:
                in_channels += 3
            if opt.use_back_normal:
                in_channels += 3
        except:
            pass
        
        self.image_filter = Filter(opt.num_stack, opt.hg_depth,in_channels,
                                   opt.hg_dim, opt.norm, opt.hg_down, False)
        self.mlp = MLP(
                filter_channels = self.opt.mlp_dim,
                merge_layer = self.opt.merge_layer,
                res_layers = self.opt.mlp_res_layers,
                norm = self.opt.mlp_norm,
                last_op = nn.Sigmoid())
        
        self.spatial_enc = DepthNormalizer(opt) #空间编码
        
        self.im_feat_list = []
        self.tmpx = None
        self.normx = None
        self.phi = None
        
        self.intermediate_preds_list = []
        
        init_net(self)
        
        self.netF = None #front net
        self.netB = None #back net
        
        try:
            if opt.use_front_normal:
                self.netF = define_G(3,3,64,'global', 4, 9, 1, 3, 'instance')
            if opt.use_back_normal:
                self.netB = define_G(3,3,64,'global', 4, 9, 1, 3, 'instance')
        except:
            pass
        self.nmlF = None
        self.nmlB = None
        
    def filter(self, images):
        '''
        将图像输入全神经网络
        args:
            images: [B, C, H, W]
        '''
        #先计算前后
        nmls = []
        with torch.no_grad():#不需要计算梯度，不反向传播
            if self.netF != None:
                self.nmlF = self.netF.forward(images).detach() #不需要回传梯度，加快速度
                nmls.append(self.nmlF)
            if self.netB != None:
                self.nmlB = self.netB.forward(images).detach()
                nmls.append(self.nmlB)
        if len(nmls)!=0:
            nmls = torch.cat(nmls,1)
            if images.size()[2:] != nmls.size()[2:]: #图像大小一致，若小了就插值法
                nmls = nn.Upsample(size=images.size()[2:], mode='bilinear', align_corners=True)(nmls)
            images = torch.cat([images,nmls],1) #训练后数据和images合并
        #传入filter 图像特征
        self.im_feat_list, self.normx = self.image_filter(images)
        # print(self.im_feat_list[0].shape)#[1, 256, 128, 128]
        if not self.training: #???
            self.im_feat_list = [self.im_feat_list[-1]]
            
    def query(self, points, calibs, transforms=None, labels = None,
              update_pred = True, update_phi = True):
        '''
        给定3d点，返回给定相机矩阵的2d投影
        args:
            points: [B, 3, N] 3d points in world space
            calibs: [B, 3, 4] 每个图像的校准矩阵
            transforms: [B, 2, 3] 图像空间坐标变换
            labels: [B, C, N] 监督用真实标签
        return:
            [B, C, N] prediction    
        '''
        xyz = self.projection(points,calibs,transforms) #将点投影到屏幕空间
        xy = xyz[:,:2,:] #xy坐标
        
        #在边界内的
        in_boundingbox = (xyz >= -1) & (xyz <= 1)
        in_boundingbox = in_boundingbox[:,0,:] & in_boundingbox[:,1,:] & in_boundingbox[:,2,:] #xyz都在边界内
        in_boundingbox = in_boundingbox[:,None,:].detach().float() #把在边界内的设为1，不在为0
        
        if labels is not None:
            self.labels = in_boundingbox *labels
        
        sp_feat = self.spatial_enc(xyz) #规格化z 空间特征
        
        intermediate_preds_list = [] #中间预测
        
        phi = None
        for i, im_feat in enumerate(self.im_feat_list):
            point_local_feat_list = [self.index(im_feat, xy), sp_feat] #提取图像特征
            point_local_feat = torch.cat(point_local_feat_list, 1)
            #print(point_local_feat.shape)
            pred, phi = self.mlp(point_local_feat)  #多层感知机预测

            pred = in_boundingbox * pred
            
            intermediate_preds_list.append(pred)
            
        if update_phi:
            self.phi = phi
        if update_pred:
            self.intermediate_preds_list = intermediate_preds_list
            self.preds = pred
        
        
    def loadFromPIFu(self, net):
        '''
        从已有网络加载
        '''
        model_dict = self.image_filter.state_dict()
        pretrained_dict = {k: v for k, v in net.image_filter.state_dict().items() if k in model_dict} #提取本模型所需
        
        for k,v in pretrained_dict.items():
            if v.size() == model_dict[k].size():
                model_dict[k] = v #初始化部分参数
        
        not_initialized = set() #储存未初始化的
        
        for k,v in model_dict.items():
            if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                not_initialized.add(k.split('.')[0])

        print('not initialized', sorted(not_initialized))
        self.image_filter.load_state_dict(model_dict) #加载参数
        
        #加载mlp
        model_mlp_dict = self.mlp.state_dict()
        pretrained_mlp_dict = {k: v for k, v in net.mlp.state_dict().items() if k in model_mlp_dict}
        
        for k,v in pretrained_mlp_dict.items():
            if v.size() == model_mlp_dict[k].size():
                model_mlp_dict[k] = v #初始化部分参数
        
        not_initialized_mlp = set() #储存未初始化的
        
        for k,v in model_mlp_dict.items():
            if k not in pretrained_mlp_dict or v.size() != pretrained_mlp_dict[k].size():
                not_initialized_mlp.add(k.split('.')[0])
        
        print('mlp not initialized', sorted(not_initialized_mlp))
        self.mlp.load_state_dict(model_mlp_dict) #加载参数
        
    def calc_normal(self, points, calibs, transforms=None, labels=None, delta=0.1):
        '''
        返回曲面法线。
        args:
            points: [B, 3, N] 3d points in world space
            calibs: [B, 3, 4] calibration matrices for each image
            transforms: [B, 2, 3] image space coordinate transforms
            delta: perturbation for finite difference 有限差分扰动
        '''
        pdx = points.clone()
        pdx[:,0,:] += delta
        pdy = points.clone()
        pdy[:,1,:] += delta
        pdz = points.clone()
        pdz[:,2,:] += delta
        
        if labels != None:
            self.labels_nml = labels
        
        points_all = torch.stack([points, pdx, pdy, pdz],3)
        points_all = points_all.view(*points.size()[:2],-1)
        xyz = self.projection(points_all, calibs, transforms)
        xy = xyz[:, :2, :]

        im_feat = self.im_feat_list[-1]
        sp_feat = self.spatial_enc(xyz)
        
        point_local_feat_list = [self.index(im_feat,xy),sp_feat]
        point_local_feat = torch.cat(point_local_feat_list,1)
        
        pred, phi = self.mlp(point_local_feat)
        pred = pred.view(*pred.size()[:2],-1,4) #(B,1,N,4)
        
        dfdx = pred[:,:,:,1] - pred[:,:,:,0]
        dfdy = pred[:,:,:,2] - pred[:,:,:,0]
        dfdz = pred[:,:,:,3] - pred[:,:,:,0]
        nml = -torch.cat([dfdx,dfdy,dfdz],1)
        nml = F.normalize(nml,dim=1,eps=1e-8)
        
        self.nml = nml
        
    def get_im_feat(self):
        '''
        返回最后一个im_feat
        return:
            [B, C, H, W]
        '''
        return self.im_feat_list[-1]
    
    def get_error(self, gamma):
        '''
        返回损失值
        '''
        error = {'Err(occ)':0}

        for preds in self.intermediate_preds_list:
            #print(preds, self.labels)
            error['Err(occ)'] += self.criteria['occ'](preds, self.labels, torch.Tensor([gamma]))
            
        error['Err(occ)'] /= len(self.intermediate_preds_list)
        
        if self.nmls != None and self.labels_nml is not None:
            error['Err(nml)'] = self.criteria['nml'](self.nmls, self.labels_nml)
        
        return error

    def forward(self, images, points, calibs, labels, gamma, 
                points_nml=None, labels_nml = None):
        self.filter(images) #将图像输入全神经网络
        self.query(points, calibs, labels=labels) #将三维点投到2维
        if points_nml != None and labels_nml != None:
            self.calc_normal(points_nml, calibs, labels = labels_nml) #返回曲面法线
            
        res = self.get_preds() #prediction
        
        err = self.get_error(gamma) #损失值
        
        return err, res