# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 20:46:50 2020

@author: Administrator
"""
import torch.nn as nn
from BasePIFuNet import BasePIFuNet
import torch
from MLP import MLP
from Filter import Filter
from net_util import init_net
import torch.nn.functional as F 

class PIFuMRNet(BasePIFuNet):
    '''
    HGPIFu使用堆叠沙漏作为图像编码器。
    '''
    def __init__(self,
                 opt,
                 netG,
                 projection_mode='otthogonal',
                 criteria={'occ':nn.MSELoss()}):
        super(PIFuMRNet, self).__init__(
            projection_mode=projection_mode,
            criteria=criteria) #调用父类 并运行本类的__init__()
        self.name = 'hg_pifu'
        
        in_channels = 3
        try:
            if netG.opt.use_front_normal:
                in_channels += 3
            if netG.opt.use_back_normal:
                in_channels += 3
        except:
            pass
        
        self.opt = opt
        self.image_filter = Filter(opt.num_stack, opt.hg_depth, in_channels, opt.hg_dim,
                                   opt.norm, 'no_down', False)
        self.mlp = MLP(filter_channels=self.opt.mlp_dim,
                       merge_layer = -1,
                       res_layers=self.opt.mlp_res_layers,
                       norm=self.opt.mlp_norm,
                       last_op = nn.Sigmoid())
        
        self.im_feat_list = []
        self.preds_interm = None
        self.preds_low = None
        self.w = None
        self.gamma = None
        
        self.intermdiate_pred_list = []
        
        init_net(self) #初始化网络
        
        self.netG = netG
        
    def train(self, mode=True):
        """
        将模块设置为训练模式。
        """
        self.training = mode
        for module in self.children():
            module.train(mode)
        if not self.opt.train_full_pifu:
            self.netG.eval()
            
        return self
    
    def filter_global(self, images):
        '''
        将图像输入全卷积神经网络
        args:
            images: [B1, C, H, W]
        '''
        if self.opt.train_full_pifu:
            self.netG.filter(images)
        else:
            with torch.no_grad():
                self.netG.filter(images)
    
    def filter_local(self, images, rect=None):
        '''
        将图像输入全卷积神经网络
        args:
            images: [B1, B2, C, H, W]
        '''
        #print(images.shape)
        nmls = [] #储存神经网络训练后的
        try:
            if self.netG.opt.use_front_normal:
                nmls.append(self.netG.nmlF)
            if self.netG.opt.use_back_normal:
                nmls.append(self.netG.nmlB)
        except:
            pass
        if len(nmls):
            nmls = nn.Upsample(size=(self.opt.loadSizeBig, self.opt.loadSizeBig),
                               mode='bilinear',align_corners=True)(torch.cat(nmls,1))
            
            if rect is None:
                images = torch.cat([images, nmls[:,None].expand(-1,images.size(1),-1,-1,-1)], 2)
            else:
                nml = []
                for i in range(rect.size(0)):
                    for j in range(rect.size(1)):
                        x1,y1,x2,y2 = rect[i,j] #人物框
                        tmp = nmls[i,:,y1:y2,x1:x2] #将人物框出来
                        nml.append(tmp)
                    nml = torch.stack(nml,0).view(*rect.shape[:2], *nml[0].size())
                    images = torch.cat([images, nml],2)
        #print(images.shape)
        self.im_feat_list, self.normx = self.image_filter(images.view(-1, *images.size()[2:])) #跑一次filter
        
        if not self.training: #不是训练 只要结果
            self.im_feat_list = [self.im_feat_list[-1]]
        #print(self.im_feat_list[0].shape) #[1, 16, 512, 512]
    def query(self, points, calib_local, calib_global=None, transforms=None, labels=None):
        '''
        给定一堆3d点，返回给定相机矩阵的2d投影，预测给定xyz下是否有人体
        args:
            points: [B1, B2, 3, N] 3d points in world space
            calibs_local: [B1, B2, 4, 4]  每个图像的校准矩阵
            calibs_global: [B1, 4, 4]
            transforms: [B1, 2, 3] 图像空间坐标变换
            labels: [B1, B2, C, N] 监督用真实标签
        return:
            [B, C, N] prediction    
        '''
        if calib_global is not None:
            B = calib_local.size(1) #B2
        else:
            B = 1
            points = points[:, None] #增加第二个维度
            calib_global = calib_local
            calib_local = calib_local[:,None]
            
        ws = []
        preds = []
        preds_interm = []
        preds_low = []
        gammas = []
        newlabels = []
        for i in range(B):
            xyz = self.projection(points[:,i], calib_local[:,i], transforms) #将点投影到图片空间
            xy = xyz[:, :2, :]
            
            #边框内
            in_boundingbox = (xyz >= -1) & (xyz <= 1)
            in_boundingbox = in_boundingbox[:,0,:] & in_boundingbox[:,1,:]#xy都在边界内
            in_boundingbox = in_boundingbox[:,None,:].detach().float() #把在边界内的设为1，不在为0
            
            self.netG.query(points=points[:,i], calibs=calib_global)
            preds_low.append(torch.stack(self.netG.intermediate_preds_list,0)) #下线 粗糙估计
            
            if labels is not None:
                newlabels.append(in_boundingbox*labels[:,i])
                with torch.no_grad():
                    ws.append(in_boundingbox.size(2)/in_boundingbox.view(in_boundingbox.size(0),-1).sum(1)) # N/三维坐标和 长度B1
                    gammas.append(1-newlabels[-1].view(newlabels[-1].size(0),-1).sum(1) \
                                  / in_boundingbox.view(in_boundingbox.size(0),-1).sum(1)) # (1-in_bb*label的坐标和)/三维坐标和 点在外的比例
            z_feat = self.netG.phi #粗糙层最后每一个的特征z （经过一次sigmoid就是每个区域的预测）
            
            if not self.opt.train_full_pifu:
                z_feat = z_feat.detach()
                
            intermediate_preds_list = []
            for j, im_feat in enumerate(self.im_feat_list):
                point_local_feat_list = [self.index(im_feat.view(-1,B,*im_feat.size()[1:])[:,i],xy),z_feat]
                point_local_feat = torch.cat(point_local_feat_list, 1)
                #print(point_local_feat.shape)
                pred = self.mlp(point_local_feat)[0]  #带着图像特征和粗糙层的z_feat一起判断
                pred = in_boundingbox * pred
                intermediate_preds_list.append(pred)
            
            preds_interm.append(torch.stack(intermediate_preds_list,0)) #每一次的中间预测记录
            preds.append(intermediate_preds_list[-1]) #最后预测记录
        self.preds = torch.cat(preds,0) #[B2,3,N]
        self.preds_interm = torch.cat(preds_interm, 1) #[B1,B2,3,N]
        self.preds_low = torch.cat(preds_low, 1) #[B1.B2,3,N]
        
        if labels is not None:
            self.w = torch.cat(ws, 0)
            self.gamma = torch.cat(gammas,0)
            self.labels = torch.cat(newlabels, 0)
            
    def calc_normal(self, points, calib_local, calib_global, transforms=None, 
                     labels=None, delta=0.001, fd_type='forward'):
        '''
        返回曲面法线。
        args:
            points: [B1, B2, 3, N] 3d points in world space
            calibs_local: [B1, B2, 4, 4] calibration matrices for each image
            calibs_global: [B1, 4, 4] calibration matrices for each image
            transforms: [B1, 2, 3] image space coordinate transforms
            labels: [B1, B2, 3, N] ground truth normal
            delta: perturbation for finite difference扰动
            fd_type: 有限差分 (forward/backward/central) 
        '''
        B = calib_local.size(1) #B2
        
        if labels is not None:
            self.labels_nml = labels.view(-1, *labels.size()[2:]) #[B1*B2,3,N]
        
        im_feat = self.im_feat_list[-1].view(-1,B,*self.im_feat_list[-1].size()[1:]) #最后一个特征，并变换维度
        
        nmls = []
        for i in range(B):
            points_sub = points[:,i] #[B1,3,N]
            pdx = points_sub.clone()
            pdx[:,0,:] += delta
            pdy = points_sub.clone()
            pdy[:,1,:] += delta
            pdz = points_sub.clone()
            pdz[:,2,:] += delta
            
            points_all = torch.stack([points_sub,pdx,pdy,pdz],3) #[B1,3,N,4]
            points_all = points_all.view(*points_sub.size()[:2],-1) #[B1,3,N*4]
            xyz = self.projection(points_all, calib_local[:,i],transforms)
            xy = xyz[:,:2,:]
            
            self.netG.query(points=points_all, calibs=calib_global, update_pred=False)
            z_feat = self.netG.phi
            if not self.opt.train_full_pifu:
                z_feat = z_feat.detach()
            
            point_local_feat_list = [self.index(im_feat[:,i], xy),z_feat]
            point_local_feat = torch.cat(point_local_feat_list, 1)
            pred = self.mlp(point_local_feat)[0]
            
            pred = pred.view(*pred.size()[:2], -1, 4) # (B1, 1, N, 4)
            
            dfdx = pred[:,:,:,1] - pred[:,:,:,0] #x方向的梯度
            dfdy = pred[:,:,:,2] - pred[:,:,:,0]
            dfdz = pred[:,:,:,3] - pred[:,:,:,0]
            
            nml = -torch.cat([dfdx,dfdy,dfdz], 1)
            nml = F.normalize(nml, dim=1, eps=1e-8)
            
            nmls.append(nml)
        
        self.nmls = torch.stack(nmls,1).view(-1, 3, points.size(3)) #[B1*B2,3,N]
    
    def get_im_feat(self):
        '''
        return the image filter in the last stack
        return:
            [B, C, H, W]
        '''
        return self.im_feat_list[-1]
    
    def get_error(self):
        '''
        返回error
        '''
        error = {}
        if self.opt.train_full_pifu:
            if not self.opt.no_intermediate_loss:
                error['Err(occ)'] = 0.0
                #print(self.preds_low.size(0),self.preds_low,self.labels)
                for i in range(self.preds_low.size(0)):
                    error['Err(occ)'] += self.criteria['occ'](self.preds_low[i], self.labels, self.gamma, self.w)
                error['Err(occ)'] /= self.preds_low.size(0)
            
        error['Err(occ:fine)'] = 0.0
        #print(self.preds_interm.size(0), self.preds_interm[i],self.labels)
        for i in range(self.preds_interm.size(0)):
            error['Err(occ:fine)'] += self.criteria['occ'](self.preds_low[i], self.labels, self.gamma, self.w)
        error['Err(occ:fine)'] /= self.preds_interm.size(0)
        if self.nmls is not None and self.labels_nml is not None:
            error['Err(nml:fine)'] = self.criteria['nml'](self.nmls, self.labels_nml)
            
        return error
    
    def forward(self, images_local, images_global, points, calib_local,
                calib_global, labels, points_nml=None, labels_nml=None, rect=None):
        self.filter_global(images_global)
        self.filter_local(images_local,rect)
        self.query(points, calib_local, calib_global, labels=labels)
        if points_nml != None and labels_nml != None:
            self.calc_normal(points_nml, calib_local, calib_global, labels=labels_nml)
            
        res = self.get_preds()
        
        err = self.get_error()
        
        return err, res
