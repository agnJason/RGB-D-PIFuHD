# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 20:20:21 2020

@author: Administrator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F 

class MLP(torch.nn.Module):
    def __init__(self, filter_channels, merge_layer=0, res_layers=[], norm='group', last_op=None):
        '''
        filter_channels: 卷积核（过滤器）数量 
        merge_layer
        res_layers: residual结构 （解决退化问题，防止参数不变，更好的优化）
        norm
        last_op
        '''
        super(MLP,self).__init__()
        
        self.filters = nn.ModuleList() #隐藏层们
        self.norms = nn.ModuleList()
        self.merge_layer = merge_layer if merge_layer > 0 else len(filter_channels) // 2
        self.res_layers = res_layers
        self.norm = norm
        self.last_op = last_op
        
        for i in range(len(filter_channels)-1):
            if i in self.res_layers: #构建残差层
                self.filters.append(nn.Conv1d(
                        filter_channels[i]+filter_channels[0],filter_channels[i+1],1))
            else:
                self.filters.append(nn.Conv1d(
                        filter_channels[i], filter_channels[i+1], 1))
            if i != len(filter_channels)-2: #插入归一化层
                if norm == 'group':
                    self.norms.append(nn.GroupNorm(32, filter_channels[i+1])) #需要划分为的groups:32
                elif norm == 'batch':
                    self.norms.append(nn.BatchNorm1d(filter_channels[i+1]))
        
    def forward(self,feature):
        '''
        前向传播
        feature may include multiple view inputs
        args:
            feature: [B, C_in, N]
        return:
            [B, C_out, N] prediction
        '''
        y = feature
        tmpy = feature
        phi = None
        for i, f in enumerate(self.filters):
            '''
            传播每一层： Conv1d + leaky_relu(norms)
            return: y, 在merge_layer的y
            '''
            #print(i)
            y = f(
                y if i not in self.res_layers
                else torch.cat([y, tmpy], 1)
            )
            if i != len(self.filters)-1:
                if self.norm not in ['batch', 'group']:
                    y = F.leaky_relu(y)
                else:
                    y = F.leaky_relu(self.norms[i](y))         
            if i == self.merge_layer:
                phi = y.clone()
        if self.last_op is not None:
            y = self.last_op(y) #sigmoid

        return y, phi
    














