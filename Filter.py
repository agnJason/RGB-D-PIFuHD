# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 11:01:10 2020

@author: Administrator
"""

import torch
import torch.nn as nn 
import torch.nn.functional as F 

def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    '''
    用3*3的卷积核进行二维卷积
    in_planes:输入样本的通道数
    out_planes:输出样本的通道数
    strd:滑动窗口大小
    padding:补洞策略
    '''
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=strd, padding=padding, bias=bias)

class ConvBlock(nn.Module):
    '''
    卷积 三层卷积核宽度为3的二维卷积 Residual模块？
    args:
        in_channels: 输入样本通道数
        out_channels: 输出样本通道数
        norm: 标准化形式 batch/group
    '''
    def __init__(self, in_channels, out_channels, norm='batch'):
        super(ConvBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, int(out_channels/2))
        self.conv2 = conv3x3(int(out_channels/2), int(out_channels/4))
        self.conv3 = conv3x3(int(out_channels/4), int(out_channels/4))
        
        if norm == 'batch':
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.bn2 = nn.BatchNorm2d(int(out_channels/2))
            self.bn3 = nn.BatchNorm2d(int(out_channels/4))
            self.bn4 = nn.BatchNorm2d(in_channels)
        elif norm == 'group':
            self.bn1 = nn.GroupNorm(32, in_channels)
            self.bn2 = nn.GroupNorm(32, int(out_channels/2))
            self.bn3 = nn.GroupNorm(32, int(out_channels/4))
            self.bn4 = nn.GroupNorm(32, in_channels)
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                    self.bn4,
                    nn.ReLU(True),
                    nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, stride=1,bias=False)
                    ) #downsample使输出维度和当前维度一致
        else:
            self.downsample = None
        
    def forward(self, x):
        residual = x #带残差的卷积
        if self.downsample != None:
            residual = self.downsample(residual) #使维度一致
            
        out1 = self.conv1(F.relu(self.bn1(x), True))    #int(out_planes / 2)
        out2 = self.conv2(F.relu(self.bn2(out1), True)) #int(out_planes / 4)
        out3 = self.conv3(F.relu(self.bn3(out2), True)) #int(out_planes / 4)
        
        out3 = torch.cat([out1, out2, out3], 1) #in_channels
        out3 += residual
        
        return out3

class HourGlass(nn.Module):
    '''
    depth阶的沙漏结构 https://www.jianshu.com/p/2bc7db188a6a
    称之为沙漏因为最中间的1层的图像宽和高因为不断的迭代是最小的，两边的2……depth越来越大，即沙漏
    '''
    def __init__(self, depth, n_features, norm='batch'):
        super(HourGlass, self).__init__()
        self.depth = depth
        self.features = n_features
        self.norm = norm
    
        self._generate_network(self.depth)
    
    def _generate_network(self, level):
        '''
        生成网络，b1_n,b2_n,b1_(n-1),b2_(n-1)...b1_1,b2_1,b2_plus_1,b3_1,b3_2,...b3_l
        args:
            level: int 层数
        '''
        self.add_module('b1_{}'.format(level), ConvBlock(self.features, self.features, norm=self.norm))
        self.add_module('b2_{}'.format(level), ConvBlock(self.features, self.features, norm=self.norm))
    
        if level > 1:
            self._generate_network(level-1) #递归建层
        else:
            self.add_module('b2_plus_{}'.format(level), ConvBlock(self.features, self.features, norm=self.norm))
        
        self.add_module('b3_{}'.format(level), ConvBlock(self.features, self.features, norm=self.norm))
    
    def _forward(self, level, inp):
        '''
        向前传播
        args:
            level: 层数
            inp：input,输入
        '''
        
        #上分支
        up1 = inp
        up1 = self._modules['b1_{}'.format(level)](up1)
        
        #下分支 先降采样（池化）/2 再升采样（插值）*2
        low1 = F.avg_pool2d(inp, 2, stride=2) #池化
        low1 = self._modules['b2_{}'.format(level)](low1)
        
        if level>1: #迭代
            low2 = self._forward(level-1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_{}'.format(level)](low2)
            
        low3 = low2
        low3 = self._modules['b3_{}'.format(level)](low3)
        
        up2 = F.interpolate(low3, scale_factor=2, mode='bicubic', align_corners=True) #二三次曲线插值，扩大一倍 因为池化了
        
        return up1+up2
    
    def forward(self, x):
        return self._forward(self.depth, x)
    
class Filter(nn.Module):
    def __init__(self, n_stack, depth, in_channels, last_channels,
                 norm = 'batch', down_type='conv64', use_sigmoid=True):
        super(Filter, self).__init__()
        self.n_stack = n_stack
        self.depth = depth
        self.in_ch = in_channels
        self.last_ch = last_channels
        self.norm = norm
        self.down_type = down_type
        self.use_sigmoid = use_sigmoid
        
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size = 7, stride = 2, padding = 3)
        last_ch = self.last_ch
        
        if self.norm == 'batch': #标准化层
            self.bn1 = nn.BatchNorm2d(64)
        elif self.norm == 'group':
            self.bn1 = nn.GroupNorm(32,64)
        
        if self.down_type == 'conv64':
            self.conv2 = ConvBlock(64, 64, self.norm)
            self.down_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        elif self.down_type == 'conv128':
            self.conv2 = ConvBlock(128, 128, self.norm)
            self.down_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        elif self.down_type == 'ave_pool' or self.down_type == 'no_down':
            self.conv2 = ConvBlock(64, 128, self.norm)

        self.conv3 = ConvBlock(128, 128, self.norm)
        self.conv4 = ConvBlock(128, 256, self.norm)
        
        #
        for stack in range(self.n_stack):
            self.add_module('m{}'.format(stack), HourGlass(self.depth, 256, self.norm))
            
            self.add_module('top_m_{}'.format(stack), ConvBlock(256,256,self.norm))
            self.add_module('conv_last{}'.format(stack),
                            nn.Conv2d(256,256, kernel_size=1, stride=1, padding=0))
            if self.norm == 'batch':
                self.add_module('bn_end' + str(stack), nn.BatchNorm2d(256))
            elif self.norm == 'group':
                self.add_module('bn_end' + str(stack), nn.GroupNorm(32, 256))
                
            self.add_module('l{}'.format(stack),
                            nn.Conv2d(256, last_ch,kernel_size=1, stride=1, padding=0))
            
            if stack < self.n_stack-1:
                self.add_module(
                        'bl{}'.format(stack),
                        nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.add_module(
                        'al{}'.format(stack),
                        nn.Conv2d(last_ch, 256, kernel_size=1, stride=1, padding=0))
    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)), True) #第一次卷积
        
        #第二次卷积 或许需要下采样 最后都是128通道
        if self.down_type == 'ave_pool':
            x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        elif self.down_type == ['conv64','conv128']:
            x = self.down_conv2(self.conv2(x))
        elif self.down_type == 'no_down':
            x = self.conv2(x)
        else:
            raise NameError('unknown downsampling type')
        
        normx = x #记录x 128
        
        x = self.conv3(x)
        x = self.conv4(x) #256
        
        previous = x #记录x
        
        outputs = []
        for i in range(self.n_stack):
            hourglass = self._modules['m{}'.format(i)](previous)
            
            ll = hourglass 
            
            ll = F.relu(self._modules['bn_end{}'.format(i)](
                    self._modules['conv_last{}'.format(i)](
                            self._modules['top_m_{}'.format(i)](ll))),True) #m_top->conv_last->norm
            
            temp_out = self._modules['l{}'.format(i)](ll) #last_ch
            
            if self.use_sigmoid:
                outputs.append(nn.Tanh()(temp_out))
            else:
                outputs.append(temp_out)
            
            if i<self.n_stack-1:
                ll = self._modules['bl{}'.format(i)](ll)
                temp_out = self._modules['al{}'.format(i)](temp_out) #256
                previous = previous + ll + temp_out #修正previous
                
        return outputs, normx
        














