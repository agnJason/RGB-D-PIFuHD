# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 21:46:07 2020

@author: Administrator
"""
import torch
from torch.nn import init
import torch.nn as nn
def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    '''
    初始化net
    '''
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net

class CustomBCELoss(nn.Module):
    def __init__(self, brock=False, gamma=None):
        super(CustomBCELoss, self).__init__()
        self.brock = brock
        self.gamma = gamma

    def forward(self, pred, gt, gamma, w=None):
        x_hat = torch.clamp(pred, 1e-5, 1.0-1e-5) # prevent log(0) from happening
        gamma = gamma[:,None,None] if self.gamma is None else self.gamma
        if self.brock:
            x = 3.0*gt - 1.0 # rescaled to [-1,2]

            loss = -(gamma*x*torch.log(x_hat) + (1.0-gamma)*(1.0-x)*torch.log(1.0-x_hat))
        else:
            loss = -(gamma*gt*torch.log(x_hat) + (1.0-gamma)*(1.0-gt)*torch.log(1.0-x_hat))

        if w is not None:
            if len(w.size()) == 1:
                w = w[:,None,None] 
            return (loss * w).mean()
        else:
            return loss.mean()