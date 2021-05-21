# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 08:40:31 2020

@author: Administrator
"""

from train import train
from options import BaseOptions

if __name__ == '__main__':
    use_gpu = True
    num_epoch = 1
    opt = BaseOptions().parse()
    print('Begin!')
    train(opt, num_epoch, use_gpu)