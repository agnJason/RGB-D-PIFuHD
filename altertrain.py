# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 11:23:18 2021

@author: Again Jason
"""
from train import train as trainMR
from pretrain_netG import train_nml, train_netG
from options import BaseOptions

resume_epoch_netG = 195
resume_epoch_netMR = 110
saveres=True
frep_show = 2
frep_save=5

opt = BaseOptions().parse()
use_gpu = True
alter_num_epoch = 5
alter_num_epochMR = 10
alter_num_time = 10
opt.resume_epoch = -1

for i in range(alter_num_time):
    print('Alternate training, time {} / {}!'.format(i, alter_num_time))
    opt.continue_train = True
    opt.resume_epoch = resume_epoch_netG + 2*i*alter_num_epoch
    train_nml(opt, alter_num_epoch, use_gpu, frep_show = frep_show, frep_save=frep_save)
    
    opt.resume_epoch = resume_epoch_netG + 2*i*alter_num_epoch + alter_num_epoch #+ i*alter_num_epoch   #2*i*alter_num_epoch + alter_num_epoch
    train_netG(opt, alter_num_epoch, use_gpu, saveres = saveres, frep_show = frep_show, frep_save= frep_save)

    if resume_epoch_netMR < 0:
        opt.continue_train = False
    else:
        opt.resume_epoch = resume_epoch_netMR + i*alter_num_epochMR
    trainMR(opt, alter_num_epochMR, use_gpu, saveres = saveres, frep_show = frep_show, frep_save= frep_save, use_crop=True)
print('Train over!')
