# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 16:39:55 2020

@author: Administrator
"""
import os
import time
import torch
import torch.nn as nn
from options import BaseOptions
from torch.utils.data import DataLoader
from TrainDataset import TrainDataset
from EvalDataset import EvalDataset
from PIFuNetwNML import PIFuNetwNML
from PIFuMRNet import PIFuMRNet
from net_util import CustomBCELoss
from reconstruction import gen_mesh
import copy
import numpy as np
import cv2

def reshape_multiview_tensors(image_tensor, calib_tensor):
    #   [B, num_views, C, W, H]
    #   [B*num_views, C, W, H]
    image_tensor = image_tensor.view(
        image_tensor.shape[0] * image_tensor.shape[1],
        image_tensor.shape[2],
        image_tensor.shape[3],
        image_tensor.shape[4]
    )
    calib_tensor = calib_tensor.view(
        calib_tensor.shape[0] * calib_tensor.shape[1],
        calib_tensor.shape[2],
        calib_tensor.shape[3]
    )

    return image_tensor, calib_tensor

def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """调整学习率"""
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


def reshape_sample_tensor(sample_tensor, num_views):
    if num_views == 1:
        return sample_tensor
    
    sample_tensor = sample_tensor.unsqueeze(dim=1)
    sample_tensor = sample_tensor.repeat(1, num_views, 1, 1)
    sample_tensor = sample_tensor.view(
        sample_tensor.shape[0] * sample_tensor.shape[1],
        sample_tensor.shape[2],
        sample_tensor.shape[3]
    )
    return sample_tensor

def train(opt,num_epoch=1,use_gpu=False, saveres = False, frep_show = 1, frep_save=1, use_crop=False):
    if use_gpu:
        cuda = torch.device('cuda:%d' % opt.gpu_id)
        
    else:
        cuda = torch.device('cpu')
    
    opt.dataroot = './traindata'
    train_dataset = TrainDataset(opt, phase='train',use_crop=use_crop)
    projection_mode = train_dataset.projection_mode
    train_data_loader = DataLoader(train_dataset)
    print('train data size: ', len(train_data_loader))
    '''
    e_opt = copy.deepcopy(opt)
    e_opt.dataroot = './evaldata'
    eval_dataset = EvalDataset(e_opt, phase='eval')
    eval_data_loader = DataLoader(eval_dataset)
    '''
    #设定预训练模型位置
    opt.load_netG_checkpoint_path= './checkpoints/pifuhd/netG_latest'

    def set_train():
        netMR.train()

    def set_eval():
        netMR.eval()
    opt.load_netMR_checkpoint_path= None    
    # load checkpoints
    if opt.load_netMR_checkpoint_path is not None:
        print('loading for net MR ...', opt.load_netMR_checkpoint_path)
        if use_gpu:
            state_dict = torch.load(opt.load_netMR_checkpoint_path)
        else:
            state_dict = torch.load(opt.load_netMR_checkpoint_path, map_location='cpu')
        opt = state_dict['opt'] #重载opt
        opt_netG = state_dict['opt_netG']
        netG = PIFuNetwNML(opt_netG, projection_mode,criteria={'occ':CustomBCELoss()}).to(device=cuda) #粗糙层 法线
        netMR = PIFuMRNet(opt, netG, projection_mode,criteria={'occ':CustomBCELoss()}).to(device=cuda) #精细层
        netMR.load_state_dict(state_dict['model_state_dict'])
    else:
        opt.hg_dim = opt.hg_dim_global
        opt.mlp_dim = opt.mlp_dim_global
        opt.mlp_res_layers = opt.mlp_res_layers_global
        opt.num_stack = opt.num_stack_global
        opt.hg_depth = opt.hg_depth_global

        netG = PIFuNetwNML(opt, projection_mode,criteria={'occ':CustomBCELoss()}).to(device=cuda) #粗糙层    
        if opt.load_netG_checkpoint_path is not None:
            print('loading for net G ...', opt.load_netG_checkpoint_path)
            netG_state_dict = torch.load(opt.load_netG_checkpoint_path, map_location='cpu')
            netG.load_state_dict(netG_state_dict['model_state_dict'])
        netG_opt = copy.deepcopy(opt)
        
        opt.num_stack=opt.num_stack_local
        opt.hg_dim = opt.hg_dim_local
        opt.mlp_dim = opt.mlp_dim_local
        opt.mlp_res_layers = opt.mlp_res_layers_local
        opt.hg_depth = opt.hg_depth_local
        netMR = PIFuMRNet(opt, netG, projection_mode,criteria={'occ':CustomBCELoss()}).to(device=cuda) #精细层

    lr = opt.learning_rate
    print('Using Network: ', netMR.name)
    # opt.continue_train = False
    # opt.resume_epoch = 0
    
    if opt.continue_train:
        if opt.resume_epoch < 0:
            #model_path = '%s/pifuhd.pt' % (opt.checkpoints_path)
            model_path = '%s/%s/netMR_latest' % (opt.checkpoints_path, opt.name)
        else:
            model_path = '%s/%s/netMR_epoch_%03d' % (opt.checkpoints_path, opt.name, opt.resume_epoch)
        print('Resuming from ', model_path)
        state_dict = torch.load(model_path, map_location='cpu')
        netMR.load_state_dict(state_dict['model_state_dict'])
        netMR.netG.load_state_dict(netG_state_dict['model_state_dict'])
        del state_dict,netG_state_dict
    else:
        del netG_state_dict
    #训练器
    optimizerMR = torch.optim.RMSprop(netMR.parameters(), lr=opt.learning_rate, momentum=0, weight_decay=0)

    os.makedirs(opt.checkpoints_path, exist_ok=True)
    os.makedirs(opt.results_path, exist_ok=True)
    os.makedirs('%s/%s' % (opt.checkpoints_path, opt.name), exist_ok=True)
    os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)
    os.makedirs('train_result', exist_ok=True)
    os.makedirs('train_result/netMR', exist_ok=True)
    
    opt.train_full_pifu=False
    opt.no_intermediate_loss = False
    
    error_list = []
    eval_error_list = []
    #开始训练
    start_epoch = 0 if not opt.continue_train else max(opt.resume_epoch+1,0)
    end_epoch = num_epoch+start_epoch
    #训练整个模型 train_full_pifu=True
    for epoch in range(start_epoch,end_epoch):
        epoch_start_time = time.time()
        error_list.append([])
        set_train()
        print(epoch,'/',end_epoch)
        iter_data_time = time.time()
        for train_idx, train_data in enumerate(train_data_loader):
            iter_start_time = time.time()

            # 准备数据 精细层用随机剪裁数据
            image_tensor = train_data['img'].to(device=cuda)
            image_tensor_global = train_data['img_512'].to(device=cuda)
            depth_tensor = train_data['depth'].to(device=cuda)
            depth_tensor_global = train_data['depth_512'].to(device=cuda)
            image_tensor =torch.cat([image_tensor, depth_tensor],2)
            image_tensor_global = torch.cat([image_tensor_global, depth_tensor_global],1)
            # print(image_tensor.shape)
            calib_tensor = train_data['calib_world'].to(device=cuda)
            calib_tensor_global = train_data['calib'].to(device=cuda)
            sample_tensor = train_data['samples'].to(device=cuda)
            label_tensor = train_data['labels'].to(device=cuda)
            # points_nml = train_data['points_nml'].to(device=cuda)
            # labels_nml = train_data['labels_nml'].to(device=cuda)
            # image_tensor = image_tensor[:,None]
            calib_tensor = calib_tensor[:,None]

            torch.cuda.empty_cache()
            err_netMR, res = netMR.forward(image_tensor, image_tensor_global, sample_tensor, calib_tensor, calib_tensor_global, labels=label_tensor)
            
            optimizerMR.zero_grad()
            #print(err_netMR['Err(occ:fine)'], netMR.w, netMR.gamma)
            err_netMR['Err(occ:fine)'].backward()        
            optimizerMR.step()
            
            iter_net_time = time.time()
            eta = ((iter_net_time - epoch_start_time) / (train_idx + 1)) * len(train_data_loader) - (
                    iter_net_time - epoch_start_time)
            error_list[-1].append(err_netMR['Err(occ:fine)'].item())
            if epoch % frep_show == 0:
                print(
                    'Name: {0} | Epoch: {1} | {2}/{3} | Err: {4:.06f} |  LR: {5:.06f} | Sigma: {6:.02f} | dataT: {7:.05f} | netT: {8:.05f} | ETA: {9:02d}:{10:02d}'.format(
                        'netMR', epoch, train_idx, len(train_data_loader), err_netMR['Err(occ:fine)'].item(), lr, opt.sigma,
                                                                            iter_start_time - iter_data_time,
                                                                            iter_net_time - iter_start_time, int(eta // 60),int(eta - 60 * (eta // 60))))
                iter_data_time = time.time()
            del err_netMR, res
            
        print(epoch, 'epoch error:', sum(error_list[-1]))
        #save    
        if epoch % frep_save == 0 : # and epoch != 0:#epoch % 4 == 0 and   #保存
            with torch.no_grad():
                set_eval()
                torch.save({'opt':opt, 'opt_netG':netG_opt, 'model_state_dict':netMR.state_dict()}, '%s/%s/netMR_latest' % (opt.checkpoints_path, opt.name))
                torch.save({'opt':opt, 'opt_netG':netG_opt, 'model_state_dict':netMR.state_dict()}, '%s/%s/netMR_epoch_%d' % (opt.checkpoints_path, opt.name, epoch))
                if saveres:
                    np.save('train_result/netMR/error_epoch_{:03d}_{:03d}'.format(start_epoch,epoch),np.array(error_list))
                    '''
                    eval_error_list.append([])
                    for eval_idx, eval_data in enumerate(eval_data_loader):
                        # 准备数据 精细层用随机剪裁数据
                        image_tensor = eval_data['img'].to(device=cuda)
                        image_tensor_global = eval_data['img_512'].to(device=cuda)
                        depth_tensor = eval_data['depth'].to(device=cuda)
                        depth_tensor_global = eval_data['depth_512'].to(device=cuda)
                        image_tensor =torch.cat([image_tensor, depth_tensor],2)
                        image_tensor_global = torch.cat([image_tensor_global, depth_tensor_global],1)
    
                        calib_tensor = eval_data['calib'].to(device=cuda)
                        calib_tensor_global = eval_data['calib'].to(device=cuda)
                        sample_tensor = eval_data['samples'].to(device=cuda)
                        label_tensor = eval_data['labels'].to(device=cuda)
    
                        calib_tensor = calib_tensor[:,None]
            
                        torch.cuda.empty_cache()
                        err_netMR, res = netMR.forward(image_tensor, image_tensor_global, sample_tensor, calib_tensor, calib_tensor_global, labels=label_tensor)
                        eval_error_list[-1].append(err_netMR['Err(occ:fine)'].item())
                    print('eval error:', sum(eval_error_list[-1]))
                    np.save('train_result/netMR/eval_error_epoch_{:03d}_{:03d}'.format(start_epoch,epoch),np.array(eval_error_list))
                    '''
                #train_data['img'] = train_data['img'][0]
                #del err_netMR
                #torch.cuda.empty_cache()
                #gen_mesh(opt.resolution, netMR, cuda, train_data,
                         #os.path.join(opt.results_path, opt.name,'train_{}_{}.obj'.format(train_data['name'],epoch)), thresh=0.5, use_octree=True)


#opt = BaseOptions().parse()
#train(opt)
