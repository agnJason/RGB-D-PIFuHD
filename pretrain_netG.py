# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 18:46:09 2020

@author: Administrator
"""

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
from PIFuNetwNML import PIFuNetwNML
from net_util import CustomBCELoss
import networks
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



def train_nml(opt,num_epoch=10,use_gpu=True, frep_show = 1, frep_save=1):
    print('Train nml network...')
    if use_gpu:
        cuda = torch.device('cuda:%d' % opt.gpu_id)
    else:
        cuda = torch.device('cpu')
    opt.dataroot = './traindata'
    train_dataset_pre = TrainDataset(opt, phase='train', load_mesh=False)
    projection_mode = train_dataset_pre.projection_mode
    train_data_loader_pre = DataLoader(train_dataset_pre,batch_size=1)
    print('train data size: ', len(train_data_loader_pre))
    
    def set_train():
        netG.train()

    def set_eval():
        netG.eval()
        
    # load checkpoints
    #opt.load_netG_checkpoint_path = 'checkpoints/pifuhd/netG_latest'

    opt.hg_dim = opt.hg_dim_global
    opt.mlp_dim = opt.mlp_dim_global
    opt.mlp_res_layers = opt.mlp_res_layers_global
    opt.num_stack = opt.num_stack_global
    opt.hg_depth = opt.hg_depth_global

    netG = PIFuNetwNML(opt, projection_mode,criteria={'occ':CustomBCELoss()}).to(device='cpu') #粗糙层    
    if opt.load_netG_checkpoint_path is not None:
        netG_state_dict = torch.load(opt.load_netG_checkpoint_path)
        netG.load_state_dict(netG_state_dict)

    criterionVGG = networks.VGGLoss()
    criterionL1 = torch.nn.L1Loss()

    
    lr = 0.0002  #opt.learning_rate
    print('Using Network: ', netG.name)
    # opt.continue_train = False
    # opt.resume_epoch = 0
    
    if opt.continue_train:
        if opt.resume_epoch < 0:
            #model_path ='checkpoints/netG.pt'
            model_path = '%s/%s/netG_latest' % (opt.checkpoints_path, opt.name)
        else:
            #model_path ='checkpoints/netG.pt'
            model_path = '%s/%s/netG_epoch_%d' % (opt.checkpoints_path, opt.name, opt.resume_epoch)
        print('Resuming from ', model_path)
        state_dict = torch.load(model_path, map_location=cuda)['model_state_dict']
        netG.load_state_dict(state_dict)
        del state_dict

    #训练器
    optimizerFG = torch.optim.Adam(netG.netF.parameters(), lr=opt.learning_rate, betas=(0.5, 0.999))
    optimizerBG = torch.optim.Adam(netG.netB.parameters(), lr=opt.learning_rate, betas=(0.5, 0.999))


    os.makedirs(opt.checkpoints_path, exist_ok=True)
    os.makedirs(opt.results_path, exist_ok=True)
    os.makedirs('%s/%s' % (opt.checkpoints_path, opt.name), exist_ok=True)
    os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)
    os.makedirs('train_result', exist_ok=True)
    os.makedirs('train_result/normal', exist_ok=True)

    #开始训练
    
    start_epoch = 0 if not opt.continue_train else max(opt.resume_epoch+1,0)
    end_epoch = num_epoch+start_epoch
    
    netG.netF.to(device=cuda)
    netG.netB.to(device=cuda)
    #训练法线预测网络
    for epoch in range(start_epoch,end_epoch ):
        epoch_start_time = time.time()
        print('epoch {}/{}'.format(epoch, end_epoch))
        set_train()
        iter_data_time = time.time()
        for train_idx, train_data in enumerate(train_data_loader_pre):
            iter_start_time = time.time()
            
            # 准备数据
            image_tensor = train_data['img'][:,0].to(device=cuda)
            depth_tensor = train_data['depth'].to(device=cuda)
            image_tensor = torch.cat([image_tensor, depth_tensor],1)
            imF = train_data['imF'].to(device=cuda)
            imB= train_data['imB'].to(device=cuda)
            Fstyle= train_data['Fstyle'].to(device=cuda)
            Bstyle= train_data['Bstyle'].to(device=cuda)

            #Front
            fake_image = netG.netF.forward(image_tensor)
            #loss G
            loss_G_l1 = criterionL1(fake_image, imF) * 5.0 #L1损失
            torch.cuda.empty_cache()
            loss_G_VGG = criterionVGG(imF, fake_image,Fstyle)  #VGG损失

            loss_G = loss_G_l1 + loss_G_VGG  # + loss_G_GAN

            # 梯度下降 生成器
            optimizerFG.zero_grad()
            loss_G.backward()
            optimizerFG.step()

            
            iter_net_time = time.time()
            eta = 2*((iter_net_time - epoch_start_time) / (train_idx + 1)) * len(train_data_loader_pre) - (
                    iter_net_time - epoch_start_time)
            if epoch % frep_show == 0 :#and epoch != 0:
                 print(
                    'Name: {0} | Epoch: {1} | {2}/{3} | Err: {4:.06f} |  LR: {5:.06f} | Sigma: {6:.02f} | dataT: {7:.05f} | netT: {8:.05f} | ETA: {9:02d}:{10:02d}'.format(
                        'netF', epoch, train_idx, len(train_data_loader_pre), loss_G, lr, opt.sigma,
                        iter_start_time - iter_data_time, iter_net_time - iter_start_time, int(eta // 60),int(eta - 60 * (eta // 60))))

            torch.cuda.empty_cache()
                        
            fake_image = netG.netB.forward(image_tensor)

            #loss G
            loss_G_l1 = criterionL1(fake_image, imB) * 5.0 #L1损失
            loss_G_VGG = criterionVGG(imB, fake_image,Bstyle)  #VGG损失
 

            loss_G = loss_G_l1 + loss_G_VGG 


            # 梯度下降 生成器
            optimizerBG.zero_grad()
            loss_G.backward()
            optimizerBG.step()

            loss_G = loss_G.item()

            
            iter_net_time = time.time()
            eta = 2^(((iter_net_time - epoch_start_time) / (train_idx + 1)) * len(train_data_loader_pre) - (
                    iter_net_time - epoch_start_time))
            
            if epoch % frep_show == 0: #and epoch != 0:
                 print(
                    'Name: {0} | Epoch: {1} | {2}/{3} | Err: {4:.06f} |  LR: {5:.06f} | Sigma: {6:.02f} | dataT: {7:.05f} | netT: {8:.05f} | ETA: {9:02d}:{10:02d}'.format(
                        'netB', epoch, train_idx, len(train_data_loader_pre), loss_G, lr, opt.sigma,
                        iter_start_time - iter_data_time, iter_net_time - iter_start_time, int(eta // 60),int(eta - 60 * (eta // 60))))
                 iter_data_time = time.time()
        if epoch % frep_save == 0:# and epoch != 0:
            set_eval()
            nmlF = netG.netF.forward(image_tensor).detach()
            nmlB = netG.netB.forward(image_tensor).detach()
            image_eval = torch.cat([image_tensor[:,:3],imF,imB,nmlF,nmlB],0)
            save_img_path = 'train_result/normal/epoch{}_{}.png'.format(epoch,train_idx)
            save_img_list = []
            for v in range(image_eval.shape[0]):#原图、正面、背面的生成图片
                save_img = (np.transpose(image_eval [v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0 
                save_img_list.append(save_img)
            save_img = np.concatenate(save_img_list, axis=1) #拼接图片
            #print(save_img_path, save_img)
            cv2.imwrite(save_img_path, save_img)
            
            torch.cuda.empty_cache()
            torch.save({'opt':opt, 'model_state_dict':netG.state_dict()}, '%s/%s/netG_latest' % (opt.checkpoints_path, opt.name))
            torch.save({'opt':opt, 'model_state_dict':netG.state_dict()}, '%s/%s/netG_epoch_%d' % (opt.checkpoints_path, opt.name, epoch))
    #return 'success!'
    
def train_netG(opt,num_epoch=10,use_gpu=True, saveres = False, frep_show = 1, frep_save=1):
    print('Train netG...')
    if use_gpu:
        cuda = torch.device('cuda:%d' % opt.gpu_id)
    else:
        cuda = torch.device('cpu')
    opt.dataroot = './traindata'
    train_dataset = TrainDataset(opt, phase='train',use_crop=False)
    projection_mode = train_dataset.projection_mode
    train_data_loader = DataLoader(train_dataset,batch_size=1)

    print('train data size: ', len(train_data_loader))
    
    def set_train():
        netG.train()

    def set_eval():
        netG.eval()
        
    # load checkpoints
    #opt.load_netG_checkpoint_path = 'checkpoints/pifuhd/netG_latest'

    opt.hg_dim = opt.hg_dim_global
    opt.mlp_dim = opt.mlp_dim_global
    opt.mlp_res_layers = opt.mlp_res_layers_global
    opt.num_stack = opt.num_stack_global
    opt.hg_depth = opt.hg_depth_global

    netG = PIFuNetwNML(opt, projection_mode,criteria={'occ':CustomBCELoss()}).to(device=cuda) #粗糙层    
    if opt.load_netG_checkpoint_path is not None:
        netG_state_dict = torch.load(opt.load_netG_checkpoint_path)
        netG.load_state_dict(netG_state_dict)
    
    lr = 0.001  #opt.learning_rate
    print('Using Network: ', netG.name)
    # opt.continue_train = False
    # opt.resume_epoch = 0
    
    if opt.continue_train:
        if opt.resume_epoch < 0:
            #model_path ='checkpoints/netG.pt'
            model_path = '%s/%s/netG_latest' % (opt.checkpoints_path, opt.name)
        else:
            #model_path ='checkpoints/netG.pt'
            model_path = '%s/%s/netG_epoch_%d' % (opt.checkpoints_path, opt.name, opt.resume_epoch)
        print('Resuming from ', model_path)
        state_dict = torch.load(model_path, map_location=cuda)['model_state_dict']
        netG.load_state_dict(state_dict)
        del state_dict

    #训练器
    optimizerG = torch.optim.RMSprop(netG.parameters(), lr=opt.learning_rate, momentum=0, weight_decay=0)   

    os.makedirs(opt.checkpoints_path, exist_ok=True)
    os.makedirs(opt.results_path, exist_ok=True)
    os.makedirs('%s/%s' % (opt.checkpoints_path, opt.name), exist_ok=True)
    os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)
    os.makedirs('train_result', exist_ok=True)
    os.makedirs('train_result/netG', exist_ok=True)

    
    error_list = []
    #开始训练
    
    start_epoch = 0 if not opt.continue_train else max(opt.resume_epoch+1,0)
    end_epoch = num_epoch+start_epoch
    
    for epoch in range(start_epoch,end_epoch):
        epoch_start_time = time.time()
        error_list.append([])
        set_train()
        iter_data_time = time.time()
        for train_idx, train_data in enumerate(train_data_loader):
            iter_start_time = time.time()
            
            # 准备数据
            #image_tensor = train_data['img'].to(device=cuda)
            image_tensor_global = train_data['img_512'].to(device=cuda)
            depth_tensor = train_data['depth_512'].to(device=cuda)
            image_tensor_global = torch.cat([image_tensor_global, depth_tensor],1)
            #calib_tensor = train_data['calib'].to(device=cuda)
            calib_tensor_global = train_data['calib'].to(device=cuda)
            sample_tensor = train_data['samples'].to(device=cuda)
            label_tensor = train_data['labels'].to(device=cuda)
            err_netG, res = netG.forward(image_tensor_global, sample_tensor[0],calib_tensor_global, labels=label_tensor,gamma=opt.gamma)
            #训练粗糙层
            optimizerG.zero_grad()
            err_netG['Err(occ)'].backward() #反向传播
            optimizerG.step()

            iter_net_time = time.time()
            eta = ((iter_net_time - epoch_start_time) / (train_idx + 1)) * len(train_data_loader) - (
                    iter_net_time - epoch_start_time)
            
            error_list[-1].append(err_netG['Err(occ)'].item())
            if epoch % frep_show == 0:
                print(
                    'Name: {0} | Epoch: {1} | {2}/{3} | Err: {4:.06f} |  LR: {5:.06f} | Sigma: {6:.02f} | dataT: {7:.05f} | netT: {8:.05f} | ETA: {9:02d}:{10:02d}'.format(
                        'netG', epoch, train_idx, len(train_data_loader), err_netG['Err(occ)'].item(), lr, opt.sigma,
                                                                            iter_start_time - iter_data_time,
                                                                            iter_net_time - iter_start_time, int(eta // 60),int(eta - 60 * (eta // 60))))
                iter_data_time = time.time()
            #save    
        if epoch % frep_save == 0:   #保存
            with torch.no_grad():
                #set_eval()
                torch.save({'opt':opt, 'model_state_dict':netG.state_dict()}, '%s/%s/netG_latest' % (opt.checkpoints_path, opt.name))
                torch.save({'opt':opt, 'model_state_dict':netG.state_dict()}, '%s/%s/netG_epoch_%d' % (opt.checkpoints_path, opt.name, epoch))
            if saveres:
                np.save('train_result/netG/error_epoch_{}_{}'.format(start_epoch,epoch),np.array(error_list))
                
                
def train(opt,num_epoch=10,use_gpu=True):
    if use_gpu:
        cuda = torch.device('cuda:%d' % opt.gpu_id)
    else:
        cuda = torch.device('cpu')
    opt.dataroot = './traindata'
    train_dataset = TrainDataset(opt, phase='train',use_crop=False)
    train_dataset_pre = TrainDataset(opt, phase='train', load_mesh=False)
    projection_mode = train_dataset.projection_mode
    train_data_loader = DataLoader(train_dataset,batch_size=1)
    train_data_loader_pre = DataLoader(train_dataset_pre,batch_size=1)
    print('train data size: ', len(train_data_loader_pre))
    
    def set_train():
        netG.train()

    def set_eval():
        netG.eval()
        
    # load checkpoints
    #opt.load_netG_checkpoint_path = 'checkpoints/pifuhd/netG_latest'

    opt.hg_dim = opt.hg_dim_global
    opt.mlp_dim = opt.mlp_dim_global
    opt.mlp_res_layers = opt.mlp_res_layers_global
    opt.num_stack = opt.num_stack_global
    opt.hg_depth = opt.hg_depth_global

    netG = PIFuNetwNML(opt, projection_mode,criteria={'occ':CustomBCELoss()}).to(device='cpu') #粗糙层    
    if opt.load_netG_checkpoint_path is not None:
        netG_state_dict = torch.load(opt.load_netG_checkpoint_path)
        netG.load_state_dict(netG_state_dict)

    criterionVGG = networks.VGGLoss()
    criterionL1 = torch.nn.L1Loss()

    
    lr = 0.0002  #opt.learning_rate
    print('Using Network: ', netG.name)
    # opt.continue_train = False
    # opt.resume_epoch = 0
    
    if opt.continue_train:
        if opt.resume_epoch < 0:
            #model_path ='checkpoints/netG.pt'
            model_path = '%s/%s/netG_latest' % (opt.checkpoints_path, opt.name)
        else:
            #model_path ='checkpoints/netG.pt'
            model_path = '%s/%s/netG_epoch_%d' % (opt.checkpoints_path, opt.name, opt.resume_epoch)
        print('Resuming from ', model_path)
        state_dict = torch.load(model_path, map_location=cuda)['model_state_dict']
        netG.load_state_dict(state_dict)
        del state_dict

    #训练器
    optimizerFG = torch.optim.Adam(netG.netF.parameters(), lr=opt.learning_rate, betas=(0.5, 0.999))
    optimizerBG = torch.optim.Adam(netG.netB.parameters(), lr=opt.learning_rate, betas=(0.5, 0.999))
    optimizerG = torch.optim.RMSprop(netG.parameters(), lr=opt.learning_rate, momentum=0, weight_decay=0)   

    os.makedirs(opt.checkpoints_path, exist_ok=True)
    os.makedirs(opt.results_path, exist_ok=True)
    os.makedirs('%s/%s' % (opt.checkpoints_path, opt.name), exist_ok=True)
    os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)
    
    
    #开始训练
    
    start_epoch = 0 if not opt.continue_train else max(opt.resume_epoch+1,0)
    end_epoch = num_epoch+start_epoch

    netG.netF.to(device=cuda)
    netG.netB.to(device=cuda)
    #训练法线预测网络
    for epoch in range(start_epoch,end_epoch ):
        epoch_start_time = time.time()
        print('epoch {}/{}'.format(epoch, end_epoch))
        set_train()
        iter_data_time = time.time()
        for train_idx, train_data in enumerate(train_data_loader_pre):
            iter_start_time = time.time()
            
            # 准备数据
            image_tensor = train_data['img'][:,0].to(device=cuda)
            depth_tensor = train_data['depth'].to(device=cuda)
            image_tensor = torch.cat([image_tensor, depth_tensor],1)
            imF = train_data['imF'].to(device=cuda)
            imB= train_data['imB'].to(device=cuda)
            Fstyle= train_data['Fstyle'].to(device=cuda)
            Bstyle= train_data['Bstyle'].to(device=cuda)

            #Front
            fake_image = netG.netF.forward(image_tensor)
            #loss G
            loss_G_l1 = criterionL1(fake_image, imF) * 5.0 #L1损失
            torch.cuda.empty_cache()
            loss_G_VGG = criterionVGG(imF, fake_image,Fstyle)  #VGG损失

            loss_G = loss_G_l1 + loss_G_VGG  # + loss_G_GAN

            # 梯度下降 生成器
            optimizerFG.zero_grad()
            loss_G.backward()
            optimizerFG.step()

            
            iter_net_time = time.time()
            eta = 2*((iter_net_time - epoch_start_time) / (train_idx + 1)) * len(train_data_loader) - (
                    iter_net_time - epoch_start_time)
            if epoch % 1 == 0 :#and epoch != 0:
                 print(
                    'Name: {0} | Epoch: {1} | {2}/{3} | Err: {4:.06f} |  LR: {5:.06f} | Sigma: {6:.02f} | dataT: {7:.05f} | netT: {8:.05f} | ETA: {9:02d}:{10:02d}'.format(
                        'netF', epoch, train_idx, len(train_data_loader_pre), loss_G, lr, opt.sigma,
                        iter_start_time - iter_data_time, iter_net_time - iter_start_time, int(eta // 60),int(eta - 60 * (eta // 60))))

            torch.cuda.empty_cache()
                        
            fake_image = netG.netB.forward(image_tensor)

            #loss G
            loss_G_l1 = criterionL1(fake_image, imB) * 5.0 #L1损失
            loss_G_VGG = criterionVGG(imB, fake_image,Bstyle)  #VGG损失
 

            loss_G = loss_G_l1 + loss_G_VGG 


            # 梯度下降 生成器
            optimizerBG.zero_grad()
            loss_G.backward()
            optimizerBG.step()

            loss_G = loss_G.item()

            
            iter_net_time = time.time()
            eta = ((iter_net_time - epoch_start_time) / (train_idx + 1)) * len(train_data_loader) - (
                    iter_net_time - epoch_start_time)
            
            if epoch % 1 == 0: #and epoch != 0:
                 print(
                    'Name: {0} | Epoch: {1} | {2}/{3} | Err: {4:.06f} |  LR: {5:.06f} | Sigma: {6:.02f} | dataT: {7:.05f} | netT: {8:.05f} | ETA: {9:02d}:{10:02d}'.format(
                        'netB', epoch, train_idx, len(train_data_loader_pre), loss_G, lr, opt.sigma,
                        iter_start_time - iter_data_time, iter_net_time - iter_start_time, int(eta // 60),int(eta - 60 * (eta // 60))))
                 iter_data_time = time.time()
        if epoch % 2 == 0:# and epoch != 0:
            set_eval()
            nmlF = netG.netF.forward(image_tensor).detach()
            nmlB = netG.netB.forward(image_tensor).detach()
            image_eval = torch.cat([image_tensor[:,:3],imF,imB,nmlF,nmlB],0)
            save_img_path = 'traindata/train/epoch{}_{}.png'.format(epoch,train_idx)
            save_img_list = []
            for v in range(image_eval.shape[0]):#原图、正面、背面的生成图片
                save_img = (np.transpose(image_eval [v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0 
                save_img_list.append(save_img)
            save_img = np.concatenate(save_img_list, axis=1) #拼接图片
            #print(save_img_path, save_img)
            cv2.imwrite(save_img_path, save_img)
            
            torch.cuda.empty_cache()
            torch.save({'opt':opt, 'model_state_dict':netG.state_dict()}, '%s/%s/netG_latest' % (opt.checkpoints_path, opt.name))
            torch.save({'opt':opt, 'model_state_dict':netG.state_dict()}, '%s/%s/netG_epoch_%d' % (opt.checkpoints_path, opt.name, epoch))
    #return 'success!'
    del loss_G, loss_G_VGG, criterionVGG

    #预训练粗糙层        
    netG.to(device=cuda)
    lr = 0.001  #opt.learning_rate
    for epoch in range(start_epoch,end_epoch):
        epoch_start_time = time.time()

        set_train()
        iter_data_time = time.time()
        for train_idx, train_data in enumerate(train_data_loader):
            iter_start_time = time.time()
            
            # 准备数据
            #image_tensor = train_data['img'].to(device=cuda)
            image_tensor_global = train_data['img_512'].to(device=cuda)
            depth_tensor = train_data['depth_512'].to(device=cuda)
            image_tensor_global = torch.cat([image_tensor_global, depth_tensor],1)
            #calib_tensor = train_data['calib'].to(device=cuda)
            calib_tensor_global = train_data['calib'].to(device=cuda)
            sample_tensor = train_data['samples'].to(device=cuda)
            label_tensor = train_data['labels'].to(device=cuda)
            err_netG, res = netG.forward(image_tensor_global, sample_tensor[0],calib_tensor_global, labels=label_tensor,gamma=opt.gamma)
            #训练粗糙层
            optimizerG.zero_grad()
            err_netG['Err(occ)'].backward() #反向传播
            optimizerG.step()

            iter_net_time = time.time()
            eta = ((iter_net_time - epoch_start_time) / (train_idx + 1)) * len(train_data_loader) - (
                    iter_net_time - epoch_start_time)
        
            if epoch % 1 == 0:
                print(
                    'Name: {0} | Epoch: {1} | {2}/{3} | Err: {4:.06f} |  LR: {5:.06f} | Sigma: {6:.02f} | dataT: {7:.05f} | netT: {8:.05f} | ETA: {9:02d}:{10:02d}'.format(
                        'netG', epoch, train_idx, len(train_data_loader), err_netG['Err(occ)'].item(), lr, opt.sigma,
                                                                            iter_start_time - iter_data_time,
                                                                            iter_net_time - iter_start_time, int(eta // 60),int(eta - 60 * (eta // 60))))
                iter_data_time = time.time()
            #save    
        if epoch % 2 == 0 and epoch != 0:   #保存
            with torch.no_grad():
                #set_eval()
                torch.save({'opt':opt, 'model_state_dict':netG.state_dict()}, '%s/%s/netG_latest' % (opt.checkpoints_path, opt.name))
                torch.save({'opt':opt, 'model_state_dict':netG.state_dict()}, '%s/%s/netG_epoch_%d' % (opt.checkpoints_path, opt.name, epoch))
        # 更新 learning rate
        lr = adjust_learning_rate(optimizerG, epoch, lr, opt.schedule, opt.gamma)

def train_depth(opt,num_epoch=10,use_gpu=True, frep_show = 1, frep_save=1):
    print('Train nml network...')
    if use_gpu:
        cuda = torch.device('cuda:%d' % opt.gpu_id)
    else:
        cuda = torch.device('cpu')
    opt.dataroot = './traindata'
    train_dataset_pre = TrainDataset(opt, phase='train', load_mesh=False)
    projection_mode = train_dataset_pre.projection_mode
    train_data_loader_pre = DataLoader(train_dataset_pre,batch_size=1)
    print('train data size: ', len(train_data_loader_pre))
    
    def set_train():
        netG.train()

    def set_eval():
        netG.eval()
        
    # load checkpoints
    #opt.load_netG_checkpoint_path = 'checkpoints/pifuhd/netG_latest'

    opt.hg_dim = opt.hg_dim_global
    opt.mlp_dim = opt.mlp_dim_global
    opt.mlp_res_layers = opt.mlp_res_layers_global
    opt.num_stack = opt.num_stack_global
    opt.hg_depth = opt.hg_depth_global

    netG = PIFuNetwNML(opt, projection_mode,criteria={'occ':CustomBCELoss()}).to(device='cpu') #粗糙层    
    if opt.load_netG_checkpoint_path is not None:
        netG_state_dict = torch.load(opt.load_netG_checkpoint_path)
        netG.load_state_dict(netG_state_dict)

    criterionVGG = networks.VGGLoss()
    criterionL1 = torch.nn.L1Loss()

    
    lr = 0.0002  #opt.learning_rate
    print('Using Network: ', netG.name)
    # opt.continue_train = False
    # opt.resume_epoch = 0
    
    if opt.continue_train:
        if opt.resume_epoch < 0:
            #model_path ='checkpoints/netG.pt'
            model_path = '%s/%s/netG_latest' % (opt.checkpoints_path, opt.name)
        else:
            #model_path ='checkpoints/netG.pt'
            model_path = '%s/%s/netG_epoch_%d' % (opt.checkpoints_path, opt.name, opt.resume_epoch)
        print('Resuming from ', model_path)
        state_dict = torch.load(model_path, map_location=cuda)['model_state_dict']
        netG.load_state_dict(state_dict)
        del state_dict

    #训练器
    optimizerDG = torch.optim.Adam(netG.netD.parameters(), lr=opt.learning_rate, betas=(0.5, 0.999))

    os.makedirs(opt.checkpoints_path, exist_ok=True)
    os.makedirs(opt.results_path, exist_ok=True)
    os.makedirs('%s/%s' % (opt.checkpoints_path, opt.name), exist_ok=True)
    os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)
    os.makedirs('train_result', exist_ok=True)
    os.makedirs('train_result/normal', exist_ok=True)

    #开始训练
    
    start_epoch = 0 if not opt.continue_train else max(opt.resume_epoch+1,0)
    end_epoch = num_epoch+start_epoch
    
    netG.netD.to(device=cuda)

    #训练法线预测网络
    for epoch in range(start_epoch,end_epoch ):
        epoch_start_time = time.time()
        print('epoch {}/{}'.format(epoch, end_epoch))
        set_train()
        iter_data_time = time.time()
        for train_idx, train_data in enumerate(train_data_loader_pre):
            iter_start_time = time.time()
            
            # 准备数据
            image_tensor = train_data['img'][:,0].to(device=cuda)
            depth_tensor = train_data['depth'].to(device=cuda)
            Dstyle= train_data['Dstyle'].to(device=cuda)

            #Front
            fake_image = netG.netD.forward(image_tensor)
            #loss G
            loss_G_l1 = criterionL1(fake_image, depth_tensor) * 5.0 #L1损失
            torch.cuda.empty_cache()
            loss_G_VGG = criterionVGG(depth_tensor, fake_image,Dstyle)  #VGG损失

            loss_G = loss_G_l1 + loss_G_VGG  # + loss_G_GAN

            # 梯度下降 生成器
            optimizerDG.zero_grad()
            loss_G.backward()
            optimizerDG.step()

            
            iter_net_time = time.time()
            eta = ((iter_net_time - epoch_start_time) / (train_idx + 1)) * len(train_data_loader_pre) - (
                    iter_net_time - epoch_start_time)
            if epoch % frep_show == 0 :#and epoch != 0:
                 print(
                    'Name: {0} | Epoch: {1} | {2}/{3} | Err: {4:.06f} |  LR: {5:.06f} | Sigma: {6:.02f} | dataT: {7:.05f} | netT: {8:.05f} | ETA: {9:02d}:{10:02d}'.format(
                        'netF', epoch, train_idx, len(train_data_loader_pre), loss_G, lr, opt.sigma,
                        iter_start_time - iter_data_time, iter_net_time - iter_start_time, int(eta // 60),int(eta - 60 * (eta // 60))))

            torch.cuda.empty_cache()
                        
            
        if epoch % frep_save == 0:# and epoch != 0:
            set_eval()
            depth = netG.netD.forward(image_tensor).detach()
            image_eval = torch.cat([image_tensor,depth_tensor,depth],0)
            save_img_path = 'train_result/normal/epoch{}_{}.png'.format(epoch,train_idx)
            save_img_list = []
            for v in range(image_eval.shape[0]):#原图、正面、背面的生成图片
                save_img = (np.transpose(image_eval [v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0 
                save_img_list.append(save_img)
            save_img = np.concatenate(save_img_list, axis=1) #拼接图片
            #print(save_img_path, save_img)
            cv2.imwrite(save_img_path, save_img)
            
            torch.cuda.empty_cache()
            torch.save({'opt':opt, 'model_state_dict':netG.state_dict()}, '%s/%s/netG_latest' % (opt.checkpoints_path, opt.name))
            torch.save({'opt':opt, 'model_state_dict':netG.state_dict()}, '%s/%s/netG_epoch_%d' % (opt.checkpoints_path, opt.name, epoch))

if __name__ == '__main__':
    opt = BaseOptions().parse()
    opt.continue_train = True
    opt.resume_epoch = 76
    num_epoch_nml = 1
    num_epoch_netG = 1
    
    train_nml(opt, num_epoch=num_epoch_nml, frep_show = 1, frep_save=1)
    opt.resume_epoch += num_epoch_nml
    train_netG(opt, num_epoch=num_epoch_netG, saveres=True, frep_show = 1, frep_save=1)
