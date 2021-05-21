# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 21:39:37 2020

@author: Administrator
"""

from options import BaseOptions
from mesh_util import save_obj_mesh_with_color, reconstruction
import numpy as np
import torch
import cv2
import os
import trimesh
from numpy.linalg import inv
from BasePIFuNet import index
from readData import readDataset
from PIFuNetwNML import PIFuNetwNML
from PIFuMRNet import PIFuMRNet
from tqdm import tqdm


parser = BaseOptions()  #加载参数列表

def gen_mesh(res, net, cuda, data, save_path, thresh=0.5, use_octree=True,components=False):
    '''
    生成结果图片和3D模型
    '''
    image_tensor_global = data['img_512'].to(device=cuda)
    image_tensor = data['img'].to(device=cuda)
    calib_tensor = data['calib'].to(device=cuda)

    net.filter_global(image_tensor_global) #粗糙
    net.filter_local(image_tensor[:,None]) #精细
    
    try:
        if net.netG.netF is not None:
            image_tensor_global = torch.cat([image_tensor_global, net.netG.nmlF],0)
        if net.netG.netB is not None:
            image_tensor_global = torch.cat([image_tensor_global, net.netG.nmlB],0)
    except:
        pass
    
    b_min = data['b_min']
    b_max = data['b_max']
    
    #try:
    save_img_path = save_path[:-4] + '.png'
    save_img_list = []
    for v in range(image_tensor_global.shape[0]):#原图、正面、背面的生成图片
        save_img = (np.transpose(image_tensor_global[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0 
        save_img_list.append(save_img)
    save_img = np.concatenate(save_img_list, axis=1) #拼接图片
    cv2.imwrite(save_img_path, save_img)

    verts, faces, _, _ = reconstruction(
        net, cuda, calib_tensor, res, b_min, b_max, thresh, use_octree=use_octree, num_samples=5000)
    verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float()
    
    color = np.zeros(verts.shape) #利用法向量代替颜色 同一个方向同一个颜色
    interval = 50000 #间隔
    for i in range(len(color) // interval + 1):
        left = i * interval
        if i == len(color) // interval:
            right = -1
        else:
            right = (i+1) * interval
        net.calc_normal(verts_tensor[:, None, :, left:right], calib_tensor[:,None], calib_tensor) #计算空间法向量
        nml = net.nmls.detach().cpu().numpy()[0] * 0.5 + 0.5
        color[left:right] = nml.T
    
    save_obj_mesh_with_color(save_path, verts, faces, color)
    #except Exception as e:
        #print('error')
        #print(e)

def gen_mesh_imgColor(res, net, cuda, data, save_path, thresh=0.5, use_octree=True, components=False):
    '''
    生成图片和3D模型，并用图片颜色上色
    '''
    image_tensor_global = data['img_512'].to(device=cuda)
    image_tensor = data['img'].to(device=cuda)
    calib_tensor = data['calib'].to(device=cuda)

    net.filter_global(image_tensor_global) #512的用于整体估计
    net.filter_local(image_tensor[:,None]) #1024用于更进一步估计

    try:
        if net.netG.netF is not None:
            image_tensor_global = torch.cat([image_tensor_global, net.netG.nmlF], 0)
        if net.netG.netB is not None:
            image_tensor_global = torch.cat([image_tensor_global, net.netG.nmlB], 0)
    except:
        pass

    b_min = data['b_min']
    b_max = data['b_max']
    try:
        save_img_path = save_path[:-4] + '.png'
        save_img_list = []
        for v in range(image_tensor_global.shape[0]):
            save_img = (np.transpose(image_tensor_global[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
            save_img_list.append(save_img)
        save_img = np.concatenate(save_img_list, axis=1)
        cv2.imwrite(save_img_path, save_img)

        verts, faces, _, _ = reconstruction(
            net, cuda, calib_tensor, res, b_min, b_max, thresh, use_octree=use_octree, num_samples=100000)

        verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float()

        # if this returns error, projection must be defined somewhere else
        xyz_tensor = net.projection(verts_tensor, calib_tensor[:1])
        uv = xyz_tensor[:, :2, :]
        color = index(image_tensor[:1], uv).detach().cpu().numpy()[0].T
        color = color * 0.5 + 0.5 #(-1,1) -> (0,1)
        #print('重新渲染颜色')
        #color = esti_color(color, xyz_tensor)
        if 'calib_world' in data:
            calib_world = data['calib_world'].numpy()[0]
            verts = np.matmul(np.concatenate([verts, np.ones_like(verts[:,:1])],1), inv(calib_world).T)[:,:3]
        
        save_obj_mesh_with_color(save_path, verts, faces, color)
        
    except Exception as e:
        print('error')
        print(e)

def gen_mesh_imgColor_plus(res, net, cuda, data, save_path, thresh=0.5, use_octree=True, components=False):
    '''
    生成图片和3D模型，并用图片颜色上色
    '''
    image_tensor_global = data['img_512'].to(device=cuda)
    image_tensor = data['img'].to(device=cuda)
    calib_tensor = data['calib'].to(device=cuda)

    net.filter_global(image_tensor_global) #512的用于整体估计
    net.filter_local(image_tensor[:,None]) #1024用于更进一步估计

    try:
        if net.netG.netF is not None:
            image_tensor_global = torch.cat([image_tensor_global, net.netG.nmlF], 0)
        if net.netG.netB is not None:
            image_tensor_global = torch.cat([image_tensor_global, net.netG.nmlB], 0)
    except:
        pass

    b_min = data['b_min']
    b_max = data['b_max']
    try:
        save_img_path = save_path[:-4] + '.png'
        save_img_list = []
        for v in range(image_tensor_global.shape[0]):
            save_img = (np.transpose(image_tensor_global[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
            save_img_list.append(save_img)
        save_img = np.concatenate(save_img_list, axis=1)
        cv2.imwrite(save_img_path, save_img)

        verts, faces, _, _ = reconstruction(
            net, cuda, calib_tensor, res, b_min, b_max, thresh, use_octree=use_octree, num_samples=100000)

        verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float()

        # if this returns error, projection must be defined somewhere else
        xyz_tensor = net.projection(verts_tensor, calib_tensor[:1])
        uv = xyz_tensor[:, :2, :]
        color = index(image_tensor[:1], uv).detach().cpu().numpy()[0].T
        color = color * 0.5 + 0.5 #(-1,1) -> (0,1)

        if 'calib_world' in data:
            calib_world = data['calib_world'].numpy()[0]
            verts = np.matmul(np.concatenate([verts, np.ones_like(verts[:,:1])],1), inv(calib_world).T)[:,:3]
        
        save_obj_mesh_with_color(save_path, verts, faces, color)
        
        #处理噪声 加强背面颜色
        meshcleaning(save_path)
        out_mesh = trimesh.load(save_path).split()[0]
        color = out_mesh.visual.vertex_colors
        verts = out_mesh.vertices
        #faces = out_mesh.faces
        out_mesh.visual.vertex_colors = np.array(esti_color(np.array(color,dtype='uint16'), net.projection(torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float(), calib_tensor[:1])),dtype='uint8') #更新颜色
        out_mesh.export(save_path)

    except Exception as e:
        print('error')
        print(e)


def esti_color(color, xyz_tensor):
    '''
    重新渲染颜色，每一个z<0的点去找y最近的左右两个点的颜色
    '''
    xyz = xyz_tensor.cpu().numpy()[0].T
    _ = list(range(xyz.shape[0]))
    xyz_1 = [[*xyz[i].tolist(),_[i]] for i in range(xyz.shape[0]) if xyz[i][2]<0]
    x_2 = np.array([[*xyz[i].tolist()[:2],_[i]] for i in range(xyz.shape[0]) if xyz[i][2]>=0 and xyz[i][2]<0.001])
    
    def find_closest(point,x_l):
        x = point[0]
        y = point[1]


        try:
            left = [int(i[2]) for i in sorted(x_l[x_l[:,0]-x<0], key=lambda x:(-abs(x[1]-y),x[0]),reverse=True)[:10]]
        except:
            left = [int(i[2]) for i in sorted(x_l[x_l[:,0]-x<0], key=lambda x:(-abs(x[1]-y),x[0]),reverse=True)]
        try:
            right = [int(i[2]) for i in sorted(x_l[x_l[:,0]-x>=0], key=lambda x:(abs(x[1]-y),x[0]))[:10]]
        except:
            right = [int(i[2]) for i in sorted(x_l[x_l[:,0]-x>=0], key=lambda x:(abs(x[1]-y),x[0]))]
        left = None if len(left)==0 else left
        right = None if len(right)==0 else right
        return [left, right]
    
    for i in range(len(xyz_1)):
        point = xyz_1[i]
        left, right = find_closest(point, x_2)
        #print(left, right)
        if right != None and left != None:
            temp_color = (sum(color[left])+sum(color[right]))/(len(left) + len(right))
        elif right == None and left != None:
            temp_color = sum(color[left])/len(left)
        elif right != None and left == None:
            temp_color = sum(color[right])/len(right)
        else:
            raise Exception('找不到最近颜色。')
        color[point[-1]] = temp_color
    return color
    
def recon(opt,use_crop,use_gpu):
    #load checkpoint
    state_dict_path = None
    if opt.load_netMR_checkpoint_path is not None:
        state_dict_path = opt.load_netMR_checkpoint_path
    elif opt.resume_epoch < 0:
        state_dict_path = '%s/%s_train_latest' % (opt.checkpoints_path, opt.name)
        opt.resume_epoch = 0
    else:
        state_dict_path = '%s/%s_train_epoch_%d' % (opt.checkpoints_path, opt.name, opt.resume_epoch)
    
    if use_gpu:
        cuda = torch.device('cuda:%d' % opt.gpu_id)
    else:
        cuda = torch.device('cpu') #显卡可以就改成 'cuda:%d' % opt.gpu_id
    
    start_id = opt.start_id
    end_id = opt.end_id
    use_color = opt.use_color
    
    state_dict = None
    if state_dict_path is not None and os.path.exists(state_dict_path):
        print('Resuming from', state_dict_path) #恢复训练的模型
        if use_gpu:
            state_dict = torch.load(state_dict_path) #显卡不行，若要调用gpu就删除map_location
        else:
            state_dict = torch.load(state_dict_path, map_location='cpu') #显卡不行，若要调用gpu就删除map_location
        print('Load {} success!'.format(state_dict_path))
        print('Warning: opt is overwritten.') #提示修改opt
        dataroot = opt.dataroot
        resolution = opt.resolution
        results_path = opt.results_path
        loadSize = opt.loadSize
        opt = state_dict['opt'] #重载opt
        opt.dataroot = dataroot
        opt.resolution = resolution
        opt.results_path = results_path
        opt.loadSize = loadSize
        print('Options has been updated!')
    else:
        raise Exception('Failed loading state dict!', state_dict_path)
    
    #parser.print_options(opt)

    
    print('Reading test images...')
    test_dataset = readDataset(opt.dataroot, opt.loadSize,use_crop=use_crop)
    print('Read success!')
    
    print('num of test images:',len(test_dataset))
    projection_mode = test_dataset.projection_mode # orthogonal 正交
    
    opt_netG = state_dict['opt_netG']

    netG = PIFuNetwNML(opt_netG, projection_mode).to(device=cuda) #粗糙层 法线
    netMR = PIFuMRNet(opt, netG, projection_mode).to(device=cuda) #精细层
    
    def set_eval(): #调用model.eval()会把所有的training属性设置为False
        netG.eval()
    
    #加载checkpoints
    netMR.load_state_dict(state_dict['model_state_dict'])
    
    #确保保存地址存在
    results_save_path = '{}/{}/recon'.format(opt.results_path, opt.name)
    os.makedirs(opt.checkpoints_path, exist_ok=True)
    os.makedirs(opt.results_path, exist_ok=True)
    os.makedirs(results_save_path,exist_ok = True)

    start_id = 0 if start_id<0 else start_id
    end_id = len(test_dataset) if end_id<0 else end_id
    
    with torch.no_grad():
        set_eval()
        
        print('generate mesh (test)...')
        for i in tqdm(range(start_id, end_id)):
            if i >= len(test_dataset):
                break
            test_data = test_dataset[i]
            
            result_save_path = results_save_path + '/result_{}_{}.obj'.format(test_data['name'],opt.resolution)
            print(result_save_path)
            if use_color == 0:
                gen_mesh(opt.resolution, netMR,cuda,test_data, result_save_path, components=opt.use_compose)
            elif use_color == 1:
                gen_mesh_imgColor(opt.resolution, netMR,cuda,test_data, result_save_path, components=opt.use_compose)
            elif use_color == 2:
                gen_mesh_imgColor_plus(opt.resolution, netMR,cuda,test_data, result_save_path, components=opt.use_compose)
            else:
                raise NameError('unknown downsampling type')
            #meshcleaning(result_save_path)


def meshcleaning(obj_path):
    '''
    只留下最大的连通图 去噪音
    '''
    print(f"Processing mesh cleaning: {obj_path}")

    mesh = trimesh.load(obj_path)
    cc = mesh.split()    

    out_mesh = cc[0]
    bbox = out_mesh.bounds
    height = bbox[1,0] - bbox[0,0]
    #找高度最大的连通体，舍弃噪音
    for c in cc:
        bbox = c.bounds
        if height < bbox[1,0] - bbox[0,0]:
            height = bbox[1,0] - bbox[0,0]
            out_mesh = c
    
    out_mesh.export(obj_path)

            
def reconWrapper(args = None,use_crop=False,use_gpu=False):
    opt = parser.parse(args)
    #print(opt)
    recon(opt,use_crop,use_gpu)
'''
if __name__ == '__main__':
    input_path = './test_images'
    out_path = './result'
    loadSize = 1024
    resolution = 512
    ckpt_path = './checkpoints/pifuhd.pt'
    start_id = -1
    end_id = -1
    use_crop = False
    use_color = 0
    
    args = ['--dataroot', input_path, '--results_path', out_path,\
           '--loadSize', str(loadSize), '--resolution', str(resolution), \
           '--load_netMR_checkpoint_path', ckpt_path,\
           '--start_id', str(start_id), '--use_color', str(use_color)]
    reconWrapper(args,use_crop,use_gpu)
'''


    
    
    
    








