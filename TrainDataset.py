# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 14:25:05 2020

@author: Administrator
"""

from torch.utils.data import Dataset
import trimesh
import os
import numpy as np
import torch
import torchvision.transforms as transforms
import cv2
from PIL import Image

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

def addrect(img, rect):
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    x, y, w, h = rect

    left = abs(x) if x < 0 else 0 #左边界
    top = abs(y) if y < 0 else 0  #上边界
    right = abs(img.shape[1]-(x+w)) if x + w >= img.shape[1] else 0 #右边界
    bottom = abs(img.shape[0]-(y+h)) if y + h >= img.shape[0] else 0 #下边界
    
    color = [0, 0, 0]
    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    x = x + left
    y = y + top

    return new_img[y:(y+h),x:(x+w),:]

class TrainDataset(Dataset):
    
    def __init__(self, opt, phase='train', load_mesh=True,use_crop=False):
        self.opt = opt
        self.projection_mode = 'orthogonal'

        self.root = self.opt.dataroot
        self.RENDER = os.path.join(self.root, 'RENDER')
        self.MASK = os.path.join(self.root, 'MASK')
        self.PARAM = os.path.join(self.root, 'PARAM')
        self.OBJ = os.path.join(self.root, 'OBJ')
        self.NORM = os.path.join(self.root, 'NORM')
        self.DEPTH = os.path.join(self.root, 'DEPTH')

        files = os.listdir(self.root+'/gen')
        self.img_files = sorted([os.path.join(self.root+'/gen', f) for f in files if f.split('.')[-1].lower() in ['png']])

        self.num_sample_inout = self.opt.num_sample_inout
        self.B_MIN = np.array([-256, -28, -562])
        self.B_MAX = np.array([0, 228, -306])
        self.load_mesh = load_mesh
        
        self.load_size = self.opt.loadSize
        self.subjects = []
        if self.load_mesh:
            self.mesh_dic = self.load_trimesh(self.OBJ)
        self.is_train = (phase == 'train')
        self.use_crop = use_crop

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]) #图像变换 to tensor

    def __len__(self):
        return len(self.img_files)
    
    def load_trimesh(self, root_dir):
        folders = os.listdir(root_dir)
        meshs = {}
        for i, f in enumerate(folders):
            if f[-9:] == '_100k.obj':
                sub_name = f
                meshs[sub_name] = trimesh.load(os.path.join(root_dir, f))
                self.subjects.append(sub_name[:-9])
        return meshs
    
    def select_sampling_method(self, subject,lims,use_crop):
        '''
        对3D样本点采样
        '''
        mesh = self.mesh_dic[subject]
        surface_points, _ = trimesh.sample.sample_surface(mesh, 4 * self.num_sample_inout)
        #print(surface_points)
        sample_points = surface_points + np.random.normal(scale=1.0, size=surface_points.shape)

        # add random points within image space
        length = self.B_MAX - self.B_MIN
        random_points = np.random.rand(self.num_sample_inout // 4, 3) * length + self.B_MIN
        sample_points = np.concatenate([sample_points, random_points], 0)
        np.random.shuffle(sample_points)

        inside = mesh.contains(sample_points)

        inside_points = sample_points[inside]
        outside_points = sample_points[np.logical_not(inside)]

        nin = inside_points.shape[0]
        inside_points = inside_points[
                        :self.num_sample_inout // 2] if nin > self.num_sample_inout // 2 else inside_points
        outside_points = outside_points[
                         :self.num_sample_inout // 2] if nin > self.num_sample_inout // 2 else outside_points[
                                                                                               :(self.num_sample_inout - nin)]
        
        samples = np.concatenate([inside_points, outside_points], 0).T

        labels = np.concatenate([np.ones((1, inside_points.shape[0])), np.zeros((1, outside_points.shape[0]))], 1)

        samples = torch.Tensor(samples).float()
        labels = torch.Tensor(labels).float()
        
        samples_crop = []
        labels_crop = []
        '''
        if use_crop:
            for i in lims: #随机剪裁
                a = i[0]
                b = i[1]
                inbox = (samples[0]>a) & (samples[0]<(a+1)) & (samples[1]>b) & (samples[1]<(b+1))
                sample_crop = samples[:,inbox]
                label_crop = labels[:,inbox]
                samples_crop.append(sample_crop)
                labels_crop.append(label_crop)
            samples_crop = torch.stack(samples_crop,0)
            labels_crop = torch.stack(labels_crop,0)
            #print(labels_crop.shape)
            #nmls_total = mesh.vertex_normals
        '''
        #randoms = np.random.choice(range(nmls_total.shape[0]),int(nmls_total.shape[0]*0.1))
        #nmls = nmls_total[randoms]
        #point_nmls = mesh.vertices[randoms]
        return {
            'samples': samples.unsqueeze(0),
            'labels': labels,
            'samples_crop':samples_crop,
            'labels_crop':labels_crop
            #'labels_nml':nmls,
            #'points_nml':point_nmls
        }
    
    def get_item(self, index):
        render_path = self.img_files[index]
        subject = '_'.join(os.path.splitext(os.path.basename(render_path))[0].split('_')[:-1]) #图片名

        param_path = os.path.join(self.PARAM, subject, '%d_%d_%02d.npy' % (0, 0, 0))
        #render_path = os.path.join(self.RENDER, subject, '%d_%d_%02d.jpg' % (0, 0, 0))
        mask_path = os.path.join(self.MASK, subject, '%d_%d_%02d.png' % (0, 0, 0))
        depth_path = os.path.join(self.DEPTH, subject, '%d_%d_%02d.png' % (0, 0, 0))
        fn_path = os.path.join(self.NORM, subject, '%d_%d_%02d.png' % (0, 0, 0))
        bn_path = os.path.join(self.NORM, subject, '%d_%d_%02d.png' % (180, 0, 0))

        # loading calibration data        
        param = np.load(param_path, allow_pickle=True)        
        # pixel unit / world unit
        ortho_ratio = param.item().get('ortho_ratio')
        # world unit / model unit
        scale = param.item().get('scale')
        # camera center world coordinate
        center = param.item().get('center')
        # model rotation
        R = param.item().get('R')
        
        translate = -np.matmul(R, center).reshape(3, 1)
        extrinsic = np.concatenate([R, translate], axis=1)
        extrinsic = np.concatenate([extrinsic, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)
        # Match camera space to image pixel space
        scale_intrinsic = np.identity(4)
        scale_intrinsic[0, 0] = scale / ortho_ratio
        scale_intrinsic[1, 1] = -scale / ortho_ratio
        scale_intrinsic[2, 2] = scale / ortho_ratio

        # Match image pixel space to image uv space
        uv_intrinsic = np.identity(4)
        uv_intrinsic[0, 0] = 1.0 / float(self.opt.loadSize // 2)
        uv_intrinsic[1, 1] = 1.0 / float(self.opt.loadSize // 2)
        uv_intrinsic[2, 2] = 1.0 / float(self.opt.loadSize // 2)

        uv_intrinsicLocal = np.identity(4)
        uv_intrinsicLocal[0, 0] = 1.0 / float(self.opt.loadSize // 2)
        uv_intrinsicLocal[1, 1] = 1.0 / float(self.opt.loadSize // 2)
        uv_intrinsicLocal[2, 2] = 1.0 / float(self.opt.loadSize // 2)
        # Transform under image pixel space
        trans_intrinsic = np.identity(4)

        mask = Image.open(mask_path).convert('L')
        render = Image.open(render_path).convert('RGB')
        depth = Image.open(depth_path).convert('RGB')
        imF = Image.open(fn_path).convert('RGB')
        imB = Image.open(bn_path).convert('RGB')

        imBig = render.resize((self.opt.loadSizeBig,self.opt.loadSizeBig))
        im = render.resize((self.opt.loadSizeLocal,self.opt.loadSizeLocal))
        depthBig = depth.resize((self.opt.loadSizeBig,self.opt.loadSizeBig))
        depth = depth.resize((self.opt.loadSizeLocal,self.opt.loadSizeLocal))

        imF = imF.resize((self.opt.loadSizeBig,self.opt.loadSizeBig))
        imB = imB.resize((self.opt.loadSizeBig,self.opt.loadSizeBig))

        intrinsic = np.matmul(trans_intrinsic, np.matmul(uv_intrinsic, scale_intrinsic))
        calib = torch.Tensor(np.matmul(intrinsic, extrinsic)).float()
        intrinsicLocal = np.matmul(trans_intrinsic, np.matmul(uv_intrinsicLocal, scale_intrinsic))

        if self.use_crop:     
            imBig = render.resize((1024,1024))
            depthBig = depth.resize((1024,1024))   
            rect = [256,np.random.randint(10,512),512,512]
            imBig = addrect(imBig, rect)
            depthBig = addrect(depthBig, rect)
            trans_mat = np.identity(4)
            scale_im2ndc = 1.0 / float(1024 // 2)
            scale = 1024 / rect[2]
            trans_mat *= scale
            trans_mat[3,3] = 1.0
            trans_mat[0, 3] = -scale*(rect[0] + rect[2]//2 - 1024//2) * scale_im2ndc
            trans_mat[1, 3] = -scale*(rect[1] + rect[3]//2 - 1024//2) * scale_im2ndc
            intrinsicLocal =  np.matmul(trans_mat, intrinsicLocal)
        calibLocal = torch.Tensor(np.matmul(intrinsicLocal, extrinsic)).float()
        extrinsic = torch.Tensor(extrinsic).float()


        im= self.to_tensor(im)
        imBig= self.to_tensor(imBig)
        depth= self.to_tensor(depth)
        depthBig= self.to_tensor(depthBig)
        imF= self.to_tensor(imF)
        imB= self.to_tensor(imB)

        lims = []

        Fstyle = cv2.resize(cv2.imread(self.root+'/normal/Fnormal.jpg'), (self.opt.loadSizeBig, self.opt.loadSizeBig))
        Fstyle = Image.fromarray(Fstyle[:,:,::-1]).convert('RGB')
        Fstyle = self.to_tensor(Fstyle)

        Bstyle = cv2.resize(cv2.imread(self.root+'/normal/Bnormal.jpg'), (self.opt.loadSizeBig, self.opt.loadSizeBig))
        Bstyle = Image.fromarray(Bstyle[:,:,::-1]).convert('RGB')
        Bstyle = self.to_tensor(Bstyle)

        res= {
            'name': subject,
            'img': imBig.view((1,*imBig.shape)),
            'img_512': im,
            'depth': depthBig.view((1,*depthBig.shape)),
            'depth_512': depth,
            'calib': calib,
            'calib_world': calibLocal,
            'b_min': self.B_MIN,
            'b_max': self.B_MAX,
            'imF':imF,
            'imB':imB,
            'Fstyle':Fstyle,
            'Bstyle':Bstyle
        }

        if self.load_mesh:
            res.update(self.select_sampling_method(subject +'_100k.obj',lims,self.use_crop))
        return res
    def __getitem__(self, index):
        return self.get_item(index)
        
