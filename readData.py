# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 14:25:41 2020

@author: Administrator
"""
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import numpy as np
import torch
from PIL import Image
from crop_img import crop_people
def addrect(img, rect):
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


class readDataset(Dataset):
    
    def __init__(self, dataroot, loadsize = 1024, projection='orthogonal', use_crop=False):
        self.projection_mode =projection
        
        self.root = dataroot #存放图像位置
        files = os.listdir(self.root)  #目录下所有文件
        self.img_files = sorted([os.path.join(self.root, f) for f in files \
                                 if f.split('.')[-1].lower() in ['jpg', 'jpeg','png'] and \
                                 os.path.exists(os.path.join(self.root, f.replace('.'+f.split('.')[-1], '_rect.txt')))]
                                ) #提取有人体姿势的图片文件
        self.load_size = loadsize
        
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]) #图像变换 to tensor
        self.use_crop = use_crop
    def get_item(self, index):
        '''
        获取index的图片数据
        '''
        img_path = self.img_files[index] #当前图片位置
        img_name = os.path.splitext(os.path.basename(img_path))[0] #图片名
        rect_path = img_path.replace('.'+img_path.split('.')[-1], '_rect.txt') #人物姿势数据
        depth_path = img_path.replace(img_name+'.'+img_path.split('.')[-1], 'depth/depth_'+img_name+'.png') #人物姿势数据
        if self.use_crop:
            im = crop_people(img_path)
        else:
            im = cv2.imread(img_path) #读取为三通道
            depth = cv2.imread(depth_path)
        h, w = im.shape[:2]
         
        rects = np.loadtxt(rect_path, dtype=np.int32) #读rect
        if len(rects.shape) == 1: #rect必须是个矩阵
            rects = np.array([rects])
        
        rect = rects[0].tolist()
        im = addrect(im, rect) #切出人位置
        depth = addrect(depth, rect)

        intrinsic = np.identity(4)
        trans_mat = np.identity(4)
        scale_im2ndc = 1.0 / float(w // 2)
        scale = w / rect[2]
        trans_mat *= scale
        trans_mat[3,3] = 1.0
        trans_mat[0, 3] = -scale*(rect[0] + rect[2]//2 - w//2) * scale_im2ndc
        trans_mat[1, 3] = scale*(rect[1] + rect[3]//2 - h//2) * scale_im2ndc
        intrinsic = np.matmul(trans_mat, intrinsic) #校准向量

        im_512 = cv2.resize(im, (512,512))
        im = cv2.resize(im, (self.load_size, self.load_size))
        depth_512 = cv2.resize(depth, (512,512))
        depth = cv2.resize(depth, (self.load_size, self.load_size))
        
        bound_min = np.array([-1, -1, -1])
        bound_max = np.array([1, 1, 1])
        projection_matrix = np.identity(4)
        projection_matrix[1, 1] = -1
        calib = torch.Tensor(projection_matrix).float()
        
        calib_world = torch.Tensor(intrinsic).float()
        
        image_512 = Image.fromarray(im_512[:,:,::-1]).convert('RGB') #RGB
        image = Image.fromarray(im[:,:,::-1]).convert('RGB')
        depth_512 = Image.fromarray(depth_512[:,:,::-1]).convert('RGB') #RGB
        depth = Image.fromarray(depth[:,:,::-1]).convert('RGB')

        # image
        image_512 = self.to_tensor(image_512)
        image = self.to_tensor(image)
        depth_512 = self.to_tensor(depth_512)
        depth = self.to_tensor(depth)
        return {
            'name': img_name,
            'img': image.unsqueeze(0),
            'img_512': image_512.unsqueeze(0),
            'depth': depth.unsqueeze(0),
            'depth_512': depth_512.unsqueeze(0),
            'calib': calib.unsqueeze(0),
            'calib_world': calib_world.unsqueeze(0),
            'b_min': bound_min,
            'b_max': bound_max,
        }
        

    def __len__(self):
        '''
        可以使用len() 获取数据集数量
        '''
        return len(self.img_files)
    
    def __getitem__(self, index):
        '''
        可以使用 data[index]获取第index个的数据
        '''
        return self.get_item(index)