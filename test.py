# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 11:49:19 2021

@author: Again Jason
One code, one world.
"""
import torch
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from EvalDataset import EvalDataset
from TrainDataset import TrainDataset
from options import BaseOptions
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def orthogonal(points, calib, transform=None):
    '''
    使用正交投影将点投影到屏幕空间
    args:
        points: [B, 3, N] 3d points in world coordinates
        calib: [B, 3, 4] projection matrix
        transform: [B, 2, 3] screen space transformation
    return:
        [B, 3, N] 3d coordinates in screen space
    '''
    rot = calib[:, :3, :3]  #[B, 3, 3]
    trans = calib[:, :3, 3:4]  #[B, 3, 1]
    #print(points.shape, calib.shape)
    pts = torch.baddbmm(trans, rot, points) #[B, 3, N]
    if transform is not None:
        scale = transform[:2, :2]
        shift = transform[:2, 2:3]
        pts[:, :2, :] = torch.baddbmm(shift, scale, pts[:, :2, :])
    return pts



opt = BaseOptions().parse()
opt.dataroot = './traindata'
train_dataset = TrainDataset(opt, phase='train',use_crop=True)
data = train_dataset[0]
print(data['name'])
p = data['samples']
calib = data['calib_world'][None,:]
p = p[None,:]

calib = calib[:,None]
p2 = orthogonal(p[:,0], calib[:,0]).tolist()
a=p2[0][0]
b=p2[0][1]
c=p2[0][2]
plt.plot(a,b)
'''
ax=plt.subplot(111,projection='3d')
ax.scatter(a,b,c,c='r')
ax.set_zlabel('Z') 
ax.set_ylabel('Y')
ax.set_xlabel('X')
plt.show()

ddd = torch.load('1')
p3 = [[],[],[]]
for i in range(256):
    for j in range(256):
        for z in range(256):
            if ddd[i][j][z].item()>0.5:
                p3[0].append(i)
                p3[1].append(j)
                p3[2].append(z)
                
a=p3[0][0]
b=p3[0][1]
c=p3[0][2]

ax=plt.subplot(111,projection='3d')
ax.scatter(a,b,c,c='r')
ax.set_zlabel('Z') 
ax.set_ylabel('Y')
ax.set_xlabel('X')
plt.show()
'''
