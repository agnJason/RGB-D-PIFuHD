# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 22:09:24 2020

@author: Administrator
"""
import numpy as np
from numpy.linalg import inv
import torch
from skimage import measure

def create_grid(resX, resY, resZ, b_min=np.array([-1, -1, -1]),
                b_max=np.array([1, 1, 1]), transform=None):
    '''
    创建具有给定分辨率和边界框的密集网格
    args:    
        resX：沿X轴的分辨率
        resY: 沿Y轴的分辨率
        resZ：沿Z轴的分辨率
        b_min:vec3（x_min，y_min，z_min）边界框角点
        b_max:vec3（x_max、y_max、z_max）边界框角
    return:
        [3，resX，resY，resZ]网格坐标，并从网格索引转换矩阵
    '''
    coords = np.mgrid[:resX,:resY,:resZ] #生成立方体坐标 n^3*3个数字
    coords = coords.reshape(3,-1)
    coords_matrix = np.eye(4)
    length = b_max - b_min
    coords_matrix[0,0]=length[0]/resX
    coords_matrix[1,1]=length[1]/resY
    coords_matrix[2,2]=length[2]/resZ
    coords_matrix[0:3, 3] = b_min
    coords = np.matmul(coords_matrix[:3, :3],coords) + coords_matrix[:3, 3:4] #转换到b_min - b_max 的空间坐标
    if transform != None:
        coords = np.matmul(transform[:3,:3],coords) + transform[:3, 3:4]
        coords_matrix = np.matmul(transform, coords_matrix)
    coords = coords.reshape(3, resX, resY, resZ)
    return coords, coords_matrix

def reconstruction(net, cuda, calib_tensor, resolution,
                   b_min, b_max, thresh=0.5, use_octree=False,
                   num_samples=10000, transform = None):
    r'''
    由网络预测的sdf重构网格。
    args:
        net:BasePixImpNet对象。在head之前调用图像过滤器。
        cuda:cuda设备
        calib_tensor:校准张量
        resolution：网格单元的分辨率
        b_min:边界框角点[x\u min，y\u min，zümin]
        b_max:边界框角点[x_max，y_max，z_max]
        use_octree：是否使用octree加速
        num_samples:查询每个gpu迭代的点数
    return:marching cubes结果。
    '''
    #首先，我们通过分辨率创建网格
    #并将网格坐标的矩阵转换为现实世界的xyz
    
    coords, mat = create_grid(resolution, resolution, resolution)
    
    calib = calib_tensor[0].cpu().numpy()
    calib_inv = inv(calib) #求逆
    coords = coords.reshape(3,-1).T
    coords = np.matmul(np.concatenate([coords, np.ones((coords.shape[0],1))], 1), calib_inv.T)[:, :3] #坐标矫正
    coords = coords.T.reshape(3, resolution, resolution, resolution)
    
    def eval_func(points):
        points = np.expand_dims(points, axis=0)
        points = np.repeat(points, 1, axis=0)
        samples = torch.from_numpy(points).to(device=cuda).float()
        
        net.query(samples, calib_tensor)
        pred = net.get_preds()[0][0]
        return pred.detach().cpu().numpy()
    
    #评估网格
    if use_octree:
        sdf = eval_grid_octree(coords, eval_func, num_samples=num_samples)
    else:
        sdf = eval_grid(coords, eval_func, num_samples=num_samples)
    
    #行进立方体算法在3d体积数据中找到表面
    try:
        verts, faces, normals, values = measure.marching_cubes_lewiner(sdf, thresh)
        #print(faces)
        #将顶点转换到坐标系
        trans_mat = np.matmul(calib_inv, mat)
        verts = np.matmul(trans_mat[:3, :3], verts.T) + trans_mat[:3, 3:4]
        verts = verts.T
        #看表面是否需要翻转
        if np.linalg.det(trans_mat[:3, :3]) < 0.0:
            faces = faces[:,::-1]
        return verts, faces, normals, values
    except:
        print('error cannot marching cubes')
        return -1    
        
def batch_eval(points, eval_func, num_samples=512 * 512 * 512):
    '''
    args:
        points: 需要分析的空间样本
        eval_func: 方法，输入点，判断这些
    '''
    num_pts = points.shape[1]
    sdf = np.zeros(num_pts)

    num_batches = num_pts // num_samples
    for i in range(num_batches):
        sdf[i * num_samples:i * num_samples + num_samples] = eval_func(
            points[:, i * num_samples:i * num_samples + num_samples])
    if num_pts % num_samples:
        sdf[num_batches * num_samples:] = eval_func(points[:, num_batches * num_samples:])

    return sdf

def eval_grid(coords, eval_func, num_samples=512 * 512 * 512):
    resolution = coords.shape[1:4]
    coords = coords.reshape([3, -1])
    sdf = batch_eval(coords, eval_func, num_samples=num_samples)
    return sdf.reshape(resolution)



def eval_grid_octree(coords, eval_func,
                     init_resolution=64, threshold=0.05,
                     num_samples=512 * 512 * 512):
    '''
    八叉树构建3d坐标网格
    '''
    resolution = coords.shape[1:4] #空间大小 分辨率

    sdf = np.zeros(resolution)

    notprocessed = np.zeros(resolution, dtype=np.bool)
    notprocessed[:-1,:-1,:-1] = True
    grid_mask = np.zeros(resolution, dtype=np.bool)

    reso = resolution[0] // init_resolution

    while reso > 0:
        # 细分网格
        grid_mask[0:resolution[0]:reso, 0:resolution[1]:reso, 0:resolution[2]:reso] = True
        # 测试样本
        test_mask = np.logical_and(grid_mask, notprocessed)
        # print('step size:', reso, 'test sample size:', test_mask.sum())
        points = coords[:, test_mask] #找到这一次测试的点

        sdf[test_mask] = batch_eval(points, eval_func, num_samples=num_samples)
        notprocessed[test_mask] = False

        # do interpolation
        if reso <= 1:
            break
        x_grid = np.arange(0, resolution[0], reso)
        y_grid = np.arange(0, resolution[1], reso)
        z_grid = np.arange(0, resolution[2], reso)

        v = sdf[tuple(np.meshgrid(x_grid, y_grid, z_grid, indexing='ij'))]
        v0 = v[:-1,:-1,:-1]
        v1 = v[:-1,:-1,1:]
        v2 = v[:-1,1:,:-1]
        v3 = v[:-1,1:,1:]
        v4 = v[1:,:-1,:-1]
        v5 = v[1:,:-1,1:]
        v6 = v[1:,1:,:-1]
        v7 = v[1:,1:,1:]

        x_grid = x_grid[:-1] + reso//2
        y_grid = y_grid[:-1] + reso//2
        z_grid = z_grid[:-1] + reso//2

        nonprocessed_grid = notprocessed[tuple(np.meshgrid(x_grid, y_grid, z_grid, indexing='ij'))]

        v = np.stack([v0,v1,v2,v3,v4,v5,v6,v7], 0)
        v_min = v.min(0)
        v_max = v.max(0)
        v = 0.5*(v_min+v_max)

        skip_grid = np.logical_and(((v_max - v_min) < threshold), nonprocessed_grid)

        xs, ys, zs = np.where(skip_grid)
        for x, y, z in zip(xs*reso, ys*reso, zs*reso):
            sdf[x:(x+reso+1), y:(y+reso+1), z:(z+reso+1)] = v[x//reso,y//reso,z//reso]
            notprocessed[x:(x+reso+1), y:(y+reso+1), z:(z+reso+1)] = False
        reso //= 2

    return sdf.reshape(resolution)
    
def save_obj_mesh_with_color(mesh_path, verts, faces, colors):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        c = colors[idx]
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()    
    
    
    
    
    
    
    
    