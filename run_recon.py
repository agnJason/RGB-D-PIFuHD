# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 08:39:59 2020

@author: Administrator
"""
from reconstruction import reconWrapper

if __name__ == '__main__':
    input_path = './test_images'
    out_path = './result'
    loadSize = 1024
    resolution = 512
    ckpt_path = './checkpoints/pifuhd.pt'
    start_id = -1
    end_id = -1
    use_crop = False
    use_gpu = False
    use_color = 0 #0:法线贴图 1：照片颜色 2：粗略上色
    args = ['--dataroot', input_path, '--results_path', out_path,\
           '--loadSize', str(loadSize), '--resolution', str(resolution), \
           '--load_netMR_checkpoint_path', ckpt_path,\
           '--start_id', str(start_id), '--end_id', str(end_id),'--use_color', str(use_color)]
    reconWrapper(args,use_crop,use_gpu)