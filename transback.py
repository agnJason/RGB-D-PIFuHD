# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 21:46:46 2020

@author: Administrator
"""


from aip import AipBodyAnalysis
import cv2
import numpy as np
import base64
import os
from PIL import Image
import copy
import argparse
from tqdm import tqdm

""" 读取图片 """
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read() 
    
    
def crop_trans_people(img_path,size,num=100):

    img_name = img_path.split('/')[-2]
    print(img_name)
    img_dirpath = img_path.replace(img_name+'/0_0_00.jpg','')
    img_name = img_name.split('.')[0]
    save_path = img_dirpath.replace('RENDER','gen1')
    try:
        os.mkdir(save_path)
    except:
        pass
    APP_ID = '21498366'
    API_KEY = 'GILkLlOqMAy1rm26O474uCcQ'
    SECRET_KEY = 'pT7G8feGiOG12yGrwClWiG7eMbzIyRhU'
    
    #client = AipBodyAnalysis(APP_ID, API_KEY, SECRET_KEY)
    image = cv2.resize(cv2.imread(img_path),(size,size))
    
    img_list = os.listdir('traindata/val2017')
    #transimg = cv2.imread('test.jpg')
    '''
    """ 调用人像分割 """
    res = client.bodySeg(image) 
    
    labelmap = base64.b64decode(res['labelmap'])
    foreground = base64.b64decode(res['foreground'])
    
    nparr_foreground = np.frombuffer(foreground,np.uint8)
    foregroundimg = cv2.imdecode(nparr_foreground,1)
    
    
    nparr_labelmap = np.frombuffer(labelmap,np.uint8)
    labelmapimg = cv2.imdecode(nparr_labelmap,1)

    
    h,w,b = labelmapimg.shape
    color = [0,0,0]
    #print(nparr_labelmap)
    top = int((size-w)/2)
    bottom = size-top-w
    left = int((size-h)/2)
    right = size-left-h
    top, bottom, left, right = abs(top), abs(bottom), abs(left), abs(right)
    foregroundimg = cv2.copyMakeBorder(foregroundimg, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    foregroundimg = cv2.resize(foregroundimg,(size,size))
    labelmapimg = cv2.copyMakeBorder(labelmapimg, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    labelmapimg = cv2.resize(labelmapimg,(size,size))    
    '''
    
   
    #随机抽样
    rand = np.random.choice(len(img_list), num)
    cv2.imwrite(save_path+'/'+img_name+'_000.png', image)
    for idx_img in rand:
        transimg = cv2.imread('traindata/val2017/{}'.format(img_list[idx_img]))
        transimg = cv2.resize(transimg,(size,size))
        #print(transimg.shape, foregroundimg.shape)
        img_ = copy.deepcopy(image)
        for i in range(size):
            for j in range(size):
                #print(labelmapimg[i,j])
                if sum(image[i,j]) > 760:
                    #print(transimg)
                    img_[i,j] = transimg[i,j]
        #cv2.imshow('1',foregroundimg)
        #cv2.waitKey(0)
        cv2.imwrite(save_path+'/'+img_name+'_{}.png'.format(idx_img), img_)
    print('gen {} success!'.format(img_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default=None)
    parser.add_argument('-ip', '--inputpath', type=str, default='traindata/OBJ')
    parser.add_argument('-s', '--size', type=int, default=1024)
    parser.add_argument('-n', '--num', type=int, default=10, help='num of images')
    args = parser.parse_args()
    if args.input != None:
        crop_trans_people(args.input,args.size,num=args.num)
    else:
        filenames=os.listdir(args.inputpath)
        for file in tqdm(filenames):
            if file[-4:] == '_OBJ':
                #print(os.path.join(args.inputpath, file))
                name = file[:-4]
                args.input = 'traindata/RENDER/{}/0_0_00.jpg'.format(name)
                crop_trans_people(args.input,args.size,num=args.num)