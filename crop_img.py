# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 21:46:46 2020

@author: Administrator
"""


from aip import AipBodyAnalysis
import cv2
import numpy as np
import base64



""" 读取图片 """
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read() 
def crop_people(img_path):
    APP_ID = '21498366'
    API_KEY = 'GILkLlOqMAy1rm26O474uCcQ'
    SECRET_KEY = 'pT7G8feGiOG12yGrwClWiG7eMbzIyRhU'
    
    client = AipBodyAnalysis(APP_ID, API_KEY, SECRET_KEY)
    image = get_file_content(img_path) 
    
    """ 调用人像分割 """
    res = client.bodySeg(image) 
    
    labelmap = base64.b64decode(res['labelmap'])
    foreground = base64.b64decode(res['foreground'])
    
    nparr_foreground = np.fromstring(foreground,np.uint8)
    foregroundimg = cv2.imdecode(nparr_foreground,1)
     
    nparr_labelmap = np.fromstring(labelmap,np.uint8)
    labelmapimg = cv2.imdecode(nparr_labelmap,1)
    im_new_labelmapimg = np.where(labelmapimg!=1, 255, foregroundimg)
    return im_new_labelmapimg
    #cv2.imwrite(img_path, im_new_labelmapimg)

