# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 20:13:12 2021

@author: Again Jason
One code, one world.
"""
import os
import numpy as np
from matplotlib import pyplot as plt

def read_npy(path):
    files = sorted(os.listdir(path))

    res = {}
    for file in files:
        if file[-3:] == 'npy':
            #print(file)
            tmp = np.load(os.path.join(path, file)).tolist()
            name = '_'.join(file.split('_')[:-3])
            if name not in res.keys():
                res[name] = tmp
            else:
                res[name] += tmp
            
    return res

def plot_error(array, groupnum=None, lim=None, array2 = None):
    array = np.array(array)
    array.resize((1,array.shape[0]*array.shape[1])) 
    array = array[0]

    if groupnum is not None:
        ids = np.arange(len(array))//groupnum
        array = np.bincount(ids,array)/np.bincount(ids)
            
    plt.plot(array.tolist(), label='course layer')
    
    if array2 is not None:
        array2 = np.array(array2)
        array2.resize((1,array2.shape[0]*array2.shape[1]))
        array2 = array2[0]
        if groupnum is not None:
            ids = np.arange(len(array2))//groupnum
            array2 = np.bincount(ids,array2)/np.bincount(ids)
        plt.plot(array2.tolist(), label='fine layer')
            
    if lim is not None:
        plt.ylim(0,lim)
    plt.title('Loss')
    plt.xlabel('times')
    plt.ylabel('loss value')
    x = range(0, 400000//groupnum, 50000//groupnum)
    plt.xticks(x,['0', '50k', '100k', '150k', '200k', '250k', '300k','350k'])
    plt.annotate("Use sliding window", (100000//groupnum,0.1), xycoords='data',
         xytext=(145000//groupnum,0.14),
         arrowprops=dict(arrowstyle='->'))
    plt.legend()
    plt.show()
    


 
netMR_errors = read_npy(r'train_result\netMR')
netMR_train_error = netMR_errors['error']
#netMR_eval_error = netMR_errors['eval_error']

plot_error(netMR_train_error,200)
#plot_error(netMR_eval_error)

netG_errors = read_npy(r'train_result/netG')
netG_train_error = netG_errors['error']

plot_error(netG_train_error,500,array2 = netMR_train_error)