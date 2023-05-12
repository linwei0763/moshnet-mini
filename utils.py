# -*- coding: utf-8 -*-
"""
Created on Thu May 11 14:04:10 2023

@author: linwe
"""


#%%
import random as rd
import numpy as np
import os
import pandas as pd


#%%


# def calibrate_intensity(pc):
    
#     pc = np.asarray(pc)
#     pc[:, 3] = pc[:, 3] - np.mean(pc[:, 3])
#     pc[:, 3] = 1/(1+np.exp(-pc[:, 3]))
#     pc[:, 3] = pc[:, 3]/max(abs(pc[:, 3]))
#     pc = pd.DataFrame(pc)
    
#     return pc



#%%

def augmentate_l1l2(fn, pc):
    
    fn_record = 'data/record_l1l2.txt'
    path_l1 = 'data/l1'
    if not os.path.exists(path_l1):
        os.mkdir(path_l1)
    path_l2 = 'data/l2'
    if not os.path.exists(path_l2):
        os.mkdir(path_l2)
    num_max = 12
    num_sample = 8192
    
    fn = fn.split('.')[0]
    num = min(int(pc.shape[0]/num_sample), num_max)
        
    for i in range(num):
        pc_new = pc.sample(n=num_sample, replace=False)
        pc = pd.concat([pc, pc_new]).drop_duplicates(keep=False)
        
        trans = np.zeros(6)
                
        pc_new = np.asarray(pc_new)
        trans[0:3] = np.mean(pc_new[:, 0:3], axis=0)
        pc_new[:, 0:3] = pc_new[:, 0:3] - trans[0:3]
        pc_new_xyz = pc_new[:, 0:3]
        trans[3] = max(np.sqrt(np.sum(pc_new_xyz**2, axis=1)))
        pc_new[:, 0:3] = pc_new[:, 0:3]/trans[3]
        trans[4] = rd.uniform(0, 360)/180*np.pi
        R = np.array([[np.cos(trans[4]),-np.sin(trans[4]),0], [np.sin(trans[4]),np.cos(trans[4]),0], [0,0,1]])
        trans[5] = max(abs(pc_new[:, 3]))
        pc_new[:, 3] = pc_new[:, 3]/trans[5]
        pc_new[:, 0:3] = np.dot(R, pc_new[:, 0:3].T).T        
        with open(fn_record, 'a+') as f:
            f.write(fn + '-' + str(i) + '.txt' + ' ' + str(trans) + '\n')

        pc_new = pd.DataFrame(pc_new, columns=['x', 'y', 'z', 'i', 'l'])
        pc_new.to_csv(path_l2 + '/' + fn + '-' + str(i) + '.txt', sep=' ', header=None, index=None)
        
        pc_new = np.asarray(pc_new)
        pc_new[:, 4] = np.minimum(pc_new[:, 4], 1)
        pc_new = pd.DataFrame(pc_new, columns=['x', 'y', 'z', 'i', 'l'])
        pc_new.to_csv(path_l1 + '/' + fn + '-' + str(i) + '.txt', sep=' ', header=None, index=None)
        


#%%

def split_set(fns):
    
    path_l2 = 'data/l2'
    if not os.path.exists(path_l2):
        os.mkdir(path_l2)
    path_split = 'data/split-l1l2'
    if not os.path.exists(path_split):
        os.mkdir(path_split)
    fn_set_train = 'set_train.txt'
    fn_set_test = 'set_test.txt'
    
    ratio_train = 0.8
    num_train = int(len(fns)*ratio_train)
    
    rd.shuffle(fns)
    fns_train = []
    fns_test = []
    
    
    for i in range(len(fns)):
        if i <num_train:
            fns_train.append(fns[i])
        else:
            fns_test.append(fns[i])
    
    fns_l2 = os.listdir(path_l2)
    
    for i in range(len(fns_l2)):
        fn_pre = fns_l2[i].rsplit('-', 1)[0] + '.txt'
        
        if fn_pre in fns_train:
            with open(path_split + '/' + fn_set_train, 'a+') as f:
                f.write(fns_l2[i] + '\n')
        if fn_pre in fns_test:
            with open(path_split + '/' + fn_set_test, 'a+') as f:
                f.write(fns_l2[i] + '\n')
        
                
                
                
                
#%%





#%%

if __name__ == '__main__':
    
    path_l2_raw = 'data/l2-raw'
    
    fns = os.listdir(path_l2_raw)
    
    for fn in fns:
        pc = pd.read_csv(path_l2_raw + '/' + fn, sep=' ', header=None, names=['x', 'y', 'z', 'i', 'l'])
        # pc = calibrate_intensity(pc)
        augmentate_l1l2(fn, pc)
    
    split_set(fns)
    
    print('yes')



#%%







#%%





#%%

