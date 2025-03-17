
# -*- coding: utf-8 -*-

import numpy as np
import os,pickle,argparse
from utils import centernorm_size,interpolate_torch

def preprocess_OLHWDB2(src_root='./data-raw/OLHWDB2',interp=4):
    writing = {}
    writers = os.listdir(src_root)
    for i,w in enumerate(writers):
        writing[i] = []
        for file in os.listdir(f'{src_root}/{w}'):
            info = []
            with open(f'{src_root}/{w}/{file}','r',encoding='utf-8') as f:
                lines = f.readlines()
                lines = lines[1:]
                lines = [l.strip() for l in lines]
            info = [list(map(lambda x:float(x),l.split()[:3])) for l in lines]
            info = np.array(info,np.float32)
            info = centernorm_size(info)
            if interp != None:
                info = interpolate_torch(info,interp_ratio=interp)
            writing[i].append(info)
    tgt_root = src_root.replace('data-raw','data')
    os.makedirs(tgt_root,exist_ok=True)
    with open(f'{tgt_root}/OLHWDB2.pkl','wb') as f:
        pickle.dump(writing,f)
    
def preprocess_DCOHE(src_root='./data-raw/DCOH-E'):
    writing = {}
    writers = os.listdir(src_root)
    for i,w in enumerate(writers):
        writing[w] = []
        for file in os.listdir(f'{src_root}/{w}'):
            info = []
            with open(f'{src_root}/{w}/{file}','r',encoding='utf-8') as f:
                lines = f.readlines()
                lines = lines[1:]
                lines = [l.strip() for l in lines]
            info = [list(map(lambda x:float(x),l.split()[:3])) for l in lines]
            info = np.array(info,np.float32)
            info = centernorm_size(info)
            if 'dcoh-e' in file:
                info = interpolate_torch(info,interp_ratio=2)
            writing[w].append(info)
    tgt_root = src_root.replace('data-raw','data')
    os.makedirs(tgt_root,exist_ok=True)
    with open(f'{tgt_root}/DCOH-E.pkl','wb') as f:
        pickle.dump(writing,f)

def preprocess_COUCH(src_root='./data-raw/COUCH09',interp=4):
    writing = {}
    writers = os.listdir(src_root)
    for i,w in enumerate(writers):
        writing[i] = []
        for file in os.listdir(f'{src_root}/{w}'):
            with open(f'{src_root}/{w}/{file}','r',encoding='utf-8') as f:
                lines = f.readlines()
                lines = lines[1:]
                lines = [l.strip() for l in lines]
            info = [list(map(lambda x:float(x),l.split()[:3])) for l in lines]
            info = np.array(info,np.float32)
            info = centernorm_size(info)
            if interp != None:
                info = interpolate_torch(info,interp_ratio=interp)
            writing[i].append(info)
    tgt_root = src_root.replace('data-raw','data')
    os.makedirs(tgt_root,exist_ok=True)
    with open(f'{tgt_root}/COUCH09.pkl','wb') as f:
        pickle.dump(writing,f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',type=str,default='olhwdb2',help='processed dataset names: [olhwdb2,dcohe,couch]')
    opt = parser.parse_args()
    func = globals()[f'preprocess_{opt.dataset.upper()}']
    print(f'start preprocessing {opt.dataset.upper()} dataset.')
    func()
    print(f'end preprocessing {opt.dataset.upper()} dataset.')
