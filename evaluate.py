# -*- coding: utf-8 -*-

import torch,os,time
from torch import nn
from scipy import io
import numpy as np
from model_utils import db_augmentation,average_query_expansion

def evaluate(qf,ql,gf,gl): # q是query，g是gallery
    # q是一个，gallery是全部19732个
    # print(qf.shape,ql.shape,qc.shape,gf.shape,gl.shape,gc.shape)
    query = qf.view(-1,1) # (512,1)
    score = torch.mm(gf,query) # (19732,1)
    # score = nn.PairwiseDistance(p=2)(gf,qf.view(1,-1))
    score = score.squeeze().cpu().numpy()
    
    idx = np.argsort(score)[::-1]
    # print(ql,gl[idx][:10],gl[idx][10:20],gl[idx][20:30])
    query_idx = np.argwhere(gl == ql) # gallery里总有当前query的类别的，而且不止一张图是这个类
    positive_idx = query_idx
    metrics = compute_mAP(idx,positive_idx)
    return metrics # (ap,CMC)

def compute_mAP(idx,positive_idx):
    ap = 0
    cmc = torch.zeros((len(idx)))
    if positive_idx.size == 0:
        cmc[0] = -1
        return ap,cmc
    len_pos = len(positive_idx)
    mask = np.in1d(idx,positive_idx) # 找到positive的index，也就是不同摄像头的同一个人
    rows_pos = np.argwhere(mask).flatten()
    # print(rows_pos,len_pos)
    cmc[rows_pos[0]:] = 1 # 赋值1的每个位置都不同，每个位置累加起来再除以总数就是Acc@1这些了
    # 注意这里有个:,是从第一个往后都赋值1，所以Rank@10会比Rank@1大
    for i in range(len_pos): # len_pos是不知道的，所以无所谓有多少个gallery样本
        precision = (i + 1) * 1. / (rows_pos[i] + 1) # 这就是每一个格子的precision
        if rows_pos[i] != 0:
            old_precision = i * 1.0 / rows_pos[i]
        else:
            old_precision = 1.0
        ap = ap + (old_precision + precision) / 2 # 不太理解为什么要old_precision然后除以2
        # ap = ap + precision
    ap = ap / len_pos

    return ap,cmc

def compute_metrics(res,logger,dba,device,verbose=True):
    query_feature = res['query_feature']
    query_label = res['query_label']
    gallery_feature = res['gallery_feature']
    gallery_label = res['gallery_label']

    if dba:
        time_start = time.time()
        query_feature,gallery_feature = db_augmentation(query_feature,gallery_feature,10)
        query_feature,gallery_feature = average_query_expansion(query_feature,gallery_feature,5)
        query_feature = query_feature / np.linalg.norm(query_feature,axis=1,keepdims=True)
        gallery_feature = gallery_feature / np.linalg.norm(gallery_feature,axis=1,keepdims=True)
        logger.info(f'DBA & AQE time consuming: {time.time() - time_start:.4f}s')

    query_feature = torch.FloatTensor(query_feature).to(device)
    gallery_feature = torch.FloatTensor(gallery_feature).to(device)
    
    CMC = torch.zeros((len(gallery_label)))
    # aps = []
    ap = 0.
    time_sum = 0.
    for i in range(len(query_label)):
        time_start = time.time()
        cur_ap,cur_CMC = evaluate(query_feature[i],query_label[i],gallery_feature,gallery_label)
        time_sum += (time.time() - time_start)
        if cur_CMC[0] == -1: continue
        CMC += cur_CMC
        ap += cur_ap
        # aps.append(cur_ap)
    # logger.info(f'evaluate time consuming: {time_sum:.4f}s')
    time_avg = time_sum / len(query_label)

    CMC /= len(query_label)
    ap /= len(query_label)
    if verbose:
        logger.info(f'[single query] Rank@1: {CMC[0] * 100.:.4f}% Rank@5: {CMC[4] * 100.:.4f}% Rank@10: {CMC[9] * 100.:.4f}%')
        logger.info(f'[single query] mAP: {ap * 100.:.4f}%')
    return time_avg,ap,CMC[0] * 100.,CMC[4] * 100.,CMC[9] * 100.

if __name__ == '__main__':
    from dataset import Writing
    import pickle
    import matplotlib.pyplot as plt
    gallery_root = f'./data/OLER/gallery-tf-optxy2.pkl'
    with open(gallery_root,'rb') as f:
        handwriting_info = pickle.load(f,encoding='iso-8859-1')
    gallery_dataset = Writing(handwriting_info,train=False)
    _,aps = compute_metrics(None,False,'cpu')
    l = gallery_dataset.user_labels
    print(aps)
    k = [len(np.where(l == i)[0]) for i in np.sort(list(set(l)))]
    print(k)
    d = {i:(len(np.where(l == i)[0]),aps[i]) for i in np.sort(list(set(l)))}
    
    d1 = [(len(np.where(l == i)[0]),aps[i]) for i in np.sort(list(set(l)))]
    d1 = sorted(d1,key=lambda x:x[0])
    # plt.hist(aps,len(aps))
    # x_array = list(set(d.values()))
    # plt.bar(range(x_array),)
    aps = [each[1] for each in d1]
    idx = [each[0] for each in d1]
    idx1 = [idx[0],idx[len(idx) // 4],idx[len(idx) // 2],idx[len(idx) // 4 * 3],idx[-1]]
    idx2 = [0,len(idx) // 4,len(idx) // 2,len(idx) // 4 * 3,len(idx)]
    idx3 = [''] * len(idx)
    idx3[0] = idx[0]
    idx3[len(idx) // 4] = idx[len(idx) // 4]
    idx3[len(idx) // 2] = idx[len(idx) // 2]
    idx3[len(idx) // 4 * 3] = idx[len(idx) // 4 * 3]
    idx3[-1] = idx[-1]
    print(d1)
    plt.bar(range(len(aps)),aps)
    plt.xticks(range(len(idx)),idx3)
    plt.savefig('./kkk.png',dpi=500)
    # print(aps)
    # a = np.array([1,2,3,4,5,1,2,1,1])
    # b = np.array([3,2])
    # c = np.in1d(a,b)
    # print(c)