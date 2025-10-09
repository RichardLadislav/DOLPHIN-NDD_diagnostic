# -*- coding:utf-8 -*-

import torch
from torch.utils.data import DataLoader
from torch.backends import cudnn
import argparse,os,time,json
from model import DOLPHIN
from dataset import Writing,collate_fn
from utils import create_logger,load_ckpt,l2_norm,fuse_all_conv_bn
import numpy as np
from evaluate import compute_metrics
import pickle
from natsort import natsorted
import matplotlib.pyplot as plt
from ptflops import get_model_complexity_info
from thop import profile
from torchstat import stat
# imported to load joblib files
from joblib import load
from tqdm.auto import tqdm

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',type=int,default=8)
parser.add_argument('--num_classes',type=int,default=1731)
parser.add_argument('--epoch',type=int,default=80)
parser.add_argument('--seed',type=int,default=123)
parser.add_argument('--cuda',type=bool,default=True)
parser.add_argument('--folder',type=str,default='./data/OLIWER')
parser.add_argument('--ngpu',type=int,default=1)
parser.add_argument('--gpu',type=str,default='0')
parser.add_argument('--weights',type=str,default='./weights/model.pth') # change 
parser.add_argument('--output_root',type=str,default='./output')
parser.add_argument('--log_root',type=str,default='./logs')
parser.add_argument('--dba',action='store_true')
parser.add_argument('--rerank',action='store_true')
parser.add_argument('--name',type=str,default='DOLPHIN')
opt = parser.parse_args()

# with open(f'{opt.weights}/settings.json','r',encoding='utf-8') as f:
#     settings = json.loads(f.read())
# opt.seed = settings['seed']
# opt.name = settings['name']
# opt.notes = settings['notes']
# opt.log_root = settings['log_root']
# # opt.folder = settings['folder']
# # opt.gpu = settings['gpu']

np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)

logger = create_logger(opt.log_root,name=opt.name,test=True)

# query_root = f'{opt.folder}/query-tf.pkl'
# with open(query_root,'rb') as f:
#     query_data = pickle.load(f,encoding='iso-8859-1')
gallery_root = f'{opt.folder}/test-tf.pkl'
# Error warining
#with open(gallery_root,'rb') as f:
#   gallery_data = pickle.load(f,encoding='iso-8859-1')
gallery_data = load(gallery_root)
# handwriting_info = {}
# min_sample = 10000
# for k in query_data:
#     handwriting_info[k] = query_data[k] + gallery_data[k]
#     min_sample = min(len(handwriting_info[k]),min_sample)
# print(min_sample,len(query_data),len(gallery_data))
gallery_dataset = Writing(gallery_data,train=False)
d_in = gallery_dataset.feature_dims

gallery_loader = DataLoader(gallery_dataset,batch_size=opt.batch_size,shuffle=False,collate_fn=collate_fn)

model = DOLPHIN(d_in,opt.num_classes)

if opt.cuda and torch.cuda.is_available():
    torch.cuda.set_device(int(opt.gpu))
    device = torch.device(f'cuda:{opt.gpu}')
else:
    device = torch.device('cpu')
model = model.to(device)

logger.info(f'\ngallery root: {gallery_root}\n'
    f'gallery loader length: {len(gallery_loader)} gallery features length: {len(gallery_dataset)}\n'
    f'model: {model.__class__.__name__}\nDBA & AQE: {opt.dba}\nRerank: {opt.rerank}')

def extract_features(model,data_loader,time_model):
    for i,(x,features_lens,user_labels) in enumerate(data_loader):
        x = torch.from_numpy(x).to(device)
        features_lens = torch.tensor(features_lens).long().to(device)
        user_labels = torch.from_numpy(user_labels).long()
        s = time.time()
        y_vector = model(x,features_lens)[0]
        # y_vector = model(x)[0]
        e = time.time()
        time_model += (e - s)
        y_vector = l2_norm(y_vector)
        if i == 0:
            features = torch.zeros(len(data_loader.dataset),y_vector.shape[1])
        start = i * opt.batch_size
        end = min((i + 1) * opt.batch_size,len(data_loader.dataset))
        features[start:end,:] = y_vector
        if i == 0:
            labels = user_labels
        else:
            labels = torch.cat([labels,user_labels],0)
    return features.cpu().numpy(),labels.cpu().numpy(),time_model

def transform_user2feat(features,labels):
    label_indices = natsorted(np.unique(labels))
    user2feat = {}
    for i in label_indices:
        pos = np.where(labels == i)[0]
        user2feat[i] = features[pos]
    return user2feat

@torch.no_grad()
def test_impl(model):
    model = model.eval()
    model = model.to(device)
    # model = fuse_all_conv_bn(model)
    
    time_elapsed_start = time.time()
    all_features,all_labels,time_model = extract_features(model,gallery_loader,0)
    user2feat = transform_user2feat(all_features,all_labels)
    repeat_times = 1
    logger.info(f'repeat times: {repeat_times}')
    gallery_labels,query_labels = [],[]
    for i in natsorted(np.unique(all_labels)):
        gallery_labels.extend([i] * (len(user2feat[i]) - 1))
        query_labels.append(i)
    gallery_labels = np.array(gallery_labels)
    query_labels = np.array(query_labels)
    aps,top1s,top5s,top10s = [],[],[],[]
    for _ in range(repeat_times):
        gallery_features,query_features = [],[]
        label_indices = natsorted(np.unique(all_labels))
        for i in label_indices:
            idx = np.random.choice(len(user2feat[i]),size=1)[0]
            gallery_features.append(user2feat[i][:idx])
            gallery_features.append(user2feat[i][idx + 1:])
            query_features.append(user2feat[i][idx])  
        gallery_features = np.concatenate(gallery_features)
        query_features = np.array(query_features)
        res = {
            'gallery_feature':gallery_features,'gallery_label':gallery_labels,
            'query_feature':query_features,'query_label':query_labels,
        }
        _,ap,top1,top5,top10 = compute_metrics(res,logger,opt.dba,device,verbose=False)
        aps.append(ap)
        top1s.append(top1)
        top5s.append(top5)
        top10s.append(top10)
    ap_mean,ap_std = np.mean(aps),np.std(aps)
    top1_mean,top1_std = np.mean(top1s),np.std(top1s)
    top5_mean,top5_std = np.mean(top5s),np.std(top5s)
    top10_mean,top10_std = np.mean(top10s),np.std(top10s)
    logger.info(f'[final] Rank@1: {top1_mean:.4f}% ({top1_std:.4f}%) Rank@5: {top5_mean:.4f}% ({top5_std:.4f}%) '
        f'Rank@10: {top10_mean:.4f}% ({top10_std:.4f}%)')
    logger.info(f'[final] mAP: {ap_mean * 100.:.4f}% ({ap_std * 100:.4f}%)')
    logger.info(f'time elapsed: {time.time() - time_elapsed_start:.5f}s\n')

def test():
    load_ckpt(model,opt.weights,device,logger,mode='test')    
    test_impl(model)

def main():
    test()

if __name__ == '__main__':
    main()
