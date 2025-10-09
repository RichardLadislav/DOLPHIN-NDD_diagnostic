# -*- coding:utf-8 -*-

from scipy import signal
from torch.nn.utils import fuse_conv_bn_eval
import numpy as np
import cv2,logging,sys,time,os
from functools import wraps,lru_cache,reduce
from termcolor import colored
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from joblib import dump,load

class ButterWorthLPF: 
    # ButterWorth filter
    def __init__(self,order=3,half_pnt=15.0,fnyquist=100.0):
        '''
            scipy.signal.butter(N, Wn, btype='low', analog=False, output='ba', fs=None)
            This is a filter function, supporting high-pass, low-pass, and band-pass filters.
            N is the filter order, Wn is the 3dB cutoff point (in radians/frequency).
            3dB refers to the point where the power drops to half, hence called half_pnt (half point).
            btype determines the filter type, analog=False means digital filter, True means analog filter.
            'ba' means the output is the numerator and denominator coefficients; 'zpk' means output zeros, poles, and gain; 'sos' means output second-order sections.
            The default is 'ba', which means the system response function's numerator coefficients (bk) and denominator coefficients (ak).
            https://wenku.baidu.com/view/adec241352d380eb62946d82.html
            order: filter order
            half_pnt: 3dB point, where the power drops to half
            fnyquist: sampling frequency, Nyquist frequency; the sampling frequency must be greater than twice the original frequency, called fs in textbooks
        '''
        fM = 0.5 * fnyquist # fM is the original frequency, naming reference: Signals and Systems, 2nd Edition, Chapter 7
        half_pnt /= fM
        b,a = signal.butter(order,half_pnt,'low')
        self.b = b # numerator
        self.a = a # denominator

    # x is the input data to be filtered
    def __call__(self,x):
        return signal.filtfilt(self.b,self.a,x)
        # Pass data through a zero-phase filter; zero-phase means the input and output signals have exactly the same phase, i.e., zero phase shift
        # As for why zero-phase filtering is used, it can only be said to be based on experience for now

lpf = ButterWorthLPF()

def difference(x): # Difference, subtract between two adjacent points
    '''
        numpy.convolve(a, v, mode='full'): a has length N, v has length M.
        The mode can be 'full', 'same', or 'valid'. 'full' means the output length is N+M-1, 
        'same' means the output length is max(M, N), and 'valid' means the output length is 
        max(M, N) - min(M, N) + 1, where only the completely overlapping points are valid, 
        and edge points are invalid (edge points are those where one sequence extends beyond the other).
        When mode='same', for example [5,8,6,9,1,2]*[0.5,0,-0.5], the 'full' length is 8, 
        then subtract one from each end, resulting in 6, and [0.5,0,-0.5] will be reversed, 
        giving [4., 0.5, 0.5, -2.5, -3.5, -0.5].
        delta_x[0] = delta_x[1]
        delta_x[-1] = delta_x[-2]
        The purpose of these two lines is to replace 4 with 0.5 and -0.5 with -3.5, 
        because those two points have nothing to subtract from.
    '''
    delta_x = np.convolve(x,[0.5,0,-0.5],mode='same')
    delta_x[0] = delta_x[1]
    delta_x[-1] = delta_x[-2]
    return delta_x
# The input x is an angle
def difference_theta(x): 
    delta_x = np.zeros_like(x)
    delta_x[1:-1] = x[2:] - x[:-2]
    delta_x[-1] = delta_x[-2]
    delta_x[0] = delta_x[1]
    t = np.where(np.abs(delta_x) > np.pi)
    delta_x[t] = np.sign(delta_x[t]) * 2 * np.pi
    delta_x *= 0.5
    return delta_x

def extract_features(handwritings,features,gpnoise=None,num=2,transform=False):
    '''
        paths: list of paths, the first dimension should be the number of points
        features: this is the list of all features, each single feature is appended to it
        num: number of information columns to use, e.g., x, y, pressure... 2 means only use the first three columns (0,1,2)
        gpnoise: unknown
        transform: unknown
        use_finger: whether it is written with a finger
    '''
    for handwriting in handwritings:
        pressure = handwriting[:,num]
        handwriting = handwriting[:,0:num] # (x,y,pressure)
        handwriting[:,0] = lpf(handwriting[:,0])
        handwriting[:,1] = lpf(handwriting[:,1])
        delta_x = difference(handwriting[:,0])
        delta_y = difference(handwriting[:,1])
        v = np.sqrt(delta_x ** 2 + delta_y ** 2) # velocity
        theta = np.arctan2(delta_y,delta_x)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        delta_v = difference(v)
        delta_theta = np.abs(difference_theta(theta))
        log_curve_radius = np.log((v + 0.05) / (delta_theta + 0.05)) # log curve radius
        delta_v2 = np.abs(v * delta_theta)
        acceleration = np.sqrt(delta_v ** 2 + delta_v2 ** 2)

        # None here is used for dimension expansion, for example [2,2] becomes [2,1,2]; concatenating them gives [2,x,2]
        single_feature = np.concatenate((delta_x[:,None],delta_y[:,None],v[:,None],
            cos_theta[:,None],sin_theta[:,None],theta[:,None],log_curve_radius[:,None],
            acceleration[:,None],delta_v[:,None],delta_v2[:,None],delta_theta[:,None],
            pressure[:,None]),axis=1).astype(np.float32)
        single_feature[:,:-1] = (single_feature[:,:-1] - np.mean(single_feature[:,:-1],axis=0)) / \
            np.std(single_feature[:,:-1],axis=0)
        # single_feature[:,:-1] = regression_based_norm(single_feature[:,:-1])
        features.append(single_feature)

def time_functions(handwriting,num=2):
    # handwriting = deepcopy(handwriting_org)
    pressure = handwriting[:,num]
    # pressure = np.ones_like(pressure)
    handwriting = handwriting[:,0:num] # (x,y,pressure)
    handwriting[:,0] = lpf(handwriting[:,0])
    handwriting[:,1] = lpf(handwriting[:,1])
    delta_x = difference(handwriting[:,0])
    delta_y = difference(handwriting[:,1])
    v = np.sqrt(delta_x ** 2 + delta_y ** 2) # velocity
    theta = np.arctan2(delta_y,delta_x)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    delta_v = difference(v)
    delta_theta = np.abs(difference_theta(theta))
    log_curve_radius = np.log((v + 0.05) / (delta_theta + 0.05)) # logaritmic curvate radius
    delta_v2 = np.abs(v * delta_theta)
    acceleration = np.sqrt(delta_v ** 2 + delta_v2 ** 2)
    delta_x2 = difference(delta_x)
    delta_y2 = difference(delta_y)
    # None here is used for dimension expansion, for example [2,2] becomes [2,1,2]; concatenating them gives [2,x,2]
    single_feature = np.concatenate((delta_x[:,None],delta_y[:,None],delta_x2[:,None],delta_y2[:,None],v[:,None],
        cos_theta[:,None],sin_theta[:,None],theta[:,None],log_curve_radius[:,None],
        acceleration[:,None],delta_v[:,None],delta_v2[:,None],delta_theta[:,None],
        pressure[:,None]),axis=1).astype(np.float32)
    
    single_feature[:,:-1] = (single_feature[:,:-1] - np.mean(single_feature[:,:-1],axis=0)) / np.std(single_feature[:,:-1],axis=0)
    return single_feature

def letterbox_image(img,target_h,target_w):
    img_h,img_w = img.shape
    scale = min(target_h / img_h,target_w / img_w) 
    # For targets smaller than original aspect ratio, scaling up is possible, though not always necessary
    new_w,new_h = int(img_w * scale),int(img_h * scale) # 这样做就依然保持了长宽比
    img = cv2.resize(img,(new_w,new_h),interpolation=cv2.INTER_AREA)
    new_img = np.ones((target_h,target_w),dtype=np.uint8) * 255
    up = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    new_img[up:up + new_h,left:left + new_w] = img
    return new_img

def interpolate_torch(org_info,interp_ratio):
    l = len(org_info)
    org_info = torch.tensor(org_info).view(1,1,l,-1)
    new_info = F.interpolate(org_info,size=(l * interp_ratio,3),mode='bicubic').squeeze().numpy()
    return new_info

def load_ckpt(model,pretrained_root,device,logger,optimizer=None,scheduler=None,mode='train',resume=False): 
    # pretrained=True means using pretrained weights from another task
    state_dict = torch.load(pretrained_root,map_location=device)
    #state_dict = load(pretrained_root) 
    if mode == 'train':
        if resume:
            optimizer.load_state_dict(state_dict['optimizer'])
            scheduler.load_state_dict(state_dict['lr_scheduler'])
            print(model.load_state_dict(state_dict['model']))
            start_epoch = state_dict['epoch'] + 1
            logger.info(f'mode: "{mode} + resume" {pretrained_root} successfully loaded.')
        else:
            state_dict = state_dict['model'] if 'model' in state_dict else state_dict
            state_dict = {k:v for k,v in state_dict.items() if k in model.state_dict().keys() and v.numel() == model.state_dict()[k].numel()}
            print(model.load_state_dict(state_dict,strict=False))
            logger.info(f'mode: "{mode} + pretrained" {pretrained_root} successfully loaded.')
            start_epoch = 0
        return start_epoch
    else:
        state_dict = state_dict['model'] if 'model' in state_dict else state_dict
        state_dict = {k:v for k,v in state_dict.items() if k in model.state_dict().keys() and v.numel() == model.state_dict()[k].numel()}
        print(model.load_state_dict(state_dict,strict=False))
        # model.load_state_dict(state_dict['model'])
        logger.info(f'mode: "{mode}" {pretrained_root} successfully loaded.')

@lru_cache()
def create_logger(log_root,name='',test=False):
    os.makedirs(f'{log_root}',exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    color_fmt = colored('[%(asctime)s %(name)s]','green') + \
        colored('(%(filename)s %(lineno)d)','yellow') + ': %(levelname)s %(message)s'
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO) # Logging level for distributed mode
    console_handler.setFormatter(logging.Formatter(fmt=color_fmt,datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console_handler)

    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    date = time.strftime('%Y-%m-%d') if not test else time.strftime('%Y-%m-%d') + '-test'
    file_handler = logging.FileHandler(f'{log_root}/log-{date}.txt',mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(fmt=fmt,datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)
    return logger

def l2_norm(x): # x:(batch_size,seq_len)
    org_size = x.size()
    x_pow = torch.pow(x,2)
    x_pow = torch.sum(x_pow,1).add_(1e-6)
    x_pow = torch.sqrt(x_pow)
    y = torch.div(x,x_pow.view(-1,1).expand_as(x)).view(org_size)
    return y
    
def centernorm_size(handwriting,coord_idx=[0,1]):
    # coord_idx is actually the index position, meaning in the 2D handwriting array, indices 0 and 1 correspond to x and y
    assert len(coord_idx) == 2
    pos = handwriting[:,coord_idx]
    minx = np.min(pos,axis=0)
    maxn = np.max(pos,axis=0)
    pos = (pos - (maxn + minx) / 2.) / np.max(maxn - minx) # Not sure, why divide liek this, empirical value
    handwriting[:,coord_idx] = pos
    return handwriting

def norm_pressure(handwriting,pressure_idx=2): # Simply scales 0–1, though not strictly necessary
    pressure = handwriting[:,pressure_idx]
    maxn = np.max(pressure)
    pressure /= maxn
    handwriting[:,pressure_idx] = pressure
    return handwriting

def fuse_all_conv_bn(model):
    stack = []
    for name,module in model.named_children():
        if list(module.named_children()):
            fuse_all_conv_bn(module)
        if isinstance(module,nn.BatchNorm1d):
            if not stack: # Empty
                continue
            if isinstance(stack[-1][1],nn.Conv1d):
                setattr(model,stack[-1][0],fuse_conv_bn_eval(stack[-1][1],module))
                setattr(model,name,nn.Identity())
        else:
            stack.append((name,module))
    return model
  
def extract_and_store(src_root='./data/OCNOLHW-granular/train.pkl'):
    import pickle
    with open(src_root,'rb') as f:
        handwriting_info = pickle.load(f,encoding='iso-8859-1')
    writing = {}
    print(len(handwriting_info))
    for i,k in enumerate(handwriting_info.keys()):
        writing[k] = []
        a = time.time()
        for each in handwriting_info[k]:
            writing[k].append(time_functions(each))
        print(time.time() - a)
        break

def clock(func):
    @wraps(func)
    def impl(*args,**kwargs):
        start = time.perf_counter()
        res = func(*args,**kwargs)
        end = time.perf_counter()
        args_list = []
        if args:
            args_list.extend([repr(arg) for arg in args])
        if kwargs:
            args_list.extend([f'{key}={value}' for key,value in kwargs.items()])
        args_str = ','.join(i for i in args_list)
        print(f'[executed in {(end - start):.5f}s, '
            f'{func.__name__}({args_str}) -> {res}]')
        return res
    return impl

def main():
    ...

if __name__ == '__main__':
    main()