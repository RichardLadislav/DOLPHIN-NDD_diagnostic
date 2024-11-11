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

class ButterWorthLPF: 
    # 巴特沃斯低通滤波器
    def __init__(self,order=3,half_pnt=15.0,fnyquist=100.0):
        '''
            scipy.signal.butter(N, Wn, btype='low', analog=False, output='ba', fs=None)
            这个是滤波器，高通低通带通
            N是滤波器阶数，Wn是3dB带宽点，dB就是radians/frequency(弧度/频率)
            3dB就是功率下降到二分之一的点，所以这里叫做half_pnt(point)
            byte决定是什么通，analog=False表示模拟滤波器，True表示数字滤波器
            'ba'表示输出分子和分母的系数;'zpk'表示输出零极点;'sos'表示输出second-order sections.
            默认是'ba'，意思应该是系统响应函数分子上的系数bk和分母上的系数ak
            https://wenku.baidu.com/view/adec241352d380eb62946d82.html
            order: 滤波器阶数
            half_pnt: 3dB点，功率降到一半的点
            fnyquist: sampling frequency，奈奎斯特采样率，采样频率必须要大于原始频率的两倍，书上叫fs
        '''
        fM = 0.5 * fnyquist # fM是原始频率，命名参考信号与系统第2版第七章
        half_pnt /= fM
        b,a = signal.butter(order,half_pnt,'low')
        self.b = b # 分子
        self.a = a # 分母

    def __call__(self,x): # x就是输入进来要滤波的数据
        return signal.filtfilt(self.b,self.a,x)
        # 将data通过零相位滤波器，零相位的意思就是输入和输出信号的相位完全相同，相移为0
        # 至于为什么是零相位滤波，暂时只能说经验值

lpf = ButterWorthLPF()

def difference(x): # 差分，两跨点之间相减
    '''
        numpy.convolve(a, v, mode='full')，a的长度是N，v是M
        mode可以取'full','same','valid',full的意思是长度为N+M-1，same的意思是长度为max(M,N)，
        valid的意思是长度为max(M,N) - min(M,N) + 1，只有完全重叠的点有效，
        边缘点无效（边缘点就是有一个序列突出去的那些）
        mode='same'的时候，比如[5,8,6,9,1,2]*[0.5,0,-0.5]，full的长度是8，然后两边减一个，
        刚刚好就是6了，而且[0.5,0,-0.5]是会反转的，[4., 0.5, 0.5, -2.5, -3.5, -0.5]
        delta_x[0] = delta_x[1]
        delta_x[-1] = delta_x[-2]
        这两句的作用就是将4代替为0.5，-0.5代替为-3.5，是因为那两个点没人减
    '''
    delta_x = np.convolve(x,[0.5,0,-0.5],mode='same')
    delta_x[0] = delta_x[1]
    delta_x[-1] = delta_x[-2]
    return delta_x

def difference_theta(x): # 输入的x是角度
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
        paths: 路径列表，第一维应该是点的个数
        features: 这个就是所有特征的列表，是单个feature append进去的
        num: 使用的信息个数，比如x,y,pressure...,2就是只用012前三个
        gpnoise: 不知道
        transform: 不知道
        use_finger: 是否是用手指写的
    '''
    for handwriting in handwritings:
        pressure = handwriting[:,num]
        handwriting = handwriting[:,0:num] # (x,y,pressure)
        handwriting[:,0] = lpf(handwriting[:,0])
        handwriting[:,1] = lpf(handwriting[:,1])
        delta_x = difference(handwriting[:,0])
        delta_y = difference(handwriting[:,1])
        v = np.sqrt(delta_x ** 2 + delta_y ** 2) # 速度
        theta = np.arctan2(delta_y,delta_x)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        delta_v = difference(v)
        delta_theta = np.abs(difference_theta(theta))
        log_curve_radius = np.log((v + 0.05) / (delta_theta + 0.05)) # log的曲线弧度       
        delta_v2 = np.abs(v * delta_theta)
        acceleration = np.sqrt(delta_v ** 2 + delta_v2 ** 2)

        # None在这里的作用是升维，比如说[2,2]会变成[2,1,2],concat起来就是[2,x,2]
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
    v = np.sqrt(delta_x ** 2 + delta_y ** 2) # 速度
    theta = np.arctan2(delta_y,delta_x)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    delta_v = difference(v)
    delta_theta = np.abs(difference_theta(theta))
    log_curve_radius = np.log((v + 0.05) / (delta_theta + 0.05)) # log的曲线弧度
    delta_v2 = np.abs(v * delta_theta)
    acceleration = np.sqrt(delta_v ** 2 + delta_v2 ** 2)
    delta_x2 = difference(delta_x)
    delta_y2 = difference(delta_y)
    # None在这里的作用是升维，比如说[2,2]会变成[2,1,2],concat起来就是[2,x,2]
    single_feature = np.concatenate((delta_x[:,None],delta_y[:,None],delta_x2[:,None],delta_y2[:,None],v[:,None],
        cos_theta[:,None],sin_theta[:,None],theta[:,None],log_curve_radius[:,None],
        acceleration[:,None],delta_v[:,None],delta_v2[:,None],delta_theta[:,None],
        pressure[:,None]),axis=1).astype(np.float32)
    
    single_feature[:,:-1] = (single_feature[:,:-1] - np.mean(single_feature[:,:-1],axis=0)) / np.std(single_feature[:,:-1],axis=0)
    return single_feature

def letterbox_image(img,target_h,target_w):
    img_h,img_w = img.shape
    scale = min(target_h / img_h,target_w / img_w) 
    # 长宽比目标size小的，可以变大，不过变大不一定有必要
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
    # pretrained=True是否基于其他任务的预训练
    state_dict = torch.load(pretrained_root,map_location=device)
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
    console_handler.setLevel(logging.INFO) # 分布式的等级
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
    # coord_idx其实是下标，就是说在handwriting这个二维数组里面是下标0和1分别是x和y
    assert len(coord_idx) == 2
    pos = handwriting[:,coord_idx]
    minx = np.min(pos,axis=0)
    maxn = np.max(pos,axis=0)
    pos = (pos - (maxn + minx) / 2.) / np.max(maxn - minx) # 不知道为什么这样除，经验值
    handwriting[:,coord_idx] = pos
    return handwriting

def norm_pressure(handwriting,pressure_idx=2): # 单纯变0到1，但是其实可以不用
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
            if not stack: # 空的
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