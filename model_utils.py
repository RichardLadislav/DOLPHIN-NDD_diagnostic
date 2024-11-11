# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import trunc_normal_
from pytorch_wavelets import DWT1D,IDWT1D

class SelectivePool1d(nn.Module):
    def __init__(self,in_features,d_head,num_heads):
        super().__init__()
        self.keys = nn.Parameter(torch.Tensor(num_heads,d_head),requires_grad=True)
        self.W_q = nn.Conv1d(in_features,d_head * num_heads,kernel_size=1)
        self.norm = 1 / np.sqrt(d_head)
        self.d_head = d_head
        self.num_heads = num_heads
        self.weights_init()

    def weights_init(self):
        nn.init.orthogonal_(self.keys,gain=1)
        nn.init.kaiming_normal_(self.W_q.weight,a=1)
        nn.init.zeros_(self.W_q.bias)

    def orthogonal_norm(self):
        keys = F.normalize(self.keys,dim=1)
        corr = torch.mm(keys,keys.transpose(0,1))
        return torch.sum(torch.triu(corr,1).abs_())

    def forward(self,x,mask):
        N,_,L = x.shape # (N,C,L)
        q = v = self.W_q(x).transpose(1,2).view(N,L,self.num_heads,self.d_head)
        if mask is not None:
            mask = mask.to(x.device)
            attn = F.softmax(torch.sum(q * self.keys,dim=-1) * self.norm - (1. - mask).unsqueeze(2) * 1000,dim=1) 
            # (N,L,num_heads)
        else:
            attn = F.softmax(torch.sum(q * self.keys,dim=-1) * self.norm,dim=1)
        y = torch.sum(v * attn.unsqueeze(3),dim=1).view(N,-1) # (N,d_head * num_heads)
        return y
   
def get_len_mask(features_lens): # mask需要更新，因为长度不一样
    features_lens = features_lens
    batch_size = len(features_lens)
    max_len = torch.max(features_lens)
    mask = torch.zeros((batch_size,max_len),dtype=torch.float32)
    for i in range(batch_size):
        mask[i,0:features_lens[i]] = 1.0
    return mask

class Swish(nn.Module):
    def forward(self,x):
        return x * torch.sigmoid(x)

class SwishImpl(torch.autograd.Function):
    @staticmethod
    def forward(ctx,i):
        res = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return res

    @staticmethod
    def backward(ctx,y_grad):
        i = ctx.saved_tensors[0]
        x_sigmoid = torch.sigmoid(i)
        return y_grad * (x_sigmoid * (1 + i * (1 - x_sigmoid)))
    
class MemoryEfficientSwish(nn.Module):
    def forward(self,x):
        return SwishImpl.apply(x)
    
class SEBlock2(nn.Module): # 通道不一样
    def __init__(self,d_in,d_hidden,act_layer=Swish): # Swish或SiLU
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(d_in,d_hidden,kernel_size=1,padding=0,stride=1),
            act_layer(),
            nn.Conv1d(d_hidden,d_in,kernel_size=1,padding=0,stride=1),
            nn.Sigmoid())

    def forward(self,x): # x: (n,c,l)
        y = self.fc(x)
        return x * y.expand_as(x)

def compute_similarity(query,gallery):
    query = query / np.linalg.norm(query,axis=1,keepdims=True)
    gallery = gallery / np.linalg.norm(gallery,axis=1,keepdims=True)
    return np.matmul(query,gallery.T)

def db_augmentation(query,gallery,topk=10):
    # DBA: Database-side feature augmentation https://link.springer.com/article/10.1007/s11263-017-1016-8
    weights = np.logspace(0,-2.,topk + 1)

    # query augmentation
    similarity = compute_similarity(query,gallery)
    indices = np.argsort(-similarity,axis=1)
    topk_gallery = gallery[indices[:,:topk],:]
    query = np.tensordot(weights,np.concatenate([query[:,None],topk_gallery],axis=1),axes=(0,1))

    # gallery augmentation
    similarity = compute_similarity(gallery,gallery)
    indices = np.argsort(-similarity,axis=1)
    topk_gallery = gallery[indices[:,:topk + 1],:]
    gallery = np.tensordot(weights,topk_gallery,axes=(0,1))
    return query,gallery

def average_query_expansion(query,gallery,topk=5):
    similarity = compute_similarity(query,gallery)
    indices = np.argsort(-similarity,axis=1)
    topk_gallery = np.mean(gallery[indices[:,:topk],:],axis=1)
    query = np.concatenate([query,topk_gallery],axis=1)

    similarity = compute_similarity(gallery,gallery)
    indices = np.argsort(-similarity,axis=1)
    topk_gallery = np.mean(gallery[indices[:,1:topk + 1],:],axis=1)
    gallery = np.concatenate([gallery,topk_gallery],axis=1)
    return query,gallery
    def __init__(self,d_feat,l):
        super().__init__()
        self.conv_depth = nn.Conv1d(d_feat,d_feat,kernel_size=3,padding=1,bias=False,groups=d_feat // 2)
        self.complex_weight = nn.Parameter(torch.randn(d_feat,l,2,dtype=torch.float32) * 0.02)
        trunc_normal_(self.complex_weight,std=.02)
        self.head = nn.Linear(d_feat,d_feat,bias=True)

    def forward(self,x):
        x1 = x[:,:,0::2]
        x2 = x[:,:,1::2]
        x1 = self.conv_depth(x1)
        _,_,l = x2.shape
        x2 = torch.fft.rfft(x2,dim=2,norm='ortho')
        weight = self.complex_weight
        if not weight.shape[1:2] == x2.shape[2:3]:
            weight = F.interpolate(weight.permute(2,0,1).unsqueeze(2),size=(1,x2.shape[2]),mode='bilinear',align_corners=True).squeeze().permute(1,2,0)
        weight = torch.view_as_complex(weight.contiguous())
        x2 *= weight
        x2 = torch.fft.irfft(x2,n=l,dim=2,norm='ortho')
        y = x1 + x2
        y = self.head(y.transpose(1,2)).transpose(1,2)
        return y

def channel_shuffle(x,groups):
    n,c,l = x.shape
    d_hidden = c // groups
    x = x.view(n,groups,d_hidden,l)
    x = x.transpose(1,2).contiguous()
    x = x.view(n,-1,l)
    return x

class ShuffleBlock(nn.Module): # 这就是ShufflNetV2的简化实现，本来就这样的，里面没有groups的
    def __init__(self,d_in,kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(d_in // 2,d_in // 2,kernel_size=kernel_size,padding=kernel_size // 2,stride=1),
            nn.BatchNorm1d(d_in // 2),
            nn.Conv1d(d_in // 2,d_in // 2,kernel_size=1,stride=1,padding=0),
            nn.SELU(True)
        )
    
    def forward(self,x):
        x1,x2 = x.chunk(2,dim=1)
        y = torch.cat((x1,self.conv(x2)),dim=1)
        return channel_shuffle(y,2)

class CBA(nn.Module):
    def __init__(self,d_in,d_out,kernel_size,stride=1,groups=1,bias=True,skip=False,act_layer=nn.ReLU):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(d_in,d_out,kernel_size=kernel_size,stride=stride,padding=padding,groups=groups,bias=bias)
        self.bn = nn.BatchNorm1d(d_out)
        # self.bn = GhostBatchNorm1d(d_out)
        self.relu = act_layer(True)
        self.skip = skip and (stride == 1) and (d_in == d_out)
    
    def forward(self,x):
        identity = x
        y = self.relu(self.bn(self.conv(x)))
        if self.skip:
            y = y + identity
        return y

class DepthwiseSeparableConv(nn.Module):
    def __init__(self,d_in,d_out,dw_kernel_size=3,stride=1,skip=True,se_rate=0.2,drop_path_rate=0.,group_size=1,):
        super().__init__()
        groups = d_in // group_size
        padding = dw_kernel_size // 2
        self.has_skip = (stride == 1 and d_in == d_out) and skip
        self.dw_conv = nn.Conv1d(d_in,d_in,dw_kernel_size,stride=stride,padding=padding,groups=groups)
        self.bn1 = nn.BatchNorm1d(d_in)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock2(d_in,int(d_in * se_rate),act_layer=nn.SELU) if se_rate else nn.Identity()
        self.pw_conv = nn.Conv1d(d_in,d_out,1,padding=0)
        self.bn2 = nn.BatchNorm1d(d_out)
        self.drop_path = nn.Dropout(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self,x):
        identity = x
        x = self.relu(self.bn1(self.dw_conv(x)))
        x = self.se(x)
        x = self.relu(self.bn2(self.pw_conv(x)))
        if self.has_skip:
            x = self.drop_path(x) + identity
        return x
  
class CAIR(nn.Module):
    def __init__(self,d_in,d_out,expand,kernel_size,stride,skip,se_rate,drop_path_rate):
        super().__init__()
        d_mid = d_in // 2 * expand
        self.expand_conv = CBA(d_in // 2,d_mid,kernel_size=1,bias=False) if expand != 1 else nn.Identity()
        self.dw_conv = CBA(d_mid,d_mid,kernel_size=kernel_size,stride=stride,groups=d_mid,bias=False)
        self.project_conv = nn.Sequential(
            nn.Conv1d(d_mid,d_out // 2,kernel_size=1,stride=1,bias=False),
            nn.SELU(True)
        )
        self.identity_conv = CBA(d_in // 2,d_out // 2,3,stride=2,groups=1,bias=False) if stride == 2 else \
            (nn.Conv1d(d_in // 2,d_out // 2,1) if d_in != d_out else nn.Identity())
        self.se = SEBlock2(d_mid,int(d_mid * se_rate),act_layer=nn.SELU) if se_rate > 0. else nn.Identity()
        self.post_conv = CBA(d_out,d_out,3,1,1,act_layer=nn.SELU)
        self.skip = (stride == 1 and d_in == d_out) and skip
        self.drop_path = nn.Dropout(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
    
    def forward(self,x):
        identity = x.clone()
        x1,x2 = x.chunk(2,dim=1)
        expand = self.expand_conv(x1)
        y1 = self.dw_conv(expand)
        y1 = self.se(y1)
        y1 = self.project_conv(y1)
        y2 = self.identity_conv(x2)
        y = torch.cat((y1,y2),dim=1)      
        y = channel_shuffle(y,2)
        y = self.post_conv(y)
        y = self.drop_path(y)
        return y

class HFGA(nn.Module):
    def __init__(self,d_feat):
        super().__init__()
        self.dwt = DWT1D(J=1,wave='bior1.1',mode='symmetric')
        self.attn_gate = nn.Parameter(torch.Tensor([0.0]))
        self.to_q = nn.Conv1d(d_feat,d_feat,1)
        self.to_k = nn.Conv1d(d_feat,d_feat,1)
        self.to_v = nn.Conv1d(d_feat,d_feat,1)
        # self.to_out = nn.Conv1d(d_feat,d_feat,1)
    
    def compute_attn_matmul(self,q,k,v): # k和v是一样的
        # q:(n,c,l1), k & v: (n,c,l2), l1比l2长
        attn = k.transpose(1,2) @ q / np.sqrt(q.shape[1]) # (n,l2,l1)
        attn = attn - attn.amax(dim=1,keepdim=True).detach()
        attn = F.softmax(attn,dim=1)
        y = v @ attn # (n,c,l1)
        return y

    def forward(self,x):
        xl,xh = self.dwt(x)
        xh = xh[0]
        q = self.to_q(x)
        k = self.to_k(xh)
        v = self.to_v(xh)
        yh = self.compute_attn_matmul(q,k,v)
        y = yh * self.attn_gate.tanh() + x
        return y

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape,)
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
            return x
    
class ContextBlock(nn.Module):
    def __init__(self,d_in,d_hidden,pooling='attn',fusions=['channel_add']):
        super().__init__()
        self.pooling = pooling
        self.conv_mask = nn.Conv1d(d_in,1,kernel_size=1) if pooling == 'attn' else nn.AdaptiveAvgPool1d(1)
        if 'channel_add' in fusions:
            self.channel_add_conv = nn.Sequential(
                nn.Conv1d(d_in,d_hidden,1),
                nn.LayerNorm([d_hidden,1]),
                nn.ReLU(True),
                nn.Conv1d(d_hidden,d_in,1)
            )
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusions:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv1d(d_in,d_hidden,1),
                nn.LayerNorm([d_hidden,1]),
                nn.ReLU(True),
                nn.Conv1d(d_hidden,d_in,1)
            )
        else:
            self.channel_mul_conv = None
        self.weights_init()
    
    def weights_init(self):
        if self.pooling == 'attn':
            nn.init.kaiming_normal_(self.conv_mask.weight,a=0,mode='fan_in',nonlinearity='relu')
            if hasattr(self.conv_mask, 'bias') and self.conv_mask.bias is not None:
                nn.init.zeros_(self.conv_mask.bias)
            self.conv_mask.inited = True
        if self.channel_add_conv is not None:
            self.last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            self.last_zero_init(self.channel_mul_conv)
    
    def last_zero_init(self,m):
        if isinstance(m,nn.Sequential):
            nn.init.zeros_(m[-1].weight)
            if hasattr(m[-1],'bias') and m[-1].bias is not None:
                nn.init.zeros_(m[-1].bias)
        else:
            nn.init.zeros_(m.weight)
            if hasattr(m,'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)

    def spatial_pool(self,x):
        if self.pooling == 'attn':
            context_mask = self.conv_mask(x) # (n,1,l)
            context_mask = F.softmax(context_mask,dim=2) # 对l维softmax
            context_mask = context_mask.squeeze().unsqueeze(-1)
            context = torch.matmul(x,context_mask) # (n,c,l) * (n,l,1) = (n,c,1)
        else:
            context = self.conv_mask(x)
        return context
    
    def forward(self,x):
        context = self.spatial_pool(x) # (n,c,1)
        if self.channel_add_conv is not None:
            channel_add = self.channel_add_conv(context)
            x = x + channel_add
        if self.channel_mul_conv is not None:
            weights = torch.sigmoid(self.channel_mul_conv(context))
            x = x * weights
        return x

def main():
    ...

if __name__ == '__main__':
    main()