# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import DepthwiseSeparableConv,CAIR,HFGA,ContextBlock,SelectivePool1d,get_len_mask

class Head(nn.Module):
    def __init__(self,d_in,d_hidden,num_classes,bias=True):
        super().__init__()
        self.fc1 = nn.Linear(d_in,d_hidden,bias=bias)
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(d_hidden,num_classes,bias=bias)
    
    def forward(self,x):
        x = self.dropout(self.fc1(x))
        y = self.head(x)
        return x,y
# Code proposed by ChatGPT
#class DiagnosisHead(nn.Module):
#    """
#    Clinical head that takes [y_vector (temporal), f3 (frequency)] concatenated.
#    Input dim = 384 + 384 = 768 in your current setup.
 #   """
#    def __init__(self, in_dim: int, num_classes: int, p_drop: float = 0.1, bias: bool = False):
#        super().__init__()
#        hidden = max(128, in_dim // 2)  # 384 here
#        self.net = nn.Sequential(
#            nn.LayerNorm(in_dim),
#            nn.Linear(in_dim, hidden, bias=bias),
#            nn.SiLU(), # nn.GELU() proposed by ChatGPT but original authors used SiLU
#            nn.Dropout(p_drop),
#            nn.Linear(hidden, num_classes, bias=bias)
#        )

#    def forward(self, z: torch.Tensor):
#        return self.net(z)  # logits [B, num_classes] (or [B,1] for binary)

class DOLPHIN(nn.Module):
    def __init__(self,d_in,num_classes):
        super().__init__()
        self.conv = DepthwiseSeparableConv(d_in,64,7,skip=False,se_rate=0.,drop_path_rate=0.)
        self.block0 = nn.Sequential(
            CAIR(64,96,1,kernel_size=5,stride=2,skip=False,se_rate=0.25,drop_path_rate=0.1),
            CAIR(96,128,3,kernel_size=3,stride=1,skip=False,se_rate=0.25,drop_path_rate=0.1)
        )
        self.block1 = nn.Sequential(
            CAIR(128,160,1,kernel_size=5,stride=2,skip=False,se_rate=0.25,drop_path_rate=0.1),
            CAIR(160,192,3,kernel_size=3,stride=1,skip=False,se_rate=0.25,drop_path_rate=0.1)
        )
        self.block2 = nn.Sequential(
            CAIR(192,224,1,kernel_size=5,stride=2,skip=False,se_rate=0.25,drop_path_rate=0.1),
            CAIR(224,256,3,kernel_size=3,stride=1,skip=False,se_rate=0.25,drop_path_rate=0.1),
        )
        self.freq1 = HFGA(128)
        self.freq_proj1 = DepthwiseSeparableConv(128,192,stride=2,skip=False,se_rate=0.,drop_path_rate=0.)
        self.freq2 = HFGA(192)
        self.freq_proj2 = DepthwiseSeparableConv(192,256,stride=2,skip=False,se_rate=0.,drop_path_rate=0.)
        self.freq3 = HFGA(256)
        self.up = nn.Upsample(scale_factor=2)
        self.conv_proj1 = nn.Conv1d(192,256,1,padding=0)
        self.conv_proj2 = nn.Conv1d(128,256,1,padding=0)
        self.context1 = ContextBlock(256,256 // 8)
        self.context2 = ContextBlock(256,256 // 8)
        self.context3 = ContextBlock(256,256 // 8)
        self.context4 = ContextBlock(256,256 // 8)
        self.head = Head(384 * 2,384,num_classes,bias=False)
        self.sel_pool1 = SelectivePool1d(256,d_head=24,num_heads=16)
        self.sel_pool2 = SelectivePool1d(256,d_head=24,num_heads=16)
        self.sel_pool3 = SelectivePool1d(256,d_head=24,num_heads=16)
        self.freq_head = nn.Sequential(
            nn.Linear(384,384,bias=False),
            nn.Dropout(0.1),
        )
        self.weights_init()

    def forward(self,x,feature_lens):
        x = x.transpose(1,2)
        x = self.conv(x)
        y0 = self.block0(x) # (n,128,l/2)
        freq1 = self.freq1(y0)

        y1 = y0 + freq1
        y1 = self.block1(y1) # (n,192,l/4)
        freq1 = self.freq_proj1(freq1)
        freq1 = freq1 + y1
        freq2 = self.freq2(freq1)

        y2 = y1 + freq2
        y2 = self.block2(y2) # (n,256,l/8)
        freq2 = self.freq_proj2(freq2)
        freq2 = freq2 + y2
        freq3 = self.freq3(freq2)

        y3 = self.context1(self.up(y2)[:,:,:y1.shape[2]]) + self.context2(self.conv_proj1(y1)) # (n,320,l/8)
        y4 = self.context3(self.up(y3)[:,:,:y0.shape[2]]) + self.context4(self.conv_proj2(y0)) # (n,320,l/4)
        y3 = F.selu(y3,inplace=True)
        y4 = F.selu(y4,inplace=True)

        feature_lens = torch.div(feature_lens + 1,2,rounding_mode='trunc')
        mask = get_len_mask(feature_lens)
        f1 = self.sel_pool1(y4,mask)
        mask = F.max_pool1d(mask,2,ceil_mode=True)
        f2 = self.sel_pool2(y3,mask)
        f3 = self.sel_pool3(freq3,None)
        y_vector = torch.cat([f1,f2],dim=1)
        y_vector,y_prob = self.head(y_vector)
        f3 = self.freq_head(f3)
        return y_vector,y_prob,f3
    
    def weights_init(self):
        for m in self.modules():
            if isinstance(m,nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',a=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight,a=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)