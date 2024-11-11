# -*- coding:utf-8 -*-

import numpy as np
from torch.utils.data import Dataset

class Writing(Dataset):
    def __init__(self,handwriting_info:dict,transform=None,train=True):
        super().__init__()
        self.users = handwriting_info.keys()
        self.users_cnt = len(self.users)
        self.train = train
        self.features = []
        self.user_labels = []
        for i,k in enumerate(self.users):
            # extract_features(handwriting_info[k],self.features)
            self.features.extend(handwriting_info[k])
            self.user_labels.extend([i] * len(handwriting_info[k]))
        assert len(self.user_labels) == len(self.features)
        self.features_cnt = len(self.features)
        self.feature_dims = np.shape(self.features[0])[1] # 就是时间函数的数量，这里是12个
        self.transform = transform

    def __len__(self):
        return self.features_cnt

    def __getitem__(self,idx):
        if self.train:
            feature = self.features[idx]
            if self.transform is not None:
                feature = self.transform(feature)
        else:
            feature = self.features[idx]
        return feature,len(feature),self.user_labels[idx]
   
def collate_fn(batch:list):
    batch_size = len(batch)
    handwriting = [i[0] for i in batch]
    hw_len = np.array([i[1] for i in batch],dtype=np.float32)
    user_labels = np.array([i[2] for i in batch])
    max_len = int(np.max(hw_len))
    time_function_cnts = np.shape(handwriting[0])[1]
    handwriting_padded = np.zeros((batch_size,max_len,time_function_cnts),dtype=np.float32)
    for i,hw in enumerate(handwriting):
        handwriting_padded[i,:hw.shape[0]] = hw
    return handwriting_padded,hw_len,user_labels

if __name__ == '__main__':
    ...