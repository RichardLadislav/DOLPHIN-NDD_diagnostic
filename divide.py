# -*- coding:utf-8 -*-

import os,pickle,json,argparse
import numpy as np
from utils import time_functions,clock

def divide_data(src_root='./data',tgt_root='./data/OLIWER'):
    if not os.path.exists(f'{tgt_root}/OLIWER.pkl'):
        print('Merging data.')
        with open(f'{src_root}/OLHWDB2/OLHWDB2.pkl','rb') as f:
            olhwdb2 = pickle.load(f,encoding='iso-8859-1')
        with open(f'{src_root}/DCOH-E/DCOH-E.pkl','rb') as f:
            dcohe = pickle.load(f,encoding='iso-8859-1')
        with open(f'{src_root}/COUCH09/COUCH09.pkl','rb') as f:
            couch = pickle.load(f,encoding='iso-8859-1')
        
        olhwdb_cnt,dcohe_cnt,couch_cnt = 0,0,0
        olhwdb_writer_cnt,dcohe_writer_cnt,couch_writer_cnt = 0,0,0
        data = {}
        for k in olhwdb2:
            if len(olhwdb2[k]) <= 20:
                continue
            data[str(k)] = olhwdb2[k]
            olhwdb_cnt += len(olhwdb2[k])
            olhwdb_writer_cnt += 1
        for k in couch:
            if len(couch[k]) <= 20:
                continue
            newk = f'couch{k}'
            data[newk] = couch[k]
            couch_cnt += len(couch[k])
            
            couch_writer_cnt += 1
        for k in dcohe:
            if len(dcohe[k]) <= 20:
                continue
            data[k] = dcohe[k]
            dcohe_cnt += len(dcohe[k])
            dcohe_writer_cnt += 1
        cnt = 0
        for k in data:
            cnt += len(data[k])
        print('user:',len(data),'sample:',cnt)
        print('dcohe samples:',dcohe_cnt,dcohe_writer_cnt)
        print('olhwdb2 samples:',olhwdb_cnt,olhwdb_writer_cnt)
        print('couch samples:',couch_cnt,couch_writer_cnt)
        os.makedirs(tgt_root,exist_ok=True)
        with open(f'{tgt_root}/OLIWER.pkl','wb') as f:
            pickle.dump(data,f)
    else:
        print('Loading existing data.')
        with open(f'{tgt_root}/OLIWER.pkl','rb') as f:
            data = pickle.load(f,encoding='iso-8859-1')
        print('user:',len(data),'sample:',np.sum([len(data[k]) for k in data.keys()]))
    
    if os.path.exists(f'./{tgt_root}/split.json'):
        with open(f'{tgt_root}/split.json') as f:
            split = json.load(f)
        train_writers = split['train_writers']
        test_writers = split['test_writers']
        print('Loading existing splits.')
    else:
        train_num = int(0.8 * len(data.keys()))
        train_writers = np.random.choice(list(data.keys()),size=train_num,replace=False)
        test_writers = list(set(list(data.keys())) - set(train_writers))
        split = {}
        split['train_writers'] = list(train_writers)
        split['test_writers'] = list(test_writers)
        with open(f'{tgt_root}/split.json','w',encoding='utf-8') as f:
            f.write(json.dumps(split,indent=4,ensure_ascii=False))
        print('Generating new splits.')
    
    train,test = {},{}
    for k in train_writers:
        train[k] = data[k]
    for k in test_writers:
        test[k] = data[k]
    
    with open(f'{tgt_root}/train.pkl','wb') as f:
        pickle.dump(train,f)
    with open(f'{tgt_root}/test.pkl','wb') as f:
        pickle.dump(test,f)

@clock
def extract_and_store(src_root='./data/OLIWER/train.pkl',tgt_root='./data/OLIWER/train-tf.pkl'):
    with open(src_root,'rb') as f:
        handwriting_info = pickle.load(f,encoding='iso-8859-1')
    writing = {}
    for i,k in enumerate(handwriting_info.keys()):
        writing[k] = []
        for each in handwriting_info[k]:
            writing[k].append(time_functions(each))
    with open(tgt_root,'wb') as f:
        pickle.dump(writing,f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--divide',action='store_true')
    parser.add_argument('--extract',action='store_true')
    opt = parser.parse_args()
    if opt.divide:
       divide_data('./data','./data/OLIWER')
    if opt.extract:
        extract_and_store('./data/OLIWER/train.pkl','./data/OLIWER/train-tf.pkl')
        extract_and_store('./data/OLIWER/test.pkl','./data/OLIWER/test-tf.pkl')