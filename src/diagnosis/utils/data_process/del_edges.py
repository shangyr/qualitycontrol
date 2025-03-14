#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   del_edges.py
@Author  :   lixin 
@Version :   1.0
@Desc    :   None
'''
import os
from tqdm import tqdm

data_path = '/home/lixin/work/prompt-gnn/data/traditional_chinese_medicine'

expand_path = os.path.join(data_path, 'triples_expand.txt')
label_path = os.path.join(data_path, 'label.txt')
labels = []
with open(label_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        labels.append(line)

expand_triples = []
with open(expand_path, 'r',encoding='utf-8') as f:
    for line in tqdm(f):
        line = line.strip()
        if not line: continue
        head_entity = line.split('\t')[0]
        tail_entity = line.split('\t')[2]
        need_del = False
        for label1 in labels:
            for label2 in labels:
                if label1 == label2: continue
                if label1 in head_entity and label2 in tail_entity:
                    need_del = True
                    break
            if need_del: break
        if not need_del:
            expand_triples.append(line)

with open(expand_path,'w',encoding='utf-8') as f:
    f.write('\n'.join(expand_triples))