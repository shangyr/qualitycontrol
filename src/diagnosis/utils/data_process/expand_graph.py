# -*- coding:utf-8 -*-

"""
利用bert计算实体相似度，向知识图谱中添加相似症状描述
"""

from transformers import AutoModel,AutoTokenizer
import torch
import numpy as np
import os
from tqdm import tqdm

data_path = '/home/lixin/work/prompt-gnn/data/traditional_chinese_medicine'
bert_path = '/home/lixin/work/prompt-gnn/PLM/Smed-bert'
label_file = os.path.join(data_path, 'label.txt')
entity_file = os.path.join(data_path, 'entities.txt')
triples_file = os.path.join(data_path, 'triples.txt')
out_triples_file = os.path.join(data_path, 'triples_expand.txt')

class GenerateEmbedding:
    def __init__(self,bert_path):
        self.bert = AutoModel.from_pretrained(bert_path)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
        if torch.cuda.is_available():
            self.bert = self.bert.cuda()
    def generate(self,entity):
        entity = '#' + entity
        tokens = self.tokenizer(entity,return_tensors = 'pt')
        with torch.no_grad():
            if torch.cuda.is_available():
                tokens['input_ids'] = tokens['input_ids'].cuda()
                tokens['attention_mask'] = tokens['attention_mask'].cuda()
            output = self.bert(tokens['input_ids'],tokens['attention_mask']).last_hidden_state[:,2:]
        return output.squeeze(0).mean(dim = 0).cpu().numpy()
    def similarity(self,vec1,vec2):
        """
            计算余弦相似度
        """
        return np.sum(vec1 * vec2) / (np.sqrt(np.sum(np.power(vec1,2))) * np.sqrt(np.sum(np.power(vec2,2))))


gm = GenerateEmbedding(bert_path = bert_path)

entities = []
entities_embed = []
entity_name2type = {}

print('loading entity file...')
with open(entity_file,'r',encoding='utf-8') as f:
    for line in tqdm(f):
        line = line.strip()
        if not line: continue
        line = line.split('\t')[0]
        entity_name = line.split('..')[0]
        entity_type = line.split('..')[1]
        if entity_name not in entity_name2type:
            entity_name2type[entity_name] = entity_type
            entities.append(entity_name)
            entities_embed.append(gm.generate(entity_name))

entities_embed = np.stack(entities_embed,axis=0)

labels = []
label_ids = []
labels_embed = []
print('loading label file...')
with open(label_file,'r',encoding='utf-8') as f:
    for line in tqdm(f):
        line = line.strip()
        if not line: continue
        labels.append(line)
        label_ids.append(entities.index(line))
        labels_embed.append(gm.generate(line))

labels_embed = np.stack(labels_embed,axis=0)

print('calculating similarity labels...')
norm_entity = np.sqrt(np.sum(np.power(entities_embed,2),axis=1,keepdims=True))
norm_label = np.sqrt(np.sum(np.power(labels_embed,2),axis=1,keepdims=True))
similarity = np.matmul(entities_embed,labels_embed.T) / (norm_entity * norm_label.T)

data = []
for i, entity_name in enumerate(entities):
    for j, label_name in enumerate(labels):
        if entity_name in labels: continue
        data.append((entity_name, label_name, similarity[i,j]))

data.sort(key=lambda x:x[2],reverse=True)
data = list(filter(lambda x:x[2]>0.95,data))

with open(out_triples_file,'w',encoding='utf-8') as f:
    cur_triples = [item[0]+'..'+entity_name2type[item[0]]+'\t' +\
                   '相似描述'+ '\t' + item[1]+'..'+entity_name2type[item[1]] \
                     for item in data]
    f.write('\n'.join(cur_triples))

print('calculating similarity entities...')
triples_set = set()
for i, entity_name1 in tqdm(enumerate(entities)):
    similarity_i = (entities_embed * np.expand_dims(entities_embed[i], axis=0)).sum(axis = 1)
    similarity_i /= (norm_entity * np.expand_dims(norm_entity[i],axis=0)).squeeze(1)
    entity_ids = list(np.where(similarity_i > 0.96)[0])
    cur_triples = []
    for entity_id in entity_ids:
        if entity_id in label_ids:
            continue
        if entity_id <= i:
            continue
        triple = entities[i] + '..' + entity_name2type[entities[i]] + \
                           '\t' + '相似描述' + '\t' + \
                            entities[entity_id] + '..' + entity_name2type[entities[entity_id]]
        if triple in triples_set:
            continue
        triples_set.add(triple)
        cur_triples.append(triple)
    if len(cur_triples) > 0:
        with open(out_triples_file,'a', encoding='utf-8') as f:
            f.write('\n')
            f.write('\n'.join(cur_triples))
