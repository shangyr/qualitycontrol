
import torch
import json
import torch.utils.data as data
import numpy as np
import os
import pickle as pkl
from datetime import datetime
from transformers import AutoTokenizer
import json

from utils.tools import get_age
from utils.distribute import is_rank_0
from graph import BaseGraph
from utils.tokenizer import Tokenizer
from tqdm import tqdm

from .base_dataset import BaseDataset

class MSLANDataset(BaseDataset):
    def __init__(self, filename,opt,graph:BaseGraph):
        super(MSLANDataset, self).__init__(filename,opt)

        self.threshold = 100
        self.graph = graph
        cache_graph_file = filename.replace('train.json','A.pkl').replace('dev.json','A.pkl').replace('test.json','A.pkl')
        self.entity2id = self.graph.match_model.entity2id
        if 'train' in filename and not os.path.exists(cache_graph_file):    
        # if 'train' in filename:
            self.A = self._A_graph(opt)
            with open(cache_graph_file,'wb') as f:
                pkl.dump(self.A, f)
        else:
            with open(cache_graph_file,'rb') as f:
                self.A = pkl.load(f)
    
    def _A_graph(self,opt):
        all_num = opt.class_num + opt.entity_num
        A = {}
        print('生成A矩阵')
        for item in tqdm(self.data):
            doc = self._get_text(item)
            entity_names = self.graph.generate_entity_name_by_text(doc)
            label_key = 'label' if 'label' in item else 'labels'
            label_names = item[label_key]
            # 标签-实体
            for label_name in label_names:
                if label_name not in self.label2id:
                    continue
                for entity_name in entity_names:
                    if entity_name not in self.entity2id:
                        continue
                    label_id = self.label2id[label_name]
                    entity_id = self.entity2id[entity_name] + opt.class_num
                    key = (label_id,entity_id)
                    if key not in A:
                        A[key] = 0
                    A[key] += 1

            # 实体-实体
            for entity_name1 in entity_names:
                if entity_name1 not in self.entity2id:
                    continue
                for entity_name2 in entity_names:
                    if entity_name2 not in self.entity2id:
                        continue
                    # if entity_name1 == entity_name2:
                    #     continue
                    
                    entity_id1 = self.entity2id[entity_name1] + opt.class_num
                    entity_id2 = self.entity2id[entity_name2] + opt.class_num
                    key = (entity_id1,entity_id2)
                    if key not in A:
                        A[key] = 0
                    A[key] += 1
        return A

    def __getitem__(self, idx):
        dic = self.data[idx]        
        raw_doc = self._get_text(dic)
        doc = self.tokenizer(raw_doc,max_length=self.max_length,truncation=True)
        if 'ents' in dic:
            ents = [e.split('\t')[0] for e in dic['ents']]
            nodes,edges = self.graph.generate_graph_by_co_occurrence(self.A, raw_doc,ner_entities=ents,threshold=self.threshold)
        else:
            nodes,edges = self.graph.generate_graph_by_co_occurrence(self.A, raw_doc,threshold=self.threshold)
        
        doc_token_ids, doc_mask = doc['input_ids'],doc['attention_mask']
        label = self._get_label(dic)
        label = label.unsqueeze(0)
        return doc_token_ids, doc_mask,nodes,edges, label
    

    def collate_fn(self, X):
        X = list(zip(*X))
        doc_token_ids, doc_mask,nodes,edges, labels = X 
        
        nodes = list(nodes)
        edges = list(edges)
        # nodes : [batch,num_node,node_len]
        for i in range(len(nodes)):
            nodes[i] = torch.LongTensor(nodes[i])
            edges[i] = torch.LongTensor(edges[i])

        doc_token_ids, doc_mask = self._pad_doc_and_mask(doc_token_ids, doc_mask)

        labels = torch.cat(labels, 0)
        # nodes,node_mask, graph
        data = {'input_ids':doc_token_ids, 
                'attention_mask':doc_mask,
                'nodes':nodes,
                'edges':edges,
                'label':labels}

        return data

