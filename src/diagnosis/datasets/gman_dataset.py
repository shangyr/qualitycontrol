import torch
import json
import os
import torch.utils.data as data
import numpy as np

from transformers import AutoTokenizer
from utils.tokenizer import Tokenizer
import json

from utils.tools import get_age
from utils.distribute import is_rank_0

from graph import BaseGraph
from .base_dataset import BaseDataset

class GMANDataset(BaseDataset):
    def __init__(self, filename, opt, graph:BaseGraph):
        super(GMANDataset, self).__init__(filename, opt)
        
        self.graph = graph
        cache_graph_file = os.path.join(opt.data_path, 'D_F_D_D.npz')
        self.entity2id = self.graph.match_model.entity2id
        # if 'train' in filename and not os.path.exists(cache_graph_file):    
        if 'train' in filename:
            self.D_F = self._D_F_graph(opt)
            self.D_D = self._D_D_graph(opt)
            np.savez(cache_graph_file,D_F = self.D_F,D_D = self.D_D)
        else:
            self.D_F = np.load(cache_graph_file)['D_F']
            self.D_D = np.load(cache_graph_file)['D_D']

    def _D_D_graph(self,opt):
        labels = []
        parent_labels = []
        edges = []
        with open(opt.label_hierarchy_path,'r',encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                edge = line.split('##')
                label_name = edge[0]
                parent_label_name = edge[1]
                edges.append(edge)
                labels.append(label_name)
                if parent_label_name not in parent_labels:
                    parent_labels.append(parent_label_name)
        all_label_num = len(labels) + len(parent_labels)
        all_labels = labels + parent_labels
        D_D = np.zeros((all_label_num,all_label_num)).astype(np.float32)
        for edge in edges:
            D_D[all_labels.index(edge[0]),all_labels.index(edge[1])] += 1
            # D_D[all_labels.index(edge[1]),all_labels.index(edge[0])] += 1
        return D_D
    
    def _D_F_graph(self,opt):
        D_F = np.zeros((opt.all_label_num,opt.entity_num)).astype(np.float32)
        for item in self.data:
            doc = self._get_text(item)
            entity_names = self.graph.generate_entity_name_by_text(doc)
            label_key = 'label' if 'label' in item else 'labels'
            label_names = item[label_key]
            for label_name in label_names:
                if label_name not in self.label2id:
                    continue
                for entity_name in entity_names:
                    if entity_name not in self.entity2id:
                        continue
                    D_F[self.label2id[label_name],self.entity2id[entity_name]] += 1
        # 计算TF-IDF
        N = len(self.data)
        tf = D_F / (D_F.sum(axis = 1,keepdims=True) + 1e-6)
        idf = np.log( N / (1 + D_F.sum(axis = 1, keepdims=True) + 1e-6) )
        D_F = tf * idf
        D_F = D_F / (D_F.sum(axis = 1,keepdims=True) + 1e-6)
        return D_F
    
    def __getitem__(self, idx):
        dic = self.data[idx]
        raw_doc = self._get_text(dic)
        if 'ents' in dic:
            ents = [e.split('\t')[0] for e in dic['ents']]
            entity_names = self.graph.generate_entity_name_by_text(raw_doc,ents)
        else:
            entity_names = self.graph.generate_entity_name_by_text(raw_doc)
        entity_ids = []
        for entity_name in entity_names:
            if entity_name not in self.entity2id:
                continue
            entity_ids.append(self.entity2id[entity_name])
        entity_mask = [1] * len(entity_ids)
        D_F = []
        for entity_id in entity_ids:
            D_F.append(self.D_F[:,entity_id])
        D_F = np.stack(D_F,axis = 1)
        doc = self.tokenizer(raw_doc,max_length=self.max_length,truncation=True)
        doc_token_ids, doc_mask = doc['input_ids'],doc['attention_mask']
        
        label = self._get_label(dic)
        label = label.unsqueeze(0)
        return doc_token_ids, doc_mask,entity_ids,entity_mask,D_F, label
        
    def collate_fn(self, X):
        X = list(zip(*X))
        doc_token_ids, doc_mask,entity_ids,entity_mask,D_F, labels = X
        doc_token_ids, doc_mask = self._pad_doc_and_mask(doc_token_ids, doc_mask)
        D_F = list(D_F)
        idxs = [entity_ids]
        masks = [entity_mask]
        for j,(idx,mask) in enumerate(zip(idxs,masks)):
            max_len = max([len(t) for t in idx])
            for i in range(len(idx)):
                idx[i].extend([self.tokenizer.pad_token_id for _ in range(max_len - len(idx[i]))])  # pad
                mask[i].extend([0 for _ in range(max_len - len(mask[i]))])
                # D_num,F_num
                D_num,F_num = D_F[i].shape
                pad_arr = np.zeros((D_num,max_len - F_num))
                D_F[i] = np.concatenate((D_F[i],pad_arr),axis = 1)
            idxs[j] = torch.tensor(idx,dtype = torch.long)
            masks[j] = torch.tensor(mask,dtype = torch.long)
        D_F = torch.tensor(np.array(D_F).astype(np.float32))

        labels = torch.cat(labels, 0)
        D_D = torch.tensor(self.D_D) # [label_num,entity_num]
        data = {'input_ids':doc_token_ids,'attention_mask':doc_mask,'nodes':idxs[0],'node_mask':masks[0],'D_F':D_F,'D_D':D_D,'label':labels}

        return data

