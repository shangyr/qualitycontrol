import torch
import json
import torch.utils.data as data
import numpy as np
import os
import math
import pickle as pkl
from tqdm import tqdm

from transformers import AutoTokenizer,BertTokenizer
from utils.tokenizer import Tokenizer
import json

from utils.distribute import is_rank_0
from graph_kge import *

from .base_dataset import BaseDataset

class PromptGNNDataset(BaseDataset):
    def __init__(self, filename, opt, graph = None):
        super(PromptGNNDataset, self).__init__(filename, opt)
        if hasattr(opt,'idf_path'):
            self.idf_path = opt.idf_path
            self.tf_idf_radio = opt.tf_idf_radio

        if graph is None:
            self.entity2id = self._entity2id(opt.entity_idx_path)
            self.id2entity = [entity for entity,_ in self.entity2id.items()]
            self.rel2id = self._entity2id(opt.relation_idx_path)
            self.id2rel = [rel for rel,_ in self.rel2id.items()]
            opt.entity_num = len(self.entity2id)
            opt.relation_num = len(self.rel2id)
            # 训练kge
            if opt.kge_trainer != '' and not os.path.exists(opt.entity_embedding_path):
                KGETrainer = eval(opt.kge_trainer)
                kge_trainer = KGETrainer(entity2id=self.entity2id,language=opt.language)
                kge_trainer.export_model(opt.entity_embedding_path)
        
        else:
            self.entity2id = graph.match_model.entity2id
            self.id2entity = [entity for entity,_ in self.entity2id.items()]
            self.rel2id = graph.rel2id
            self.id2rel = [rel for rel,_ in self.rel2id.items()]
            opt.entity_num = graph.entity_num()
            opt.relation_num = graph.relation_num()

        self.path_type = opt.path_type
        self.graph = graph
        
        self.label_length = 0
        with open(opt.label_name_path,'r',encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line : continue
                self.label_length += len(self.tokenizer.tokenize(line))
        self._preprocess()

    def _preprocess(self):
        super()._preprocess()
        if hasattr(self,'idf_path') and not os.path.exists(self.idf_path):
            assert type(self.data_dir) is str and 'train' in self.data_dir
            counter = {}
            print('loading idf...')
            for item in tqdm(self.data):
                raw_doc = self._get_text(item)
                entity_names = self.graph.generate_entity_name_by_text(raw_doc)
                # uniq_entity_names = list(set(entity_names))
                # _,_,_,_,_,entity_names = self.graph.generate_graph_by_text(raw_doc,path_type = self.path_type)
                entity_names = set(entity_names)
                for entity_name in entity_names:
                    if entity_name not in counter:
                        counter[entity_name] = 0
                    counter[entity_name] += 1
            self.idf = {entity_name: math.log(len(self.data) / (entity_count + 1e-3)) for entity_name,entity_count in counter.items()}
            with open(self.idf_path,'wb') as f:
                pkl.dump(self.idf,f)
        elif hasattr(self,'idf_path'):
            with open(self.idf_path,'rb') as f:
                self.idf = pkl.load(f)

    def _entity2id(self,entity_idx_path):
        entity2id = {}
        with open(entity_idx_path,'r',encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                entity2id[line] = len(entity2id)
        return entity2id
    
    def __getitem__(self, idx):
        dic = self.data[idx]
        raw_doc = self._get_text(dic)
        doc = self.tokenizer(raw_doc,max_length=self.max_length-self.class_num,truncation=True)
        doc_idxes, doc_mask = doc['input_ids'],doc['attention_mask']
        str_paths = None
        if self.graph is None:
            nodes = [i for i in range(self.class_num)]
            edges = [[],[]]
            edges_types = []
            paths = dic['path_T'] + dic['path_F']
            edges_set = set()
            pre_edge_type = 0
            for path in paths:
                pre_node_id = None
                for node in path:
                    if '\\' in node:
                        pre_edge_type = self.rel2id.get(node,0)
                        continue
                    cur_node_id = self.entity2id[node]
                    if cur_node_id not in nodes:
                        nodes.append(cur_node_id)
                    edge = pre_node_id,cur_node_id
                    if pre_node_id is not None and edge not in edges_set:
                        edges[0].append(nodes.index(pre_node_id))
                        edges[1].append(nodes.index(cur_node_id))
                        edges_types.append(pre_edge_type)
                        edges_set.add(edge)
                    pre_node_id = cur_node_id
        else:
            entity_names = self.graph.generate_entity_name_by_text(raw_doc)
            uniq_entity_names = list(set(entity_names))
            uniq_entity_names.sort(key=lambda x: entity_names.count(x)/(len(entity_names) + 1e-3) * self.idf.get(x,0.01),reverse=True)
            uniq_entity_names = uniq_entity_names[:int(len(uniq_entity_names) * self.tf_idf_radio)]
            nodes,edges,node_types,edges_types,str_paths = self.graph._generate_graph_by_entity(uniq_entity_names,path_type = self.path_type)
            
        nodes = torch.LongTensor(nodes)
        edges = torch.LongTensor(edges)
        edges_types = torch.LongTensor(edges_types)
        
        label = self._get_label(dic)
        label = label.unsqueeze(0)

        return doc_idxes, doc_mask, nodes, edges, edges_types, label, str_paths

    def collate_fn(self, X):
        X = list(zip(*X))
        
        doc_idxes, doc_mask,ent_ids,edges,edges_types, labels, str_paths = X

        idxs = [doc_idxes]
        masks = [doc_mask]
        for j,(idx,mask) in enumerate(zip(idxs,masks)):
            max_len = max([len(t) for t in idx])
            for i in range(len(idx)):
                idx[i].extend([self.tokenizer.pad_token_id for _ in range(max_len - len(idx[i]))])  # pad
                mask[i].extend([0 for _ in range(max_len - len(mask[i]))])
            idxs[j] = torch.tensor(idx,dtype = torch.long)
            masks[j] = torch.tensor(mask,dtype = torch.long)
            # 不使用框架
            masks[j] = torch.cat((torch.ones(len(doc_idxes),self.class_num),masks[j]),dim = 1)
            # masks[j] = torch.cat((torch.ones(len(doc_idxes),self.class_num + self.label_length),masks[j]),dim = 1)

        ent_ids = list(ent_ids)
        edges = list(edges)
        edges_types = list(edges_types)

        labels = torch.cat(labels, 0)
        data = {'input_ids':idxs[0],'attention_mask':masks[0],
                'nodes':ent_ids,'edges':edges,'edges_types':edges_types,
                'label':labels, 'paths':str_paths}

        return data
