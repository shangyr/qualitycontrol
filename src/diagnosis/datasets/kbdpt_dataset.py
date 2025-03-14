import random
import torch
import json
import torch.utils.data as data
import numpy as np
import os


from transformers import AutoTokenizer,BertTokenizer
from utils.tokenizer import Tokenizer
import json

from utils.tools import get_age
from utils.distribute import is_rank_0
from .base_dataset import BaseDataset

class KBDPTDataset(BaseDataset):
    def __init__(self, filename:str or dict, opt, graph = None):
        super(KBDPTDataset, self).__init__(filename, opt)
        self.train_mode = type(filename) is str and 'train' in filename 
        if not self.train_mode: self.batch_size = 1
        self.num_hop = opt.num_hop
        self.graph = graph

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
        self.relation_num = opt.relation_num

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
        doc = self.tokenizer(raw_doc,add_special_tokens=False)
        _doc_token_ids, _doc_mask = doc['input_ids'],doc['attention_mask']
        label,label_names = self._get_label(dic, return_label_name = True)
        label = label.unsqueeze(0)
        path_t,path_f = self._find_all_path(dic, raw_doc, label_names)

        if self.train_mode:
            if len(path_t) == 0: path_t.append('')
            if len(path_f) == 0: path_f.append('')
            if random.random() <= len(path_t) / (len(path_t) + len(path_f)):
                path = random.choice(path_t)
                positive_path = 1
            else:
                path = random.choice(path_f)
                positive_path = 0
            prompt = self.tokenizer(path,add_special_tokens=False)
            prompt_token_ids, prompt_attention_mask = prompt['input_ids'], prompt['attention_mask']

            if len(_doc_token_ids) + len(prompt_token_ids) + 3 >= self.max_length:
                origin_len = len(_doc_token_ids) + len(prompt_token_ids) + 3
                _doc_token_ids = _doc_token_ids[:self.max_length - origin_len]
                _doc_mask = _doc_mask[:self.max_length - origin_len]
            # doc_token_ids = [self.tokenizer.cls_token_id] + _doc_token_ids + [self.tokenizer.mask_token_id] + prompt_token_ids + [self.tokenizer.sep_token_id]
            # doc_mask = [1] + _doc_mask + [1] + prompt_attention_mask + [1]
            # mask_idx = len(_doc_token_ids) + 1
            doc_token_ids = [self.tokenizer.cls_token_id]  + _doc_token_ids  + [self.tokenizer.mask_token_id] +prompt_token_ids  + [self.tokenizer.sep_token_id]
            doc_mask = [1] + _doc_mask + [1] + prompt_attention_mask + [1]
            mask_idx = len(_doc_token_ids) + 1
            label_prompt = torch.zeros(2)
            label_prompt[positive_path] = 1

            return doc_token_ids, doc_mask, mask_idx, label, label_prompt
        else:
            paths = path_t + path_f
            label2path_id = {}
            for path_id, path in enumerate(paths):
                path_label = path.split('->')[-1]
                if path_label not in label2path_id:
                    label2path_id[path_label] = []
                label2path_id[path_label].append(path_id)
            if len(label2path_id) == 0:
                label2path_id[''] = 0
                paths = ['']
            
            doc_token_ids,doc_mask,mask_idx = [],[],[]
            for label_name in label2path_id:
                path = paths[random.choice(label2path_id[label_name])]
                prompt = self.tokenizer(path,add_special_tokens=False)
                prompt_token_ids, prompt_attention_mask = prompt['input_ids'], prompt['attention_mask']
                if len(_doc_token_ids) + len(prompt_token_ids) + 3 >= self.max_length:
                    origin_len = len(_doc_token_ids) + len(prompt_token_ids) + 3
                    _doc_token_ids = _doc_token_ids[:self.max_length - origin_len]
                    _doc_mask = _doc_mask[:self.max_length - origin_len]
                doc_token_ids_i = [self.tokenizer.cls_token_id] + _doc_token_ids + [self.tokenizer.mask_token_id] + prompt_token_ids + [self.tokenizer.sep_token_id]
                doc_mask_i = [1] + _doc_mask + [1] + prompt_attention_mask + [1]
                doc_token_ids.append(doc_token_ids_i)
                mask_idx.append(len(_doc_mask) + 1)
                doc_mask.append(doc_mask_i)
                
            return tuple(doc_token_ids), tuple(doc_mask), mask_idx, label

    def _find_all_path(self, dic, raw_doc, label_names):
        """
        找到所有关联路径
        """
        if self.graph is None:
            path_t = ['->'.join(path) for path in dic['path_T']]
            path_f = ['->'.join(path) for path in dic['path_F']]
        else:
            entity_names = self.graph.generate_entity_name_by_text(raw_doc)
            nodes,edges,node_types,edges_types,paths = self.graph._generate_graph_by_entity(entity_names,path_type = self.path_type, filter_node = True)
            if self.path_type == 'v1':
                path_t,path_f = [],[]
                for i,path in enumerate(paths):
                    if i % 2 == 1: path = path.replace('_',' ')
                    if path.split('->')[-1] in label_names:
                        path_t.append(path)
                    else:
                        path_f.append(path)
            else: # self.path_type = 'v2'
                path_t,path_f = [],[]
                for head,edge_type,tail in zip(edges[0],edges_types,edges[1]):
                    head_name = self.id2entity[nodes[head]]
                    tail_name = self.id2entity[nodes[tail]]
                    _rel = ''
                    for _ in range(self.num_hop):
                        cur_edge_type = edge_type % (self.relation_num + 1)
                        edge_type //= self.relation_num + 1
                        if cur_edge_type == self.relation_num:
                            continue
                        if len(_rel) > 0: _rel = '->' + _rel
                        _rel = self.id2rel[cur_edge_type] + _rel
                    _rel = _rel.replace('_',' ')
                    path = head_name + '->' + _rel + '->' + tail_name
                    if tail_name in label_names:
                        path_t.append(path)
                    else:
                        path_f.append(path)
        return path_t, path_f

    def collate_fn(self, X):
        if self.train_mode:
            return self._train_collate_fn(X)
        else:
            return self._test_collect_fn(X)
    
    def _train_collate_fn(self, X):
        doc_token_ids, doc_mask, mask_idx, label, label_prompt = list(zip(*X))
        doc_token_ids, doc_mask = self._pad_doc_and_mask(doc_token_ids, doc_mask)
        mask_idx = torch.LongTensor(mask_idx)
        label = torch.cat(label, 0)
        label_prompt = torch.stack(label_prompt, 0)
        data = {'input_ids':doc_token_ids,'attention_mask':doc_mask,'mask_idx':mask_idx,'label':label,'label_prompt':label_prompt}
        return data
    
    def _test_collect_fn(self, X):
        doc_token_ids, doc_mask, mask_idx, label = list(zip(*X))
        assert len(doc_token_ids) == 1, f'测试或者开发集batch_size应为1，实际为{len(doc_token_ids)}'
        doc_token_ids, doc_mask, mask_idx, label = doc_token_ids[0], doc_mask[0], mask_idx[0], label[0]

        doc_token_ids, doc_mask = self._pad_doc_and_mask(doc_token_ids, doc_mask)
        mask_idx = torch.LongTensor(mask_idx)
        data = {'input_ids':doc_token_ids,'attention_mask':doc_mask,'mask_idx':mask_idx,'label':label}
        return data

