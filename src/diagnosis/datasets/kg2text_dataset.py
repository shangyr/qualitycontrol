import torch
import json
import torch.utils.data as data
import numpy as np

from transformers import AutoTokenizer
from utils.tokenizer import Tokenizer
import json

from graph import BaseGraph

from utils.tools import get_age
from utils.distribute import is_rank_0
from .base_dataset import BaseDataset

class KG2TextDataset(BaseDataset):
    def __init__(self, filename, opt,graph:BaseGraph):
        super(KG2TextDataset, self).__init__(filename, opt)
        self.graph = graph
    
    def __getitem__(self, idx):
        dic = self.data[idx]
        raw_doc = self._get_text(dic)
        doc = self.tokenizer(raw_doc,max_length=self.max_length,truncation=True)
        doc_token_ids, doc_mask = doc['input_ids'],doc['attention_mask']
        if 'ents' in dic:
            ents = [e.split('\t')[0] for e in dic['ents']]
            kg_text = self.graph.generate_kg_text_by_text(raw_doc,ents)
        else:
            kg_text = self.graph.generate_kg_text_by_text(raw_doc)
        
        kg_text_doc = self.tokenizer(kg_text,max_length=self.max_length,truncation=True)
        kg_doc_token_ids, kg_doc_mask = kg_text_doc['input_ids'],kg_text_doc['attention_mask']
        
        label = self._get_label(dic)
        label = label.unsqueeze(0)

        return doc_token_ids, doc_mask,kg_doc_token_ids,kg_doc_mask, label
        
    def collate_fn(self, X):
        X = list(zip(*X))
        doc_token_ids, doc_mask,kg_doc_token_ids,kg_doc_mask, labels = X

        doc_token_ids, doc_mask = self._pad_doc_and_mask(doc_token_ids, doc_mask)
        kg_doc_token_ids, kg_doc_mask = self._pad_doc_and_mask(kg_doc_token_ids, kg_doc_mask)

        
        labels = torch.cat(labels, 0)
        data = {'input_ids':doc_token_ids,'attention_mask':doc_mask,
                'kg_input_ids':kg_doc_token_ids,'kg_attention_mask':kg_doc_mask,
                'label':labels}

        return data

