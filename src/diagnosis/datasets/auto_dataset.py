import torch
import json
import torch.utils.data as data
import numpy as np

from transformers import AutoTokenizer,BertTokenizer
from utils.tokenizer import Tokenizer
import json

from utils.tools import get_age
from utils.distribute import is_rank_0

from .base_dataset import BaseDataset

class AutoDataset(BaseDataset):
    def __init__(self, filename:str or dict, opt):
        super(AutoDataset, self).__init__(filename, opt)
        
    def __getitem__(self, idx):
        dic = self.data[idx]
        raw_doc = self._get_text(dic)
        doc = self.tokenizer(raw_doc,max_length=self.max_length,truncation=True)
        doc_token_ids, doc_mask = doc['input_ids'],doc['attention_mask']
        label = self._get_label(dic)
        label = label.unsqueeze(0)
        return doc_token_ids, doc_mask, label
        
    def collate_fn(self, X):
        X = list(zip(*X))
        doc_token_ids, doc_mask, labels = X

        doc_token_ids, doc_mask = self._pad_doc_and_mask(doc_token_ids, doc_mask)

        labels = torch.cat(labels, 0)
        data = {'input_ids':doc_token_ids,'attention_mask':doc_mask,'label':labels}

        return data