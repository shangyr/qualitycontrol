import torch
import json 
import torch.utils.data as data
import numpy as np

from transformers import AutoTokenizer,BertTokenizer
from utils.tokenizer import Tokenizer
import json

from utils.tools import get_age
from utils.distribute import is_rank_0

class BaseDataset(data.Dataset):
    def __init__(self, filename:str or dict, opt):
        super(BaseDataset, self).__init__()
        self.data_dir = filename  # data_path
        self.label_idx_path = opt.label_idx_path
        self.label_name_path = opt.label_name_path
        self.label_smooth_lambda = opt.label_smooth_lambda
        self.label2id = opt.label2id

        self.bert_path = opt.bert_path
        self.class_num = opt.class_num
        self.batch_size = opt.batch_size
        self.data = []
        if '.embeds' in self.bert_path:
            # word2vec 版本
            self.tokenizer = Tokenizer.from_pretrained(self.bert_path)
            if opt.use_pretrain_embed_weight:
                opt.embedding_dim = self.tokenizer.embedding_dim()
        else:
            # 预训练模型版本
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.bert_path)
            except:
                self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.max_length = opt.max_length
        self.label_names= []
        with open(self.label_name_path,'r',encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                self.label_names.append(line)

        self._preprocess()

    def _preprocess(self):
        if is_rank_0():
            print("Loading data file...")

        if type(self.data_dir) is str:
            with open(self.data_dir, 'r', encoding='UTF-8')as f:
                self.data = json.load(f)
        else:
            self.data = self.data_dir

    def __len__(self):
        return len(self.data)

    def _get_text(self,dic):
        if '主诉' in dic:
            if dic["主诉"] is None:
                dic['主诉'] = ''
            if dic["现病史"] is None:
                dic['现病史'] = ''
            if dic["既往史"] is None:
                dic['既往史'] =  ''

            if '主诉' not in dic["主诉"]:
                dic['主诉'] = '主诉：'+dic['主诉']
            if '现病史' not in dic["现病史"]:
                dic['现病史'] = '现病史：'+dic['现病史']
            if '既往史' not in dic["既往史"]:
                dic['既往史'] = '既往史：'+dic['既往史']

            # chief_complaint = '性别：' + dic['性别'] + '；年龄：'+ dic['年龄'] + '；'+ dic["主诉"]
            chief_complaint =  dic["主诉"]
            now_history, past_history = dic["现病史"], dic["既往史"]
            raw_doc = chief_complaint + '[SEP]' + now_history + '[SEP]' + past_history
        else:
            raw_doc = dic['doc'].replace('[CLS]','').replace('[SEP]','')
        return raw_doc

    def _get_label(self, dic, return_label_name = False):
        if '出院诊断' in dic and type(dic['出院诊断']) is str:
            label = torch.tensor([self.label_smooth_lambda if label != dic['出院诊断'] else 1-self.label_smooth_lambda \
                                        for label in self.label2id])
            if return_label_name:
                label_name = [self.label_names[id_] for label,id_ in self.label2id.items() if label == dic['出院诊断']]
        elif '出院诊断' in dic and type(dic['出院诊断']) is list:
            label = torch.tensor([self.label_smooth_lambda if label not in dic['出院诊断'] else 1-self.label_smooth_lambda \
                                        for label in self.label2id])
            if return_label_name:
                label_name = [self.label_names[id_] for label,id_ in self.label2id.items() if label in dic['出院诊断']]
        elif 'labels' in dic:
            label = torch.tensor([self.label_smooth_lambda if label not in dic['labels'] else 1-self.label_smooth_lambda \
                                        for label in self.label2id])
            if return_label_name:
                label_name = [self.label_names[id_] for label,id_ in self.label2id.items() if label in dic['labels']]
        elif 'label' in dic:
            label = torch.tensor([self.label_smooth_lambda if label not in dic['label'] else 1-self.label_smooth_lambda \
                                        for label in self.label2id])
            if return_label_name:
                label_name = [self.label_names[id_] for label,id_ in self.label2id.items() if label in dic['label']]
        else:
            raise NotImplementedError
        if return_label_name:
            return label,label_name
        else:
            return label
    
    def set_new_data(self, data):
        self.data = data
    
    def _pad_doc_and_mask(self, sentence, attention_mask):
        """
        Args:
            sentence : (tensor([1,2,3,4]), ...) 
            attention_mask : (tensor([1,1,1,1]), ...) 
        Return:
            sentence : tensor([[1,2,3,4], ...])
            attention_mask : tensor([[1,1,1,1], ...])
        """
        idxs = [sentence]
        masks = [attention_mask]
        for j,(idx,mask) in enumerate(zip(idxs,masks)):
            max_len = max([len(t) for t in idx])
            for i in range(len(idx)):
                idx[i].extend([self.tokenizer.pad_token_id for _ in range(max_len - len(idx[i]))])  # pad
                mask[i].extend([0 for _ in range(max_len - len(mask[i]))])
            idxs[j] = torch.tensor(idx,dtype = torch.long)
            masks[j] = torch.tensor(mask,dtype = torch.long)
        return idxs[0], masks[0]
