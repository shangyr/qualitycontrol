from abc import ABC

import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM,LongformerForMaskedLM
from transformers import BertModel,AutoModel
from fengshen import LongformerModel
# from transformers import LongformerModel
import numpy as np

class PLMTokenEncoder(nn.Module, ABC):
    def __init__(self, opt):
        super(PLMTokenEncoder, self).__init__()
        self.model_name = 'AutoModel'
        self.bert_path = opt.bert_path
        self.class_num = opt.class_num
        if 'Erlangshen-Longformer-110M'.lower() in self.bert_path.lower():
            self.word_embedding = LongformerModel.from_pretrained(opt.bert_path)
        else:
            # self.word_embedding = AutoModelForMaskedLM.from_pretrained(opt.bert_path)
            self.word_embedding = AutoModel.from_pretrained(opt.bert_path)
        for param in self.word_embedding.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(opt.dropout_rate)

    def forward(self, X, masks,mask_idx = None,inputs_embeds=None,use_disease_token=True,output_attentions=False):
        if 'longformer' in self.bert_path.lower():
            batch_size,seq_len = masks.size()
            global_attention_mask = np.zeros((batch_size,seq_len))
            global_attention_mask[:,0] = 1 # 前面有 50 个类别的分类
            if mask_idx is not None:
                global_attention_mask[np.arange(batch_size), mask_idx.data.cpu().numpy()] = 1
            if use_disease_token:
                global_attention_mask[:,1:self.class_num + 1] = 1 # 前面有 50 个类别的分类
            
            global_attention_mask = torch.tensor(global_attention_mask,device=masks.device)
            output = self.word_embedding(X, attention_mask=masks,\
                                         global_attention_mask=global_attention_mask,\
                                         inputs_embeds = inputs_embeds,\
                                         output_attentions = output_attentions)
        else:
            output = self.word_embedding(X, attention_mask=masks,\
                                         inputs_embeds = inputs_embeds,\
                                         output_attentions = output_attentions)
        embed = output.last_hidden_state
        pooled = self.dropout(embed)
        if output_attentions:
            return pooled, output.attentions
        else:
            return pooled

    def forward_logit(self, X, masks,mask_idx = None,use_disease_token=True,inputs_embeds=None):
        if 'longformer' in self.bert_path.lower():
            batch_size,seq_len = masks.size()
            global_attention_mask = np.zeros((batch_size,seq_len))
            # global_attention_mask[:,1:self.class_num + 1] = 1 # 前面有 50 个类别的分类
            # global_attention_mask = torch.tensor(global_attention_mask,device=masks.device)
            batch_size,seq_len = masks.size()
            global_attention_mask = np.zeros((batch_size,seq_len))
            if mask_idx is not None:
                global_attention_mask[np.arange(batch_size), mask_idx.data.cpu().numpy()] = 1
            if use_disease_token:
                global_attention_mask[:,1:self.class_num + 1] = 1 # 前面有 50 个类别的分类
            
            embed = self.word_embedding(X, attention_mask=masks,global_attention_mask=global_attention_mask,inputs_embeds = inputs_embeds).logits
        else:
            embed = self.word_embedding(X, attention_mask=masks,inputs_embeds = inputs_embeds).logits
        pooled = self.dropout(embed)
        return pooled

    def get_embedding(self):
        return self.word_embedding.get_input_embeddings()