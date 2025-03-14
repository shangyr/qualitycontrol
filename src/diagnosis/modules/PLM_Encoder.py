from abc import ABC

import torch
import torch.nn as nn
from transformers import AutoModel
from transformers import AutoModelForMaskedLM
from transformers import BertModel
from fengshen import LongformerModel
from fengshen import LongformerForMaskedLM
import numpy as np

class PLMEncoder(nn.Module, ABC):
    def __init__(self, opt):
        super(PLMEncoder, self).__init__()
        self.model_name = 'AutoModel'
        self.bert_path = opt.bert_path
        if 'Erlangshen-Longformer-110M'.lower() in self.bert_path.lower():
            self.word_embedding = LongformerModel.from_pretrained(opt.bert_path)
        else:
            self.word_embedding = AutoModel.from_pretrained(opt.bert_path)
        for param in self.word_embedding.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(opt.dropout_rate)

    def forward(self, X, masks,inputs_embeds=None):
        if 'longformer' in self.bert_path.lower():
            batch_size,seq_len = masks.size()
            global_attention_mask = np.zeros((batch_size,seq_len))
            global_attention_mask[:,0] = 1
            global_attention_mask = torch.tensor(global_attention_mask,device=masks.device)
            embed = self.word_embedding(X, attention_mask=masks,global_attention_mask=global_attention_mask,inputs_embeds = inputs_embeds).pooler_output
        else:
            embed = self.word_embedding(X, attention_mask=masks,inputs_embeds = inputs_embeds).pooler_output
        pooled = self.dropout(embed)
        return pooled

    def get_embedding(self):
        return self.word_embedding.get_input_embeddings()

class PLMWithMaskEncoder(nn.Module, ABC):
    def __init__(self, opt):
        super(PLMWithMaskEncoder, self).__init__()
        self.model_name = 'AutoModel'
        self.bert_path = opt.bert_path
        if 'Erlangshen-Longformer-110M'.lower() in self.bert_path.lower():
            self.word_embedding = LongformerForMaskedLM.from_pretrained(opt.bert_path)
        else:
            self.word_embedding = AutoModelForMaskedLM.from_pretrained(opt.bert_path)
        for param in self.word_embedding.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(opt.dropout_rate)

    def forward(self, X, masks):
        base_plm = getattr(self.word_embedding,self.word_embedding.base_model_prefix)
        if 'longformer' in self.bert_path.lower():
            batch_size,seq_len = masks.size()
            global_attention_mask = np.zeros((batch_size,seq_len))
            global_attention_mask[:,0] = 1
            global_attention_mask = torch.tensor(global_attention_mask,device=X.device)
            embed = base_plm(X, attention_mask=masks,global_attention_mask=global_attention_mask).last_hidden_state[:,0]
        else:
            embed = base_plm(X, attention_mask=masks,).last_hidden_state[:,0]
        pooled = self.dropout(embed)
        return pooled

    def forward_mask(self, input_ids, attention_mask,labels):
        if 'longformer' in self.bert_path.lower():
            batch_size,seq_len = attention_mask.size()
            global_attention_mask = np.zeros((batch_size,seq_len))
            global_attention_mask[:,0] = 1
            global_attention_mask = torch.tensor(global_attention_mask,device=input_ids.device) #global_attention_mask=global_attention_mask,
            ans = self.word_embedding(input_ids, attention_mask=attention_mask,labels=labels,output_hidden_states=True)
        else:
            ans = self.word_embedding(input_ids, attention_mask=attention_mask,labels=labels,output_hidden_states=True)
        embed = ans.hidden_states[-1] # 最后一层的输出
        pooled = self.dropout(embed)
        loss = ans.loss
        return pooled,loss