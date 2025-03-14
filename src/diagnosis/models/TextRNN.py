from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from modules import EmbeddingLayer,PLMTokenEncoder

class Model(nn.Module):
    def __init__(self,opt):
        super(Model, self).__init__()
        if '.embeds' in opt.bert_path:
            self.word_embedding = EmbeddingLayer(opt)
        else:
            self.base_layer = PLMTokenEncoder(opt)
        self.lstm = nn.GRU(opt.embedding_dim, opt.embedding_dim, 2,
                            bidirectional=False, batch_first=True)
        self.fc = nn.Linear(opt.embedding_dim,opt.class_num)

    def forward(self, x, attention_mask):
        if hasattr(self,'word_embedding'):
            x = self.word_embedding(x)
        else:
            x = self.base_layer(x,attention_mask)
        x = x * attention_mask.unsqueeze(2)
        H,(_,_) = self.lstm(x)         # [batch_size,seq_len,2*hidden_size]
        out = torch.relu(H)
        logits = self.fc(out[:,-1,:])
        return logits