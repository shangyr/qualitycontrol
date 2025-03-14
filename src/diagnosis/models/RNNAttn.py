from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from modules import EmbeddingLayer,PLMTokenEncoder

'''Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification'''

class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        if '.embeds' in opt.bert_path:
            self.word_embedding = EmbeddingLayer(opt)
        else:
            self.base_layer = PLMTokenEncoder(opt)
        self.lstm = nn.LSTM(opt.embedding_dim, 512, 1, bidirectional=True, batch_first=True, dropout=opt.dropout_rate)
        self.tanh1 = nn.Tanh()
        # self.u = nn.Parameter(torch.Tensor(opt.hidden_size * 2, opt.hidden_size * 2))
        self.w = nn.Parameter(torch.zeros(1024))
        self.tanh2 = nn.Tanh()

        self.fc1 = nn.Linear(1024 ,768)
        self.fc = nn.Linear(768, opt.class_num)
        
    def forward(self, x, attention_mask):
        if hasattr(self,'word_embedding'):
            x = self.word_embedding(x)
        else:
            x = self.base_layer(x,attention_mask)
        emb = x * attention_mask.unsqueeze(2)
        H, _ = self.lstm(emb)  # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]

        M = self.tanh1(H)  # [128, 32, 256]
        # M = torch.tanh(torch.matmul(H, self.u))
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [128, 32, 1]
        out = H * alpha  # [128, 32, 256]
        out = torch.sum(out, 1)  # [128, 256]
        
        out = torch.relu(self.fc1(out))
        out = self.fc(out)  # [128, 64]
        return out