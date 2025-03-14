from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from modules import EmbeddingLayer,PLMTokenEncoder


'''Recurrent Convolutional Neural Networks for Text Classification'''


class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        if '.embeds' in opt.bert_path:
            self.word_embedding = EmbeddingLayer(opt)
        else:
            self.base_layer = PLMTokenEncoder(opt)

        self.lstm = nn.LSTM(opt.embedding_dim, 512, 1,bidirectional=True, batch_first=True, dropout=opt.dropout_rate)
        self.fc = nn.Linear(1024+opt.embedding_dim ,opt.class_num)

    def forward(self, x,attention_mask):
        if hasattr(self,'word_embedding'):
            x = self.word_embedding(x)
        else:
            x = self.base_layer(x,attention_mask)
        embed = x * attention_mask.unsqueeze(2)
        out, _ = self.lstm(embed)
        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out,_ = torch.max(out,dim=2)
        out = self.fc(out)
        return out