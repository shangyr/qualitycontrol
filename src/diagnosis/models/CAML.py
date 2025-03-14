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

        num_filter_maps = 100
        kernel_size = 3
        self.conv = nn.Conv1d(opt.embedding_dim,num_filter_maps,kernel_size=kernel_size,padding = kernel_size//2)
        self.U = nn.Linear(num_filter_maps, opt.class_num)
        self.final = nn.Linear(num_filter_maps, opt.class_num)

    def forward(self, x, attention_mask):
        if hasattr(self,'word_embedding'):
            x = self.word_embedding(x)
        else:
            x = self.base_layer(x,attention_mask)
        x = x * attention_mask.unsqueeze(2)
        x = x.transpose(1, 2) # [batch_size, embed_size ,seq_len]
        #apply convolution and nonlinearity (tanh)
        x = F.tanh(self.conv(x).transpose(1,2)) # [batch_size, seq_len, hidden_size]
        #apply attention
        alpha = F.softmax(self.U.weight.matmul(x.transpose(1,2)), dim=2) # [batch_size,num_class,seq_len]
        #document representations are weighted sums using the attention. Can compute all at once as a matmul
        m = alpha.matmul(x) # [batch_size, num_class, hidden_size]
        #final layer classification
        y_hat = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
        return y_hat