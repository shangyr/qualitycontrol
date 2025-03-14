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

        ##  256 
        best_u = 256
        best_da = 256

        self.bi_lstm = nn.LSTM(opt.embedding_dim, best_u, 2,
                            bidirectional=True, batch_first=True)

        self.w = nn.Parameter(torch.FloatTensor(best_u * 2,best_da))
        nn.init.xavier_normal_(self.w)
        self.u = nn.Parameter(torch.FloatTensor(best_da,opt.class_num))
        nn.init.xavier_normal_(self.u)

        self.final1 = nn.Linear(best_u * 2, 512)
        nn.init.xavier_normal_(self.final1.weight)
        self.final2 = nn.Linear(512, opt.class_num)
        nn.init.xavier_normal_(self.final2.weight)

    def forward(self, x, attention_mask):

        if hasattr(self,'word_embedding'):
            embed = self.word_embedding(x)
        else:
            embed = self.base_layer(x,attention_mask)
        embed = embed * attention_mask.unsqueeze(2)
        # bi-lstm
        H,(_,_) = self.bi_lstm(embed)         # [batch_size,seq_len,2*hidden_size]
        
        # label attn layer
        Z = torch.tanh(H @ self.w)            # [batch_size,seq_len,attn_d]
        A = torch.softmax(Z @ self.u,dim = 1) # [batch_size,seq_len,labels_num]
        V = A.transpose(1,2) @ H              # [batch_size,labels_num,2*hidden_size]

        # output layer
        V = torch.relu(self.final1(V))        # [batch_size,labels_num,ffn_size]
        y_hat = self.final2.weight.mul(V).sum(dim=2).add(self.final2.bias) # [batch_size,labels_num]

        return y_hat