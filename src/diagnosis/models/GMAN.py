from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from modules import EmbeddingLayer,PLMTokenEncoder
import math


class Model(nn.Module):
    def __init__(self,opt):
        super(Model, self).__init__()
        if '.embeds' in opt.bert_path:
            self.word_embedding = EmbeddingLayer(opt)
        else:
            self.base_layer = PLMTokenEncoder(opt)
        self.class_num = opt.class_num
        
        self.label_embed = nn.Embedding(opt.all_label_num,opt.embedding_dim)
        self.W1 = nn.Linear(opt.embedding_dim,opt.embedding_dim,bias=False)
        self.W2 = nn.Linear(opt.embedding_dim,opt.embedding_dim,bias=False)
        self.W3 = nn.Linear(opt.embedding_dim,opt.embedding_dim,bias=False)
        self.W4 = nn.Linear(opt.embedding_dim,opt.embedding_dim,bias=False)
        self.W5 = nn.Linear(opt.embedding_dim,opt.embedding_dim,bias=False)
        self.W6 = nn.Linear(2*opt.embedding_dim,opt.embedding_dim)
        self.W7 = nn.Linear(2*opt.embedding_dim,opt.embedding_dim)
        self.V1 = nn.parameter.Parameter(torch.randn(opt.embedding_dim))
        self.V2 = nn.parameter.Parameter(torch.randn(opt.embedding_dim))
        self.entity_embed = nn.Embedding(opt.entity_num,opt.embedding_dim)
        self.bi_gru = nn.GRU(opt.embedding_dim,opt.embedding_dim//2,bidirectional=True,batch_first=True)
        
        self.conv1 = nn.Conv1d(opt.embedding_dim,opt.embedding_dim,kernel_size=3)
        self.conv2 = nn.Conv1d(opt.embedding_dim,opt.embedding_dim,kernel_size=4)
        self.conv3 = nn.Conv1d(opt.embedding_dim,opt.embedding_dim,kernel_size=5)
        
        self.cls_layer1 = nn.Linear(opt.embedding_dim*2, 1024)
        nn.init.xavier_normal_(self.cls_layer1.weight)
        self.cls_layer2 = nn.Linear(1024, opt.class_num)
        nn.init.xavier_normal_(self.cls_layer2.weight)        

        nn.init.xavier_normal_(self.W1.weight)
        nn.init.xavier_normal_(self.W2.weight)
        nn.init.xavier_normal_(self.W3.weight)
        nn.init.xavier_normal_(self.W4.weight)
        nn.init.xavier_normal_(self.W5.weight)
        nn.init.xavier_normal_(self.W6.weight)
        nn.init.xavier_normal_(self.W7.weight)
        

    def forward(self, x:torch.Tensor,
                     attention_mask:torch.Tensor,
                     nodes:torch.Tensor,
                     node_mask:torch.Tensor,
                     D_D:torch.Tensor,
                     D_F:torch.Tensor
                     ):

        batch_size = x.size(0)
        # GCN Encoding
        D = self.label_embed.weight
        F = self.entity_embed(nodes) * node_mask.unsqueeze(2) #[B,E,H]
        D_hat = torch.relu(self.W1(D) + 
                           D_D @ self.W2(D) / (D_D.sum(dim = 1,keepdim = True) + 1e-5)+
                           D_D.T @ self.W3(D) / (D_D.T.sum(dim = 1,keepdim = True) + 1e-5))
        F_hat = torch.relu(self.W4(F) + 
                           (D_F.transpose(1,2)) @ (D_hat.unsqueeze(0).repeat(batch_size,1,1)) / \
                           (D_F.transpose(1,2).sum(dim = 2,keepdim=True)+ 1e-5)) \
                           * node_mask.unsqueeze(2)
        
        F_num = F_hat.shape[1]
        # D_hat [B,D,H]
        # F_hat [B,E,H]
        if hasattr(self,'base_layer'):
            x = self.base_layer(x,attention_mask)
        else:
            x = self.word_embedding(x)
        H1,_ = self.bi_gru(x)
        H1 = torch.mean(H1,dim=1)
        U = torch.tanh(self.W6(torch.cat((F_hat,H1.unsqueeze(1).repeat(1,F_num,1)),dim = 2)))# [B,E,H]
        alpha1 = torch.softmax((U * self.V1.unsqueeze(0).unsqueeze(0)).sum(dim = 2),dim = 1) # [B,E]
        Hf = (alpha1.unsqueeze(1) @ U).squeeze(1)
        # [B,H]
        Y1 = self.conv1(x.transpose(1,2)).transpose(1,2) #[B,S,H]
        Y2 = self.conv2(x.transpose(1,2)).transpose(1,2)
        Y3 = self.conv3(x.transpose(1,2)).transpose(1,2)
        Y = torch.cat((Y1,Y2,Y3),dim = 1)
        Y_num = Y.shape[1]
        Z = torch.tanh(self.W6(torch.cat((Y,Hf.unsqueeze(1).repeat(1,Y_num,1)),dim = 2)))# [B,E,H]
        alpha2 =  torch.softmax((Z * self.V2.unsqueeze(0).unsqueeze(0)).sum(dim = 2),dim = 1) # [B,E]
        Ht = (alpha2.unsqueeze(1) @ Z).squeeze(1)

        H = torch.cat((Hf,Ht),dim = 1)
        hidden_vec = torch.relu(self.cls_layer1(H))
        y_hat = self.cls_layer2(hidden_vec)
        return y_hat