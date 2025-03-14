from abc import ABC

import torch
import torch.nn as nn

from modules import PLMTokenEncoder

import numpy as np

class Model(nn.Module):
    def __init__(self,opt):
        super(Model, self).__init__()

        self.class_num = opt.class_num
        self.base_layer = PLMTokenEncoder(opt)
        self.word_embedding = self.base_layer.get_embedding()

        self.cls_layer1 = nn.Linear(opt.embedding_dim, 1024)
        nn.init.xavier_normal_(self.cls_layer1.weight)
        self.cls_layer2 = nn.Linear(1024, opt.class_num)
        nn.init.xavier_normal_(self.cls_layer2.weight)

        self.class_embedding = nn.Embedding(opt.class_num,opt.embedding_dim)

        self.prompt_cls_layer1 = nn.Linear(opt.embedding_dim, 256)
        nn.init.xavier_normal_(self.prompt_cls_layer1.weight)
        self.prompt_cls_layer2 = nn.Linear(256, 2)
        nn.init.xavier_normal_(self.prompt_cls_layer2.weight)

    def forward(self,sentence,mask,mask_idx):
        batch_size = sentence.size(0)
        word_embed = self.word_embedding(sentence)
        disease_mask = torch.ones(batch_size, self.class_num, dtype = torch.long).to(mask.device)
        mask = torch.cat((mask[:,:1], disease_mask,mask[:,1:]),dim = 1)
        disease_features = self.class_embedding.weight.unsqueeze(0).repeat(batch_size,1,1)
        merge_embed = torch.cat((word_embed[:,:1],disease_features,word_embed[:,1:]),dim = 1)
        # text_vec = self.base_layer(sentence,mask,mask_idx = mask_idx) # [batch_size,seq_len, hidden_size]
        text_vec = self.base_layer(None,mask,inputs_embeds=merge_embed,mask_idx=mask_idx)
        prompt_vec = text_vec[np.arange(batch_size),mask_idx] # [batch_size,hidden_size]
        
        hidden_vec = torch.relu(self.cls_layer1(text_vec))[:,1:self.class_num+1] # [batch_size,class_num, hidden_size]
        y_hat = (self.cls_layer2.weight * hidden_vec).sum(dim = 2)# [batch_size,class_num]

        prompt_hidden_vec = torch.relu(self.prompt_cls_layer1(prompt_vec))
        y_hat_prompt = self.prompt_cls_layer2(prompt_hidden_vec)
        return y_hat,y_hat_prompt

