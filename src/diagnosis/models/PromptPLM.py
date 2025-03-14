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

        self.cls_layer1 = nn.Linear(opt.embedding_dim, 1024)
        nn.init.xavier_normal_(self.cls_layer1.weight)
        self.cls_layer2 = nn.Linear(1024, opt.class_num)
        nn.init.xavier_normal_(self.cls_layer2.weight)

        self.prompt_cls_layer1 = nn.Linear(opt.embedding_dim, 256)
        nn.init.xavier_normal_(self.prompt_cls_layer1.weight)
        self.prompt_cls_layer2 = nn.Linear(256, 2)
        nn.init.xavier_normal_(self.prompt_cls_layer2.weight)

    def forward(self,sentence,mask,mask_idx):
        batch_size = sentence.size(0)
        text_vec = self.base_layer(sentence,mask,mask_idx = mask_idx,use_disease_token=False) # [batch_size,seq_len, hidden_size]
        prompt_vec = text_vec[np.arange(batch_size),mask_idx] # [batch_size,hidden_size]

        text_vec = text_vec[np.arange(batch_size),0] # [batch_size,hidden_size]
        
        hidden_vec = torch.relu(self.cls_layer1(text_vec))
        y_hat = self.cls_layer2(hidden_vec)

        prompt_hidden_vec = torch.relu(self.prompt_cls_layer1(prompt_vec))
        y_hat_prompt = self.prompt_cls_layer2(prompt_hidden_vec)
        return y_hat,y_hat_prompt

