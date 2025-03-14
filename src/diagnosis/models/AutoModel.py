from abc import ABC

import torch
import torch.nn as nn
from modules import PLMEncoder

import numpy as np

class Model(nn.Module):
    def __init__(self,opt):
        super(Model, self).__init__()

        self.class_num = opt.class_num
        self.base_layer = PLMEncoder(opt)

        self.cls_layer1 = nn.Linear(opt.embedding_dim, 1024)
        nn.init.xavier_normal_(self.cls_layer1.weight)
        self.cls_layer2 = nn.Linear(1024, opt.class_num)
        nn.init.xavier_normal_(self.cls_layer2.weight)

    def forward(self,sentence,mask):
        text_vec = self.base_layer(sentence,mask) # [batch_size, hidden_size]
        hidden_vec = torch.relu(self.cls_layer1(text_vec))
        y_hat = self.cls_layer2(hidden_vec)
        return y_hat

