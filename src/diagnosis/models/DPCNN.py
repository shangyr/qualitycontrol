from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from modules import EmbeddingLayer,PLMTokenEncoder

class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        if '.embeds' in opt.bert_path:
            self.word_embedding = EmbeddingLayer(opt)
        else:
            self.base_layer = PLMTokenEncoder(opt)
        self.conv_region = nn.Conv2d(1, opt.embedding_dim, (3, opt.embedding_dim), stride=1)
        self.conv = nn.Conv2d(opt.embedding_dim, opt.embedding_dim, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()
        self.fc = nn.Linear(opt.embedding_dim, opt.class_num)


    def forward(self, x, attention_mask):
        if hasattr(self,'word_embedding'):
            x = self.word_embedding(x)
        else:
            x = self.base_layer(x,attention_mask)
        x = x * attention_mask.unsqueeze(2)
        x = x.unsqueeze(1)  # [batch_size, 250, seq_len, 1]
        x = self.conv_region(x)  # [batch_size, 250, seq_len-3+1, 1]

        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        while x.size()[2] >= 2:
            x = self._block(x)
        
        xx = x.squeeze(-1).squeeze(-1)  # [batch_size, num_filters(250)] 最后一个数据时 batch维度可能也被squeeze掉了
        xxx = self.fc(xx)

        return xxx

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)

        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)

        # Short Cut
        x = x + px
        return x