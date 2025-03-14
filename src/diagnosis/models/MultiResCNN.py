# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_ as xavier_uniform
import json
from math import floor
import numpy as np
from modules import EmbeddingLayer,PLMTokenEncoder

class OutputLayer(nn.Module):
    def __init__(self,input_size, num_classes):
        super(OutputLayer, self).__init__()

        self.U = nn.Linear(input_size, num_classes)
        xavier_uniform(self.U.weight)

        self.final = nn.Linear(input_size, num_classes)
        xavier_uniform(self.final.weight)

    def forward(self, x):
        alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)
        m = alpha.matmul(x)
        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
        return y


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride, use_res, dropout):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=int(floor(kernel_size / 2)), bias=False),
            nn.BatchNorm1d(outchannel),
            nn.Tanh(),
            nn.Conv1d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=int(floor(kernel_size / 2)), bias=False),
            nn.BatchNorm1d(outchannel)
        )

        self.use_res = use_res
        if self.use_res:
            self.shortcut = nn.Sequential(
                        nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm1d(outchannel)
                    )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.left(x)
        if self.use_res:
            out += self.shortcut(x)
        out = torch.tanh(out)
        out = self.dropout(out)
        return out


class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()

        if '.embeds' in opt.bert_path:
            self.word_embedding = EmbeddingLayer(opt)
        else:
            self.base_layer = PLMTokenEncoder(opt)

        self.conv = nn.ModuleList()

        filter_sizes = [3,5,9,15,19,25]
        num_filter_maps = 50

        self.filter_num = len(filter_sizes)
        for filter_size in filter_sizes:
            one_channel = nn.ModuleList()
            tmp = nn.Conv1d(opt.embedding_dim, opt.embedding_dim, kernel_size=filter_size,
                            padding=int(floor(filter_size / 2)))
            xavier_uniform(tmp.weight)
            one_channel.add_module('baseconv', tmp)


            conv_dimension = [opt.embedding_dim, num_filter_maps]
            for idx in range(len(conv_dimension) - 1):
                tmp = ResidualBlock(conv_dimension[idx], conv_dimension[idx + 1], filter_size, 1, True,
                                    opt.dropout_rate)
                one_channel.add_module('resconv-{}'.format(idx), tmp)

            self.conv.add_module('channel-{}'.format(filter_size), one_channel)

        self.output_layer = OutputLayer(self.filter_num * num_filter_maps,opt.class_num)

    def forward(self, x,attention_mask):
        if hasattr(self,'word_embedding'):
            x = self.word_embedding(x)
        else:
            x = self.base_layer(x,attention_mask)
        
        x = x * attention_mask.unsqueeze(2)

        x = x.transpose(1, 2)

        conv_result = []
        for conv in self.conv:
            tmp = x
            for idx, md in enumerate(conv):
                if idx == 0:
                    tmp = torch.tanh(md(tmp))
                else:
                    tmp = md(tmp)
            tmp = tmp.transpose(1, 2)
            conv_result.append(tmp)
        x = torch.cat(conv_result, dim=2)
        y_hat = self.output_layer(x)
        return y_hat

