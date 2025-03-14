import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_ as xavier_uniform
import json
from math import floor
from modules import EmbeddingLayer
from modules import PLMTokenEncoder
from torch_geometric.nn import GCNConv
from torch_geometric.utils import remove_self_loops
# device = torch.device('cpu')


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

class AttentionLayer(nn.Module):
    def __init__(self, num_filter_maps,class_num):
        super(AttentionLayer, self).__init__()
        attn_d = 256

        self.w = nn.Parameter(torch.FloatTensor(attn_d, num_filter_maps))  # da x embed_size
        xavier_uniform(self.w)
        self.u = nn.Parameter(torch.FloatTensor(class_num, attn_d))  # L x da
        xavier_uniform(self.u)

    def forward(self, x):
        # x:
        Z = torch.tanh(torch.matmul(self.w, x))            # [batch_size,seq_len,attn_d]
        A = torch.softmax(torch.matmul(self.u, Z), dim=2)  # [batch_size,seq_len,labels_num]
        V = torch.matmul(x, A.transpose(1, 2))             # [batch_size,labels_num,filter_num*feature_size]
        V = V.transpose(1, 2)

        return V


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride, use_res, dropout):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=int(floor(kernel_size / 2)), bias=False),
            nn.BatchNorm1d(outchannel),
        )

        self.use_res = use_res
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.left(x)
        if self.use_res:
            out += x
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
        self.entity_embed = nn.Embedding(opt.entity_num+opt.class_num,opt.embedding_dim)
        self.class_num = opt.class_num
        self.encoders = nn.ModuleList()
        self.num_entity = opt.entity_num

        filter_sizes = [3,5,9]
        num_filter_maps = 100

        filter_num = len(filter_sizes)
        for filter_size in filter_sizes:
            filter_size = int(filter_size)
            encoder = nn.ModuleList()

            # Convolution Layer
            tmp = nn.Conv1d(opt.embedding_dim, num_filter_maps, kernel_size=filter_size,
                            padding=int(floor(filter_size / 2)))
            xavier_uniform(tmp.weight)
            encoder.add_module('baseconv', tmp)

            # Batch Normalization
            norm = nn.BatchNorm1d(num_filter_maps)
            encoder.add_module('norm', norm)

            # Residual Block
            conv_dimension = [num_filter_maps,num_filter_maps, num_filter_maps] # 2
            for idx in range(len(conv_dimension) - 1):
                tmp = ResidualBlock(conv_dimension[idx], conv_dimension[idx + 1], filter_size, 1, True,
                                    opt.dropout_rate)
                encoder.add_module('resconv-{}'.format(idx), tmp)

            # Label Attention
            attention = AttentionLayer(num_filter_maps,opt.class_num)
            encoder.add_module('labelattn', attention)

            self.encoders.add_module('encoder-{}'.format(filter_size), encoder)

        # GCN
        self.gc1 = GCNConv(opt.embedding_dim,opt.embedding_dim)
        self.gc2 = GCNConv(opt.embedding_dim, opt.embedding_dim)

        # Output Layer
        ffn_size = 128
        self.final1 = nn.Linear(num_filter_maps * filter_num + opt.embedding_dim, ffn_size)
        xavier_uniform(self.final1.weight)
        self.final2 = nn.Linear(ffn_size, opt.class_num)
        xavier_uniform(self.final2.weight)

        self.dropout = nn.Dropout(p=opt.dropout_rate)

    def forward(self, x,attention_mask, nodes,edges):

        if hasattr(self,'base_layer'):
            x = self.base_layer(x,attention_mask)
        else:
            x = self.word_embedding(x)

        x = x.transpose(1, 2)

        encoder_result = []
        for encoder in self.encoders:
            tmp = x
            for idx, md in enumerate(encoder):
                if idx == 0:
                    tmp = torch.tanh(md(tmp))
                else:
                    tmp = md(tmp)
            encoder_result.append(tmp)

        V = torch.cat(encoder_result, dim=2)    # concat encoder results

        out_graphs = []
        for _nodes,_edges in zip(nodes,edges):
            # GCN
            node_embedding = self.entity_embed(_nodes)
            _edges = remove_self_loops(_edges)[0]
            out_graph = F.relu(self.gc1(node_embedding, _edges))
            out_graph = self.dropout(out_graph)
            out_graph = self.gc2(out_graph, _edges)
            out_graph = out_graph[:self.class_num, :].unsqueeze(0)
            out_graphs.append(out_graph)

        out_graphs = torch.cat(out_graphs,dim = 0)
        out = torch.cat((out_graphs, V), dim=2)

        # output layer
        V = torch.relu(self.final1(out))
        y_hat = self.final2.weight.mul(V).sum(dim=2).add(self.final2.bias)
        y_hat = self.dropout(y_hat)
        return y_hat
