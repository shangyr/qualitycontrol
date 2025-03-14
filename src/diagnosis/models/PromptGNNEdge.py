from abc import ABC
import math

import torch
import torch.nn as nn
from modules import PLMEncoder
from modules import PLMTokenEncoder
# from torch_geometric.nn import GATConv
from modules import NavieGAT
from torch_geometric.nn import GCNConv
import numpy as np

class Model(nn.Module):
    def __init__(self,opt):
        super(Model, self).__init__()

        self.class_num = opt.class_num
        self.base_layer = PLMTokenEncoder(opt)
        self.word_embedding = self.base_layer.get_embedding()
        

        if opt.kge_trainer != '':
            kge = torch.load(opt.entity_embedding_path)
            self.node_embedding = nn.Embedding.from_pretrained(kge['entity_embedding'])
            edge_embedding = kge['relation_embedding']
        else:
            self.node_embedding = nn.Embedding(opt.entity_num,opt.embedding_dim)
            edge_embedding = torch.randn(opt.relation_num, opt.embedding_dim)

        # v2 版本需要对关系进行编码
        if opt.path_type == 'v2':
            weight = torch.cat((edge_embedding, torch.zeros(1, opt.embedding_dim)),dim = 0)
            # + 1 指的是pad
            merge_weight = None
            for i in range(opt.num_hop):
                shape = [1 if i!=j else opt.relation_num + 1 for j in range(opt.num_hop)]
                rev_shape = [opt.relation_num + 1 if j == 1 else 1 for j in shape]
                shape.append(opt.embedding_dim)
                rev_shape.append(1)
                _weight = weight.view(tuple(shape))
                _weight = _weight.repeat(tuple(rev_shape))
                if merge_weight is None:
                    merge_weight = _weight.view(int(math.pow(opt.relation_num + 1,opt.num_hop)),opt.embedding_dim)
                else:
                    merge_weight += _weight.view(int(math.pow(opt.relation_num + 1,opt.num_hop)),opt.embedding_dim)

            self.edge_embedding = nn.Embedding.from_pretrained(merge_weight.contiguous())
        else:
            self.edge_embedding = nn.Embedding.from_pretrained(edge_embedding)

        self.class_embedding = nn.Embedding(opt.class_num,opt.embedding_dim)
        self.gnn_layer = GCNConv(opt.embedding_dim,opt.embedding_dim,concat=False)

        self.cls_layer1 = nn.Linear(opt.embedding_dim, 1024)
        nn.init.xavier_normal_(self.cls_layer1.weight)
        self.cls_layer2 = nn.Linear(1024, opt.class_num)
        nn.init.xavier_normal_(self.cls_layer2.weight)

    def forward(self,sentence,mask,nodes,edges,edge_attr,output_attentions=False):
        word_embed = self.word_embedding(sentence)
        batch_size = word_embed.size(0)
        features = []
        for _nodes,_edges,_edge_attr in zip(nodes,edges,edge_attr):
            _features = self.node_embedding(_nodes)
            _edge_attr = self.edge_embedding(_edge_attr)
            _features = torch.tanh(self.gnn_layer(_features,_edges))
            features.append(_features[:self.class_num])
        features = torch.stack(features,dim = 0)
        features = features + self.class_embedding.weight.unsqueeze(0)
        # features = self.class_embedding.weight.unsqueeze(0).repeat(batch_size,1,1)

        merge_embed = torch.cat((word_embed[:,:1],features,word_embed[:,1:]),dim = 1)

        if output_attentions:
            graph_attentions = None
            text_vec,text_attentions = self.base_layer(None,mask,inputs_embeds=merge_embed, output_attentions = output_attentions) # [batch_size,seq_len, hidden_size]
            hidden_vec = torch.relu(self.cls_layer1(text_vec))[:,1:self.class_num+1] # [batch_size,class_num, hidden_size]
            y_hat = (self.cls_layer2.weight * hidden_vec).sum(dim = 2)# [batch_size,class_num]
            return y_hat,text_attentions,graph_attentions
        else:
            text_vec = self.base_layer(None,mask,inputs_embeds=merge_embed, output_attentions = output_attentions) # [batch_size,seq_len, hidden_size]
            hidden_vec = torch.relu(self.cls_layer1(text_vec))[:,1:self.class_num+1] # [batch_size,class_num, hidden_size]
            y_hat = (self.cls_layer2.weight * hidden_vec).sum(dim = 2)# [batch_size,class_num]
            return y_hat