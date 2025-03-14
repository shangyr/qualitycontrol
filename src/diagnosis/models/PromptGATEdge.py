from abc import ABC
import math

import torch
import torch.nn as nn
from modules import PLMEncoder
from modules import PLMTokenEncoder
# from torch_geometric.nn import GATConv
from modules import NavieGAT
import numpy as np

class Model(nn.Module):
    def __init__(self,opt):
        super(Model, self).__init__()

        self.class_num = opt.class_num
        self.relation_num = opt.relation_num
        self.path_type = opt.path_type
        self.num_hop = opt.num_hop
        self.base_layer = PLMTokenEncoder(opt)
        self.word_embedding = self.base_layer.get_embedding()
        
        if opt.kge_trainer != '':
            kge = torch.load(opt.entity_embedding_path)
            self.node_embedding = nn.Embedding.from_pretrained(kge['entity_embedding'])
            edge_embedding = torch.cat((kge['relation_embedding'].data.cpu(),torch.zeros(1, opt.embedding_dim)),dim = 0) 
            self.edge_embedding = nn.Embedding.from_pretrained(edge_embedding)
        else:
            self.node_embedding = nn.Embedding(opt.entity_num,opt.embedding_dim)
            edge_embedding = torch.randn(opt.relation_num, opt.embedding_dim)
            edge_embedding = torch.cat((edge_embedding,torch.zeros(1, opt.embedding_dim)),dim = 0) 
            self.edge_embedding = nn.Embedding.from_pretrained(edge_embedding)
        
        self.class_embedding = nn.Embedding(opt.class_num,opt.embedding_dim)
        self.gat_layer = NavieGAT(opt.embedding_dim,opt.embedding_dim,opt.embedding_dim,concat=False)

        self.cls_layer1 = nn.Linear(opt.embedding_dim, 1024)
        nn.init.xavier_normal_(self.cls_layer1.weight)
        self.cls_layer2 = nn.Linear(1024, opt.class_num)
        nn.init.xavier_normal_(self.cls_layer2.weight)

    def forward(self,sentence,mask,nodes,edges,edge_attr,output_attentions=False):
        word_embed = self.word_embedding(sentence)
        batch_size = word_embed.size(0)
        features = []
        graph_attentions = []
        for _nodes,_edges,_edge_attr in zip(nodes,edges,edge_attr):
            _features = self.node_embedding(_nodes)
            if self.path_type == 'v1':
                _edge_embed = self.edge_embedding(_edge_attr)
            else: # 'v2'
                _edge_embed = None
                for _ in range(self.num_hop):
                    embed = self.edge_embedding(_edge_attr % (self.relation_num + 1))
                    _edge_attr //= (self.relation_num + 1)
                    if _edge_embed is None:
                        _edge_embed = embed
                    else:
                        _edge_embed += embed

            if output_attentions:
                _features,(_,_graph_attentions) = self.gat_layer(_features,_edges,_edge_embed, output_attentions)
                graph_attentions.append(_graph_attentions)
            else:
                _features = self.gat_layer(_features,_edges,_edge_embed, output_attentions)
            _features = torch.tanh(_features)
            features.append(_features[:self.class_num])
        features = torch.stack(features,dim = 0)
        features = features + self.class_embedding.weight.unsqueeze(0)
        # features = self.class_embedding.weight.unsqueeze(0).repeat(batch_size,1,1)

        merge_embed = torch.cat((word_embed[:,:1],features,word_embed[:,1:]),dim = 1)
        if output_attentions:
            text_vec,text_attentions = self.base_layer(None,mask,inputs_embeds=merge_embed, output_attentions = output_attentions) # [batch_size,seq_len, hidden_size]
            hidden_vec = torch.relu(self.cls_layer1(text_vec))[:,1:self.class_num+1] # [batch_size,class_num, hidden_size]
            y_hat = (self.cls_layer2.weight * hidden_vec).sum(dim = 2)# [batch_size,class_num]
            return y_hat,text_attentions,graph_attentions
        else:
            text_vec = self.base_layer(None,mask,inputs_embeds=merge_embed, output_attentions = output_attentions) # [batch_size,seq_len, hidden_size]
            hidden_vec = torch.relu(self.cls_layer1(text_vec))[:,1:self.class_num+1] # [batch_size,class_num, hidden_size]
            y_hat = (self.cls_layer2.weight * hidden_vec).sum(dim = 2)# [batch_size,class_num]
            return y_hat
