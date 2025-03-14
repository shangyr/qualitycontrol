from abc import ABC

import torch
import torch.nn as nn
from graph import BaseGraph

# from fengshen import LongformerModel

import numpy as np

# from modules import GAT
from torch_geometric.nn import GATConv
import time
from modules import DiagGNN
from modules import PLMWithMaskEncoder
from transformers import AutoTokenizer
from graph import BaseGraph

class Model(nn.Module, ABC):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.model_name = 'GraphModel'
        self.class_num = opt.class_num
        self.embedding_dim = opt.embedding_dim

        self.label_nodes = opt.label_nodes
        
        node_type_embed_dim = 100

        self.base_layer = PLMWithMaskEncoder(opt)
        # self.base_layer_textcnn = TextCNNEncoder(opt)
        
        self.entity_embed = nn.Embedding(opt.entity_num, opt.embedding_dim)
        self.entity_type_embed = nn.Embedding(opt.entity_num, node_type_embed_dim)
        self.logic_entity_mlp = nn.Sequential(
            nn.Linear(opt.embedding_dim,1024),
            nn.ReLU(),
            nn.Linear(1024,opt.embedding_dim)
        )
        self.node_type_w = nn.Linear(opt.embedding_dim,opt.embedding_dim)

        self.dropout = nn.Dropout(opt.dropout_rate)

        # GCN融合标签共现关系
        # self.graph_embedding = nn.Embedding(self.graph_nodes_num, self.embedding_dim)
        self.gnn = DiagGNN(k_layer=3,node_dim=opt.embedding_dim,type_dim=node_type_embed_dim,
                           score_dim=100,edge_dim=node_type_embed_dim,n_edges_type=50)

        self.multi_head_attn = nn.MultiheadAttention(opt.embedding_dim,1,batch_first=True)

        self.cls_layer = nn.Linear(self.embedding_dim,self.embedding_dim)
        # self.gcn = nn.parameter.Parameter(torch.eye(self.class_num))
        self.cls_layer2 = nn.Linear(self.embedding_dim,self.class_num)

        self.logic_entity_mlp[0].apply(self.init_weights)
        self.logic_entity_mlp[2].apply(self.init_weights)
        self.node_type_w.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self,sentence,
                     mask,
                     nodes,
                     node_mask,
                     node_types,
                     edges,
                     edge_types,
                     prompts):
        # [1 * (class_num+entity_num) * 768] * [batch_size, (class_num+entity_num),1]
        # -> [batch_size * (class_num+entity_num) * 768]
        text_vec = self.base_layer(sentence,mask) # [batch_size, hidden_size]
        pool = []
        for i,(_nodes,_node_mask,_node_types,_edges,_edge_types) in enumerate(zip(
                                                nodes,node_mask,node_types,edges,edge_types)):
            if _nodes.shape[1] > 1:
                logic_node_idx = torch.where(_node_mask[:,1].bool())[0]
                kg_node_num = logic_node_idx[0]
            else:
                kg_node_num  = _nodes.shape[0]
            
            node_embeds = self.entity_embed(_nodes) # [node_num,3]
            logic_embeds= self.logic_entity_mlp(node_embeds[kg_node_num:])
            node_embeds = torch.cat((node_embeds[:kg_node_num],logic_embeds))
            node_embeds = node_embeds @ self.node_type_w.weight # 
            node_embeds = (node_embeds * _node_mask.unsqueeze(2)).sum(dim =1)
            
            plm_output = text_vec[i].unsqueeze(0)
            node_embeds = torch.cat((plm_output,node_embeds),dim =0)
            node_type_embeds = self.entity_type_embed(_node_types)
            if self.training:
                gnn_nodes, _, _ = self.gnn(node_embeds,node_type_embeds,_edges,_edge_types)
            else:
                gnn_nodes, _ =self.gnn(node_embeds,node_type_embeds,_edges,_edge_types)
            gnn_nodes = gnn_nodes.unsqueeze(0)

            plm_output = plm_output.unsqueeze(0)
            _pool = plm_output[:,0] + gnn_nodes[:,0]
            # _pool,_ = self.multi_head_attn(plm_output,gnn_nodes,gnn_nodes)
            _pool = self.dropout(_pool)
            # _pool = _pool.squeeze(0)
            pool.append(_pool)
        pool = torch.cat(pool,dim = 0)
    
        pool = self.cls_layer(pool)
        pool = torch.relu(pool)
        logic = self.cls_layer2(pool)

        return logic
