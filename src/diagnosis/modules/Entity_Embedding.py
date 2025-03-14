import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EntityEmbeddingLayer(nn.Module):
    def __init__(self,opt):
        super(EntityEmbeddingLayer, self).__init__()
        if hasattr(opt,'pretrain_entity_path') and '.npz' in opt.pretrain_entity_path:
            word_embeddings = np.load(opt.pretrain_entity_path)['entity_embed']
            pretrain_embed = torch.tensor(word_embeddings,dtype=torch.float32)
            self.word_embedding = nn.Embedding.from_pretrained(pretrain_embed)
        else:
            self.word_embedding = nn.Embedding(opt.entity_num,opt.embedding_dim)
        self.dropout = nn.Dropout(opt.dropout_rate)
    def forward(self, entity_ids):
        embed = self.word_embedding(entity_ids)
        return self.dropout(embed)
