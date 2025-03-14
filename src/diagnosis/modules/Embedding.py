import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.tokenizer import Tokenizer

class EmbeddingLayer(nn.Module):
    def __init__(self,opt):
        super(EmbeddingLayer, self).__init__()
        assert '.embeds' in opt.bert_path
        tokenizer = Tokenizer.from_pretrained(opt.bert_path)
        pretrain_embed = torch.tensor(tokenizer.embedding_weight(),dtype=torch.float32)
        vocab_size,_ = pretrain_embed.shape
        if opt.use_pretrain_embed_weight:
            self.word_embedding = nn.Embedding.from_pretrained(pretrain_embed,padding_idx=tokenizer.pad_token_id)
        else:
            self.word_embedding = nn.Embedding(vocab_size,768,padding_idx=tokenizer.pad_token_id)
        self.dropout = nn.Dropout(opt.dropout_rate) 

    def forward(self, sentence):
        embed = self.word_embedding(sentence)
        return self.dropout(embed)
    