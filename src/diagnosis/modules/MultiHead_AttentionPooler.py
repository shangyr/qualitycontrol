from abc import ABC

import torch.nn as nn
import numpy as np


class MatrixVectorScaledDotProductAttention(nn.Module, ABC):

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q, k, v, mask=None):
        """
        q: tensor of shape (n*b, d_k)
        k: tensor of shape (n*b, l, d_k)
        v: tensor of shape (n*b, l, d_v)

        returns: tensor of shape (n*b, d_v), tensor of shape(n*b, l)
        """
        attn = (q.unsqueeze(1) * k).sum(2)  # (n*b, l)
        attn = attn / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = (attn.unsqueeze(2) * v).sum(1)
        return output, attn


class MultiHeadAttPoolLayer(nn.Module, ABC):

    def __init__(self, n_head, d_q_original, d_k_original, dropout=0.1):
        super().__init__()
        assert d_k_original % n_head == 0  # make sure the output dimension equals to d_k_origin
        self.n_head = n_head
        self.d_k = d_k_original // n_head
        self.d_v = d_k_original // n_head

        self.w_qs = nn.Linear(d_q_original, n_head * self.d_k)
        self.w_ks = nn.Linear(d_k_original, n_head * self.d_k)
        self.w_vs = nn.Linear(d_k_original, n_head * self.d_v)

        self.attention = MatrixVectorScaledDotProductAttention(temperature=np.power(self.d_k, 0.5))
        self.dropout = nn.Dropout(dropout)

        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.zeros_(self.w_qs.bias)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.zeros_(self.w_ks.bias)
        nn.init.xavier_uniform_(self.w_vs.weight)
        nn.init.zeros_(self.w_vs.bias)

    def forward(self, q, k, mask=None):
        """
        q: tensor of shape (b, d_q_original)
        k: tensor of shape (b, l, d_k_original)
        mask: tensor of shape (b, l) (optional, default None)
        returns: tensor of shape (b, n*d_v)
        """
        n_head, d_k, d_v = self.n_head, self.d_k, self.d_v

        bs, _ = q.size()
        bs, len_k, _ = k.size()

        qs = self.w_qs(q).view(bs, n_head, d_k)  # (b, n, dk)
        ks = self.w_ks(k).view(bs, len_k, n_head, d_k)  # (b, l, n, dk)
        vs = self.w_vs(k).view(bs, len_k, n_head, d_v)  # (b, l, n, dv)

        qs = qs.permute(1, 0, 2).contiguous().view(n_head * bs, d_k)
        ks = ks.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_k)
        vs = vs.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1)
        output, attn = self.attention(qs, ks, vs, mask=mask)
        output = output.view(n_head, bs, d_v)
        output = output.permute(1, 0, 2).contiguous().view(bs, n_head * d_v)  # (b, n*dv)
        output = self.dropout(output)

        return output, attn