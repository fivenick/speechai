import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from libs.nnet.transformer.modules import ScaledDotProductAttention
from libs.nnet.transformer.modules import DFsmn

__author__ = 'yxy'

class MultiHeadAttention(nn.Module):
    ''' multi-head attention module '''
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, has_dfsmn=False):
        super().__init__()
        
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
 
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        # dfsmn parameters
        self.has_dfsmn = has_dfsmn
        self.l_order = 20
        self.r_order = 20
        self.l_stride = 2
        self.r_stride = 2
        self.dfsmn = DFsmn(self.l_order, self.r_order, self.l_stride, self.r_stride, d_v)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v) 

        q,k,v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)

        if mask is not None:
            mask = mask.unsqueeze(1)
        q, attn = self.attention(q,k,v,mask=mask)
        # add dfsmn - yxy
        if self.has_dfsmn:
            p = self.dfsmn(v)
            q += p

        q = q.transpose(1,2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)
        
        return q, attn
class PositionwiseFeedForward(nn.Module):
    ''' a two-feed-forward-layer module '''
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        residual = x
        
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)
        return x
