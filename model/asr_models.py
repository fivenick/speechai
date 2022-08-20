''' define the transformer model '''
import torch
import torch.nn as nn
import numpy as np
from libs.nnet.transformer.layers import EncoderLayer, DecoderLayer 

__author__ = 'yxy'

def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)
def get_src_pad_mask(src_seq, pad_idx):
    return (src_seq != pad_idx)[:,:,:1].reshape(src_seq.shape[0],1, -1)

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        #print(f"position encoding, x shape: {x.shape}, position shape: {self.pos_table.shape}")
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Encoder(nn.Module):
    ''' a encoder model with self attention mechanism. '''
    def __init__(
           self, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1, n_position=2000, n_sublayers= 1):
        super().__init__()
        
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position) 
        self.dropout = nn.Dropout(p=dropout)

        self.beg_layer = EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)

        self.n_sublayers = n_sublayers
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src_seq, src_mask, return_attns = False):
        enc_slf_attn_list = []
        
        enc_output = self.dropout(self.position_enc(src_seq))
        enc_output = enc_output.to(torch.float32)
        enc_output = self.layer_norm(enc_output)

        enc_output, enc_slf_attn = self.beg_layer(enc_output, slf_attn_mask=src_mask)
        
        for enc_layer in self.layer_stack:
            for _ in range(self.n_sublayers):
                enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
                enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,

class Decoder(nn.Module):
    ''' a decoder model with self attention mechanism.  '''
    def __init__(self,  n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
                   d_model, d_inner, pad_idx, n_position=1000, dropout=0.1, scale_emb=False, n_sublayers= 1):
        super().__init__()
        
        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)        
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)

        self.n_sublayers = n_sublayers
        
        self.layer_stack = nn.ModuleList([
             DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout) 
             for _ in range(n_layers)])
        
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):
        dec_slf_attn_list, dec_enc_attn_list = [], []
    
        dec_output = self.trg_word_emb(trg_seq)
        if self.scale_emb:
            dec_output *= self.d_model ** 0.5
        dec_output = self.dropout(self.position_enc(dec_output))
        dec_output = self.layer_norm(dec_output)
        
        
        for dec_layer in self.layer_stack:
            for _ in range(self.n_sublayers):
                dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                    dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
                dec_slf_attn_list += [dec_slf_attn] if return_attns else []
                dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,



class Transformer(nn.Module):
    ''' a sequence to sequence model with attention mechanism. '''
    def __init__(self, n_trg_vocab, trg_pad_idx,
            d_word_vec=480, d_model=480, d_inner=2048,
            n_head=4, d_k=64, d_v=64, dropout=0.1, n_position=200,
            trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True,
            scale_emb_or_prj='prj', n_enc_layers=2, n_dec_layers=3, 
            n_dec_sublayers=1, n_enc_sublayers=3):

        super().__init__()
        
        self.trg_pad_idx = trg_pad_idx
 
        assert scale_emb_or_prj in ['emb', 'prj', 'none']
        scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False

        self.d_model = d_model
        
        self.encoder = Encoder(
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_enc_layers, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout, n_sublayers= n_enc_sublayers)

        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_dec_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, dropout=dropout, scale_emb=scale_emb, n_sublayers=n_dec_sublayers) 

        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)
     
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        assert d_model == d_word_vec
        if trg_emb_prj_weight_sharing:
            self.trg_word_prj.weight = self.decoder.trg_word_emb.weight
        if emb_src_trg_weight_sharing:
            self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight

    def forward(self, src_seq, trg_seq):
        #print(f"trg pad idx: {self.trg_pad_idx}")
        #src_mask = (src_seq != 0)[:,:,:1].view(src_seq.shape[0],1, -1)
        src_mask = get_src_pad_mask(src_seq, 0)
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

        enc_output, *_ = self.encoder(src_seq, src_mask)
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)

        seq_logit = self.trg_word_prj(dec_output)
        if self.scale_prj:
            seq_logit *= self.d_model ** -0.5
        return seq_logit.view(-1, seq_logit.size(2))




