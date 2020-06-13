import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.function as F

from util import sequence_mask, tag_mask, init_weights
from transformer import *


class PNAIC(nn.Module):

    def __init__(self, enc_layers, dec_layers, hidden_size, head_num, ffn_size,
                 src_emb_conf, tgt_emb_conf=None, eot_id=None, rdt_id=None,
                 tag_num=4, dropout=0.1, use_label_smoothing=True, smooth_rate=0.15):
        super(PNAIC, self).__init__()
        if tgt_emb_conf is None:
            self.embedding = self._init_embedding(src_emb_conf)
            self.src_embedding = self.embedding
            self.tgt_embedding = self.embedding
        else:
            self.src_embedding = self._init_embedding(src_emb_conf)
            self.tgt_embedding = self._init_embedding(tgt_emb_conf)
        self.tag_embedding = nn.Embedding(tag_num, hidden_size)
        self.pos_embedding = PositionEmbedding(hidden_size)
        self.encoder = TransformerEncoder(enc_layers, hidden_size, head_num, ffn_size, dropout)
        self.decoder = TransformerDecoder(dec_layers, hidden_size, head_num, ffn_size, dropout, causal=False)
        self.dropout = nn.Dropout(dropout)
        self.eot_id = eot_id
        self.rdt_id = rdt_id
        self.tag_num = tag_num
        self.hidden_size = hidden_size
        self.use_label_smoothing = use_label_smoothing
        self.smooth_rate = smooth_rate
        self.apply(init_weights)

    def _init_embedding(self, emb_conf):
        vocab_size = emb_conf['vocab_size']
        emb_size = emb_conf['emb_size']
        padding_idx = emb_conf.get('padding_idx', None)
        return nn.Embedding(vocab_size, emb_size, padding_idx)

    def forward(self, src_seq, tgt_seq, src_lens, label, tag_id=None, tgt_pos=None, tag_lens=None):
        max_len = max(int(src_lens.max()), 1)
        src_seq = src_seq[:, :max_len]
        src_enc, src_mask = self.encode(src_seq, src_lens)
        tgt_dec, logit = self.decode(tgt_seq, src_enc, src_mask, pos=tgt_pos, tag_id=tag_id, tag_lens=tag_lens)
        log_prob = F.log_softmax(logit, dim=-1).transpose(1,2)
        loss = self.comput_loss(log_prob, label)

        return loss

    def compute_loss(self, log_prob, label):
        if self.use_label_smoothing:
            loss = smooth_loss(log_prob, label, pad=self.tgt_embedding.padding_idx, eps=self.smooth_rate)
        else:
            loss_func = nn.NLLLoss(ignore_index=self.tgt_embedding.padding_idx, reduction='sum')
            loss = loss_func(log_prob, label)

        return loss

    def embed(self, input, embedding, step=None, pos=None, tag_id=None):
        batch_size, length = input.size()
        emb = embedding(input) * self.hidden_size ** 0.5

        if pos is not None:
            step = pos.contiguous().view(-1)

        pos_emb = self.pos_embedding(length, step=step).to(input.device)

        if pos is not None:
            pos_emb = pos_emb.view(batch_size, length, -1)
        else:
            pos_emb = pos_emb.unsqueeze(0).expand(batch_size, -1, -1)

        if tag_id is not None:
            tag_emb = self.tag_embedding(tag_id)
            pos_emb = pos_emb + tag_emb

        emb = self.dropout(emb + pos_emb)

        return emb

    def encode(self, src, src_lens):
        src_emb = self.embed(src, self.src_embedding)
        src_enc, src_mask = self.encoder(src_emb, src_lens)
        return src_enc, src_mask

    def decode(self, tgt, src_enc, src_mask=None, step=None, pos=None, tag_id=None, tag_lens=None):
        padding_idx = self.tgt_embedding.padding_idx
        tgt_mask = tgt.data.eq(padding_idx)
        tgt_mask = tgt_mask | tgt.data.eq(self.eot_id)
        tgt_mask = tgt_mask | tgt.data.eq(self.rdt_id)
        tgt_mask = tgt_mask.unsqueeze(1)

        if step is None:
            tgt_max_len = tgt.size(1)
            tgt_len = tgt_max_len - torch.sum(tgt_mask, dim=-1)
            tgt_len = tgt_len.view(-1)
            tgt_mask = 1 - tag_mask(tgt_len, tgt_max_len, self.tag_num, tag_lens)

        tgt_emb = self.embed(tgt, self.tgt_embedding, step=step, pos=pos, tag_id=tag_id)
        tgt_dec = self.decoder(tgt_emb, src_enc, src_pad_mask=src_mask, tgt_pad_mask=tgt_mask, step=step)
        logit = F.linear(tgt_dec, self.tgt_embedding.weight)

        return tgt_dec, logit


