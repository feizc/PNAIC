import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from util import sequence_mask, init_weights


def smooth_loss(log_prob, label, pad, eps=0.15):
    mask = label == pad

    nclass = log_prob.size(1)
    smoothed_label = torch.zeros_like(log_prob)
    smoothed_label = smoothed_label + eps / (nclass - 1)

    class_range = torch.arange(0, nclass).to(log_prob.device)
    selected_mask = class_range.view(1, nclass, 1).expand_as(log_prob) == label.unsqueeze(1).expand_as(log_prob)

    smoothed_label = smoothed_label.masked_fill(selected_mask, 1-eps)
    smoothed_label = smoothed_label.masked_fill(mask.unsqueeze(1), 0.)

    loss = torch.sum(-log_prob * smoothed_label)

    return loss


# Step means that if the tag length is L, we can encode position i, i+L, i+3L... rather than continue i, i+1, i+2...
class PositionEmbedding(nn.Module):

    def __init__(self, emb_size, max_timescale=1.0e4):
        super(PositionEmbedding, self).__init__()
        self.emb_size = emb_size
        self.max_timescale = max_timescale

    def forward(self, length=None, step=None):
        assert length is not None or step is not None
        if length is not None:
            pos = torch.arange(0., length).unsqueeze(-1)
        if isinstance(step, int):
            pos = torch.tensor([[step]], dtype=torch.float)
        elif step is not None:
            pos = step.unsqueeze(-1).float()

        dim = torch.arange(0., self.emb_size, 2.).unsqueeze(0) / self.emb_size
        dim = dim.to(pos.device)

        sin = torch.sin(pos / torch.pow(self.max_timescale, dim))
        cos = torch.cos(pos / torch.pow(self.max_timescale, dim))

        pos_emb = torch.stack((sin, cos), -1).view(pos.size(0), -1)
        pos_emb = pos_emb[:, :self.emb_size]

        return pos_emb


class FeedForward(nn.Module):

    def __init__(self, input_size, hidden_size, dropout=0.1):
        super(FeedForward, self).__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.layer_2 = nn.Linear(hidden_size, input_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        mid = F.relu(self.layer_1(inputs))
        return self.layer_2(self.dropout(mid))


class MultiHeadAttention(nn.Module):

    def __init__(self, head_num, hidden_size, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert hidden_size % head_num == 0
        self.head_size = hidden_size // head_num
        self.hidden_size = hidden_size
        self.head_num = head_num

        self.linear_keys = nn.Linear(hidden_size, head_num * self.head_size)
        self.linear_values = nn.Linear(hidden_size, head_num * self.head_size)
        self.linear_query = nn.Linear(hidden_size, head_num * self.head_size)
        self.final_linear = nn.Linear(hidden_size, hidden_size)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, key, value, query, mask=None, layer_cache=None, att_type=None, mask_self=False):
        batch_size = key.size(0)
        head_size = self.head_size
        head_num = self.head_num

        # Dimension adjust.
        def shape(x):
            return x.view(batch_size, -1, head_num, head_size).transpose(1,2)

        def unshape(x):
            return x.transpose(1,2).contiguous().view(batch_size, -1, head_num * head_size)

        # Project key, value, and query.
        if layer_cache is not None:
            if att_type == 'self':
                query = self.linear_query(query)
                key = self.linear_keyss(key)
                value = self.linear_values(value)
                key, value = shape(key), shape(value)

                if layer_cache['self_keys'] is not None:
                    key = torch.cat((layer_cache['self_keys'], key), dim=2)
                if layer_cache['self_values'] is not None:
                    value = torch.cat((layer_cache['self_values'], value), dim=2)

                layer_cache['self_keys'] = key
                layer_cache['self_values'] = value

            elif att_type == 'context':
                query = self.linear_query(query)
                if layer_cache['memory_keys'] is None:
                    key = self.linear_keys(key)
                    value = self.linear_values(value)
                    key, value = shape(key), shape(value)

                    layer_cache['memory_keys'] = key
                    layer_cache['memory_values'] = value
                else:
                    key, value = layer_cache['memory_keys'], layer_cache['memory_values']
        else:
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)
            key, value = shape(key), shape(value)
        query = shape(query)

        # Calculate and scale dot product score.
        query = query / head_size ** 0.5
        scores = torch.matmul(query, key.transpose(2, 3)).float()
        if mask is not None:
            if att_type == 'self' and layer_cache is not None:
                if layer_cache['mask'] is not None:
                    mask = torch.cat((layer_cache['mask'], mask), dim=2)
                layer_cache['mask'] = mask

            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask, -1e18)

        if mask_self and att_type == 'self' and layer_cache is None:
            diag_mask = torch.diagflat(torch.ones(key.size(2), dtype=torch.uint8, device=scores.device))
            diag_mask = diag_mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(diag_mask, -1e18)

        # Apply attention dropout and compute context vector
        att = self.softmax(scores).to(query.dtype)
        drop_att = self.dropout(att)
        context_original = torch.matmul(drop_att, value)
        context = unshape(context_original)
        output = self.final_linear(context)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, hidden_size, heads, ffn_size, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.self_att = MultiHeadAttention(heads, hidden_size, dropout=dropout)
        self.feed_forward = FeedForward(hidden_size, ffn_size, dropout)
        self.layer_norm_1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        input_norm = self.layer_norm_1(inputs)
        context = self.self_att(input_norm, input_norm, input_norm, mask=mask, att_type='self')
        mid = self.dropout(context) + inputs

        mid_norm = self.layer_norm_2(mid)
        ffn_out = self.feed_forward(mid_norm)
        output = self.dropout(ffn_out) + mid

        return output
















