import torch
import torch.nn as nn


def sequence_mask(lengths, max_len=None):
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size,1)
            .lt(lengths.unsqueeze(1)))


# At each time step, our PNAIC generates a word for each incomplete tag.
def tag_mask(lengths, max_len, tag_num, tag_lens):
    batch_size = lengths.size(0)
    lengths = lengths.view(-1)
    max_len = max_len or lengths.max()
    seq_mask = sequence_mask(lengths, max_len)
    seq_mask = seq_mask.unsqueeze(1)*seq_mask.unsqueeze(-1)

    tag_lens = tag_lens.to(lengths.device)
    cumlen = torch.cumsum(tag_lens, dim=-1)
    zero_tensor = torch.zeros((batch_size,1), dtype=cumlen.dtype, device=cumlen.device)
    tag_start = torch.cat((zero_tensor, cumlen[:, :-1]), dim=-1)
    tag_end = cumlen

    len_range = torch.arange(0, max_len).to(lengths.device)
    len_range = len_range.view(1, 1, max_len).expand(batch_size, tag_num, -1)

    upper_bound_mask = len_range < tag_end.unsqueeze(-1)
    lower_bound_mask = len_range >= tag_start.unsqueeze(-1)

    gen_order = len_range - tag_start.unsqueeze(-1)
    gen_order = gen_order * upper_bound_mask.long() * lower_bound_mask.long()
    gen_order = torch.sum(gen_order, dim=1)

    mask = gen_order.unsqueeze(-1) >= gen_order.unsqueeze(1)
    mask = mask * seq_mask

    return mask


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()



