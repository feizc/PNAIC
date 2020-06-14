import torch

class GreedySearch():

    def __init__(self, bot_id, eot_id, batch_size, tag_num=4, rdt_id=None, pad=None,
                 max_len_list=None, device=None):
        self.hypotheses = [[] for _ in range(batch_size)]
        self._batch_offset = torch.arange(batch_size, dtype=torch.long, device=device)
        self.alive_seq = torch.full([batch_size, tag_num, 1], bot_id,
                                    dtype=torch.long, device=device)
        self.is_tag_finished = torch.zeros([batch_size, tag_num], dtype=torch.uint8,
                                           device=device)
        self.is_finished = None

        assert len(max_len_list)  == batch_size * tag_num
        self.max_len_th = torch.tensor(max_len_list, dtype=torch.long, device=device)
        self.max_len_th = self.max_len_th.view(batch_size, tag_num)

        self.eot_id = eot_id
        self.rdt_id = rdt_id
        self.pad = pad
        self.bath_size = batch_size
        self.tag_num = tag_num
        self.device = device

        self.selected_indices = None
        self.done = False

    def search_one_step(self, word_score):
        topk_scores, topk_ids = word_score.max(dim=-1)
        topk_ids = topk_ids.masked_fill(self.is_tag_finished, self.pad)

        self.alive_seq = torch.cat([self.alive_seq, topk_ids.view(-1, self.tag_num, 1)],
                                   dim=-1)
        len_exceed = self.max_len_th <= (self.alive_seq.size(-1) - 1)
        cur_tag_finised = topk_ids.eq(self.eot_id) | len_exceed

        if self.rdt_id is not None:
            rdt_flag = topk_ids.eq(self.rdt_id)
            cur_tag_finised = cur_tag_finised | rdt_flag

        self.is_tag_finished = self.is_tag_finished | cur_tag_finised
        self.is_finished = self.is_tag_finished.all(dim=1)

    def update_finished(self):
        finished = self.is_finished.nonzero().view(-1)
        for i in finished:
            b = self._batch_offset[i]
            self.hypotheses[b].append((None, self.alive_seq[i, :, 1:]))

        self.done = self.is_tag_finished.all()
        if self.done:
            return

        is_alive = ~self.is_finished.view(-1)
        self.alive_seq = self.alive_seq[is_alive]
        self.is_tag_finished = self.is_tag_finished[is_alive]
        self._batch_offset = self._batch_offset[is_alive]
        self.max_len_th = self.max_len_th[is_alive]
        self.selected_indices = is_alive.nonzero().view(-1)

    def get_final_results(self):
        return self.hypotheses
    





