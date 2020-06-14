import torch
import torch.nn.functional as F


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


class BeamSearch():

    def __init__(self, bot_id, eot_id, batch_size, device, beam_size=4, max_len_list=None):
        self.hypotheses = [[] for _ in range(batch_size)]
        self.alive_seq = torch.full([batch_size * beam_size, 1], bot_id, dtype=torch.long, device=device)
        self.topk_scores = torch.tensor([0.0]+[float('-inf')] * (beam_size - 1), device=device).repeat(batch_size)
        self.top_ids = torch.empty((batch_size, beam_size), dtype=torch.long, device=device)
        self.is_finished = torch.zeros([batch_size, beam_size], dtype=torch.uint8, device=device)
        self._batch_offset = torch.arange(batch_size, dtype=torch.long, device=device)
        self._beam_offset = torch.arange(0, batch_size * beam_size, step=beam_size, dtype=torch.long, device=device)

        assert len(max_len_list) == batch_size
        self.max_len_th = torch.tensor(max_len_list, dtype=torch.long, device=device)
        self.max_len_th = self.max_len_th.unsqueeze(-1).expand(-1, beam_size)

        self.eot_id = eot_id
        self.batch_size = batch_size
        self.beam_size = beam_size

        self.device = device
        self.selected_indices = None
        self.done = False
        self.alpha = 0.6

    def _lenth_normalize_score(self, score, length, alpha):
        return score / ((length+5.) / (1. + 5.)) ** alpha

    def search_one_step(self, log_prob):
        vocab_size = log_prob.size(-1)
        cur_batch_size = log_prob.size(0) // self.beam_size

        scores = self.topk_scores.view(-1, 1) + log_prob
        self.topk_scores, self.topk_ids = torch.topk(scores.view(cur_batch_size, -1), self.beam_size, dim=-1)

        self.selected_indices = torch.div(self.topk_ids, vocab_size) + self._beam_offset[:cur_batch_size].unsqueeze(1)
        self.selected_indices = self.selected_indices.view(cur_batch_size * self.beam_size)

        self.topk_ids = torch.fmod(self.topk_ids, vocab_size)
        self.alive_seq = torch.cat([self.alive_seq.index_select(0, self.selected_indices),
                                    self.topk_ids.view(-1, 1)], dim=1)

        len_exceed = self.max_len_th <= (self.alive_seq.size(-1) - 1)
        self.is_finished = self.topk_ids.eq(self.eot_id) | len_exceed

    def update_finished(self):
        cur_batch_size = self.alive_seq.size(0) // self.beam_size
        length = self.alive_seq.size(-1) - 1
        predictions =self.alive_seq.view(cur_batch_size, self.beam_size, -1)

        normalize_score = self._lenth_normalize_score(self.topk_scores, length, self.alpha)

        self.topk_scores = self.topk_scores.masked_fill(self.is_finished, -1e10)
        best_alive_score, _ = torch.max(self._lenth_normalize_score(self.topk_scores, self.max_len_th.float(),self.alpha),dim=1)

        non_finished = []
        for i in range(self.is_finished.size(0)):
            b = self._batch_offset[i]
            finished_hyp = self.is_finished[i].nonzero().view(-1)
            for j in finished_hyp:
                self.hypotheses[b].append((float(normalize_score[i,j]), predictions[i, j, 1:]))

            if len(self.hypotheses[b]) >= self.beam_size:
                self.hypotheses[b] = sorted(self.hypotheses[b], key=lambda x:x[0], reverse=True)[:, self.beam_size]
                worst_finished_score = self.hypotheses[b][-1][0]
                if worst_finished_score < best_alive_score[i]:
                    non_finished.append(i)
            else:
                non_finished.append(i)

        if len(non_finished) == 0:
            self.done = True
            return

        non_finished = torch.tensor(non_finished, device=self.device)
        self._batch_offset = self._batch_offset.index_select(0, non_finished)
        self.topk_scores = self.topk_scores.index_select(0, non_finished)
        self.selected_indices = self.selected_indices.view(-1, self.beam_size).index_select(0, non_finished).view(-1)
        self.alive_seq = predictions.index_select(0, non_finished).view(-1, self.alive_seq.size(-1))
        self.max_len_th = self.max_len_th.index_select(0, non_finished)

    def get_final_results(self):
        length = self.alive_seq.size()-1
        normalized_score = self._lenth_normalize_score(self.topk_scores, length, self.alpha)

        self.alive_seq = self.alive_seq.view(-1, self.beam_size, self.alive_seq.size(-1))
        unfinished = ~self.alive_seq[:, :, -1].eq(self.eot_id)
        for i in range(unfinished.size(0)):
            b = self._batch_offset[i]
            unfinished_hyp = unfinished[i].nonzero().view(-1)
            for j in unfinished_hyp:
                self.hypotheses[b].append((float(normalized_score[i, j]), self.alive_seq[i, j, 1:]))

        for b in range(self.batch_size):
            if len(self.hypotheses[b]) > self.beam_size:
                self.hypotheses[b] = sorted(
                    self.hypotheses[b], key=lambda x: x[0], reverse=True)[:self.beam_size]

        return self.hypotheses















