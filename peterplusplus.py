import math
import torch
import torch.nn as nn

from module import PETER


class PETERPlusPlus(PETER):
    def __init__(self, number_of_rating_classes, *args, **kwargs):
        super(PETERPlusPlus, self).__init__(*args, **kwargs)
        self.rating_embeddings = nn.Embedding(number_of_rating_classes, self.emsize)
        self.rating_embeddings.weight.data.uniform_(-self.initrange, self.initrange)

    def forward(self, user, item, text, gt_rating=None, seq_prediction=True, context_prediction=True, rating_prediction=True):
        '''
        :param user: (batch_size,), torch.int64
        :param item: (batch_size,), torch.int64
        :param text: (total_len - ui_len, batch_size), torch.int64
        :param gt_rating: added for Peter++ compatibility - ground truth rating for teacher forcing, torch.int64
        :param seq_prediction: bool
        :param context_prediction: bool
        :param rating_prediction: bool
        :return log_word_prob: target tokens (tgt_len, batch_size, ntoken) if seq_prediction=True; the last token (batch_size, ntoken) otherwise.
        :return log_context_dis: (batch_size, ntoken) if context_prediction=True; None otherwise.
        :return rating: (batch_size,) if rating_prediction=True; None otherwise.
        :return attns: (nlayers, batch_size, total_len, total_len)
        '''
        device = user.device
        batch_size = user.size(0)
        total_len = self.ui_len + 1 + text.size(0)  # deal with generation when total_len != src_len + tgt_len
        # see nn.MultiheadAttention for attn_mask and key_padding_mask
        attn_mask = self.attn_mask[:total_len, :total_len].to(device)  # (total_len, total_len)
        left = torch.zeros(batch_size, self.ui_len + 1).bool().to(device)  # (batch_size, ui_len)
        right = text.t() == self.pad_idx  # replace pad_idx with True and others with False, (batch_size, total_len - ui_len)
        key_padding_mask = torch.cat([left, right], 1)  # (batch_size, total_len)

        u_src = self.user_embeddings(user.unsqueeze(0))  # (1, batch_size, emsize)
        i_src = self.item_embeddings(item.unsqueeze(0))  # (1, batch_size, emsize)
        w_src = self.word_embeddings(text)  # (total_len - ui_len, batch_size, emsize)
        dummy_rating_src = torch.zeros((1, batch_size, self.emsize)).float().to(device)
        src = torch.cat([u_src, i_src, dummy_rating_src, w_src], 0)  # (total_len, batch_size, emsize)
        src = src * math.sqrt(self.emsize)
        src = self.pos_encoder(src)
        hidden, attns = self.transformer_encoder(src, attn_mask, key_padding_mask)  # (total_len, batch_size, emsize) vs. (nlayers, batch_size, total_len_tgt, total_len_src)
        rating = self.predict_rating(hidden)  # (batch_size,)
        if self.training:
            rating_embedded = self.rating_embeddings((gt_rating - 1).long().to(device).unsqueeze(0))
        else:
            rating_embedded = self.rating_embeddings(torch.clamp(rating - 1, 0, 4).round().long().to(device).unsqueeze(0))
        new_src = torch.cat([u_src, i_src, rating_embedded, w_src], 0)
        new_src = new_src * math.sqrt(self.emsize)
        new_src = self.pos_encoder(new_src)
        hidden, attns = self.transformer_encoder(new_src, attn_mask, key_padding_mask)
        if context_prediction:
            log_context_dis = self.predict_context(hidden)  # (batch_size, ntoken)
        else:
            log_context_dis = None
        if seq_prediction:
            log_word_prob = self.predict_seq(hidden)  # (tgt_len, batch_size, ntoken)
        else:
            log_word_prob = self.generate_token(hidden)  # (batch_size, ntoken)
        return log_word_prob, log_context_dis, rating, attns