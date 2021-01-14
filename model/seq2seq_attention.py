from typing import List

import torch

from torch import nn
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
import numpy as np

import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, encoder_direction=2, dec_num_layers=2):
        super().__init__()

        self.encoder_direction = encoder_direction
        self.dec_num_layers = dec_num_layers
        self.attn = nn.Linear((enc_hid_dim * self.encoder_direction) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [batch size, encoder_inputs len, enc hid dim * direction]
        # mask = [bach_size, encoder_inputs len]

        encoder_inputs_len = encoder_outputs.shape[1]

        hidden = hidden.unsqueeze(1).repeat(1, encoder_inputs_len, 1)
        # hidden = [batch size, encoder_inputs len, dec hid dim]

        # print(hidden.shape, encoder_outputs.shape)
        attention = self.v(torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))).squeeze(2)
        # attention = [batch size, encoder_inputs len]

        attention = attention.masked_fill(mask == 0, -1e10)

        return F.softmax(attention, dim=1)


class Seq2SeqAttentionModel(nn.Module):
    def __init__(self, src_embedding: nn.Embedding, dst_embedding: nn.Embedding, config):
        super(Seq2SeqAttentionModel, self).__init__()
        self.src_embedding = src_embedding
        self.dst_embedding = dst_embedding

        self.src_embedding_dim = self.src_embedding.embedding_dim
        self.dst_embedding_dim = self.dst_embedding.embedding_dim

        self.output_dim = self.dst_embedding.num_embeddings

        self.bos_idx = config.bos_idx
        self.eos_idx = config.eos_idx

        self.lstm_dim = config.lstm_dim
        self.encoder_direction = config.direction
        # self.encoder_direction = 1
        self.bidirectional = True if self.encoder_direction == 2 else False
        self.num_layers = config.num_layers
        self.max_decoder_inputs_length = config.max_attention_len
        self.max_encoder_inputs_length = config.max_attention_len

        self.encoder = nn.LSTM(self.src_embedding_dim, self.lstm_dim, batch_first=True,
                               bidirectional=self.bidirectional,
                               num_layers=self.num_layers)
        self.attention_layers = Attention(self.lstm_dim, self.lstm_dim * self.encoder_direction,
                                          self.encoder_direction, self.num_layers)
        self.decoder = nn.LSTM(self.dst_embedding_dim, self.lstm_dim * self.encoder_direction, batch_first=True,
                               num_layers=self.num_layers)

        self.linear = nn.Linear(self.lstm_dim * self.encoder_direction + self.lstm_dim * self.encoder_direction,
                                self.output_dim)

        self.softmax = nn.Softmax(dim=-1)
        self.loss_ignore_idx = config.loss_ignore_idx
        self.loss = nn.CrossEntropyLoss(ignore_index=self.loss_ignore_idx)

        self.beam_size = config.beam_size
        self.device = config.device

    def init_weights(self):
        def init_weights_(model):
            for name, param in model.named_parameters():
                if 'weight' in name:
                    nn.init.normal_(param.data, mean=0, std=0.01)
                else:
                    nn.init.constant_(param.data, 0)
        self.apply(init_weights_)

    def forward_and_get_loss(self, x: List[torch.LongTensor], y: List[torch.LongTensor]):
        encoder_outputs, hidden, mask = self.encoder_forward(x)

        for i in range(len(y)):
            if len(y[i]) > self.max_decoder_inputs_length:
                y[i] = torch.cat([y[i][:self.max_decoder_inputs_length], torch.LongTensor([self.eos_idx])])
        decoder_inputs = [sent[:-1] for sent in y]
        decoder_target_outputs = [sent[1:] for sent in y]
        decoder_outputs, hidden = self.decoder_forward(decoder_inputs, hidden, encoder_outputs, mask)

        if self.device == 'cuda':
            decoder_target_outputs = [i.cuda() for i in decoder_target_outputs]

        decoder_target_outputs = pad_sequence(decoder_target_outputs, batch_first=True,
                                              padding_value=self.loss_ignore_idx)

        loss = self.loss(decoder_outputs.permute(0, 2, 1), decoder_target_outputs)

        return self.softmax(decoder_outputs), loss

    def predict(self, x, max_len=20, beam_size=5):
        outputs = []
        for x_i in x:
            outputs.append(self.predict_one_sentence(x_i, max_len, beam_size))

        return outputs

    def predict_one_sentence_(self, x, max_len=20):
        encoder_outputs, hidden, mask = self.encoder_forward([x])
        decoder_inputs = [torch.LongTensor([self.bos_idx])]
        # decoder_inputs shape (1, 1)
        outputs = []

        for i in range(1, max_len):
            decoder_outputs, hidden = self.decoder_forward(decoder_inputs, hidden, encoder_outputs, mask)
            decoder_outputs = torch.topk(decoder_outputs.reshape((-1,)), k=1)
            decoder_outputs = decoder_outputs.indices.tolist()[0]

            outputs.append(decoder_outputs)
            if decoder_outputs == self.eos_idx:
                break
            decoder_inputs = [torch.LongTensor([decoder_outputs])]

        return outputs

    def normalize_prob(self, prob):
        return np.log(1 + prob)

    def predict_one_token(self, decoder_inputs, hidden, encoder_outputs, mask, beam_size=5):
        decoder_outputs, hidden = self.decoder_forward(decoder_inputs, hidden, encoder_outputs, mask)
        decoder_outputs = torch.topk(self.softmax(decoder_outputs).reshape((-1,)), k=beam_size)
        topk_output_indices = decoder_outputs.indices.tolist()
        topk_output_values = decoder_outputs.values.tolist()

        return hidden, topk_output_indices, topk_output_values

    def predict_one_sentence(self, x, max_len=50, beam_size=5):
        encoder_outputs, hidden, mask = self.encoder_forward([x])
        decoder_inputs = [torch.LongTensor([self.bos_idx])]
        # decoder_inputs shape (1, 1)

        hidden, topk_output_indices, topk_output_values = self.predict_one_token(decoder_inputs, hidden,
                                                                                 encoder_outputs, mask, beam_size)
        res = []
        for i in range(len(topk_output_indices)):
            res.append([[topk_output_indices[i]], self.normalize_prob(topk_output_values[i]), hidden])

        for i in range(1, max_len):
            candidates = res[:]
            for pos in range(len(res)):
                input_ids, accumulate_prob, hidden = res[pos]
                input_id = input_ids[-1]
                decoder_inputs = [torch.LongTensor([input_id])]
                if input_id != self.eos_idx:
                    new_hidden, topk_output_indices, topk_output_values = self.predict_one_token(decoder_inputs, hidden,
                                                                                                 encoder_outputs, mask,
                                                                                                 beam_size)
                    for i in range(len(topk_output_indices)):
                        candidates.append([input_ids[:] + [topk_output_indices[i]],
                                           (accumulate_prob + self.normalize_prob(topk_output_values[i])),  # normalize with len
                                           new_hidden])

            candidates.sort(key=lambda x: x[1], reverse=True)
            res = candidates[:beam_size]

        return res[0][0]

    def create_mask(self, encoder_inputs):
        mask = (encoder_inputs != self.src_embedding.padding_idx)
        return mask

    def encoder_forward(self, x):
        x = [i[:self.max_encoder_inputs_length] for i in x]
        if self.device == 'cuda':
            x = [i.cuda() for i in x]

        lens = [len(sent) for sent in x]

        # padding
        x = pad_sequence(x, batch_first=True, padding_value=self.src_embedding.padding_idx)
        mask = self.create_mask(x)
        x = self.src_embedding(x)  # shape: batch * max(lens) * embedding_dim

        # packing
        x = pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)

        # forward
        out_packed, (h, c) = self.encoder(x)
        out, lens_unpack = pad_packed_sequence(out_packed, batch_first=True,
                                               padding_value=self.src_embedding.padding_idx)
        h = h.reshape(self.num_layers, self.encoder_direction, len(lens), -1).transpose(2, 1).reshape(self.num_layers,
                                                                                              len(lens), -1)
        # h = [num layers, seq len, enc hid dim * direction] = c
        c = c.reshape(self.num_layers, self.encoder_direction, len(lens), -1).transpose(2, 1).reshape(self.num_layers,
                                                                                              len(lens), -1)

        return out, (h, c), mask

    def decoder_forward(self, decoder_inputs, hidden, encoder_outputs, mask):
        # decoder_inputs: list of tensor
        # hidden = [num layers, batch size, dec hid dim]
        # encoder_outputs = [batch size, encoder_inputs len, enc hid dim * direction]
        # mask = [batch size, encoder_inputs len]

        # decoder_inputs = [i[:self.max_decoder_inputs_length] for i in decoder_inputs]
        if self.device == 'cuda':
            decoder_inputs = [i.cuda() for i in decoder_inputs]

        decoder_inputs_lens = [len(sent) for sent in decoder_inputs]
        decoder_inputs = pad_sequence(decoder_inputs, batch_first=True,
                                      padding_value=self.dst_embedding.padding_idx)
        # print(decoder_inputs)
        decoder_inputs = self.dst_embedding(decoder_inputs)
        decoder_inputs = pack_padded_sequence(decoder_inputs, decoder_inputs_lens, batch_first=True,
                                              enforce_sorted=False)

        out_packed, hidden = self.decoder(decoder_inputs, hidden)

        out, lens_unpack = pad_packed_sequence(out_packed, batch_first=True,
                                               padding_value=self.dst_embedding.padding_idx)
        # out = [batch size, decoder_inputs seq len, dec hid dim]

        all_attention = []
        attention_hidden_inputs = out.permute(1, 0, 2)
        for i in range(len(attention_hidden_inputs)):
            attention_hidden_input = attention_hidden_inputs[i]
            attention_outputs = self.attention_layers(attention_hidden_input, encoder_outputs, mask).unsqueeze(1)
            # attention_outputs = [batch size, 1, encoder_inputs len]

            weighted = torch.bmm(attention_outputs, encoder_outputs)
            # weighted = [batch size, 1, enc hid dim * direction]
            all_attention.append(weighted)

        if attention_hidden_inputs.requires_grad:
            attention_hidden_inputs.retain_grad()
        all_attention = torch.cat(all_attention, dim=1)
        # all_attention = [batch size, decoder_inputs seq len, enc hid dim * direction]

        # linear forward
        out = self.linear(torch.cat([out, all_attention], dim=-1))
        return out, hidden

    def decoder_forward_get_attention(self, decoder_inputs, hidden, encoder_outputs, mask):
        # decoder_inputs: list of tensor
        # hidden = [num layers, batch size, dec hid dim]
        # encoder_outputs = [batch size, encoder_inputs len, enc hid dim * direction]
        # mask = [batch size, encoder_inputs len]

        # decoder_inputs = [i[:self.max_decoder_inputs_length] for i in decoder_inputs]
        if self.device == 'cuda':
            decoder_inputs = [i.cuda() for i in decoder_inputs]

        decoder_inputs_lens = [len(sent) for sent in decoder_inputs]
        decoder_inputs = pad_sequence(decoder_inputs, batch_first=True,
                                      padding_value=self.dst_embedding.padding_idx)
        # print(decoder_inputs)
        decoder_inputs = self.dst_embedding(decoder_inputs)
        decoder_inputs = pack_padded_sequence(decoder_inputs, decoder_inputs_lens, batch_first=True,
                                              enforce_sorted=False)

        out_packed, hidden = self.decoder(decoder_inputs, hidden)

        out, lens_unpack = pad_packed_sequence(out_packed, batch_first=True,
                                               padding_value=self.dst_embedding.padding_idx)
        # out = [batch size, decoder_inputs seq len, dec hid dim]

        all_attention = []
        attention_hidden_inputs = out.permute(1, 0, 2)
        all_attention_softmax = []
        for i in range(len(attention_hidden_inputs)):
            attention_hidden_input = attention_hidden_inputs[i]
            attention_outputs = self.attention_layers(attention_hidden_input, encoder_outputs, mask).unsqueeze(1)
            all_attention_softmax.append(attention_outputs)
            # attention_outputs = [batch size, 1, encoder_inputs len]

            weighted = torch.bmm(attention_outputs, encoder_outputs)
            # weighted = [batch size, 1, enc hid dim * direction]
            all_attention.append(weighted)

        all_attention = torch.cat(all_attention, dim=1)
        # all_attention = [batch size, decoder_inputs seq len, enc hid dim * direction]

        # linear forward
        out = self.linear(torch.cat([out, all_attention], dim=-1))
        return out, hidden, all_attention_softmax

    def forward_and_get_attention(self, x: List[torch.LongTensor], y: List[torch.LongTensor]):
        encoder_outputs, hidden, mask = self.encoder_forward(x)

        for i in range(len(y)):
            if len(y[i]) > self.max_decoder_inputs_length:
                y[i] = torch.cat([y[i][:self.max_decoder_inputs_length], torch.LongTensor([self.eos_idx])])
        decoder_inputs = [sent[:-1] for sent in y]
        decoder_outputs, hidden, attention_softmax = self.decoder_forward_get_attention(decoder_inputs, hidden,
                                                                                        encoder_outputs, mask)

        return torch.topk(self.softmax(decoder_outputs), dim=-1, k=1), attention_softmax

# class EncoderAttention(nn.Module):
#     def __init__(self, embedding: nn.Embedding, lstm_dim, bidirectional=True, num_layers=2, device='cuda'):
#         super(EncoderAttention, self).__init__()
#         self.embedding = embedding
#
#         self.lstm_dim = lstm_dim
#         self.bidirectional = bidirectional
#         self.encoder_direction = 1 if not self.bidirectional else 2
#         self.num_layers = num_layers
#         self.network = nn.LSTM(self.embedding.embedding_dim, self.lstm_dim, batch_first=True,
#                                bidirectional=self.bidirectional, num_layers=self.num_layers)
#         self.device = device
#
#     def forward(self, inputs, lens, hidden=None):
#         if self.device == 'cuda':
#             inputs = inputs.cuda()
#         inputs = self.embedding(inputs)  # shape: batch * max(lens) * embedding_dim
#
#         # packing
#         inputs = pack_padded_sequence(inputs, lens, batch_first=True, enforce_sorted=False)
#
#         output, (h, c) = self.network(inputs, hidden)
#
#         output, lens_unpack = pad_packed_sequence(output, batch_first=True, padding_value=self.embedding.padding_idx)
#         # output = [batch size, seq len, enc hid dim * direction]
#         # h = [n layers, seq len, enc hid dim * direction] = c
#         h = h.reshape(self.num_layers, self.encoder_direction, len(lens), -1).transpose(2, 1).reshape(self.num_layers,
#                                                                                               len(lens), -1)
#         c = c.reshape(self.num_layers, self.encoder_direction, len(lens), -1).transpose(2, 1).reshape(self.num_layers,
#                                                                                               len(lens), -1)
#         # print(output.shape)
#
#         return output, (h, c)
#
#
# class DecoderAttention(nn.Module):
#     def __init__(self, embedding: nn.Embedding, attention, enc_hid_dim, dec_hid_dim, encoder_direction=2,
#                  num_layers=2, dropout=0.1, device='cuda'):
#         super().__init__()
#
#         self.attention = attention
#
#         self.embedding = embedding
#         self.output_dim = self.embedding.num_embeddings
#
#         self.encoder_direction = encoder_direction
#         self.num_layers = num_layers
#         self.network = nn.LSTM((enc_hid_dim * self.encoder_direction) + self.embedding.embedding_dim, dec_hid_dim,
#                                batch_first=True, num_layers=self.num_layers)
#         self.fc_out = nn.Linear(dec_hid_dim, self.output_dim)
#         self.dropout = nn.Dropout(dropout)
#         self.device = device
#
#     def forward(self, input, hidden, encoder_outputs, mask):
#         # input = [batch size]
#         # hidden = [batch size, dec hid dim]
#         # encoder_outputs = [batch size, encoder_inputs len, enc hid dim * direction]
#         # mask = [batch size, encoder_inputs len]
#         if self.device == 'cuda':
#             input = input.cuda()
#
#         input = input.unsqueeze(0)
#         # input = [1, batch size]
#
#         embedded = self.dropout(self.embedding(input))
#         # embedded = [1, batch size, emb dim]
#
#         a = self.attention(hidden, encoder_outputs, mask)
#         # a = [batch size, encoder_inputs len]
#
#         a = a.unsqueeze(1)
#         # a = [batch size, 1, encoder_inputs len]
#
#         weighted = torch.bmm(a, encoder_outputs)
#         # weighted = [batch size, 1, enc hid dim * 2]
#         weighted = weighted.permute(1, 0, 2)
#         # weighted = [1, batch size, enc hid dim * 2]
#
#         input = torch.cat((embedded, weighted), dim=2)
#         # input = [1, batch size, (enc hid dim * 2) + emb dim]
#
#         output, hidden = self.network(input, hidden.unsqueeze(0))
#         # output = [seq len, batch size, dec hid dim * n layers]
#         # hidden = [n layers, batch size, dec hid dim]
#
#         assert (output == hidden).all()
#
#         # embedded = embedded.squeeze(0)
#         output = output.squeeze(0)
#         # weighted = weighted.squeeze(0)
#
#         # prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
#         prediction = self.fc_out(output)
#         # prediction = [batch size, output dim]
#
#         return prediction, hidden.squeeze(0), a.squeeze(1)
#
#
# class Seq2Seq(nn.Module):
#     def __init__(self, encoder: EncoderAttention, decoder: DecoderAttention,
#                  encoder_inputs_pad_idx, decoder_inputs_pad_idx, device):
#         super().__init__()
#
#         self.encoder = encoder
#         self.decoder = decoder
#         self.encoder_inputs_pad_idx = encoder_inputs_pad_idx
#         self.decoder_inputs_pad_idx = decoder_inputs_pad_idx
#         self.device = device
#
#     def create_mask(self, encoder_inputs):
#         mask = (encoder_inputs != self.encoder.embedding.padding_idx)
#         return mask
#
#     def init_weights(self):
#         def init_weights_(model):
#             for name, param in model.named_parameters():
#                 if 'weight' in name:
#                     nn.init.normal_(param.data, mean=0, std=0.01)
#                 else:
#                     nn.init.constant_(param.data, 0)
#         self.apply(init_weights_)
#
#     def forward(self, encoder_inputs, decoder_inputs, teacher_forcing_ratio=0.5, max_len=30):
#         # encoder_inputs = list of encoder input tensor
#         # decoder_inputs = list of decoder input tensor
#         # teacher_forcing_ratio is probability to use teacher forcing
#
#         encoder_inputs_lens = [len(sent) for sent in encoder_inputs]
#         encoder_inputs = pad_sequence(encoder_inputs, batch_first=True, padding_value=self.encoder.embedding.padding_idx)
#
#         decoder_inputs = pad_sequence(decoder_inputs, batch_first=True, padding_value=self.decoder.embedding.padding_idx)
#
#         if self.device == 'cuda':
#             encoder_inputs = encoder_inputs.cuda()
#             decoder_inputs = decoder_inputs.cuda()
#         decoder_inputs_len = decoder_inputs.shape[1]
#
#         outputs = []
#
#         encoder_outputs, (h, c) = self.encoder(encoder_inputs, encoder_inputs_lens)
#
#         # first input to the decoder is the <sos> tokens
#         input = decoder_inputs[:, 0]
#
#         mask = self.create_mask(encoder_inputs)
#
#         # mask = [batch size, encoder_inputs len]
#
#         if teacher_forcing_ratio == 0 and decoder_inputs.shape[1] == 1:
#             decoder_inputs_len = max_len
#
#         for t in range(1, decoder_inputs_len):
#             output, hidden, _ = self.decoder(input, (h, c), encoder_outputs, mask)
#             # output = [batch size, dec hid dim]
#
#             outputs.append(output.unsqueeze(1))
#
#             teacher_force = np.random.random() < teacher_forcing_ratio
#
#             top1 = output.argmax(1)
#
#             if teacher_force:
#                 input = decoder_inputs[:, t]
#             else:
#                 input = top1
#
#         outputs = torch.cat(outputs, dim=1)
#         # outputs = [batch size, dec input seq length, dec hid dim]
#
#         return outputs, decoder_inputs[:, 1:].contiguous()

