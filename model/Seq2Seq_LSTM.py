import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
from typing import List
import numpy as np


class Seq2SeqModel(nn.Module):
    def __init__(self, src_embedding: nn.Embedding, dst_embedding: nn.Embedding, device='cuda'):
        super(Seq2SeqModel, self).__init__()
        self.src_embedding = src_embedding
        self.dst_embedding = dst_embedding

        self.src_embedding_dim = self.src_embedding.embedding_dim
        self.dst_embedding_dim = self.dst_embedding.embedding_dim

        self.output_dim = self.dst_embedding.num_embeddings

        self.bos_idx = 2
        self.eos_idx = 3

        self.lstm_dim = 128
        self.direction = 2
        self.encoder = nn.LSTM(self.src_embedding_dim, self.lstm_dim, batch_first=True, bidirectional=True, num_layers=1)
        self.decoder = nn.LSTM(self.dst_embedding_dim, self.lstm_dim, batch_first=True, bidirectional=True, num_layers=1)
        self.linear = nn.Linear(self.lstm_dim * self.direction, self.output_dim)

        self.softmax = nn.Softmax(dim=-1)
        self.loss_ignore_idx = -100
        self.loss = nn.CrossEntropyLoss(ignore_index=self.loss_ignore_idx)

        self.beam_size = 5
        self.device = device

    def forward(self, x: List[torch.LongTensor], y:  List[torch.LongTensor] = None, max_len=20, beam_size=None):

        lens = [len(sent) for sent in x]

        # padding
        x = pad_sequence(x, batch_first=True, padding_value=self.src_embedding.padding_idx).to(self.device)
        x = self.src_embedding(x)  # shape: batch * max(lens) * embedding_dim

        # packing
        x = pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)

        # forward
        out_packed, (h, c) = self.encoder(x)
        # print(out_packed)
        # out, out_lens = pad_packed_sequence(out_packed, batch_first=True)
        if y is not None:
            decoder_inputs = [sent[:-1].clone() for sent in y]
            decoder_outputs = [sent[1:].clone() for sent in y]

            decoder_inputs_lens = [len(sent) for sent in decoder_inputs]
            decoder_inputs = pad_sequence(decoder_inputs, batch_first=True,
                                          padding_value=self.dst_embedding.padding_idx)
            decoder_inputs = self.dst_embedding(decoder_inputs)
            decoder_inputs = pack_padded_sequence(decoder_inputs, decoder_inputs_lens, batch_first=True,
                                                  enforce_sorted=False)

            decoder_outputs = pad_sequence(decoder_outputs, batch_first=True, padding_value=self.loss_ignore_idx)
            out_packed, (h, c) = self.decoder(decoder_inputs, (h, c))
            # unpack
            out, lens_unpack = pad_packed_sequence(out_packed, batch_first=True,
                                                   padding_value=self.dst_embedding.padding_idx)
            # linear forward
            print(out.shape)
            out = self.linear(out)
            loss = self.loss(out, decoder_outputs)
            return loss
        else:
            # h_n of shape (num_layers * num_directions, batch, hidden_size)
            if beam_size is None:
                beam_size = beam_size
            res = []
            for batch_i in range(h.shape[1]):
                h_i = h[:, batch_i: batch_i + 1, :]
                c_i = c[:, batch_i: batch_i + 1, :]
                print(batch_i)
                res.append(self.forward_sent((h_i.contiguous(), c_i.contiguous()), max_len=max_len, beam_size=beam_size))
            return res

    # def forward_sent(self, states, max_len=200, beam_size=5):
    #     # h, c = states
    #     temp_input = torch.LongTensor([self.bos_idx]).unsqueeze(0).to(self.device)
    #     res = []
    #     while True:
    #         temp_output, states = self.decoder(self.embeddings(temp_input), states)
    #         temp_output_idx = torch.argmax(self.softmax(temp_output.squeeze(0)), dim=-1)
    #         res.append(temp_output_idx)
    #         if temp_output_idx == self.eos_idx or len(res) == max_len:
    #             break
    #         temp_input = torch.LongTensor([temp_output_idx]).unsqueeze(0).to(self.device)
    #
    #     return torch.LongTensor(res).to(self.device)

    def normalize_prob(self, prob):
        return np.log(prob)

    def forward_one_token(self, token, states, beam_size=None):
        if beam_size is None:
            beam_size = self.beam_size
        input_id = torch.LongTensor([[token]]).to(self.device)
        output, states = self.decoder(self.dst_embedding(input_id), states)
        topk_output = torch.topk(self.softmax(output.squeeze(0).squeeze(0)), k=beam_size, dim=-1)
        topk_output_indices = topk_output.indices.tolist()  # for next token
        topk_output_values = topk_output.values.tolist()  # for probability of next token

        return topk_output_indices, topk_output_values, states

    def forward_sent(self, states, max_len=50, beam_size=None):
        # h, c = states
        if beam_size is None:
            beam_size = self.beam_size

        # initialize
        topk_output_indices, topk_output_values, states = self.forward_one_token(self.bos_idx, states, beam_size)
        res = []
        for i in range(len(topk_output_indices)):
            res.append([[topk_output_indices[i]], self.normalize_prob(topk_output_values[i]), states])

        while True:
            candidates = []
            count_eos_token = 0
            for pos in range(len(res)):
                input_ids, accumulate_prob, states = res[pos]
                input_id = input_ids[-1]
                # print(len(input_ids))
                if input_id != self.eos_idx and len(input_ids) < max_len:
                    topk_output_indices, topk_output_values, new_states = self.forward_one_token(self.bos_idx, states,
                                                                                                 beam_size)
                    for i in range(len(topk_output_indices)):
                        candidates.append([input_ids + [topk_output_indices[i]],
                                           (accumulate_prob * len(input_ids) +
                                            self.normalize_prob(topk_output_values[i])) / (len(input_ids)+1),  # normalize with len
                                           new_states])
                elif input_id == self.eos_idx:
                    count_eos_token += 1
                else:
                    input_ids.append(self.eos_idx)
            if count_eos_token == beam_size or len(candidates) == 0:
                break
            candidates.sort(key=lambda x: x[1], reverse=True)
            res = candidates[:beam_size]

        return torch.LongTensor(res[0][0][:-1]).to(self.device)

