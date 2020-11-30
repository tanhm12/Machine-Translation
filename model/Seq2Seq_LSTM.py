import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
from typing import List
import numpy as np


class Seq2SeqModel(nn.Module):
    def __init__(self, vocab_size=10000, device='cuda'):
        super(Seq2SeqModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = 100
        self.lstm_dim = 256
        self.output_dim = self.vocab_size
        self.bos_idx = 2
        self.eos_idx = 3
        self.unk_idx = 1

        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        self.direction = 2
        self.encoder = nn.LSTM(self.embedding_dim, self.lstm_dim, batch_first=True, bidirectional=True, num_layers=1)
        self.decoder = nn.LSTM(self.embedding_dim, self.lstm_dim, batch_first=True, bidirectional=True, num_layers=1)
        self.linear = nn.Linear(self.lstm_dim * self.direction, self.output_dim)

        self.softmax = nn.Softmax(dim=-1)
        self.loss_ignore_idx = -100
        self.loss = nn.CrossEntropyLoss(ignore_index=self.loss_ignore_idx)

        self.device = device

    def forward(self, x: List[torch.LongTensor], y:  List[torch.LongTensor] = None, max_len=20):

        lens = [len(sent) for sent in x]

        # padding
        x = pad_sequence(x, batch_first=True, padding_value=self.embeddings.padding_idx)
        x = self.embeddings(x)  # shape: batch * max(lens) * embedding_dim

        # packing
        x = pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)

        # forward
        out_packed, (h, c) = self.encoder(x)
        # out, out_lens = pad_packed_sequence(out_packed, batch_first=True)
        if y is not None:
            decoder_inputs = [sent[:-1].clone() for sent in y]
            decoder_outputs = [sent[1:].clone() for sent in y]

            decoder_inputs_lens = [len(sent) for sent in decoder_inputs]
            decoder_inputs = pad_sequence(decoder_inputs, batch_first=True, padding_value=self.embeddings.padding_idx)
            decoder_inputs = self.embeddings(decoder_inputs)
            decoder_inputs = pack_padded_sequence(decoder_inputs, decoder_inputs_lens, batch_first=True,
                                                  enforce_sorted=False)

            decoder_outputs = pad_sequence(decoder_outputs, batch_first=True, padding_value=self.loss_ignore_idx)
            out_packed, (h, c) = self.decoder(decoder_inputs, (h, c))
            # unpack
            out, lens_unpack = pad_packed_sequence(out_packed, batch_first=True,
                                                   padding_value=self.embeddings.padding_idx)
            # linear forward
            print(out.shape)
            out = self.softmax(self.linear(out)).
            loss = self.loss(out, decoder_outputs)
            return loss
        else:
            # h_n of shape (num_layers * num_directions, batch, hidden_size)
            res = []
            for batch_i in range(h.shape[1]):
                h_i = h[:, batch_i: batch_i + 1, :]
                c_i = c[:, batch_i: batch_i + 1, :]
                res.append(self.forward_sent((h_i.contiguous(), c_i.contiguous()), max_len=max_len))
            return res

    def forward_sent(self, states, max_len=200):
        # h, c = states
        temp_input = torch.LongTensor([self.bos_idx]).unsqueeze(0).to(self.device)
        res = []
        while True:
            temp_output, states = self.decoder(self.embeddings(temp_input), states)
            temp_output_idx = torch.argmax(self.softmax(temp_output.squeeze(0)), dim=-1)
            res.append(temp_output_idx)
            if temp_output_idx == self.eos_idx or len(res) == max_len:
                break
            temp_input = torch.LongTensor([temp_output_idx]).unsqueeze(0).to(self.device)

        return torch.LongTensor(res).to(self.device)


