import torch

from torch import nn
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
import numpy as np

import torch.nn.functional as F


# class Encoder(nn.Module):
#     def __init__(self, embedding: nn.Embedding, lstm_dim, bidirectional=True, num_layers=2):
#         super(Encoder, self).__init__()
#         self.embedding = embedding
#
#         self.lstm_dim = lstm_dim
#         self.bidirectional = bidirectional
#         self.direction = 1 if not self.bidirectional else 2
#         self.num_layers = num_layers
#         self.lstm = nn.LSTM(self.embedding.embedding_dim, self.lstm_dim, batch_first=True, bidirectional=self.bidirectional,
#                                num_layers=self.num_layers)
#
#     def forward(self, inputs, hidden=None):
#
#         lens = [len(sent) for sent in inputs]
#
#         # padding
#         inputs = pad_sequence(inputs, batch_first=True, padding_value=self.embedding.padding_idx)
#         inputs = self.embedding(inputs)  # shape: batch * max(lens) * embedding_dim
#
#         # packing
#         inputs = pack_padded_sequence(inputs, lens, batch_first=True, enforce_sorted=False)
#
#         output, (h, c) = self.lstm(inputs, hidden)
#         h = h.reshape(self.num_layers, self.direction, len(lens), -1).transpose(2, 1).reshape(self.num_layers,
#                                                                                               len(lens), -1)
#         c = c.reshape(self.num_layers, self.direction, len(lens), -1).transpose(2, 1).reshape(self.num_layers,
#                                                                                               len(lens), -1)
#
#         return output, (h, c)
#
#
# class Decoder(nn.Module):
#     def __init__(self, embedding: nn.Embedding, lstm_dim, num_layers=1, dropout=0.1):
#         super(Decoder, self).__init__()
#         self.embedding = embedding
#
#         self.lstm_dim = lstm_dim
#         self.num_layers = num_layers
#         self.network = nn.LSTM(self.embedding.embedding_dim, self.lstm_dim, batch_first=True,
#                                num_layers=self.num_layers)
#         self.linear = nn.Linear(self.lstm_dim, self.embedding.num_embeddings)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, inputs, hidden_states):
#         decoder_inputs = inputs
#
#         decoder_inputs_lens = [len(sent) for sent in decoder_inputs]
#         decoder_inputs = pad_sequence(decoder_inputs, batch_first=True,
#                                       padding_value=self.embedding.padding_idx)
#
#         decoder_inputs = self.embedding(decoder_inputs)
#
#         decoder_inputs = pack_padded_sequence(decoder_inputs, decoder_inputs_lens, batch_first=True,
#                                               enforce_sorted=False)
#         out_packed, hidden_states = self.network(decoder_inputs, hidden_states)
#         # unpack
#         out, lens_unpack = pad_packed_sequence(out_packed, batch_first=True,
#                                                padding_value=self.embedding.padding_idx)
#         # linear forward
#
#         return out
#
#
# class BaseSeq2SeqModel(nn.Module):
#     def __init__(self, encoder: Encoder, decoder: Decoder, bos_idx: int = 0, eos_idx: int = 2, device='cuda'):
#         super(BaseSeq2SeqModel, self).__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#         self.bos_idx = bos_idx
#         self.eos_idx = eos_idx
#
#         self.softmax = nn.Softmax()
#         self.loss = nn.CrossEntropyLoss()
#         self.device = device
#
#     def forward(self, encoder_inputs, decoder_inputs):
#         encoder_outputs, hidden_states = self.encoder(encoder_inputs)
#
#         decoder_outputs = self.decoder(decoder_inputs, hidden_states)
#
#         return decoder_outputs
#
#     def forward_with_loss(self, encoder_inputs, decoder_inputs, decoder_decoder_inputss):
#         decoder_outputs = self.forward(encoder_inputs, decoder_inputs).transpose(2, 1)
#
#         return self.loss(decoder_outputs, decoder_decoder_inputss)
#
#     def predict(self, sentences, max_len=20, beam_size=5):
#         encoder_outputs, hidden_states = self.encoder(sentences)
#
#         # h_n of shape (num_layers, batch, hidden_size * num_directions)
#         result = []
#         h, c = hidden_states
#         for i in range(len(sentences)):
#             h_i = h[:, i: i + 1, :]
#             c_i = c[:, i: i + 1, :]
#             result.append(self.beam_search((h_i.contiguous(), c_i.contiguous()), max_len=max_len, beam_size=beam_size))
#         return result
#
#     def forward_one_token(self, token, states, beam_size=None):
#         if beam_size is None:
#             beam_size = self.beam_size
#         input_id = torch.LongTensor([[token]]).to(self.device)
#         output, states = self.decoder(input_id, states)
#         topk_output = torch.topk(self.softmax(output.squeeze(0).squeeze(0)), k=beam_size, dim=-1)
#         topk_output_indices = topk_output.indices.tolist()  # for next token
#         topk_output_values = topk_output.values.tolist()  # for probability of next token
#
#         return topk_output_indices, topk_output_values, states
#
#     def normalize_prob(self, prob):
#         return np.log(0.5 + prob)
#
#     def search_token(self, token, states, beam_size=5):
#         if self.device == 'cuda':
#             input_id = torch.LongTensor([[token]]).cuda()
#         else:
#             input_id = torch.LongTensor([[token]]).to(self.device)
#
#         output, states = self.decoder(input_id, states)
#         topk_output = torch.topk(self.softmax(output.squeeze(0).squeeze(0)), k=beam_size, dim=-1)
#         topk_output_indices = topk_output.indices.tolist()  # for next token
#         topk_output_values = topk_output.values.tolist()  # for probability of next token
#
#         return topk_output_indices, topk_output_values, states
#
#     def beam_search(self, hidden_states, max_len=20, beam_size=5):
#         # initialize
#         topk_output_indices, topk_output_values, hidden_states = self.forward_one_token(self.bos_idx, hidden_states,
#                                                                                         beam_size)
#         res = []
#         for i in range(len(topk_output_indices)):
#             res.append([[topk_output_indices[i]], self.normalize_prob(topk_output_values[i]), hidden_states])
#
#         while True:
#             candidates = []
#             count_eos_token = 0
#             for pos in range(len(res)):
#                 input_ids, accumulate_prob, hidden_states = res[pos]
#                 input_id = input_ids[-1]
#                 print(len(input_ids))
#                 if input_id != self.eos_idx and len(input_ids) < max_len:
#                     topk_output_indices, topk_output_values, new_hidden_states = self.search_token(self.bos_idx, hidden_states,
#                                                                                                  beam_size)
#                     for i in range(len(topk_output_indices)):
#                         candidates.append([input_ids + [topk_output_indices[i]],
#                                            (accumulate_prob + self.normalize_prob(topk_output_values[i])),
#                                            # normalize with len
#                                            new_hidden_states])
#                 elif input_id == self.eos_idx:
#                     count_eos_token += 1
#                 else:
#                     input_ids.append(self.eos_idx)
#             if count_eos_token == beam_size or len(candidates) == 0:
#                 break
#             candidates.sort(key=lambda x: x[1], reverse=True)
#             res = candidates[:beam_size]
#
#         return np.array(res[0][0])


class EncoderAttention(nn.Module):
    def __init__(self, embedding: nn.Embedding, lstm_dim, bidirectional=True):
        super(EncoderAttention, self).__init__()
        self.embedding = embedding

        self.lstm_dim = lstm_dim
        self.bidirectional = bidirectional
        self.direction = 1 if not self.bidirectional else 2
        self.network = nn.GRU(self.embedding.embedding_dim, self.lstm_dim, batch_first=True,
                              bidirectional=self.bidirectional)

    def forward(self, inputs, lens, hidden=None):

        inputs = self.embedding(inputs)  # shape: batch * max(lens) * embedding_dim

        # packing
        inputs = pack_padded_sequence(inputs, lens, batch_first=True, enforce_sorted=False)

        output, hidden = self.network(inputs, hidden)
        output, lens_unpack = pad_packed_sequence(output, batch_first=True, padding_value=self.embedding.padding_idx)
        hidden = torch.cat([hidden[0, :, :], hidden[1, :, :]], dim=1)
        # print(output.shape)

        return output, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [batch size, encoder_inputs len, enc hid dim * 2]
        # mask = [bach_size, max_enc_inputs_length]

        encoder_inputs_len = encoder_outputs.shape[1]

        hidden = hidden.unsqueeze(1).repeat(1, encoder_inputs_len, 1)
        # hidden = [batch size, encoder_inputs len, dec hid dim]
        # print(hidden.shape, encoder_outputs.shape)
        attention = self.v(torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))).squeeze(2)
        # attention = [batch size, encoder_inputs len]

        attention = attention.masked_fill(mask == 0, -1e10)

        return F.softmax(attention, dim=1)


class DecoderAttention(nn.Module):
    def __init__(self, embedding: nn.Embedding, attention, enc_hid_dim, dec_hid_dim, dropout=0.1):
        super().__init__()

        self.attention = attention

        self.embedding = embedding
        self.output_dim = self.embedding.num_embeddings

        self.network = nn.GRU((enc_hid_dim * 2) + self.embedding.embedding_dim, dec_hid_dim)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + self.embedding.embedding_dim, self.output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs, mask):
        # input = [batch size]
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [batch size, encoder_inputs len, enc hid dim * 2]
        # mask = [batch size, encoder_inputs len]

        input = input.unsqueeze(0)
        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, emb dim]

        a = self.attention(hidden, encoder_outputs, mask)
        # a = [batch size, encoder_inputs len]

        a = a.unsqueeze(1)
        # a = [batch size, 1, encoder_inputs len]

        weighted = torch.bmm(a, encoder_outputs)
        # weighted = [batch size, 1, enc hid dim * 2]
        weighted = weighted.permute(1, 0, 2)
        # weighted = [1, batch size, enc hid dim * 2]

        input = torch.cat((embedded, weighted), dim=2)
        # input = [1, batch size, (enc hid dim * 2) + emb dim]

        output, hidden = self.network(input, hidden.unsqueeze(0))
        # output = [seq len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]

        assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        # prediction = [batch size, output dim]

        return prediction, hidden.squeeze(0), a.squeeze(1)


class Seq2Seq(nn.Module):
    def __init__(self, encoder: EncoderAttention, decoder: DecoderAttention,
                 encoder_inputs_pad_idx, decoder_inputs_pad_idx, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.encoder_inputs_pad_idx = encoder_inputs_pad_idx
        self.decoder_inputs_pad_idx = decoder_inputs_pad_idx
        self.device = device

    def create_mask(self, encoder_inputs):
        mask = (encoder_inputs != self.encoder.embedding.padding_idx)
        return mask

    def init_weights(self):
        def init_weights_(model):
            for name, param in model.named_parameters():
                if 'weight' in name:
                    nn.init.normal_(param.data, mean=0, std=0.01)
                else:
                    nn.init.constant_(param.data, 0)
        self.apply(init_weights_)

    def forward(self, encoder_inputs, decoder_inputs, teacher_forcing_ratio=0.5, max_len=30):
        # encoder_inputs = list of encoder input tensor
        # decoder_inputs = list of decoder input tensor
        # teacher_forcing_ratio is probability to use teacher forcing

        encoder_inputs_lens = [len(sent) for sent in encoder_inputs]
        encoder_inputs = pad_sequence(encoder_inputs, batch_first=True, padding_value=self.encoder.embedding.padding_idx)

        decoder_inputs = pad_sequence(decoder_inputs, batch_first=True, padding_value=self.decoder.embedding.padding_idx)

        decoder_inputs_len = decoder_inputs.shape[1]

        outputs = []

        encoder_outputs, hidden = self.encoder(encoder_inputs, encoder_inputs_lens)

        # first input to the decoder is the <sos> tokens
        input = decoder_inputs[:, 0]

        mask = self.create_mask(encoder_inputs)

        # mask = [batch size, encoder_inputs len]

        if teacher_forcing_ratio == 0 and decoder_inputs.shape[1] == 1:
            decoder_inputs_len = max_len

        for t in range(1, decoder_inputs_len):
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs, mask)
            # output = [batch size, dec hid dim]

            outputs.append(output.unsqueeze(1))

            teacher_force = np.random.random() < teacher_forcing_ratio

            top1 = output.argmax(1)

            if teacher_force:
                input = decoder_inputs[:, t]
            else:
                input = top1

        outputs = torch.cat(outputs, dim=1)
        # outputs = [batch size, dec input seq length, dec hid dim]

        return outputs, decoder_inputs[:, 1:].contiguous()
