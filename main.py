import numpy as np
import torch
import os
import gensim

from model.Seq2Seq_LSTM import Seq2SeqModel as Seq2Seq_LSTM
from tokenizer.BPE import BPE_VI, BPE_EN
from tokenizer._tokenizer import Tokenizer
from torch import nn
from config import config


bpe_en = BPE_EN(padding=False)
bpe_vi = BPE_VI(padding=False)

tokenizer_en = Tokenizer(bpe_en.symbols, bpe_en)
tokenizer_vi = Tokenizer(bpe_vi.symbols, bpe_vi)


def get_embedding_models(dir='/content/drive/MyDrive/MT/bpe_vi'):
    return gensim.models.KeyedVectors.load(os.path.join(dir, 'word2vec.kv'))


vi_embedding = get_embedding_models(config.bpe_vi_embedding)
en_embedding = get_embedding_models(config.bpe_en_embedding)

src_embedding = en_embedding
dst_embedding = vi_embedding

device = 'cpu'
model = Seq2Seq_LSTM(src_embedding=nn.Embedding.from_pretrained(torch.FloatTensor(vi_embedding.vectors), padding_idx=1),
                     dst_embedding=nn.Embedding.from_pretrained(torch.FloatTensor(en_embedding.vectors), padding_idx=1),
                     config=config)
model.to(device)


