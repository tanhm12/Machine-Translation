import numpy as np
import torch
import os
import gensim

from sklearn.model_selection import train_test_split

from model.Seq2Seq_LSTM import Seq2SeqModel as Seq2Seq_LSTM
from tokenizer.BPE import BPE_VI, BPE_EN
from tokenizer._tokenizer import Tokenizer
from torch import nn


bpe_en = BPE_EN(padding=False)
bpe_vi = BPE_VI(padding=False)


def get_embedding_models(dir='/content/drive/MyDrive/MT/bpe_vi'):
    return gensim.models.KeyedVectors.load(os.path.join(dir, 'word2vec.kv'))


def get_text_data(test_ratio=0.1, valid_ratio=0.05, seed=222):
    def get_text(dir):
        with open(dir, encoding='utf8') as f:
            text = f.read().split('\n')
        return text

    data_en = []
    data_vi = []
    en_dirs = ['./MT-EV-VLSP2020/basic/data.en', './MT-EV-VLSP2020/evbcorpus/data.en',
              './MT-EV-VLSP2020/indomain-news/dev.en', './MT-EV-VLSP2020/indomain-news/train.en',
              './MT-EV-VLSP2020/indomain-news/tst.en', './MT-EV-VLSP2020/openSub/data.en',
              './MT-EV-VLSP2020/ted-like/data.en', './MT-EV-VLSP2020/wiki-alt/data.en']

    for en_dir in en_dirs:
        vi_dir = en_dir[:-2] + 'vi'
        en_text = get_text(en_dir)
        vi_text = get_text(vi_dir)
        assert len(en_text) == len(vi_text)

        data_en.extend(en_text)
        data_vi.extend(vi_text)

    train_en, test_en, train_vi, test_vi = train_test_split(data_en, data_vi, test_size=test_ratio, random_state=seed)
    train_en, valid_en, train_vi, valid_vi = train_test_split(train_en, test_en, test_size=valid_ratio,
                                                              random_state=seed)

    return train_en, train_vi, valid_en, valid_vi, test_en, test_vi
