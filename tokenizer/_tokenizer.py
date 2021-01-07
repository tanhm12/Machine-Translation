import torch
import numpy as np

from typing import Union, List
from abc import ABC
from tqdm import tqdm
from tokenizer.utils import *
from tokenizer.BPE import BPE_EN, BPE_VI
from tokenizer.preprocess import VnSegmentNLP


class SpaceTokenizer(ABC):
    def __init__(self, vocab: dict, unk_token='<unk>'):
        self.vocab = vocab
        self.unk_token = unk_token

    def _tokenize(self, sent: str):
        sent = sent.split()
        for i, word in enumerate(sent):
            if word not in self.vocab:
                sent[i] = self.unk_token
        return sent

    def tokenize(self, sent: Union[list, str]):
        if type(sent) is str:
            return [self._tokenize(sent)]
        else:
            tokenized_sent = []
            for token in tqdm(sent):
                tmp = self._tokenize(token)
                tokenized_sent.append(tmp)
            return tokenized_sent

    def _merge(self, token: List[str]):
        return ' '.join(token)

    def merge(self, tokens: Union[list, str]):
        if type(tokens) is str:
            return [self._merge(tokens)]
        else:
            res = []
            for sent_id in tokens:
                res.append(self._merge(sent_id))
            return res


class Tokenizer(ABC):
    def __init__(self, vocab: dict, tokenizer=None, preprocess=False):
        self.tokenizer = tokenizer

        self.vocab = vocab
        self.bos = self.vocab['<s>']
        self.eos = self.vocab['</s>']
        self.unk = self.vocab['<unk>']
        self.index2word = {self.vocab[i]: i for i in self.vocab}
        if preprocess:
            self.vnSegment = VnSegmentNLP()
        else:
            self.vnSegment = None

    def from_pretrained(self, lang='en', tokenizer_type='bpe'):
        if tokenizer_type == 'bpe':
            if lang == 'en':
                self.tokenizer = BPE_EN(padding=False)
            else:
                self.tokenizer = BPE_VI(padding=False)
        elif tokenizer_type == 'space':
            self.tokenizer = SpaceTokenizer(self.vocab)

    def tokenize(self, sent: Union[list, str]):
        if self.vnSegment is not None:
            if type(sent) is str:
                sent = self.vnSegment.word_segment(sent)
            else:
                n_sent = [self.vnSegment.word_segment(s) for s in sent]
                sent = n_sent
        sent_tokenized = self.tokenizer.tokenize(sent)
        return self.sent2id(sent_tokenized)

    def merge(self, tokens: Union[list, np.ndarray]):
        tokens = self.id2sent(tokens)
        return self.tokenizer.merge(tokens)

    def sent2id(self, sent: Union[list, str]):
        """
        convert a list (batch) of sentence to a list of LongTensor
        :param sent:
        :return:
        """
        if type(sent) is str:
            return [self._sent2id(sent)]
        else:
            res = []
            for s in sent:
                res.append(self._sent2id(s))
            return res

    def _sent2id(self, sent: str):
        """
        support for sent2id func
        :param sent:
        :return:
        """
        _l = sent.strip().split()
        res = np.zeros(len(_l), dtype='int')
        for i, char in enumerate(_l):
            if char in self.vocab:
                res[i] = self.vocab[char]
            else:
                res[i] = self.unk
        return torch.from_numpy(res).type(torch.LongTensor)

    def _id2sent(self, _id: np.ndarray):
        """
        support for id2sent func
        :param _id:
        :return:
        """
        res = []
        for i in _id:
            res.append(self.index2word[i])
        return ' '.join(res)

    def id2sent(self, _id: Union[list, np.ndarray]):
        """
        convert a list of numpy array of ID to a list of sentences
        :param _id:
        :return:
        """
        if type(_id) is np.ndarray:
            return [self._id2sent(_id)]
        else:
            res = []
            for i in _id:
                res.append(self._id2sent(i))
            return res




