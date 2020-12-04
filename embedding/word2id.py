from gensim.models import KeyedVectors
import torch
import numpy as np


def load_word2vec_model(tokenizer, lang):
    return KeyedVectors.load('./embedding/' + tokenizer + "_" + lang + '/word2vec.kv')


class WordIdConversion:
    def __init__(self, tokenizer: str, lang: str):
        """
        Word - ID Conversion
        :param tokenizer: name of tokenizer type. 'space' or 'bpe'
        :param lang:    language 'vi' or 'en'
        """
        assert tokenizer == 'space' or tokenizer == 'bpe'
        assert lang == 'vi' or lang == 'en'

        self.tokenizer = tokenizer
        self.lang = lang
        self.model = load_word2vec_model(tokenizer, lang)
        self.bos = self.model.vocab['<s>'].index
        self.eos = self.model.vocab['</s>'].index
        self.unk = self.model.vocab['<unk>'].index

    def sent2id(self, sent: list):
        """
        convert a list (batch) of sentence to a list of LongTensor
        :param sent:
        :return:
        """
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
            if char in self.model.vocab:
                res[i] = self.model.vocab[char].index
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
            res.append(self.model.index2word[i])
        return ' '.join(res)

    def id2sent(self, _id: list):
        """
        convert a list of numpy array of ID to a list of sentences
        :param _id:
        :return:
        """
        res = []
        for i in _id:
            res.append(self._id2sent(i))
        return res

