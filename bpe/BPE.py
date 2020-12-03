from abc import ABC
from typing import Union

from tqdm import tqdm

from bpe.utils import utils


class BPE(ABC):
    def __init__(self, vocab_file='./bpe/resources/vocab', decode_file='./bpe/resources/inv_vocab', max_length=256, padding=True, lang='vi'):
        self.symbols = utils.read_vocab(vocab_file, lang)
        self.decode = utils.read_decode(decode_file, lang)
        self.padding = padding
        if not padding:
            self.max_length = -1
        else:
            self.max_length = max_length

    def segment_BPE(self, tokens):
        """
        :param tokens: a list of word
        :return: a tokenized sentence, a list ID tokenized words
        """
        pass

    def _tokenize(self, sent: str):
        pass

    def tokenize(self, sent: Union[list, str]):
        if type(sent) is str:
            return [self._tokenize(sent)]
        else:
            tokenized_sent = []
            for token in tqdm(sent):
                tmp = self._tokenize(token)
                tokenized_sent.append(tmp)
            return tokenized_sent

    def _merge(self, token):
        pass

    def merge(self, tokens: Union[list, str]):
        if type(tokens) is str:
            return [self._merge(tokens)]
        else:
            res = []
            for sent_id in tokens:
                res.append(self._merge(sent_id))
            return res


