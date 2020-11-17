from abc import ABC
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

    def tokenizer(self, sent: str, return_sent=False):
        pass

    def tokenizers(self, sent: str, return_sent=False):
        pass

    def merge(self, sent_id):
        pass

    def merges(self, sent_ids):
        pass