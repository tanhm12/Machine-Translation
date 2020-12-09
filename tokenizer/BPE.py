import re

from abc import ABC
from typing import Union
from tqdm import tqdm
from tokenizer import utils


class BPE(ABC):
    def __init__(self, vocab_file='./tokenizer/resources/vocab', decode_file='./tokenizer/resources/inv_vocab',
                 max_length=256, padding=True, lang='vi'):
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


class BPE_EN(BPE):
    def __init__(self, vocab_file='./tokenizer/resources/vocab', decode_file='./tokenizer/resources/inv_vocab', max_length=256, padding=True):
        super().__init__(vocab_file=vocab_file, decode_file=decode_file, max_length=max_length, padding=padding, lang='en')

    def segment_BPE(self, tokens):
        """
        :param tokens: a list of word
        :return: a tokenized sentence, a list ID tokenized words
        """
        outputs = []
        for token in tokens:
            start, end = 0, len(token)
            cur_output = []
            # Segment token with the longest possible sub words from symbols
            while start < len(token) and start < end:
                if token[start: end] in self.symbols:
                    cur_output.append(token[start: end])
                    start = end
                    end = len(token)
                else:
                    end -= 1
            if start < len(token):
                cur_output.append('<unk>')
            outputs.append(' '.join(cur_output))
        return ' '.join(outputs)

    def _tokenize(self, sent: str):
        sent = re.sub(r'\s+', ' ', sent.strip())
        sent = re.sub(r' ', ' Ġ', sent)
        sent = sent.split()
        tokens = ['<s>'] + sent + ['</s>']
        tokenized_sent = self.segment_BPE(tokens)
        return tokenized_sent

    def _merge(self, token):
        token = re.sub(r'(<s> |<\/s>)', '', token)
        token = re.sub(r' ', '', token)
        return re.sub(r'Ġ', ' ', token)


class BPE_VI(BPE):
    def __init__(self, vocab_file='./tokenizer/resources/vocab', decode_file='./tokenizer/resources/inv_vocab', max_length=256, padding=True):
        super().__init__(vocab_file=vocab_file, decode_file=decode_file, max_length=max_length, padding=padding, lang='vi')

    def segment_BPE(self, tokens):
        """
        :param tokens: a list of word
        :return: a tokenized sentence, a list ID tokenized words
        """
        outputs = []
        for token in tokens:
            start, end = 0, len(token)
            cur_output = []
            # Segment token with the longest possible sub words from symbols
            while start < len(token) and start < end:
                if end == len(token) and token[start:end] in self.symbols.keys():
                    cur_output.append(token[start:end])
                    start = end
                    break
                elif end < len(token) and token[start:end] + '@@' in self.symbols.keys():
                    cur_output.append(token[start: end] + '@@')
                    start = end
                    end = len(token)
                else:
                    end -= 1
            if start < len(token):
                cur_output.append('<unk>')
            outputs.append(' '.join(cur_output))
        return ' '.join(outputs)

    def _tokenize(self, sent: str):
        sent = sent.strip().split()
        tokens = ['<s>'] + sent + ['</s>']
        tokenized_sent = self.segment_BPE(tokens)
        return tokenized_sent

    def _merge(self, token):
        token = re.sub(r'(\<s\> | \<\/s\>)', '', token)
        return re.sub(r'@@ ', '', token).strip()

