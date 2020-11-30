import re
from bpe.BPE import BPE


class BPE_VI(BPE):
    def __init__(self, vocab_file='./bpe/resources/vocab', decode_file='./bpe/resources/inv_vocab', max_length=256, padding=True):
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

    def tokenizer(self, sent: str):
        sent = sent.strip().split()
        tokens = ['<s>'] + sent + ['</s>']
        tokenized_sent = self.segment_BPE(tokens)
        return tokenized_sent

    def tokenizers(self, sent: list):
        tokenized_sent = []
        for token in sent:
            tmp = self.tokenizer(token)
            tokenized_sent.append(tmp)
        return tokenized_sent

    def merge(self, token):
        token = re.sub(r'(\<s\> | \<\/s\>)', '', token)
        return re.sub(r'@@ ', '', token).strip()

    def merges(self, tokens):
        res = []
        for sent_id in tokens:
            res.append(self.merge(sent_id))
        return res
