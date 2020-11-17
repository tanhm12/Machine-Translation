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
        outputs, outputs_id = [], []
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
            outputs_id += [self.symbols[s] for s in cur_output]
        return ' '.join(outputs), outputs_id

    def tokenizer(self, sent: str, return_sent=False):
        sent = sent.strip().split()
        tokens = ['<s>'] + sent + ['</s>']
        tokenized_sent, outputs_id = self.segment_BPE(tokens)
        if self.max_length >= 0:
            if len(outputs_id) > self.max_length:
                tmp = len(outputs_id) - self.max_length
                outputs_id = outputs_id[:-tmp-1] + outputs_id[-1]
            else:
                tmp = self.max_length - len(outputs_id)
                outputs_id += [1] * tmp
        if return_sent:
            return tokenized_sent, outputs_id
        else:
            return outputs_id

    def tokenizers(self, sent: list, return_sent=False):
        tokenized_sent, outputs_id = [], []
        for token in sent:
            if return_sent:
                tmp1, tmp2 = self.tokenizer(token, return_sent=return_sent)
                tokenized_sent.append(tmp1)
            else:
                tmp2 = self.tokenizer(token, return_sent=return_sent)
            outputs_id.append(tmp2)
        if return_sent:
            return tokenized_sent, outputs_id
        else:
            return outputs_id

    def merge(self, sent_id):
        i = 1
        token = ''
        while sent_id[i] != 2:
            token += self.decode[str(sent_id[i])] + ' '
            i += 1
        return re.sub(r'@@ ', '', token).strip()

    def merges(self, sent_ids):
        res = []
        for sent_id in sent_ids:
            res.append(self.merge(sent_id))
        return res
