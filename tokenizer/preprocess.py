from vncorenlp import VnCoreNLP
import logging

logging.basicConfig(level=logging.WARNING)


class VnSegmentNLP:
    def __init__(self, jar_file='./tokenizer/VnCoreNLP-1.1.1.jar'):
        self.annotator = VnCoreNLP(jar_file, annotators="wseg", max_heap_size='-Xmx2g')

    def word_segment(self, inp: str):
        word_segmented_text = self.annotator.tokenize(inp)
        sentences = [' '.join(word) for word in word_segmented_text]
        return ' '.join(sentences)
