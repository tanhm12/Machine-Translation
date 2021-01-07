from vncorenlp import VnCoreNLP
import logging

logging.basicConfig(level=logging.DEBUG)


class VnSegmentNLP:
    def __init__(self, jar_file='./tokenizer/VnCoreNLP-1.1.1.jar'):
        self.annotator = VnCoreNLP(jar_file, annotators="wseg", max_heap_size='-Xmx2g')

    def word_segment(self, inp: str):
        word_segmented_text = self.annotator.tokenize(inp)
        if len(word_segmented_text) > 1:
            logging.warning("VnSegmentNLP: This sentence was split to more one sentence:" + inp)
        word_segmented_text = word_segmented_text[0]

        return ' '.join(word_segmented_text)