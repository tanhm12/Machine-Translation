import gensim
import os

DATA_DIR = '/content/drive/My Drive/aimesoft_training/Flair/corpus/train'
SAVE_DIR = "/content/drive/My Drive/aimesoft_training/week 1/model/model"

files = os.listdir(DATA_DIR)

def get_corpus(file):
    with open(os.path.join(DATA_DIR, file), encoding='utf8') as f:
        text = f.read().split('\n')
    for i in range(len(text)):
        text[i] = text[i].split()
    return text

file = files[0]
corpus = get_corpus(file)  # corpus is [[sent1, sent2, ...], [sent1, sent2, ..],...], sent is list of word
print(corpus[1])

n = 10
embedding_size = 300
vocab_size = 70000
epochs = 10
model = gensim.models.Word2Vec(corpus, size=embedding_size, window=10, iter=epochs,
                               min_count=n, compute_loss=True, seed=22)
for i in range(1, len(files)):
    model.save(SAVE_DIR)
    print('Length of vocab: {}'.format(len(model.wv.vocab)))
    # if "tốt" in model.wv.vocab:
    #     print(model.most_similar("tốt"))
    # if "bắc" in model.wv.vocab:
    #     print(model.most_similar("bắc"))
    file = files[i]
    corpus = get_corpus(file)
    print('First sentence: {}'.format(corpus[1]))
    model.build_vocab(corpus, update=True)
    model.train(corpus, total_examples=model.corpus_count, epochs=epochs)