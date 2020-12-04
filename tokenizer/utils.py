import json

from gensim.models import KeyedVectors


def read_vocab(vocab_file, lang):
    return json.load(open(vocab_file + '_' + lang + '.json', mode='r', encoding='utf8'))


def read_decode(decode_file, lang):
    return json.load(open(decode_file + '_' + lang + '.json', mode='r', encoding='utf8'))


def load_json(dir):
    return json.load(open(dir, encoding='utf8'))


def dump_json(obj, dir):
    json.dump(obj, open(dir, 'w', encoding='utf8'), ensure_ascii=False)


def load_word2vec_model(tokenizer_type, lang):
    return KeyedVectors.load('./embedding/' + tokenizer_type + "_" + lang + '/word2vec.kv')
