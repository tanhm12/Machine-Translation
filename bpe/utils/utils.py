import json


def read_vocab(vocab_file, lang):
    return json.load(open(vocab_file + '_' + lang + '.json', mode='r', encoding='utf8'))


def read_decode(decode_file, lang):
    return json.load(open(decode_file + '_' + lang + '.json', mode='r', encoding='utf8'))