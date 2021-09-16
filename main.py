import numpy as np
import torch
import os
import gensim
import copy
from sklearn.metrics import classification_report
from torch import argmax
from sklearn.metrics import confusion_matrix


from model.base_seq2seq import Seq2SeqModel as Seq2Seq_LSTM
from tokenizer.BPE import BPE_VI, BPE_EN
from tokenizer._tokenizer import Tokenizer
from torch import nn
from tqdm import tqdm
from config import config

from utils import get_text_data


def get_embedding_models(dir):
    return gensim.models.KeyedVectors.load(os.path.join(dir, 'word2vec.kv'))


vi_embedding = get_embedding_models(config.bpe_vi_embedding)
en_embedding = get_embedding_models(config.bpe_en_embedding)

src_embedding = en_embedding
dst_embedding = vi_embedding

bpe_en = BPE_EN(padding=False)
bpe_vi = BPE_VI(padding=False)

tokenizer_en = Tokenizer(dict(zip(en_embedding.index2word, range(len(en_embedding.index2word)))), bpe_en)
tokenizer_vi = Tokenizer(dict(zip(vi_embedding.index2word, range(len(vi_embedding.index2word)))), bpe_vi)

model = Seq2Seq_LSTM(src_embedding=nn.Embedding.from_pretrained(torch.FloatTensor(en_embedding.vectors), padding_idx=1),
                     dst_embedding=nn.Embedding.from_pretrained(torch.FloatTensor(vi_embedding.vectors), padding_idx=1),
                     config=config)
model.to(config.device)

print(model)
#
s = ['But lets face it: At the core of this line of thinking isnt safety -- its sex',
     '--These are parts of their cars.']

# test_len = 10
max_generated_len = 10

x = tokenizer_en.tokenize(s)
print(x)
print(tokenizer_vi.merge([i for i in model.predict(x, max_len=max_generated_len)]))


optimizer = torch.optim.AdamW(model.parameters(), lr=0.001,)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.82)


train_en_text, train_vi_text, valid_en_text, valid_vi_text, test_en_text, test_vi_text = get_text_data()
train_en, valid_en, test_en = [tokenizer_en.tokenize(i) for i in [train_en_text, valid_en_text, test_en_text]]
train_vi, valid_vi, test_vi = [tokenizer_vi.tokenize(i) for i in [train_vi_text, valid_vi_text, test_vi_text]]


def eval(valid_en=valid_en, valid_vi=valid_vi, full_detail=False, confusion=False):
    y_true = []
    y_pred = []
    total_loss = []
    batch_size = config.batch_size

    s = ['But lets face it: At the core of this line of thinking isnt safety -- its sex',
         'Process finished with exit code 0']
    max_generated_len = 20

    x = tokenizer_en.tokenize(s)

    print("Test sample {} -----> {}".format(s, tokenizer_vi.merge(
        [i for i in model.predict(x, max_len=max_generated_len)])))
    print("Validating ...")
    with torch.no_grad():
        for i in tqdm(range(0, len(valid_en), config.batch_size)):
            # print(valid_vi_text[i: i + batch_size])
            outputs, loss = model.forward_and_get_loss(valid_en[i: i + batch_size],
                                                       valid_vi[i: i + batch_size],
                                                       )

            total_loss.append(loss.item())
        # y_pred.extend(argmax(prob, dim=-1).tolist())
        # y_true.extend(batch_labels[i: i + batch_size].tolist())

    # report = classification_report(y_true, y_pred, output_dict=True)
    # print(report['macro avg'])
    # print(report['weighted avg'])
    # if full_detail:
    #     print(report)
    # if confusion:
    #     return confusion_matrix(y_true, y_pred, labels=list(range(87)))

    return np.mean(total_loss)


def train(data=train_en, labels=train_vi, epochs=config.epochs):
    model.train()
    best_dev_loss = 1e9
    total_loss = []
    all_total_loss, all_dev_loss = [], []
    state_dict = copy.copy(model.state_dict())
    print_each = int(len(data) * config.print_interval)
    batch_size = config.batch_size
    for epoch in range(epochs):
        print('epoch:', epoch)
        print_counter = 0
        print_counter_ubound = print_each
        for i in tqdm(range(0, len(data), batch_size)):
            print_counter += batch_size
            outputs, loss = model.forward_and_get_loss(data[i: i + batch_size],
                                                       labels[i: i + batch_size],
                                                       )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())
            if print_counter > print_counter_ubound:
                print('step: {}, loss: {}'.format(i, loss.item()))
                print_counter_ubound += print_each
        scheduler.step()
        print('train loss:', np.mean(total_loss))
        dev_loss = eval(valid_en, valid_vi)
        all_dev_loss.append(dev_loss)
        print('dev_loss:', dev_loss)
        print('\n')
        if dev_loss < best_dev_loss:
            torch.save(model.state_dict(), os.path.join(config.save_dir, 'best-model.pt'))
            best_dev_loss = dev_loss
        all_total_loss.append(total_loss)
        total_loss = []
    model.load_state_dict(torch.load(os.path.join(config.save_dir, 'best-model.pt')))
    model.to(config.device)
    model.eval()

    return all_total_loss, all_dev_loss


train()
eval()

