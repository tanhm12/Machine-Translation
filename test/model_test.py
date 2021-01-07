import numpy as np
import torch

from model.Seq2Seq_LSTM import Seq2SeqModel as Seq2Seq_LSTM
from tokenizer.BPE import BPE_VI, BPE_EN
from tokenizer._tokenizer import Tokenizer
from torch import nn

bpe_en = BPE_EN(padding=False)
bpe_vi = BPE_VI(padding=False)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
model = Seq2Seq_LSTM(src_embedding=nn.Embedding(80000, 128, padding_idx=1),
                     dst_embedding=nn.Embedding(80000, 128, padding_idx=1),
                     device=device)
model.to(device)

test_len = 10
max_generated_len = 20

# s = ['Bánh rán là một món ăn ưa thích của Doraemon', 'Có một vài LeeSin đá sóng âm hụt']
# tokenizer = BPE_VI(padding=False)
s = ['But lets face it: At the core of this line of thinking isnt safety -- its sex', 'Process finished with exit code 0']

tokenizer_en = Tokenizer(bpe_en.symbols, bpe_en)
tokenizer_vi = Tokenizer(bpe_vi.symbols, bpe_vi)
x = tokenizer_en.tokenize(s)
print(x)
print(tokenizer_vi.merge([np.array(i.tolist()) for i in model(x, max_len=max_generated_len)]))

# print(tokenizer.merge(x))