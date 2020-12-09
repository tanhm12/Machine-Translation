import numpy as np
import torch

from model.Seq2Seq_LSTM import Seq2SeqModel as Seq2Seq_LSTM
from tokenizer.BPE import BPE_VI, BPE_EN
from tokenizer.tokenizer import Tokenizer

bpe = BPE_EN(padding=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
vocab_size = len(bpe.symbols)
model = Seq2Seq_LSTM(vocab_size=vocab_size, device=device)
model.to(device)

test_len = 10
max_generated_len = 20

# s = ['Bánh rán là một món ăn ưa thích của Doraemon', 'Có một vài LeeSin đá sóng âm hụt']
# tokenizer = BPE_VI(padding=False)
s = ['But lets face it: At the core of this line of thinking isnt safety -- its sex', 'Process finished with exit code 0']

tokenizer = Tokenizer(bpe.symbols, bpe)
x = tokenizer.tokenize(s)
print(x)
print(tokenizer.merge([np.array(i.tolist()) for i in model(x, max_len=max_generated_len)]))

# print(tokenizer.merge(x))