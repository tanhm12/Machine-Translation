import numpy as np
import torch

from model.Seq2Seq_LSTM import Seq2SeqModel as Seq2Seq_LSTM
from tokenizer.BPE import BPE_VI, BPE_EN

device = 'cuda' if torch.cuda.is_available() else 'cpu'
vocab_size = 10000
model = Seq2Seq_LSTM(vocab_size=vocab_size, device=device)
model.to(device)

test_len = 10
max_generated_len = 30

# s = ['Bánh rán là một món ăn ưa thích của Doraemon', 'Có một vài LeeSin đá sóng âm hụt']
# tokenizer = BPE_VI(padding=False)
s = ['But lets face it: At the core of this line of thinking isnt safety -- its sex', 'Process finished with exit code 0']
bpe = BPE_EN(padding=False)
# x = [torch.randint(0, vocab_size, [np.random.randint(1, 30)]).to(device) for _ in range(test_len)]
# print(x[0].dtype)
x = bpe.tokenize(s)
# x = [torch.LongTensor(_).to(device) for _ in x]
print(x)

# print(tokenizer.merge(model(x, max_len=max_generated_len)))

# print(tokenizer.merge(x))