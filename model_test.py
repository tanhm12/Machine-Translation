import numpy as np
import torch

from model.Seq2Seq_LSTM import Seq2SeqModel as Seq2Seq_LSTM
from bpe.BPE_VI import BPE_VI

device = 'cuda' if torch.cuda.is_available() else 'cpu'
vocab_size = 100000
model = Seq2Seq_LSTM(vocab_size=vocab_size, device=device)
model.to(device)

test_len = 10
max_generated_len = 30

s = ['Bánh rán là một món ăn ưa thích của Doraemon', 'Có một vài LeeSin đá sóng âm hụt']
bpe = BPE_VI(padding=False)
# x = [torch.randint(0, vocab_size, [np.random.randint(1, 30)]).to(device) for _ in range(test_len)]
# print(x[0].dtype)
x = bpe.tokenizers(s, return_sent=False)
# x = [torch.LongTensor(_).to(device) for _ in x]
# print(x)

# print(bpe.merges(model(x, max_len=max_generated_len)))

print(bpe.merges(x))