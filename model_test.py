import numpy as np
import torch

from model.Seq2Seq_LSTM import Seq2SeqModel as Seq2Seq_LSTM

device = 'cuda' if torch.cuda.is_available() else 'cpu'
vocab_size = 10000
model = Seq2Seq_LSTM(vocab_size=vocab_size, device=device)
model.to(device)

test_len = 10
max_generated_len = 20
x = [torch.randint(0, vocab_size, [np.random.randint(1, 30)]).to(device) for _ in range(test_len)]

print(model(x, max_len=max_generated_len))
