import numpy as np
import torch

from model.Seq2Seq_LSTM import Seq2SeqModel as Seq2Seq_LSTM
from tokenizer.BPE import BPE_VI, BPE_EN
from tokenizer._tokenizer import Tokenizer
from torch import nn


bpe_en = BPE_EN(padding=False)
bpe_vi = BPE_VI(padding=False)



