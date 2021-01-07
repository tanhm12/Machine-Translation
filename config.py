from argparse import Namespace


class Config:
    bpe_vi_embedding='./embedding/models/bpe_vi'
    bpe_en_embedding='./embedding/models/bpe_en'
    bos_token='<s>'
    eos_token='</s>'
    pad_token='<pad>'
    unk_token='<unk>'
    bos_idx=0
    eos_idx=2
    pad_idx=1
    unk_idx=3
    lstm_dim=128
    direction=2
    num_layers=1
    loss_ignore_idx=-100
    batch_size=128
    beam_size=5
    epochs = 5
    print_interval = 1/50
    device='cpu'


config = Config()


