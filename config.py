from argparse import Namespace

config = Namespace(
    bpe_vi_embedding='/content/drive/MyDrive/MT/bpe_vi',
    bpe_en_embedding='/content/drive/MyDrive/MT/bpe_en',
    bos_token='<s>',
    eos_token='</s>',
    pad_token='<pad>',
    unk_token='<unk>',
    lstm_dim=128,
    direction=2,
    num_layers=1,
    loss_ignore_index=-100,
    beam_size=5,
    device='cuda',
)


