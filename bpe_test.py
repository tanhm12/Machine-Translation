from bpe.BPE_EN import BPE_EN
from bpe.BPE_VI import BPE_VI
from time import time

bpe = BPE_VI(padding=False)
print(bpe.tokenizers(
    ['Cuối cùng thì ta cũng không thể win the champition', 'Cuối cùng Thì Ta cũng không thể win the champition'],
    return_sent=True))
a = [0, 3223, 81, 54, 237, 32, 17, 4623, 16209, 2292, 58098, 30435, 2049, 2]
print(bpe.merge(a))

print('-----------------------------------------------------------------------------------------------------')

bpe = BPE_EN(padding=False)
s = time()
print(bpe.tokenizers(['Anyway, I think Onepiece is not a game'], return_sent=True))
t = time()
print(t - s)
a = [0, 46871, 6, 38, 206, 509, 10449, 16, 45, 10, 177, 2]
print(bpe.merge(a))
