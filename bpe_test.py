from bpe.BPE_EN import BPE_EN
from bpe.BPE_VI import BPE_VI
from time import time

bpe = BPE_VI(padding=False)
print(bpe.tokenizers(
    ['Cuối cùng thì ta cũng không thể win the champition', 'Cuối cùng Thì Ta cũng không thể win the champition']))
a = '<s> Cuối cùng Thì Ta cũng không thể win the champ@@ iti@@ on </s>'
print(bpe.merge(a))

print('-----------------------------------------------------------------------------------------------------')

bpe = BPE_EN(padding=False)
s = time()
print(bpe.tokenizers(['Anyway, I think Onepiece is not a game']))
t = time()
print(t - s)
a = '<s> Anyway , ĠI Ġthink ĠOne piece Ġis Ġnot Ġa Ġgame </s>'
print(bpe.merge(a))
