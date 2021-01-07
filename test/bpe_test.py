from tokenizer.BPE import BPE_VI, BPE_EN
from tokenizer._tokenizer import Tokenizer
from time import time

# bpe = BPE_VI(padding=False)
# print(bpe.tokenize(
#     ['Cuối cùng thì ta cũng không thể win the champition', 'Cuối cùng Thì Ta cũng không thể win the champion']))
# a = '<s> Cuối cùng Thì Ta cũng không thể win the champ@@ iti@@ on </s>'
# print(bpe.merge(a))
#
# print('-----------------------------------------------------------------------------------------------------')

bpe = BPE_EN(padding=False)
tokenizer = Tokenizer(bpe.symbols, bpe)
s = time()
print(tokenizer.tokenize(''))
x = tokenizer.tokenize(['', 'But lets face it: At the core of this line of thinking isnt safety -- its sex'])
print(x)
print(tokenizer.merge([i.numpy() for i in x]))
t = time()
print(t - s)
# a = '<s> Anyway , ĠI Ġthink ĠOne piece Ġis Ġnot Ġa Ġgame </s>'
# print(tokenizer.merge(a))

