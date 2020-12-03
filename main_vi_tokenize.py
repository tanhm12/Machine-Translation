import os
from tqdm import tqdm

# os.system('wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \'https://docs.google.com/uc?export=download&id=1A19YjV4Jo8c5diwpBjJOImxvdPmfb4tG\' -O- | sed -rn \'s/.*confirm=([0-9A-Za-z_]+).*/\1\n/p\')&id=1A19YjV4Jo8c5diwpBjJOImxvdPmfb4tG" -O MT-Vi-Mono-VLSP2020.zip && rm -rf /tmp/cookies.txt')
# os.system('unzip MT-Vi-Mono-VLSP2020.zip')
os.system('mkdir vi_data')
os.system("split -l 10000 './corpus.2M.shuf.txt' 'vi_data/vi'")

from os import listdir
from os.path import isfile, join
_path = 'vi_data'
link = [join(_path, f) for f in listdir(_path) if isfile(join(_path, f))]


os.system("pip install vncorenlp")
from vncorenlp import VnCoreNLP
os.system("wget 'https://github.com/vncorenlp/VnCoreNLP/archive/v1.1.1.zip' -O ./models.$$ && unzip -o ./models.$$ && rm -r ./models.$$.")
from vncorenlp import VnCoreNLP
segmentNLP = VnCoreNLP('./VnCoreNLP-1.1.1/VnCoreNLP-1.1.1.jar', port=9001, annotators="wseg,pos,ner,parse", quiet=False)

f_w = open('corpus.tokened.2M.shuf.txt', 'w', encoding='utf8')
for file in tqdm(link):
    f = open(file, 'r', encoding='utf8')
    a = f.read().strip().split('\n')
    for i in a:
        s = ''
        tmp=segmentNLP.tokenize(i)
        for j in tmp:
            s += ' '.join(j) + ' '
        f_w.write(s + '\n')
    f.close()
f_w.close()

