# Machine Translation for English - Vietnamese (Deep Learning Project)

## Word2vec pretained

There are following word2vec models.

Model | Data | Note
----------- | --------- | -----------
Space Vi | MT-Vi-Mono-VLSP2020 using space tokeniner for preprocessing | VnCoreNLP for pretokenizer
Space En | CNN-DailyMail using space tokeniner for preprocessing | StandfordCoreNLP for sentence segment
BPE Vi | MT-Vi-Mono-VLSP2020 using bpe tokeniner for preprocessing | VnCoreNLP for pretokenizer
BPE En  | CNN-DailyMail using space bpe for preprocessing | StandfordCoreNLP for sentence segment

Notebook: embedding/word2vec_pretrain_machine_trans_en_vi.ipynb

Link models: [Google Drive](https://drive.google.com/drive/folders/1VAZFWtKEeh0NnYsyXntOWFZHI6TqVYfi?usp=sharing)
models word2vec will be saved in /embedding/model_name

## Data

Extract MT-EV-VLSP2020.zip to ./

## Seq2Seq model

Check model/

## Training, evaluating and inferencing

Check 2 main file or these [Colab notebooks](https://drive.google.com/drive/folders/1VAZFWtKEeh0NnYsyXntOWFZHI6TqVYfi?usp=sharing)
