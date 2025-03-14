import csv
import gensim.models.word2vec as w2v
import pandas as pd
import os
from gensim.models.callbacks import CallbackAny2Vec
from tqdm import tqdm

# 中文电子病历训练word2vec
class LossLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 0:
            print("Loss after epoch {}: {}".format(self.epoch, loss))
        else:
            print("Loss after epoch {}: {}".format(self.epoch, loss - self.loss_previous_step))
        self.epoch += 1
        self.loss_previous_step = loss

import jieba

Chinese42_DIR = '/root/prompt-gnn/data/traditional_chinese_medicine'

def word_embeddings(sentences,out_file, embedding_size, min_count, n_iter):
    model = w2v.Word2Vec(vector_size=embedding_size,
                         min_count=min_count,
                         workers=64,
                         sg=0,
                         negative=5,
                         window=5,
                         callbacks=[LossLogger()])
    model.build_vocab(sentences)
    print("training...")
    model.train(sentences, total_examples=model.corpus_count, epochs=n_iter)
    print("writing embeddings to %s" % (out_file))
    model.save(out_file)
    return out_file

import json
if __name__ == "__main__":
    if not os.path.exists(os.path.join(Chinese42_DIR,"embeds")):
        os.mkdir(os.path.join(Chinese42_DIR,"embeds"))
    train_path = os.path.join(Chinese42_DIR,'train.json')
    sentences = []
    with open(train_path,'r',encoding='utf-8') as f:
        train = json.load(f)
        for item in tqdm(train):
            if '主诉' in item:
                raw_doc = item['主诉']+'[SEP]'+item['现病史']+'[SEP]'+item['既往史']
            else:
                raw_doc = item['doc']
            sentences.append(list(jieba.cut(raw_doc)))

    word_embeddings(sentences,os.path.join(Chinese42_DIR,"embeds","128_0_10_cb_5n_5w.embeds"), 128, 0, 10)
    embed_file = os.path.join(Chinese42_DIR,"embeds","128_0_10_cb_5n_5w.embeds")
    model = w2v.Word2Vec.load(embed_file)
    similar_words = model.wv.most_similar("咳嗽")
    print(similar_words)
