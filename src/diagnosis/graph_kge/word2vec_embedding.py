# -*- coding:utf-8 -*-
import sys
from transformers import BertModel
from transformers import AutoModel,AutoTokenizer
import torch
import numpy as np
from tqdm import tqdm
import json
from gensim.models.callbacks import CallbackAny2Vec
import gensim.models.word2vec as w2v

from graph import BaseGraph
from tqdm import tqdm

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

class Word2vecTrainer:
    def __init__(self,graph:BaseGraph=None,entity2id:dict=None,language='zh',train_path=None) -> None:
        # from graph import ChineseGraph
        # self.graph = ChineseGraph('/home/lixin/prompt-gnn/data/chinese-small/triplesv2.txt',
        #                           '/home/lixin/prompt-gnn/data/chinese-small/entities.txt',
        #                           '/home/lixin/prompt-gnn/data/chinese-small/label2id.txt')
        if graph is None:
            raise NotImplementedError
        sentences = []
        with open(train_path,'r',encoding='utf-8') as f:
            data = json.load(f)
            for item in tqdm(data):
                nodes,edges,nodes_type,edges_type,all_paths,_ = graph.generate_graph_by_text(item['doc'])
                for path in all_paths:
                    sentence = [str(path[0])]
                    for head,tail in zip(path[:-1],path[1:]):
                        rel = graph.ht2rel.get((head,tail),None)
                        if rel is None:
                            sentence.append(str(tail))
                        else:
                            sentence.append('rel:' + str(graph.rel2id[rel]))
                            sentence.append(str(tail))
                    sentences.append(sentence)

        model = w2v.Word2Vec(vector_size=768,
                         min_count=0,
                         workers=64,
                         sg=0,
                         negative=5,
                         window=3,
                        callbacks=[LossLogger()])
        model.build_vocab(sentences)
        print("training...")
        model.train(sentences, total_examples=model.corpus_count, epochs=10)
        self.entity_embedding = np.random.randn(graph.entity_num(),768)
        self.relation_embedding = np.random.randn(graph.relation_num(),768)
        vocab = model.wv.key_to_index.copy()
        weights = model.wv.vectors.copy()
        for raw_word in vocab:
            if 'rel:' in raw_word:
                word = int(raw_word.replace('rel:',''))
                self.relation_embedding[word] = weights[vocab[raw_word]]
            else:
                word = int(raw_word)
                self.entity_embedding[word] = weights[vocab[raw_word]]
        self.relation_embedding = torch.tensor(self.relation_embedding).float()
        self.entity_embedding = torch.tensor(self.entity_embedding).float()

    def export_model(self,output_path):
        torch.save({
            'entity_embedding':self.entity_embedding.data.cpu(),
            'relation_embedding':self.relation_embedding.data.cpu()
        },output_path)
