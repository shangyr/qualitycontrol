import os
import json
import numpy as np
import datetime
from parsers import PROJECT_DIR
from transformers import Trainer

class MIMICConfig(object):

    def __init__(self, args):
        # self.bert_path = PROJECT_DIR + "data/mimic-3/embeds/128_0_10_cb_5n_5w.embeds"
        # self.bert_path = "yikuan8/Clinical-Longformer"
        # self.bert_path = "bert-base-uncased"
        self.data_path = PROJECT_DIR + "data/mimic-3"
        self.train_path = os.path.join(self.data_path, f"train.json")
        self.dev_path = os.path.join(self.data_path, f"dev.json")
        self.test_path = os.path.join(self.data_path, f"test.json")
        self.label_idx_path = os.path.join(self.data_path, "label2id.txt")

        self.kg_graph_path = os.path.join(self.data_path, "triples.txt")
        self.entity_path = os.path.join(self.data_path, "entities.txt")
        self.emr2kg_path = os.path.join(self.data_path, "emr2kg.pkl")
        self.label_name_path = os.path.join(self.data_path, "label_name.txt")
        self.idf_path = os.path.join(self.data_path, "idf.pkl")
        self.tf_idf_radio = 0.9
        self.entity_embedding_path = os.path.join(self.data_path, "entities_embed_kge.pth")
        self.cache_entity_trie_tree_path = os.path.join(self.data_path, "entity_trie_tree.pkl")
        # self.kge_trainer = ''
        self.kge_trainer = 'KGETrainer'
        self.language = 'en'

        self.max_length = 3500
        self.label_smooth_lambda = 0.0
        self.bert_lr = 2e-5
        self.other_lr = 1e-3
        self.batch_size = 2
        self.logic_node_num = 8
        self.entity_embedding_dim = 100
        self.gnn_layer = 2
        self.num_hop = 2
        self.accumulation_steps = 2
        self.path_type = 'v2'

        for attr in dir(args):
            if attr[:1] == '_':
                continue
            setattr(self,attr,getattr(args,attr))

        self.label2id = {}
        with open(self.label_idx_path, "r", encoding="UTF-8")as f:
            for line in f:
                lin = line.strip()
                self.label2id[lin] = len(self.label2id)
        self.class_num = len(self.label2id)
        self.entity_num = 0
    def __str__(self):
        ans = "====================Configuration====================\n"
        for key, value in self.__dict__.items():
            if key in ['label2id','entity2id']:
                continue
            ans += key + ":" + (value if type(value) == str else str(value)) + "\n"
        ans += "====================Configuration====================\n"

        return ans