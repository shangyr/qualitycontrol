import os
import json
import numpy as np
import datetime
from parsers import PROJECT_DIR

kgetrainer2path = {
    '':'',
    'KGETrainer':'entities_embed_kge.pth',
    'BertKGETrainer':'entities_embed_bert.pth',
    'Word2vecTrainer':'entities_embed_word2vec.pth',
}

class TraditionalChineseConfig(object):

    def __init__(self, args):

        # self.bert_path = 'nghuyong/ernie-health-zh'
        # self.bert_path = 'ValkyriaLenneth/longformer_zh'
        # self.bert_path = 'bert-base-chinese'
        # self.bert_path = 'IDEA-CCNL/Erlangshen-Longformer-110M'
        # self.bert_path = PROJECT_DIR + 'data/electronic-medical-record-42/embeds/128_0_10_cb_5n_5w.embeds'
        self.data_path = PROJECT_DIR + "data/traditional_chinese_medicine"
        self.train_path = os.path.join(self.data_path, "train.json")
        self.dev_path = os.path.join(self.data_path, "dev.json")
        self.test_path = os.path.join(self.data_path, "test.json")
        # self.test_path = os.path.join(self.data_path, f"tiny.json")
        self.label_idx_path = os.path.join(self.data_path, "label.txt")
        self.label_name_path = os.path.join(self.data_path, "label.txt")
        self.idf_path = os.path.join(self.data_path, "idf.pkl")
        self.tf_idf_radio = 0.9
        self.kg_graph_path = os.path.join(self.data_path, "triples.txt")
        self.entity_path = os.path.join(self.data_path, "entities.txt")
        self.cache_entity_trie_tree_path = os.path.join(self.data_path, "entity_trie_tree.pkl")
        self.emr2kg_path = None

        # self.kge_trainer = ''
        # self.kge_trainer = 'KGETrainer'
        self.language = 'zh'

        self.max_length = 500
        self.label_smooth_lambda = 0.0
        self.bert_lr = 1e-5
        self.other_lr = 5e-4
        self.batch_size = 12
        self.entity_embedding_dim = 100
        self.gnn_layer = 2
        self.num_hop = 2
        self.accumulation_steps = 1
        # self.path_type = 'v2'
        
        for attr in dir(args):
            if attr[:1] == '_':
                continue
            setattr(self,attr,getattr(args,attr))
        
        self.entity_embedding_path = os.path.join(self.data_path, kgetrainer2path[self.kge_trainer])

        self.save_model_path = PROJECT_DIR + 'data/traditional_chinese_medicine/PromptGATEdge_Smed-bert_BertKGETrainer_v1_seed_1_2023-12-12-23-12-32.pth'
        assert os.path.exists(self.save_model_path)
        
        # self.save_model_path = '/home/lixin/diagnose2/logs/checkpoints/DILHv2_seed_1_2023-06-14-10-23-04.pth'
        if 'longformer' in self.bert_path.lower():
            self.max_length = 1000
        self.label2id = {}
        with open(self.label_idx_path, "r", encoding="UTF-8") as f:
            for line in f:
                lin = line.strip().split()
                self.label2id[lin[0]] = len(self.label2id)
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