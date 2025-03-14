import os
import json
import numpy as np
import datetime
from parsers import PROJECT_DIR

class Chinese42Config(object):

    def __init__(self, args):

        # self.bert_path = 'nghuyong/ernie-health-zh'
        # self.bert_path = 'ValkyriaLenneth/longformer_zh'
        # self.bert_path = 'bert-base-chinese'
        # self.bert_path = 'IDEA-CCNL/Erlangshen-Longformer-110M'
        # self.bert_path = PROJECT_DIR + 'data/electronic-medical-record-42/embeds/128_0_10_cb_5n_5w.embeds'
        self.data_path = PROJECT_DIR + "data/chinese42"
        self.train_path = os.path.join(self.data_path, "train.json")
        self.dev_path = os.path.join(self.data_path, "dev.json")
        self.test_path = os.path.join(self.data_path, "test.json")
        # self.test_path = os.path.join(self.data_path, f"tiny.json")
        self.label_idx_path = os.path.join(self.data_path, "label2id.txt")
        self.label_name_path = os.path.join(self.data_path, "label2id.txt")
        self.idf_path = os.path.join(self.data_path, "idf.pkl")
        self.tf_idf_radio = 0.9
        self.kg_graph_path = os.path.join(self.data_path, "triplesv2.txt")
        self.entity_path = os.path.join(self.data_path, "entities.txt")
        self.emr2kg_path = None
        self.entity_embedding_path = os.path.join(self.data_path, "entities_embed_kge.pth")
        self.cache_entity_trie_tree_path = os.path.join(self.data_path, "entity_trie_tree.pkl")
        # self.kge_trainer = ''
        self.kge_trainer = 'KGETrainer'
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
        self.path_type = 'v2'
        
        for attr in dir(args):
            if attr[:1] == '_':
                continue
            setattr(self,attr,getattr(args,attr))
        # self.save_model_path = 'C:\\Users\\asndj\\Desktop\\新建文件夹\\data\\chinese-42\\PromptGATEdge_seed_1_2023-12-04-22-51-13.pth'
        # self.save_model_path = '/home/lixin/diagnose2/logs/checkpoints/DILHv2_seed_1_2023-06-14-10-23-04.pth'
        # if 'longformer' in self.bert_path.lower():
        #     self.max_length = 1000
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