import os
import json
import numpy as np

class TestConfig(object):

    def __init__(self, args):

        self.train_path = os.path.join(args.data_path, f"tiny.json")
        self.dev_path = os.path.join(args.data_path, f"tiny.json")
        self.test_path = os.path.join(args.data_path, f"tiny.json")
        self.label_idx_path = os.path.join(args.data_path, "label2id.txt")
        for attr in dir(args):
            if attr[:1] == '_':
                continue
            setattr(self,attr,getattr(args,attr))
            
        self.graph_path1 = os.path.join(args.data_path, "anss.txt")
        self.graph_path2 = os.path.join(args.data_path, "triples.txt")
        self.entity_path = os.path.join(args.data_path, "entities.txt")

        for attr in dir(args):
            if attr[:1] == '_':
                continue
            setattr(self,attr,getattr(args,attr))
        
        self.label2id = {}
        with open(self.label_idx_path, "r", encoding="UTF-8")as f:
            for line in f:
                lin = line.strip().split()
                self.label2id[lin[0]] = len(self.label2id)
        self.class_num = len(self.label2id)
        self.entity_num = 0