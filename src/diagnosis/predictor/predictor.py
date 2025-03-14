from importlib import import_module
import torch
import sys
import os
# sys.path.append("..")

# from parsers import get_args
from graph import ChineseGraph,MIMICGraph
from strategies import NaiveStrategy
from trainers import BaseTrainer

import configs
import datasets
import trainers
 

class Predictor:
    def __init__(self, config) -> None:
        # args = get_args()
        # Config = eval('configs.' + config.config)
        self.Dataset = eval('datasets.' + config.dataset)
        Trainer:BaseTrainer = eval('trainers.' + config.trainer)
        data = [{"emr_id":'1',"doc":"","label":[]}]
        self.opt = config
        self._init_graph()
        self._init_model()
        self._init_stagegy()
        loss_fn = None
        self.trainer = Trainer(self.model,self.opt,loss_fn,self.strategy)
        if self.graph is not None:
            self.dataset = self.Dataset(data, self.opt, self.graph)
        else:
            self.dataset = self.Dataset(data, self.opt)
        datasets = [self.dataset]
        self.trainer.setup_dataset(datasets)

    def _init_graph(self):
        # 加载数据集
        if self.opt.dataset in ['GMANDataset','MSLANDataset','KG2TextDataset','PromptGNNDataset'] and hasattr(self.opt,'kg_graph_path'):
            if 'mimic' in self.opt.kg_graph_path:
                self.graph = MIMICGraph(kg_graph_path=self.opt.kg_graph_path,
                                entity_path=self.opt.entity_path,
                                label_path=self.opt.label_name_path,
                                emr2kg_path=self.opt.emr2kg_path,
                                max_hop=self.opt.num_hop+1)
            else:
                self.graph = ChineseGraph(kg_graph_path=self.opt.kg_graph_path,
                                entity_path=self.opt.entity_path,
                                label_path=self.opt.label_name_path,
                                emr2kg_path=self.opt.emr2kg_path,
                                max_hop=self.opt.num_hop+1)
            self.opt.entity_num = self.graph.entity_num()
            self.opt.relation_num = self.graph.relation_num()
        else:
            self.graph = None
        
    def _init_model(self):
        self.model = import_module('models.' + self.opt.model_name).Model(self.opt)
        
    def _init_stagegy(self):
        self.strategy = NaiveStrategy()

    def predict(self, data):
        self.dataset.set_new_data(data)
        ans = self.trainer.predict(self.dataset, data)
        return ans

