# -*- coding:utf-8 -*-
import sys
from transformers import BertModel
from transformers import AutoModel,AutoTokenizer
import torch
import numpy as np
from tqdm import tqdm

from graph import BaseGraph

class GenerateEmbedding:
    def __init__(self,bert_path):
        self.bert = AutoModel.from_pretrained(bert_path)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
        if torch.cuda.is_available():
            self.bert = self.bert.cuda()
    def generate(self,entity):
        tokens = self.tokenizer(entity,return_tensors = 'pt')
        with torch.no_grad():
            if torch.cuda.is_available():
                tokens['input_ids'] = tokens['input_ids'].cuda()
                tokens['attention_mask'] = tokens['attention_mask'].cuda()
            output = self.bert(tokens['input_ids'],tokens['attention_mask']).last_hidden_state[0,0]
        output = output.cpu()
        output = (output - output.mean()) / (output.std() +1e-6)
        output[output > 5] = 5.0
        output[output < -5] = -5.0
        return output

    def similarity(self,vec1,vec2):
        """
            计算余弦相似度
        """
        vec1 = (vec1-np.mean(vec1)) / np.std(vec1)
        vec2 = (vec2-np.mean(vec2)) / np.std(vec2)
        return np.sum(vec1 * vec2) / (np.sqrt(np.sum(np.power(vec1,2))) + np.sqrt(np.sum(np.power(vec2,2))))

class BertKGETrainer:
    def __init__(self,graph:BaseGraph=None,entity2id:dict=None,language='zh',train_path=None) -> None:
        # from graph import ChineseGraph
        # self.graph = ChineseGraph('/home/lixin/prompt-gnn/data/chinese-small/triplesv2.txt',
        #                           '/home/lixin/prompt-gnn/data/chinese-small/entities.txt',
        #                           '/home/lixin/prompt-gnn/data/chinese-small/label2id.txt')
        
        if language == 'zh':
            bert_path = '/home/lixin/work/prompt-gnn/PLM/Smed-bert'
        else:
            bert_path = 'bert-base-uncased'

        entities,relations = {},{}
        if graph is not None:
            self.graph = graph
            entities = self.graph.match_model.entity2id
            relations = self.graph.rel2id
        elif entity2id is not None:
            entities = entity2id

        
        self.model = GenerateEmbedding(bert_path)
        print('generating entity embedding...')
        
        self.entity_embedding = torch.zeros(len(entities),768,dtype=torch.float32)
        for entity,entity_id in entities.items():
            if entity_id % 1000 == 0:
                print(f'cur/all: {entity_id}/{len(entities)}')
            self.entity_embedding[entity_id] = self.model.generate(entity)
        
        self.relation_embedding = torch.zeros(len(relations),768,dtype=torch.float32)
        for relation,relation_id in relations.items():
            if relation_id % 1000 == 0:
                print(f'cur/all: {relation_id}/{len(relations)}')
            self.relation_embedding[relation_id] = self.model.generate(relation)
        
        print(f'{len(entities)} entities, {len(relations)} relations.')
           
    def export_model(self,output_path):
        torch.save({
            'entity_embedding':self.entity_embedding.data.cpu(),
            'relation_embedding':self.relation_embedding.data.cpu()
        },output_path)
