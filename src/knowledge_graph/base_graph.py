from abc import ABC,abstractmethod
import pickle as pkl
import os
import random
import copy
import time
import queue
from tqdm import tqdm
import datetime

class BaseGraph(ABC):
    """
    图
    """
    def __init__(self,kg_graph_path,match_model) -> None:
        # 匹配模型
        self.match_model = match_model
        self.entityname2type = self.match_model.entityname2type
        self.id2entity = self.match_model.id2entity

        # 图模型
        self.h2t = {} # head_id:[tail_id1,tail_id2,..]
        self.ht2rel = {} # {(h,t):relation(str),...}
        self.rel2id = {} # {'伴随':0,...}
        self.id2rel = [] # ['伴随',...]
        cache_path = kg_graph_path.replace('.txt','.pkl')
        if os.path.exists(cache_path):
            with open(cache_path,'rb') as f:
                data = pkl.load(f)
            print('knowledge graph using cache, cache date:' ,data['date'])
            self.h2t = data['h2t'] # head_id:[tail_id1,tail_id2,..]
            self.ht2rel = data['ht2rel'] # {(h,t):relation(str),...}
            self.rel2id = data['rel2id'] # {'伴随':0,...}
            self.id2rel = data['id2rel'] # ['伴随',...]
        else:
            self._read_kg_file(kg_graph_path)
            cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
            print('knowledge graph making cache, cache date:',cur_time)
            data = {
                'h2t':self.h2t,
                'ht2rel':self.ht2rel,
                'rel2id':self.rel2id,
                'id2rel':self.id2rel,
                'date': cur_time
            }
            with open(cache_path,'wb') as f:
                data = pkl.dump(data,f)
    
    def _check_correct_name(self, triple:list) -> tuple:
        """
        检查三元组中实体名字是否在匹配模型中存在
        Args:
            triple : [h:str, r:str, t:str]
        Return:
            存在返回 (head_id, tail_id), 否则 (False, False)
        """
        head_name = self.match_model.process_entity_name(triple[0].split('..')[0])
        tail_name = self.match_model.process_entity_name(triple[2].split('..')[0])
        if head_name not in self.match_model.entity2id or tail_name not in self.match_model.entity2id:
            return False, False
        head_id = self.match_model.entity2id[head_name]
        tail_id = self.match_model.entity2id[tail_name]
        return head_id, tail_id

    def _add_h2t(self, head_id, rel_name, tail_id):
        """
        添加三元组
        """
        # h -> t
        if head_id not in self.h2t:
            self.h2t[head_id] = set()
        self.h2t[head_id].add(tail_id)
        
        if rel_name not in self.rel2id:
            self.rel2id[rel_name] = len(self.rel2id)
            self.id2rel.append(rel_name)

        if (head_id, tail_id) not in self.ht2rel:
            self.ht2rel[(head_id,tail_id)] = self.rel2id[rel_name]

    def _read_kg_file(self,kg_graph_path):
        triples = []
        print('reading:', kg_graph_path)
        with open(kg_graph_path,'r',encoding='utf-8') as f:
            for line in tqdm(f):
                line = line.strip()
                if not line: continue
                triple = line.split('\t')
                head_id, tail_id = self._check_correct_name(triple)
                if head_id == False: continue
                triples.append([head_id, triple[1], tail_id])
        print('adding triples:', kg_graph_path)
        # h -> t
        for triple in tqdm(triples):
            self._add_h2t(triple[0], triple[1], triple[2])

        # t -> h
        for triple in tqdm(triples):
            self._add_h2t(triple[2], self.reverse_word + triple[1], triple[0])
        
        # TODO 添加对别名的处理

    def read_kg_file(self,kg_graph_path):
        self._read_kg_file(kg_graph_path)
        if os.path.exists(kg_graph_path.replace('triples.txt','triples_expand.txt')):
            self._read_kg_file(kg_graph_path.replace('triples.txt','triples_expand.txt'))
    

    def _search_paths(self, head_id, path, paths, num_hop):
        if head_id not in self.h2t:
            return
        # if self._caculate_path_hop(path) == num_hop:
        #     return
        if len(path) // 2 == num_hop:
            return
        for tail_id in self.h2t[head_id]:
            tail_name = self.match_model.id2entity[tail_id]
            if tail_name in path: continue
            path.append(self.id2rel[self.ht2rel[(head_id, tail_id)]])
            path.append(tail_name)
            paths.append(path[:])
            self._search_paths(tail_id, path, paths, num_hop)
            path.pop()
            path.pop()

    def search_paths(self, raw_entity_name, num_hop = 1):
        """
        找临近节点
        """
        assert num_hop <= 2,'大于两跳会引入很多无关噪音'
        paths = []
        entity_name = self.match_model.process_entity_name(raw_entity_name)
        if entity_name not in self.match_model.entity2id:
            return paths
        entity_id = self.match_model.entity2id[entity_name]
        path = [entity_name]
        self._search_paths(entity_id, path, paths, num_hop)
        return paths

    def search_link_paths(self, raw_entity_name1,raw_entity_name2, num_hop = 2):
        """
        查找临近节点
        """
        _paths = []
        entity_name1 = self.match_model.process_entity_name(raw_entity_name1)
        if entity_name1 not in self.match_model.entity2id:
            return _paths
        entity_name2 = self.match_model.process_entity_name(raw_entity_name2)
        if entity_name2 not in self.match_model.entity2id:
            return _paths
        entity_id1 = self.match_model.entity2id[entity_name1]
        entity_id2 = self.match_model.entity2id[entity_name2]
        
        if num_hop == 1:
            path = [entity_name1]
            _paths,paths = [],[]
            self._search_paths(entity_id1, path, _paths, num_hop)
            for path in _paths:
                if path[-1] == entity_name2:
                    paths.append(path)
        elif num_hop == 2:
            path = [entity_name1]
            _paths1,_paths2,paths = [],[],[]
            self._search_paths(entity_id1, path, _paths1, 1)
            path = [entity_name2]
            self._search_paths(entity_id2, path, _paths2, 1)
            for path1 in _paths1:
                for path2 in _paths2:
                    if path1[-1] != path2[-1]:
                        continue
                    path = path1[:]
                    path2_head = self.match_model.entity2id[path2[-1]]
                    path2_tail = self.match_model.entity2id[path2[0]]
                    path2_rel = self.id2rel[self.ht2rel[(path2_head,path2_tail)]]
                    path.append(path2_rel)
                    path.append(path2[0])
                    paths.append(path)
        else:
            raise Exception('大于两跳会引入很多无关噪音')

        return paths


    def _caculate_path_hop(self,path):
        relation = {'name':'','num':0}
        hop = 0
        for i in range(1,len(path),2):
            if path[i] != relation['name']:
                relation['name'] = path[i]
                relation['num'] = 1
                hop += 1
            else:
                if relation['name'] not in self.repeatable_relations:
                    relation['num'] += 1
                    hop += 1
                elif relation['num'] <= 1:
                    relation['num'] += 1
                else:
                    relation['num'] += 1
                    hop += 1
        return hop
