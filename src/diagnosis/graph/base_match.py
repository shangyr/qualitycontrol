from abc import ABC,abstractclassmethod
import os
import re
import pickle

class TrieTreeMatch(ABC):
    """字典树匹配算法"""
    def __init__(self,entity_file_path, entity_cache_path):
        """
        Args:
            entity_file_path:实体列表
            格式如下
            实体1..类型1
            实体2..类型1
            ...
        """

        super(TrieTreeMatch, self).__init__()

        self.trie_tree = {}
        self.entity2id = {}
        self.id2entity = {}
        self.type2id = {}
        self.entityname2type = {}
        self.end_sign = None
        
        if entity_cache_path is None or not os.path.exists(entity_cache_path):
            # 构建字典树
            self.build_tree(entity_file_path)
            self.entity_size = len(self.entity2id) # 实体数量
            if entity_cache_path is not None:
                with open(entity_cache_path, 'wb') as f:
                    pickle.dump({'trie_tree':self.trie_tree, 'entity2id':self.entity2id, \
                                'type2id':self.type2id, 'entityname2type':self.entityname2type}, f)
        else:
            with open(entity_cache_path, 'rb') as f:
                data = pickle.load(f)
                self.trie_tree = data['trie_tree']
                self.entity2id = data['entity2id']
                self.id2entity = [entity for entity in self.entity2id]
                self.type2id = data['type2id']
                self.entityname2type = data['entityname2type']
                self.entity_size = len(self.entity2id)

    def _add_entity_to_tree(self,entity_name:str):
        """
        添加实体到字典树中
        """
        tree_node = self.trie_tree
        # entity = re.split(r"[ ]",entity)
        for word in entity_name:
            if tree_node.get(word) is None:
                tree_node[word] = {}
            tree_node = tree_node[word]
        tree_node[self.end_sign] = None
    
    def _add_entity_to_list(self,entity_name:str,entity_type):
        """
        添加实体/类型到列表里面
        Args:
            entity_name:实体名字
            entity_type:实体类型
        """
        if entity_name not in self.entity2id:
            self.entity2id[entity_name] = len(self.entity2id)
            self.id2entity[len(self.id2entity)] = entity_name
        if entity_type not in self.type2id:
            self.type2id[entity_type] = len(self.type2id)
        self.entityname2type[entity_name] = entity_type
    
    def _process_entity_name(self,entity_name):
        entity_name = entity_name.lower()
        entity_name = re.sub(r'[\-,\./]',' ',entity_name) # 一些特殊符号转化为空格
        entity_name = re.sub(r'\s+', ' ', entity_name)   # 多个连续空格转化为单个空格
        return entity_name
    
    def build_tree(self,file_path):
        """
        建立字典树
        """
        with open(file_path,'r',encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line == '': continue
                entity = line.split('\t')[0]
                entity_type = entity.split('..')[1]
                entity_name = entity.split('..')[0]
                entity_name = self._process_entity_name(entity_name)
                self._add_entity_to_tree(entity_name)
                self._add_entity_to_list(entity_name,entity_type)
    
    def _match_entity(self,start_idx:int,sentence:str) -> list:
        """
        从start位置匹配实体
        Args:
            start_idx:实体匹配位置
            sentence:文档
        Return:
            end_idxs:实体结束位置+1列表
        """
        tree_node = self.trie_tree
        end_idx = start_idx
        token_idx = start_idx
        end_idxs = []
        while token_idx < len(sentence) and (sentence[token_idx] in tree_node):
            
            if sentence[token_idx] in tree_node: # 字符
                tree_node = tree_node[sentence[token_idx]]
                if self.end_sign in tree_node:
                    end_idxs.append(token_idx + 1)
            token_idx += 1
        
        return end_idxs

    @abstractclassmethod
    def _remove_neg_entities(self,entities:list,sentence:str) -> list:
        """
        去除否定实体
        Args:
            entities:实体列表
            sentence:句子
        Return:
            filter_entities:去否定实体之后的实体列表
        """
        pass
    
    def find_entities(self,sentence:str) -> list:
        """
        获取实体列表
        Args:
            sentence:句子/文档
        Return:
            entities:匹配实体列表(去掉否定实体/去重)
            ['实体1','实体2',...]
        """
        start_idx = -1
        end_idx = -1
        entities = []
        sentence = sentence.lower()
        # sentence = re.split(r'[\. ]',sentence)
        for start_idx,token in enumerate(sentence):
            end_idxs = self._match_entity(start_idx,sentence)
            for end_idx in end_idxs:
                entities.append(sentence[start_idx:end_idx])

        # entities = list(set(entities)) # 去重
        entities = self._remove_neg_entities(entities,sentence)
        entities = [entity for entity in entities if len(entity) > 1]
        return entities
