from abc import ABC,abstractclassmethod
import os
import re
import json
from .base_extract import BaseExtract


"""
字符串匹配算法得到实体
"""
class MatchExtract(BaseExtract):
    """字典树匹配算法"""
    def __init__(self,entity_file_path):
        """
        Args:
            entity_file_path:实体列表
            格式如下
            实体1..类型1
            实体2..类型1
            ...
        """

        super(MatchExtract, self).__init__()

        self.trie_tree = {}
        self.entity2id = {}
        self.id2entity = {}
        self.type2id = {}
        self.entityname2type = {}
        self.end_sign = None

        self.ignore_set = {
            '体重',
            '精神',
            '手术',
            '不适',
            '休息',
            '明显增减',
            '出生时',
            'mg',
            '鸡蛋',
            '持续',
            '反复发作',
            '明显异常',
            '右侧',
            '第3',
            '全身',
            '明显变化',
            '受限',
            '转移',
            '刺激',
            '明显改变',
            '减轻',
            '大小',
            '量少',
            '中度',
            '为主',
            '突发',
            '呼吸',
            '明显减轻',
            '慢性',
            '量多',
            '脱出',
            '晚期',
            '渐增大',
            '重度',
            '中晚',
            '明显下降',
            '第三',
            '感染',
            '完全'
        }

        self.entity2syn = json.load(open(os.path.join(os.path.dirname(entity_file_path), 'sym2syn.json'), 'r', encoding='utf-8'))
        # self.sym2idx = {}
        # self.idx2sym = {}

        # 构建字典树
        self.build_tree(entity_file_path)
        self.entity_size = len(self.entity2id) # 实体数量
        self.distinguish_pos_neg = True


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
    
    def process_entity_name(self,entity_name):
        """
        标准化实体名字
        """
        entity_name = entity_name.lower()
        entity_name = entity_name.strip('等伴感')
        return entity_name
    
    def _valid_entity_name(self, entity_name):
        """
        判断是否是合理的实体
        """
        if entity_name in self.ignore_set: return False
        if '无' in self.ignore_set: return False
        return True


    def build_tree(self,file_path):
        """
        建立字典树
        """
        with open(file_path,'r',encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line == '': continue
                entity = line.split('\t')[0]
                entity_type = entity.split('..')[1]
                entity_name = entity.split('..')[0]
                entity_name = self.process_entity_name(entity_name)
                # if entity_type == '症状' or entity_type == '疾病':
                #     self.sym2idx[entity_name] = len(self.sym2idx)
                #     self.idx2sym[len(self.idx2sym)] = [entity_name, entity_type]
                if self._valid_entity_name(entity_name):
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
        token_idx = start_idx
        end_idxs = []
        while token_idx < len(sentence) and (sentence[token_idx] in tree_node):
            tree_node = tree_node[sentence[token_idx]]
            if self.end_sign in tree_node:
                end_idxs.append(token_idx + 1)
            token_idx += 1
        
        return end_idxs
    
    def _find_entities(self,sentence:str) -> list:
        """
        获取实体列表
        Args:
            sentence:句子/文档
        Return:
            entities:匹配实体列表 
            [(实体名, 实体类型, 实体开始位置, 实体结束位置),...]
        """
        start_idx = -1
        end_idx = -1
        entities = []
        sentence = sentence.lower()
        # sentence = re.split(r'[\. ]',sentence)
        for start_idx,token in enumerate(sentence):
            end_idxs = self._match_entity(start_idx,sentence)
            for end_idx in end_idxs:
                entity = sentence[start_idx:end_idx]
                entities.append((entity,self.entityname2type[entity], start_idx, end_idx))
        # print('test',entities)

        return entities

    def _remove_neg_entities(self, entities: list, sentence: str) -> list:
        """
        去除否定实体
        Args:
            entities:实体列表
            sentence:句子
        Return:
            filter_entities:去否定实体之后的实体列表
        """

        neg_entities = re.findall(r"((否认)|(无))(.*?)(，|。|；|,|$|(\[SEP\]))", sentence)
        # 否认“肝炎，伤寒，结核”
        # neg_entities.extend(re.findall(r'((否认)|(无))“(.*?)”',sentence))
        neg_entities = ''.join([''.join(list(item)) for item in neg_entities])  # 电子病历中的否定实体

        # 也有一些正确的
        pos_entities = re.findall(r"无明显诱因(.*?)(，|。|；|,|$|(\[SEP\]))", neg_entities)
        pos_entities = ''.join([''.join(list(item)) for item in pos_entities])
        filter_entities = []
        for entity in entities:
            if entity not in neg_entities:
                filter_entities.append(entity)
            if entity in pos_entities:
                filter_entities.append(entity)
        return filter_entities


    def find_entities(self, sentence: str) -> list:
        """
        获取实体列表
        Args:
            sentence:句子/文档
        Return:
            entities:匹配实体列表
            distinguish_pos_neg:
            {'pos': [(实体名, 实体类型, 实体开始位置, 实体结束位置),...] ,'neg':[]}
            not distinguish_pos_neg
            [(实体名, 实体类型, 实体开始位置, 实体结束位置),...]
        """
        entities = self._find_entities(sentence)
        if not self.distinguish_pos_neg:
            return entities

        entities = [entity for entity in entities if len(entity[0]) > 1]

        include_entities = []
        for entity1 in entities:
            for entity2 in entities:
                if entity1[0] == entity2[0]:
                    continue
                if entity1[2] >= entity2[2] and entity1[3] <= entity2[3]:
                    include_entities.append(entity1[0])
                    break
        entities = [entity for entity in entities if entity[0] not in include_entities]

        entity_names = [entity[0] for entity in entities]
        pos_entities = self._remove_neg_entities(entity_names, sentence)
        neg_entities = [entity for entity in entities if entity[0] not in pos_entities]
        pos_entities = [entity for entity in entities if entity[0] in pos_entities]
        extend_entities = []
        for entity in pos_entities:
            if entity[1] == '疾病' or entity[1] == '症状':
                if entity[0] in self.entity2syn:
                    extend_entities.extend((item, entity[1], -1, -1) for item in self.entity2syn[entity[0]])

        pos_entities.extend(extend_entities)
        # 相似度模块，计算太耗时了
        # extend_entities = []
        # for entity in pos_entities:
        #     if entity[1] == '疾病' or entity[1] == '症状':
        #         if entity[0] in self.sym2idx:
        #             ori_idx = self.sym2idx[entity[0]]
        #             res = cos_similarity(self.entity_embedding[ori_idx], self.entity_embedding)
        #             print(1)
        #             sorted = np.argsort(res)
        #             filtered = sorted[res[sorted] > 0.94]
        #             # print(len(self.idx2sym))
        #             for idx in filtered:
        #                 if idx != ori_idx:
        #                     extend_entities.append((self.idx2sym[idx][0], self.idx2sym[idx][1], entity[2], entity[3]))
        #     pos_entities.extend(extend_entities)

        return {
            'pos': pos_entities,
            'neg': neg_entities
        }
