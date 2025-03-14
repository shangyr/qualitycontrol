# -*- coding:utf-8 -*-
import os
import pickle as pkl
import datetime

from .base_graph import BaseGraph
from ..extract_entity import MatchExtract

class ChineseGraph(BaseGraph):

    def __init__(self,kg_graph_path:str,match_model:MatchExtract) -> None:
        """
        初始化一些数据
        Args:
            kg_graph_path:str
            内部格式如下
            普鲁斯病..疾病\t别名\t波型热..疾病
            慢性肺原性心脏病..疾病\t伴随疾病\t室性期前收缩..疾病
            ...

        Return:
            None
        """
        
        self.reverse_word = '反向'
        self.repeatable_relations = {'相似描述'} # 可以重复出现算一个hop的关系，最多连续重复1次

        super().__init__(kg_graph_path, match_model)

