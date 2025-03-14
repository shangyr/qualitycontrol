# -*- coding:utf-8 -*-

"""
既往史及其它病史
26.既往史复制主诉内容。
27.既往史存在连续字符或语段（3个字及以上）重复。
28.既往史只有符号、字母、数字。
29.当次就诊删除过敏史信息，但未在既往史中描述相关情况。#缺少数据
31.既往史记录有错别字
"""
from ..extract_entity import BaseExtract
from ..knowledge_graph import BaseGraph
from fuzzywuzzy import fuzz
import numpy as np
import random
import re

class PastIllness:
    def __init__(self, match_model: BaseExtract, ner_model: BaseExtract, spacy_model: BaseExtract,
                 graph_model: BaseGraph, corrector_model) -> None:
        self.match_model = match_model
        self.ner_model = ner_model
        self.spacy_model = spacy_model
        self.graph_model = graph_model
        self.corrector = corrector_model

    def check(self, emr, item, parm):
        """
        Args:
            emr:{'姓名':'***', '性别':'*', '接诊时间':'*****',...}
        Return:
            { 'score' : int 该部分扣除的分数,
                'reason': str 扣分原因
            }
        """
        fields = [
            '_item_28',
            # '_item_29',
        ]
        if item not in fields:
            return {}
        ans = getattr(self, item)(emr, parm)
        return ans


    def _item_28(self, emr, parm):
        """
        既往史及其他病史只有符号、数字、字母。
        """
        score = 0
        reason = ''
        for item in parm.get("Parameter0",[]):
            keyword = emr.get(item, '')

            if not (re.search(r"[\u4e00-\u9fff]", keyword)):
                score -= 2
                reason += f'{item}只有符号、数字、字母。\n'
        if score < 0:
            return {"item": 28,"是否扣分": "是", "score": -2, "reason": reason, "knowledge_graph": {}}
        else:
            return {"item": 28, "是否扣分": "否", "score": 0, "reason": reason, "knowledge_graph": {}}

    def _item_29(self, emr):
        """
        当次就诊删除过敏史信息，但未在既往史中描述相关情况。
        """
        reason = ''
        score = 0

        ans = {'score': score, 'reason': reason}
        return ans


