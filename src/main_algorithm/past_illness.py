# -*- coding:utf-8 -*-

"""
既往史及其它病史
26.既往史复制主诉内容。
27.既往史存在连续字符或语段（3个字及以上）重复。
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
    def __init__(self, match_model:BaseExtract, ner_model:BaseExtract, spacy_model:BaseExtract,graph_model:BaseGraph,corrector_model) -> None:
        self.match_model = match_model
        self.ner_model = ner_model
        self.spacy_model = spacy_model
        self.graph_model = graph_model
        self.corrector = corrector_model

    def check(self,emr):
        """
        Args:
            emr:{'姓名':'***', '性别':'*', '接诊时间':'*****',...}
        Return:
            { 'score' : int 该部分扣除的分数,
                'reason': str 扣分原因
            }
        """
        fields = [
            '_item_26',
            '_item_27',
            '_item_29',
            '_item_31',
        ]
        reason = ''
        score = 0
        for field in fields:
            field_ans = getattr(self, field)(emr)
            reason += field_ans['reason']
            score += field_ans['score']
            
        ans = {'score': score, 'reason':reason}
        return ans

    def _item_26(self, emr):
        """
        26.既往史复制主诉内容。
        """
        reason = ''
        score = 0
        present = emr['既往史及其他病史']
        chief = emr['主诉']
        symbol = re.compile(r'[\u4e00-\u9fffA-Za-z0-9]')

        text_pre = symbol.findall(present)
        text_chief = symbol.findall(chief)

        if text_pre == text_chief:
            score -= 15
            reason += "既往史及其他病史复制主诉内容，扣15分\n"

        ans = {'score': score, 'reason': reason}
        return ans


    def _item_27(self, emr):
        """
        既往史存在连续字符或语段（3个字及以上）重复，但只考虑中文字符、数字和英文字母。
        """
        reason = ''
        score = 0
        min_length = 3

        removed_text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', emr['既往史及其他病史'])
        repeated_segments = []

        for i in range(len(removed_text) - min_length + 1):
            segment = removed_text[i:i + min_length]

            pattern = re.compile(r"(.)\1{2,}")  # 使用rf字符串和re.escape来避免特殊字符问题
            repeat_word = re.findall(pattern, removed_text)



            if removed_text.count(segment) > 1 or (len(repeat_word) != 0):
                repeated_segments.append(segment)
                score -= 15
                reason += "既往史及其他病史存在连续字符或语段重复，扣15分\n"
                break  # 一旦找到重复就跳出循环

        ans = {'score': score, 'reason': reason}
        return ans

    def _item_29(self, emr):
        """
        当次就诊删除过敏史信息，但未在既往史中描述相关情况。
        """
        reason = ''
        score = 0
        
        ans = {'score': score, 'reason': reason}
        return ans

    def _item_31(self, emr):
        """
        既往病史记录有错别字
        """
        score = 0
        reason = ''
        ans = self.corrector.correct(emr.get("'既往史及其他病史'", ''))
        if ans['errors']:
            score -= 2
            reason += '既往史记录有错别字。'
        ans = {'score': score, 'reason': reason}
        return ans