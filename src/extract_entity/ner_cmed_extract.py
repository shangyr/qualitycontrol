# -*- coding:utf-8 -*-

from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

from .base_extract import BaseExtract
from .cmed_tools import CmedMedicalNer
"""
huggingface命名实体识别模型得到实体, 地点时间等
"""
# "xiaxy/elastic-bert-chinese-ner"
# https://github.com/king-yyf/CMeKG_tools
# model_path: NER: 链接: https://pan.baidu.com/s/16TPSMtHean3u9dJSXF9mTw 密码:shwh
class NERCmedExtract(BaseExtract):
    """
    实体命名实体识别
    """
    def __init__(self,model_path:str) -> None:
        """
        Args:
            model_path:模型地址
        """
        self.model = CmedMedicalNer(model_path)
        self.distinguish_pos_neg = True

    def _find_entities(self,sentence:str) -> list:
        """
        命名实体识别
        Args:
            sentence:要识别的句子
        Return:
            实体列表:[{'type':'LOC','tokens':[...]},...]
        """
        ans = []
        for i in range(0,len(sentence),300):
            ans = ans + self._ner(sentence[i:i+300], start_pos = i)
        return ans
    
    def _ner(self,sentence:str, start_pos:int = 0) -> list:
        if len(sentence) == 0: return []
        entities = self.model.predict_sentence(sentence)
        
        entities = [(entity[0], entity[1], sentence.index(entity[0]) + start_pos,\
                      sentence.index(entity[0]) + len(entity[0]) + start_pos) for entity in entities]

        return entities
