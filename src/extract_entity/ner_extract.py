# -*- coding:utf-8 -*-

from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

from .base_extract import BaseExtract
"""
huggingface命名实体识别模型得到实体, 地点时间等
"""
# "xiaxy/elastic-bert-chinese-ner"
# https://huggingface.co/iioSnail/bert-base-chinese-medical-ner
class NERExtract(BaseExtract):
    """
    实体命名实体识别
    """
    def __init__(self,model_path:str) -> None:
        """
        Args:
            model_path:模型地址
        """
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.special_tokens = {self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token}
        self.distinguish_pos_neg = False

    def _find_entities(self,sentence:str) -> list:
        """
        命名实体识别
        Args:
            sentence:要识别的句子
        Return:
            实体列表:[{'type':'LOC','tokens':[...]},...]
        """
        ans = []
        for i in range(0,len(sentence),500):
            ans = ans + self._ner(sentence[i:i+500], start_pos = i)
        return ans
    
    def _ner(self,sentence:str, start_pos:int = 0) -> list:
        if len(sentence) == 0: return []
        inputs = self.tokenizer(
            sentence, add_special_tokens=True, return_tensors="pt"
        )
        
        if torch.cuda.is_available():
            self.model = self.model.to(torch.device('cuda:0'))
            for key in inputs:
                inputs[key] = inputs[key].to(torch.device('cuda:0'))
            
        with torch.no_grad():
            logits = self.model(**inputs).logits
        predicted_token_class_ids = logits.argmax(-1)
        predicted_tokens_classes = [self.model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]
        entities = []
        entity = {}
        cur_len = 0
        
        for idx, token in enumerate(self.tokenizer.tokenize(sentence,add_special_tokens=True)):
            if token in self.special_tokens: continue
            if token == self.tokenizer.unk_token: token = 'U' # 默认unk只有1个
            if len(token) >= 3 and token[:2] == '##':  token = token[2:]

            if 'B-' in predicted_tokens_classes[idx] or 'S-' in predicted_tokens_classes[idx]:
                if len(entity) != 0:
                    entity['end'] = cur_len
                    entities.append(entity)
                entity = {}
                entity['type'] = predicted_tokens_classes[idx].replace('B-','').replace('S-','')
                entity['tokens'] = [token]
                entity['start'] = cur_len
            elif 'I-' in predicted_tokens_classes[idx] or 'E-' in predicted_tokens_classes[idx] or 'M-' in predicted_tokens_classes[idx]:
                if len(entity) == 0:
                    entity['type'] = predicted_tokens_classes[idx].replace('I-','').replace('E-','').replace('M-','')
                    entity['tokens'] = []
                    entity['start'] = cur_len
                entity['tokens'].append(token)
            else:
                if len(entity) != 0:
                    entity['end'] = cur_len
                    entities.append(entity)
                    entity = {}
            cur_len += len(token)

        if len(entity) > 0:
            entity['end'] = cur_len
            entities.append(entity)
        
        entities = [(''.join(entity['tokens']), entity['type'], entity['start'] + start_pos, entity['end'] + start_pos) for entity in entities]

        return entities
