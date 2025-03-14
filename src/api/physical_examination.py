# -*- coding:utf-8 -*-
"""
体格检查（医生临床接触听诊器、电筒等等)
34.体格检查复制主诉内容。
35.体格检查存在连续字符或语段（3个字及以上）重复。
36.体格检查内容与既往3个月内主要诊断相同的初诊病历体格检查记录重复率为100%。
38.体格检查描述有缺陷
39.体格检查的数值（如体温、心率、血压等）超过合理范围
40.主诉中提及体检发现体征异常的，现病史和体格检查未作异常结果的具体情况记录

"""
from ..extract_entity import BaseExtract
from ..knowledge_graph import BaseGraph
from fuzzywuzzy import fuzz
import numpy as np
import re

class PhysicalExamination:
    def __init__(self, match_model:BaseExtract, ner_model:BaseExtract, spacy_model:BaseExtract,graph_model:BaseGraph, corrector_model) -> None:
        self.match_model = match_model
        self.ner_model = ner_model
        self.spacy_model = spacy_model
        self.graph_model = graph_model
        self.corrector = corrector_model

    def check(self,emr, item, parm):
        """
        Args:
            emr:{'姓名':'***', '性别':'*', '接诊时间':'*****',...}
        Return:
            { 'score' : int 该部分扣除的分数,
                'reason': str 扣分原因
            }
        """
        fields = [
            '_item_36',
            '_item_39',
            '_item_40',
            '_item_41',
        ]
        if item not in fields:
            return {}

        ans = getattr(self, item)(emr, parm)
        return ans

    def _item_36(self, emr, parm):
        """
        体格检查内容与既往3个月内主要诊断相同的初诊病历体格检查记录重复率为100%。
        """
        reason = ''
        score = 0
        flag = 0 if '是否初诊' not in emr.keys() else emr['是否初诊']
        #如果是复诊病历
        if flag != 0:
            emr_past_month = {'体格检查':'患者自述头痛发热咳嗽'}
            date = '2019年5月2日'#假设这是病例时间
            str1 = set(emr_past_month['体格检查'].split())
            str2 = set(emr['体格检查'].split())
            intersection_len = len(str1.intersection(str2))
            union_len = len(str1.union(str2))
            similarity = intersection_len/union_len
        
            if similarity == 1:
                score -= 10
                reason += f'不合理复制体格检查记录，与{date}病历记录雷同\n'
        ans = {'score': score, 'reason': reason}
        return ans


    def _item_39(self, emr, parm):

        """
        检查体温、心率和血压是否在正常范围内。
        """
        score = 0
        reason = ''
        phy_list =  parm.get("Parameter0")
        for item in parm.get("Parameter0",[]):
            physical_exam_results = emr.get(item, "")

            # 检查体温
            temp_patterns = [
                r"体温[:：]?\s*(\d{2,4}(?:\.\d)?)(°?[C|摄氏度]?)",  # 摄氏度，单位可选
                r"温度[:：]?\s*(\d{2,4}(?:\.\d)?)(°?F)?"  # 华氏度，单位可选
            ]
            temp = None
            for pattern in temp_patterns:
                match = re.search(pattern, physical_exam_results)
                if match:
                    temp = float(match.group(1))
                    if "F" in pattern:  # 如有必要，将华氏度转换为摄氏度
                        temp = (temp - 32) * 5.0 / 9.0
                    break

            if temp is not None:
                if temp < 34 or temp > 42:
                    score -= 2
                    reason += f'{item}体温不在合理范围内。'


            # 检查心率
            hr_patterns = [
                r"心(率|跳)[:：]?\s*(\d{2,4})(?:次/分钟|bpm|次)?",
                r"HR[:：]?\s*(\d{2,4})(?:bpm|次)?"
            ]
            hr = None
            for pattern in hr_patterns:
                match = re.search(pattern, physical_exam_results)
                if match:
                    if match.group(1).isdigit():
                        hr = int(match.group(1))  # 匹配规则不同，这是第一个匹配规则
                    else:
                        hr = int(match.group(2)) # 对应于第二个匹配规则
                    break

            if hr is not None:
                if hr < 30 or hr > 250:
                    score -= 2
                    reason += f'{item}心率不在正常范围内。'

            # 检查血压
            bp_patterns = [
                r"(?:血压|BP)[:：]?\s*(\d{2,3})/(\d{2,3})(?: mmHg)?",
                r"收缩压[:：]?\s*(\d{2,3}) mmHg.*舒张压[:：]?\s*(\d{2,3}) mmHg",
                r"家测[:：]?\s*(\d{2,3})/(\d{2,3})(?: mmHg)?"
            ]

            systolic = diastolic = None
            for pattern in bp_patterns:
                match = re.search(pattern, physical_exam_results)
                if match:
                    systolic, diastolic = map(int, match.groups()[:2])
                    break

            if systolic is not None and diastolic is not None and \
                    (systolic < 50 or systolic > 250 or diastolic < 30 or diastolic > 150):
                score -= 2
                reason += f'{item}血压不在正常范围内。'
                # 检查脉搏
                pulse_patterns = [
                    r"脉搏[:：]?\s*(\d{2,4})(?:次/分钟|bpm|次)?",
                    r"Pulse[:：]?\s*(\d{2,4})(?:bpm|次)?"
                ]
                pulse = None
                for pattern in pulse_patterns:
                    match = re.search(pattern, physical_exam_results)
                    if match:
                        pulse = int(match.group(1))  # 正确的分组编号是 1
                        break

                if pulse is not None:
                    if pulse < 60 or pulse > 100:
                        score -= 2
                        reason += f'{item}脉搏不在正常范围内。'
        if score < 0:
            return{"item": 39, "是否扣分": "是", "score": score, "reason": reason, "knowledge_graph": {}}
        else:
            return{"item": 39, "是否扣分": "否", "score":0, "reason":"", "knowledge_graph": {}}

    def _item_40(self, emr, parm):
        """
        主诉中提及体检发现体征异常的，现病史和体格检查未作异常结果的具体情况记录
        """
        reason = ''
        score = 0
        kg = {}
        cheif_list =  parm.get("Parameter0")
        project_list = parm.get("Parameter1")
        entities = self.match_model.find_entities(emr[cheif_list[0]])['pos']
        entities_1 = [item for item in entities]
        if len(entities_1):
            if entities_1[0][1] not in kg:
                kg[entities_1[0][1]] = []
            kg[entities_1[0][1]].append(entities_1[0][0])
        
        for entity in entities:
            entity_text = entity[0]
            in_history = entity_text in emr[project_list[0]]
            in_exam = entity_text in emr[project_list[2]]
            
            if not in_history or not in_exam:
                score -= 10
                reason += '未记录与本病相关的阳性体征\n'
                break

        if score < 0:
            return{"item": 40, "是否扣分": "是", "score": score, "reason": reason, "knowledge_graph": kg}
        else:
            return{"item": 40, "是否扣分": "否", "score":0, "reason":"", "knowledge_graph": {}}

    def _item_41(self, emr, parm):

        """
        体格检查只写无特殊、无异常、正常。
        """
        score = 0
        reason = ''

        list1 = ["无特殊","无异常","正常"]

        for item in parm.get("Parameter0",[]):
            text = emr.get(item, "")
            if text in list1:
                score -= 10
                reason += f'{item}内容简单。\n'

        if score < 0:
            return{"item": 41, "是否扣分": "是", "score": score, "reason": reason, "knowledge_graph": {}}
        else:
            return{"item": 41, "是否扣分": "否", "score":0, "reason": "", "knowledge_graph": {}}

    