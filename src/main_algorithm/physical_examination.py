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
import json
import json
import pandas as pd
from datetime import datetime

class PhysicalExamination:
    def __init__(self, match_model:BaseExtract, ner_model:BaseExtract, spacy_model:BaseExtract,graph_model:BaseGraph, corrector_model) -> None:
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
            '_item_34',
            '_item_35',
            '_item_36',
            '_item_37',
            '_item_38',
            '_item_39',
            '_item_40',
            '_item_401'

        ]
        reason = ''
        score = 0
        for field in fields:
            field_ans = getattr(self, field)(emr)
            reason += field_ans['reason']
            score += field_ans['score']
            
        ans = {'score': score, 'reason':reason}
        return ans

    def _item_34(self, emr):
        """
        体格检查复制主诉内容。
        """
        reason = ''
        score = 0
        physical = emr['体格检查']
        chief = emr['主诉']
        symbol = re.compile(r'[\u4e00-\u9fffA-Za-z0-9]')

        text_phy = symbol.findall(physical)
        text_chief = symbol.findall(chief)
    
        if text_phy == text_chief:
            score -= 15
            reason += "体格检查复制主诉内容，扣15分\n"
        
        ans = {'score': score,'reason': reason}
        return ans



    def _item_35(self, emr):
        """
        体格检查存在连续字符或语段（3个字及以上）重复。
        """
        reason = ''
        score = 0

        removed_text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', emr['体格检查'])

        length = len(removed_text)
        repeated_substrings = []

        
        repeated_substrings = set()
        
        for i in range(length - 1):
            # 检查从当前位置开始的所有可能的子串
            for length in range(3, length - i + 1):  # 子串长度至少为3
                substring = removed_text[i:i + length]
                # 检查这个子串是否直接在后面连续重复出现
                if substring == removed_text[i + length:i + 2 * length]:
                    repeated_substrings.add(substring)
                    break  # 找到重复子串后，跳出内层循环

        a = list(repeated_substrings)

        if len(a)!= 0:
            score -= 15
            reason += '体格检查存在连续字符或语段（3个字及以上）重复，重复语段为{}，扣15分\n'.format(a)
        ans = {'score': score, 'reason': reason}
        return ans
   

    def _item_36(self, emr):
        """
        体格检查内容与既往3个月内主要诊断相同的初诊病历体格检查记录重复率为100%。
        """
        reason = ''
        score = 0
        # 当前病历为复诊病历
        if emr.get('是否初诊', 0):
            emr_past_month = emr['既往病历']

            for emr_past in emr_past_month:
                if emr_past.get('体格检查', 0):
                    # 既往病历中的初诊病历
                    if not emr_past.get('是否初诊', 0):
                        symbol_diagnosis = re.compile(r'1、(.*?)；')
                        diagnosis_pre = symbol_diagnosis.search(emr['初步诊断'])
                        diagnosis_past = symbol_diagnosis.search(emr_past['初步诊断'])

                        # 主要诊断相同
                        if diagnosis_past.group(1) == diagnosis_pre.group(1):
                            past_pre = emr_past['体格检查']
                            present_pre = emr['体格检查']

                            symbol_text = re.compile(r'[\u4e00-\u9fffA-Za-z0-9]')
                            text_past = symbol_text.findall(past_pre)
                            text_present = symbol_text.findall(present_pre)

                            # 现病史内容相同
                            if text_past == text_present:
                                score -= 10
                                reason += '不合理复制体格检查记录，与既往初诊病历记录雷同\n'
        ans = {'score': score, 'reason': reason}
        return ans

    def _item_37(self, emr):
        """
        体格检查描述有错别字
        """

        # 初值
        score = 0
        reason = ''
        if True:
            return {'score': score, 'reason': reason}
        ans = self.corrector.correct(emr.get("体格检查", ''))
        if ans['errors']:
            score -= 2
            reason += '体格检查描述有错别字。'

        ans = {'score': score, 'reason': reason}
        return ans

    def _item_38(self, emr):
        """
        体格检查描述有缺陷
        """
        reason = ''
        score = 0

        ans = {'score': score, 'reason': reason}
        return ans

    def _item_39(self, emr):

        """
        检查体温、心率和血压是否在正常范围内。
        """
        score = 0
        reason = ''

        # 提取体格检查结果
        # physical_exam_results = emr.get("体格检查", "") + " " + \
        #                         emr.get("主诉", "") + " " + \
        #                         emr.get("现病史", "") + " " + \
        #                         emr.get("既往史及其他病史", "") + " " + \
        #                         emr.get("辅助检查", "")
        physical_exam_results = emr.get("体格检查", "")

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
                reason += '体温不在合理范围内。'


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
                reason += '心率不在正常范围内。'

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
            reason += '血压不在正常范围内。'
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
                    reason += '脉搏不在正常范围内。'
        return {'score': score, 'reason': reason}

    def _item_40(self, emr):
        """
        主诉中提及体检发现体征异常的，现病史和体格检查未作异常结果的具体情况记录
        """
        reason = ''
        score = 0
        if '体检' in emr['主诉']:
            cheif_entities = self.match_model.find_entities(emr['主诉'])['pos'] 
            entities_1 = [item for item in cheif_entities if item[1] == '症状']


            # print('从主诉中提取的异常体征为',entities_1)
            #修改json_adr为json的绝对地址
            json_adr = r"F:\广医质检\pdf\qualitycontrol_service\disease_classify.json"
            df = pd.read_json(json_adr)
            jslist = []
            for i in range(len(df)):
                    for j in range(len(df.columns)):
                        element = df.at[i, df.columns[j]]
                        my_dict = json.loads(element)
                        jslist.append(my_dict)
            # print(jslist)
            my_dict = {}
            for data_string in jslist:
                data_dict = json.loads(data_string)
                my_dict.update(data_dict)
            # print(my_dict)
            for entity in entities_1:
                entity_text= entity[0]
                # print(entity_text)
                if entity_text in my_dict.keys():
                    # print('yes')
                    classify = my_dict[entity_text]
                    # print('分类为',classify)
                    if classify == '体格检查':
                        history = emr['现病史']
                        # 
                
                        # text = history + exam
                        his_enti = self.match_model.find_entities(history)['pos'] + self.match_model.find_entities(history)['neg']
                        # print('从体格检查和现病史中匹配到的实体列表为：',his_enti)
                        exam =  emr['体格检查']
                        exa_enti = self.match_model.find_entities(exam)['pos'] + self.match_model.find_entities(exam)['neg']
                        paths_his = 0
                        paths_exa = 0 
                        for enti in his_enti:
                            if enti[0] == entity_text:
                                # print('enti[0]',enti[0])
                                paths_his += 1
                                break
                            elif self.graph_model.search_link_paths(enti[0],entity_text):
                                # print('找到体格检查和现病史中的实体{}和主诉中的异常体征{}有路径'.format(enti[0],entity_text),self.graph_model.search_link_paths(enti[0],entity_text))
                                paths_his += 1
                                break
                        for enti in exa_enti:
                            if enti[0] == entity_text:
                                # print('enti[0]',enti[0])
                                paths_exa += 1
                                break
                            elif self.graph_model.search_link_paths(enti[0],entity_text):
                                # print('找到体格检查和现病史中的实体{}和主诉中的异常体征{}有路径'.format(enti[0],entity_text),self.graph_model.search_link_paths(enti[0],entity_text))
                                paths_exa += 1
                                break


                        if paths_his  == 0:
                            score -= 10
                            reason += '主诉中提及体检发现体征{}异常，但现病史未记录与本病相关的阳性体征\n'.format(entity_text)
                            break
                        if paths_exa  == 0:
                            score -= 10
                            reason += '主诉中提及体检发现体征{}异常，但体格检查未记录与本病相关的阳性体征\n'.format(entity_text)
                            break
                    # print('paths_his',paths_his)
                    # print('paths_exa',paths_exa)

                    # if not in_history and not in_exam:
                    #     score -= 10
                    #     reason += '主诉中提及体检发现体征{}异常，但现病史和体格检查未记录与本病相关的阳性体征\n'.format(entity_text)
                    #     break
        ans = {'score': score, 'reason': reason}
        return ans
    # def _item_401(self, emr):
    #     """
    #     体格检查中需要有 诊断结果应该涉及到的检查内容（阳性或阴性）。
    #     1.体格检查只写到正常、无异常或无特殊，直接扣10分
    #     根据诊断的疾病，根据疾病发生的部位，再去体格检查里面看有没有相应部位的内容
    #     """
    #     reason = ''
    #     score = 0
    #     removed_text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', emr['体格检查'])
    #     if removed_text == '正常' or removed_text == '无异常' or removed_text == '无特殊':
    #         score -= 10
    #         reason += '体格检查只写到正常、无异常或无特殊，直接扣10分\n'
    #     if emr['初步诊断']:
    #         # diagnoses = emr['初步诊断'].split("；")
    #         # print('初步诊断',diagnoses)
    #         # formatted_diagnoses = [re.sub(r"\d+、", "", item).strip() for item in diagnoses if item]
           
            
    #         #从诊断中提取阳性和阴性的所有症状和疾病和人体实体
    #         # for item in formatted_diagnoses:
    #         dia_enti = self.match_model.find_entities(emr['初步诊断'])['pos'] + self.match_model.find_entities(emr['初步诊断'])['neg']
    #         phy_enti = self.match_model.find_entities(emr['体格检查'])['pos'] + self.match_model.find_entities(emr['体格检查'])['neg']
    #         for enti_ in dia_enti:
    #             paths = []
    #             for enti in phy_enti:
    #                 path = self.graph_model.search_link_paths(enti[0], enti_[0])
    #                 if path:
    #                     paths.append(path)
    #             if not paths:
    #                 score -= 10
    #                 reason += '体格检查中需要有诊断结果应该涉及到的检查内容（阳性或阴性）\n'
    #                 break
    #     ans = {'score': score, 'reason': reason}
    #     return ans
    
    def _item_401(self, emr):
        """
        体格检查中需要有 诊断结果应该涉及到的检查内容（阳性或阴性）。
        1.体格检查只写到正常、无异常或无特殊，直接扣10分
        根据诊断的疾病，根据疾病发生的部位，再去体格检查里面看有没有相应部位的内容
        """
        reason = ''
        score = 0
        text = emr['体格检查']
        # removed_text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', emr['体格检查'])
        wrong_list = ['正常','无异常','无特殊']
        if text in wrong_list:
            score -= 10
            reason += '体格检查内容太简单，直接扣10分\n'
        elif not(re.search(r"[\u4e00-\u9fffA-Za-z]'", text)):#如果没有中文
            sig_list = ['(-)','(+)','-','+',]
            for sig in sig_list:
                if sig in text:
                    removed_text = text.replace(sig,'')

                    stripped_text = re.sub(r"[^\w\s]", "", removed_text)#移除空格和字符
                    stripped_text = re.sub(r"\s+", "", stripped_text)
                    if bool(re.fullmatch(r"[A-Za-z]*", stripped_text)):
                        score -= 10
                        reason += '体格检查内容太简单，直接扣10分\n'
                        break
        if emr['初步诊断']:   
            entities = []
            #从诊断中提取阳性和阴性的所有症状和疾病和人体实体
            pos_enti = self.match_model.find_entities(emr['初步诊断'])['pos'] 
            neg_enti = self.match_model.find_entities(emr['初步诊断'])['neg']
            enti = pos_enti + neg_enti
            # print(enti)
            type_list = ['疾病','疾病a','疾病39','人体']
            
            buwei = []
            for i in range(len(enti)):
                entities.append(enti[i][0]) if enti[i][1] in type_list  else None
                if enti[i][1] == '人体':
                    # print('有人体实体')
                    buwei.append(enti[i][0])
            # print('诊断中所有实体为：',entities)
            
            for entity in entities:
                #找到这个实体的所有关系
                all_path_list = self.graph_model.search_paths(entity)
                label = 0
                for path in all_path_list:
                    if path[1] == '发生部位':
                        # print('实体{}发生部位为{}'.format(entity,path[2]))
                        buwei.append(path[2])
                        label += 1
                        break
                if label == 0:
                    # print('实体{}没有找到发生部位'.format(entity))
                    buwei.append(entity)
            buwei = list(set(buwei))
            # print('匹配到的发生部位以及没有匹配到的实体名称为：', buwei)
            #寻找体格检查中是否有相应部位的内容
            phy_entities1 = self.match_model.find_entities(emr['体格检查'])['pos'] + self.match_model.find_entities(emr['体格检查'])['neg']
            
            phy_entities = []

            for i in range(len(phy_entities1)):
                phy_entities.append(phy_entities1[i][0]) 
            
            if 'T' or '℃' in emr['体格检查']:
                phy_entities.append('发热')
            # print('体格检查中的实体为：',phy_entities)
            if phy_entities == []:
                score += 0
                reason += '判断体格检查中需要有诊断结果应该涉及到的检查内容（阳性或阴性）时，在体格检查中没有匹配到实体\n'
            if buwei :
                for buwei_ in buwei:# phy_buwei = []
                    paths = []
                    for entity in phy_entities:
                        #如果本身就是部位
                        if entity in buwei:
                            paths.append(entity)
                            break
                        else:
                    #         phy_path_list = self.graph_model.search_paths(entity)

                    #         for path in phy_path_list:
                    #             label = 0
                    #             if path[1] == '发生部位':
                    #                 print('体格检查中实体{}发生部位为{}'.format(entity,path[2]))
                    #                 phy_buwei.append(path[2])
                    #                 label += 1
                    #                 break
                    #         if label == 0:
                    #             print('体格检查中实体{}的路径中没有找到发生部位'.format(entity))
                    #             phy_buwei.append(entity)
                    # phy_buwei = list(set(phy_buwei))            
                    #现在得到phy_buwei和buwei两个列表，如果buwei中的元素在phy_buwei 中找不到就扣分
                    # for entity in buwei:
                    #     if entity not in phy_buwei:
                    #         score -= 10
                    #         reason += '体格检查中需要有诊断结果应该涉及到的检查内容（阳性或阴性）时，没有找到实体{}的发生部位\n'.format(entity)
                    #         break
                            # print('肺部感染和湿性啰音的路径为',self.graph_model.search_link_paths('肺部感染','湿性啰音'))

                            path = self.graph_model.search_link_paths(buwei_,entity)
                            paths.append(path)
                            # print('体格检查中的实体{}与诊断中的发生部位{}有如下路径'.format(entity,buwei_),path)
                    if not paths:
                        score -= 10
                        reason += '体格检查中需要有诊断结果应该涉及到的检查内容（阳性或阴性）\n'


        ans = {'score': score, 'reason': reason}
        return ans

    