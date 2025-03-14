# -*- coding:utf-8 -*-
"""
诊断 (10分)
4.主诉、现病史、既往史、体格检查、辅助检查之间的一致性检测。
49.诊断是症状的未列出可能性最大的诊断。

"""
from ..extract_entity import BaseExtract
from ..knowledge_graph import BaseGraph
import re
from ..tool.cy_tool import CyTool
class Diagnosis:
    def __init__(self, match_model:BaseExtract, ner_model:BaseExtract, spacy_model:BaseExtract,graph_model:BaseGraph) -> None:
        self.match_model = match_model
        self.ner_model = ner_model
        self.spacy_model = spacy_model
        self.graph_model = graph_model
        actor_model = CyTool()
        self.actor_model = actor_model

    def check(self, emr):
        """
        Args:
            emr:{'姓名':'***', '性别':'*', '接诊时间':'*****',...}
        Return:
            { 'score' : int 该部分扣除的分数,
                'reason': str 扣分原因
            }
        """
        fields = [
            '_item_4',
            '_item_49',
            '_item_491',

        ]
        reason = ''
        score = 0
        for field in fields:
            field_ans = getattr(self, field)(emr)
            reason += field_ans['reason']
            score += field_ans['score']
            
        ans = {'score': score, 'reason':reason}
        return ans

    def _item_4(self, emr):
        """
        主诉、现病史、既往史、体格检查、辅助检查之间的一致性检测。
        利用match_model 对医疗实体进行提取
        主诉 - 现病史 可通过graph_model.search_link_paths找是不是有路径
        体格检查 - 诊断 可通过graph_model.search_link_paths找是不是有路径
        辅助检查 - 诊断 可通过graph_model.search_link_paths找是不是有路径
        现病史 - 诊断 可通过graph_model.search_link_paths找是不是有路径
        """
        reason = ''
        score = 0
        # 不同模块之间的一致性检查
        match_pair = [["主诉", "现病史"], ['体格检查', '初步诊断'], ['辅助检查', '初步诊断'], ['现病史', '初步诊断']]
        for pair in match_pair:
            # 匹配出每一个模块有的实体
            entites1 = self.match_model.find_entities(emr[pair[0]])['pos']
            entites2 = self.match_model.find_entities(emr[pair[1]])['pos']
            flag = 0
            count = 0
            # 若未匹配出实体，直接判断未不一致，后续可进行修改
            if (pair[0] != '辅助检查' and len(entites1) == 0) or (pair[1] != '辅助检查' and len(entites2) == 0):
                score -= 2
                temp = ''
                if len(entites1) == 0:
                    temp += pair[0] + '模块，'
                if len(entites2) == 0:
                    temp += pair[1] + '模块'
                temp += '未识别到相关信息。'
                reason += '一致性检查：{}-{}间不存在一致性，扣2分；原因:{}\n'.format(pair[0], pair[1], temp)
                continue
            # 进行遍历，查找各个实体之间是否有路径
            elif len(entites1) != 0 and len(entites2) != 0:
                for entity1 in entites1:
                    for entity2 in entites2:
                        res = self.graph_model.search_link_paths(entity1[0], entity2[0], 2)
                        is_anti=self.actor_model.judge(entity1[0],entity2[0])
                    # 存在路径 跳出循环
                        if len(res) > 0: 
                           count += 1
                        if is_anti:
                           flag = 1 
                           score -= 2
                           temp1 = ''
                           temp1 += pair[0] + '模块实体如下:' + str([item[:2] for item in entites1])
                           temp1 += pair[1] + '模块实体如下:' + str([item[:2] for item in entites2])
                           reason += '一致性检查：{}-{}间不存在一致性，扣2分；原因:病情描述之间出现矛盾,{}\n'.format(pair[0], pair[1], temp1)
                           break
                    if flag:
                       break
                if flag:
                   continue
                if count == 0 :  
                   score -= 2
                   temp1 = ''
                   temp1 += pair[0] + '模块实体如下:' + str([item[:2] for item in entites1])
                   temp1 += pair[1] + '模块实体如下:' + str([item[:2] for item in entites2])
                   reason += '一致性检查：{}-{}间不存在一致性，扣2分；原因:未匹配到路径,{}\n'.format(pair[0], pair[1], temp1)
        ans = {'score': score, 'reason': reason}
        return ans

    def _item_49(self, emr):
        """
        诊断是症状但未列出可能性最大的诊断。
        1.判断诊断是否为症状
        2.有无列出待查诊断
        """
        reason = ''
        score = 0
        matched_entities = self.match_model.find_entities(emr['初步诊断'])['pos']
        symptom_entities = [item[0] for item in matched_entities if '症状' in item[1]]
        dis = [item[0] for item in matched_entities if '疾病' in item[1]]
        if len(symptom_entities) != 0:
          if not dis:
              score -= 2
              reason = '诊断是症状但未列出可能性最大的诊断'
        ans = {'score': score, 'reason': reason}
        return ans

    def _item_491(self, emr):
        """
        第25条和28条改为对诊断的判断，扣分扣在诊断那里，合并为一条：
        次要诊断未在主诉、现病史和既往史及其他病史，体格检查，辅助检查中记录。
        只要有一个描述了就不扣分，否则在诊断处扣除5分（诊断无依据）。（每存在一个次要诊断都扣五分）
        """
        reason = ''
        score = 0

        flag = 0 if '是否初诊' not in emr.keys() else emr['是否初诊']
        #如果是初诊病历
        if flag == 0:
            # diagnosis_pattern = r"；(.*?)$" 
                # 使用正则表达式去除数字和顿号
                cleaned_text = re.sub(r"\d+、", "", emr['初步诊断'])
                # 去除分号
                cleaned_text1 = re.sub(r"；", ";", cleaned_text)
                # 使用分号分割字符串，得到列表
                diagnosis_list1 = cleaned_text1.split(";")
                # 去除列表中每个元素的首尾空白字符
                diagnosis_list = [item.strip() for item in diagnosis_list1 if item]
                if diagnosis_list:
                    del diagnosis_list[0]
                    # print('次要诊断是',diagnosis_list)
                    text = emr['主诉'] + emr['现病史'] + emr['既往史及其他病史'] + emr['体格检查'] + emr['辅助检查']
                    
                    entities = self.match_model.find_entities(text)['pos']+self.match_model.find_entities(text)['neg']
                    # print('主诉现病史和既往史、体格检查辅助检查中的实体',entities)
                    for diagnosis in diagnosis_list:
                        sym = 0
                        for i in range(len(diagnosis) - 1):
                            
                            if diagnosis[i]  in text and diagnosis[i+1]  in text:
                                sym += 1
                        if sym == 0:

                            paths = []
                            for entity in entities:
                                paths.extend(self.graph_model.search_link_paths(entity[0], diagnosis, 2))
                                # print('存在的路径',paths)
                            if len(paths) == 0:   
                                score -= 5
                                reason += '次要诊断{}未在主诉、现病史和既往史及其他病史，体格检查，辅助检查中记录。扣5分。\n'.format(diagnosis)


        ans = {'score': score,'reason': reason}
        return ans

