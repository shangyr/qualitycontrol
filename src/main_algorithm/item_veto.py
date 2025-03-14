# -*- coding:utf-8 -*-

"""
二期质控规则中所有会产生单项否决的规则
1.病历记录创建日期与接诊日期相同，判断病历是否在就诊时及时完成。  初步完成
2.同一患者当次病历记录与既往3个月病历记录内容重复，主诉、现病史、既往史、辅助检查、体格检查所有文字重复率为100%。 11分
8.主诉缺项或只有符号等无意义内容。已完成
17.现病史缺项或只有符号等无意义内容。已完成
33.体格检查缺项或只有符号等无意义内容。已完成
50.主要诊断与主诉等病史不符。 已完成  (10分，乙级)
51.无处理内容，同时无嘱托记录。已完成  (15分，乙级)
52.手术记录创建时间超过手术结束时间24小时。
55.有创操作记录创建时间超过操作时间24小时。

注意仔细查看初诊和复诊的区分
"""


import random
import math
import re
import jieba
import difflib
from fuzzywuzzy import fuzz
import numpy as np
from collections import Counter
import datetime
from ..extract_entity import BaseExtract
from ..knowledge_graph import BaseGraph
from ..diagnosis import Predictor, ChineseALLConfig
class ItemVeto:
    def __init__(self, match_model: BaseExtract, ner_model: BaseExtract, spacy_model: BaseExtract,
                 graph_model: BaseGraph,diagnosis_path:str) -> None:
        self.match_model = match_model
        self.ner_model = ner_model
        self.spacy_model = spacy_model
        self.graph_model = graph_model
        self.diagnosis_path = diagnosis_path

    def check(self, emr):
        """
        Args:
            emr:{'姓名':'***', '性别':'*', '接诊时间':'*****',...}
        Return:
            { 'vote' : str, 乙级病历 / 丙级病历 / 无否决
                'reason': str 扣分原因
            }
        """
        fields = [
            '_item_1',
            '_item_2',
            '_item_5',
            '_item_8', # 形式
            '_item_17',# 形式
            '_item_33',# 形式
            '_item_50',# 形式
            '_item_51',
            '_item_52',
            '_item_55',
        ]

        reason = ''
        vote = '无否决'
        score = 0

        for field in fields:
            field_ans = getattr(self, field)(emr)
            score += field_ans['score']

            if reason == ' ':
                reason += field_ans['reason']
            elif field_ans != '' and reason != ' ':  
                reason += field_ans['reason']


            if field_ans['vote'] == '乙级病历' and vote == '无否决':
                vote = '乙级病历'
            if field_ans['vote'] == '丙级病历':
                vote = '丙级病历'
            if field_ans['vote'] == -5:
                score = field_ans['vote']
            
        ans = {'vote': vote, 'reason':reason,'score':score}
        return ans

    def _item_1(self, emr):
        """
        病历记录创建日期与接诊日期相同，判断病历是否在就诊时及时完成。
        简单通过正则匹配提取日期，然后判断是否在一天即可
        """
        # 不清楚时间格式，这里使用re库做正则匹配，如果格式确定的话可以更简单一点，
        pattern = re.compile(r'([0-9]{4}).([0-9]{2}).([0-9]{2})')
        res1 = pattern.search(emr.get('就诊时间', ''))
        res2 = pattern.search(emr.get('病历创建时间', ''))
        # 匹配到对应日期进行判断
        if res1 and res2:
            for i in range(3):
                # print(res1.group(i+1))
                # print(res2.group(i+1))
                if res1.group(i + 1) != res2.group(i + 1):
                    ans = {'vote': '丙级病历', 'reason': '病历记录创建日期与接诊日期不同,丙级否决。\n','score':-100}
                    return ans

        ans = {'vote': '无否决', 'reason': '','score':0}

        return ans

    def _item_2(self, emr):
        """
        同一患者当次病历记录与既往3个月病历记录内容重复，主诉、现病史、既往史、辅助检查、体格检查所有文字重复率为100%。
        100%重复可以用简单的==判断，或者其他方法
        """
        past_record = emr.get('既往病历', '')

        # 未找到既往病历
        if not past_record or past_record[0]=={}:
            ans = {'vote': '无否决', 'reason': '','score':0}
            return ans

        emr_list = {'主诉', '现病史', '既往史及其他病史', '体格检查', '辅助检查'}
        # emr_new = {key: value for key, value in emr.items() if key in emr_list}

        for record in past_record:
            if record == {}:
                break
            bj = 1

            for k in emr_list:
                symbol = re.compile(r'[\u4e00-\u9fffA-Za-z0-9]')
                present_string = symbol.findall(emr[k])
                past_string = symbol.findall(record[k])
                if present_string != past_string:
                    bj = 0

            if bj == 1:
                ans = {'vote': '乙级病历','reason': '同一患者当次病历记录与既往3个月病历记录内容重复，主诉、现病史、既往史、辅助检查、体格检查所有文字重复率为100%。','score':-11}
                return ans

        ans = {'vote': '无否决', 'reason': '','score': 0}
        return ans


    def _item_5(self, emr):
        """
        同一章节内存在连续字符或语段（5个字及以上）重复。
        可以基于之前的匹配规则
        """

        emr_list = {'主诉', '现病史', '既往史及其他病史', '体格检查', '辅助检查', '初步诊断', '处理','嘱托'}  # 仅需要对这些章节进行判断
        emr_new = {key: value for key, value in emr.items() if key in emr_list}
        round = 5  # 阈值1：对于一个章节截取的次数
        for i in emr_new:
            length_i = len(emr_new[i])
            lim_len = min(math.ceil(length_i / 3), 10)  # 阈值：截取的随机片段的字符串长度
            for k in range(length_i - lim_len):
                # start = random.randint(0, length_i - lim_len)
                sample_string = emr_new[i][k:k + lim_len]
                if emr_new[i].count(sample_string) > 1 and emr_new[i].count(" ") < 2:
                    ans = {'vote': '无否决',
                           'reason': '','score': 0}
                    return ans

        ans = {'vote': '无否决', 'reason': '','score': 0}
        return ans

    def _item_8(self, emr):
        """
        主诉缺项或只有符号等无意义内容，乙级
        基于之前的规则
        """
        keyword = emr['主诉']
        if keyword == '':
            ans = {'vote': '乙级病历', 'reason': '主诉缺项，乙级否决。\n', 'score':-10}
            return ans
        elif not(re.search(r"[\u4e00-\u9fa5]", keyword)): #判断是否有中文 有就不全是符号、数字，字母等字符 (这里未将字母看成符号，单独判断了一下)
            ans = {'vote': '乙级病历', 'reason': '主诉只有“-、。.”等无意义内容，乙级否决。\n','score':-10}
            return ans

        ans = {'vote': '无否决', 'reason': '','score':0}
        return ans


    def _item_17(self, emr):
        """
        现病史缺项或只有符号等无意义内容，乙级
        基于之前的规则
        """
        keyword = emr['现病史']
        if keyword == '':
            ans = {'vote': '乙级病历', 'reason': '现病史缺项，乙级否决。\n','score':-15}
            return ans
        elif not (re.search(r"[\u4e00-\u9fa5]", keyword)):  # 判断是否有中文或者字母 有就不全是符号、数字等字符 (这里未将字母看成符号，单独判断了一下)
            ans = {'vote': '乙级病历', 'reason': '现病史只有“-、。.”等无意义内容，乙级否决。\n','score':-15}
            return ans

        ans = {'vote': '无否决', 'reason': '','score':0}
        return ans

    def _item_33(self, emr):
        """
        体格检查缺项或只有符号等无意义内容，乙级
        体格检查（同时有字母和数字）或者中文或者包含（-）（全角和半角都算），则不列入1.6.1规则；只写了无，也进行1.6.1扣分
        """
        keyword = emr['体格检查']

        exclude_list = [ '(-)',  '(+)', 'T','Bp']

        if keyword == '':
            ans = {'vote': '乙级病历', 'reason': '体格检查只有“-、。.”等无意义内容，乙级否决。\n','score':-15}
            return ans

        if not(re.search(r"[\u4e00-\u9fff]", keyword)) or keyword == '无':#没有中文或者只有无
            # print('没有中文或者只有无')
            if not(re.search(r"[a-zA-Z]", keyword) and re.search(r"[0-9]", keyword)):#如果也不是同时有字母和数字
                # print('不是同时有字母和数字')
                sym = 0
                for exclude_item in exclude_list:
                    if exclude_item in keyword:
                        sym += 1
                if sym == 0:
                    ans = {'vote': '乙级病历', 'reason': '体格检查只有“-、。.”等无意义内容，乙级否决。\n','score':-15}
                    return ans

        ans = {'vote': '无否决', 'reason': '','score':0}
        return ans


    def _item_50(self, emr):
        """
        主要诊断与主诉等病史不符。
        基于知识图谱/诊断模型判断？
        """
        def string_similar(s1, s2):
            return difflib.SequenceMatcher(None, s1, s2).quick_ratio()


        flag = 0 if '是否初诊' not in emr.keys() else emr['是否初诊']
        # print(flag)
        if not flag:
            entity_type_list = ['疾病','症状','症状39', '疾病39','症状a', '疾病a','疾病icd10']
            entities_1_list = []
            entities_2_list = []

            entities_1 = self.match_model.find_entities(emr['主诉'])['pos']
            entities_2 = self.match_model.find_entities(emr['初步诊断'])['pos']
            # print(entities_1)
            for entity in entities_1:
                if entity[1] in entity_type_list:
                    entities_1_list.append(entity[0])
            for entity in entities_2:
                if entity[1] in entity_type_list:
                    entities_2_list.append(entity[0])
            print(entities_1_list)
            print(entities_2_list)

            bj = 0
            # 实体间关系匹配
            if entities_2_list != []:
                entity_y = entities_2_list[0]
                for entity_x in entities_1_list:
                    if string_similar(entity_x, entity_y) >= 0.5:
                        ans = {'vote': '无否决', 'reason':'', 'score': 0}
                        return ans

                    else:
                        attr = self.graph_model.search_link_paths(entity_y, entity_x, 1)
                        if attr != []:
                            ans = {'vote': '无否决', 'reason':'', 'score': 0}
                            return ans

            cfg = ChineseALLConfig(data_dir=self.diagnosis_path)
            p = Predictor(cfg)
            data = [
                {
                    "emr_id": '123456',
                    "doc": emr.get("主诉", "") + " " + \
                           emr.get("现病史", "") + " " + \
                           emr.get("既往史及其他病史", "") + " " + \
                           emr.get("体格检查", "") + " " + \
                           emr.get("辅助检查", ""),
                    "label": []
                }
            ]
            raw_ans = p.predict(data)  # 调用一次执行一次
            predict_ans = []
            for key in raw_ans[0]['predict_score']:
                predict_ans.append((key, raw_ans[0]['predict_score'][key]))
                predict_ans.sort(key=lambda x: x[1], reverse=True)
            predict_ans = predict_ans[:5]


            for key in predict_ans:
                entity_y = entities_2_list[0]
                print(key[0])
                if entity_y in key[0]:
                    ans = {'vote': '无否决', 'reason': '', 'score': 0}
                    return ans


            if entities_2_list == [] or '复诊' in emr['主诉']:
                ans = {'vote': '无否决', 'reason':'', 'score': 0}
            else:
                ans = {'vote': '乙级病历', 'reason':'主要诊断与主诉等病史不符,乙级否决。'+'(第一诊断为['+entities_2_list[0]+']，主诉中匹配到的症状是'+ str(entities_1_list)+'，知识图谱未匹配到关系)\n', 'score': -10}
            return ans

        else:
            ans = {'vote': '无否决', 'reason': '', 'score': 0}
            return ans


    def _item_51(self, emr):
        """
        无处理内容，同时无嘱托记录。乙级
        可基于之前的指控规则
        """
        if emr['嘱托'] !='':
            ans = {'vote': '无否决', 'reason': '', 'score': 0}
            return ans
        elif emr['处理'] == '{}{ 请录入处方！}':
            ans = {'vote': '乙级病历', 'reason': '无处理内容，同时无嘱托记录。', 'score': -15}
            return ans
        else:
            one_site = emr['处理'].find("1  ")
            two_site = emr['处理'].find("2  ")

            if one_site != -1 and two_site == -1:
                num = len(re.findall(r'\d', emr['处理']))-len(re.findall(r'\d\s\s', emr['处理']))
                if '诊查费' in emr['处理'] and emr['处理'].count('诊查费') == emr['处理'].count('*') and emr['处理'].count('诊查费') == num:
                    ans = {'vote': '乙级病历', 'reason': '无处理内容，同时无嘱托记录。', 'score': -15}
                    return ans
        ans = {'vote': '无否决', 'reason': '', 'score': 0}
        return ans

    def _item_52(self, emr):
        """
        手术记录创建时间超过手术结束时间24小时。
        同item_1
        """

        # 同上，正则匹配
        pattern = re.compile(r'([0-9]{4}).([0-9]{2}).([0-9]{2}).([0-9]{2}).([0-9]{2})')
        med_record = emr.get('手术记录', '')

        # 未找到手术记录
        if not med_record:
            ans = {'vote': '无否决', 'reason': '','score':0}
            return ans

        res1 = pattern.search(med_record.get('创建时间', ''))
        res2 = pattern.search(med_record.get('手术结束时间', ''))

        # 匹配到对应日期进行判断
        if res1 and res2:
            date1 = datetime.datetime(int(res1.group(1)), int(res1.group(2)), int(res1.group(3)), int(res1.group(4)),
                                      int(res1.group(5)))
            date2 = datetime.datetime(int(res2.group(1)), int(res2.group(2)), int(res2.group(3)), int(res2.group(4)),
                                      int(res2.group(5)))
            if (date1 - date2).days >= 1:
                ans = {'vote': '乙级病历', 'reason': '手术记录创建时间超过手术结束时间24小时，乙级否决。\n','score':-15}
                return ans

        ans = {'vote': '无否决', 'reason': '','score':0}

        return ans


    def _item_55(self, emr):
        """
        有创操作记录创建时间超过操作时间24小时。
        同item_1
        """

        # 同上，正则匹配
        pattern = re.compile(r'([0-9]{4}).([0-9]{2}).([0-9]{2}).([0-9]{2}).([0-9]{2})')
        med_record = emr.get('有创操作记录', '')

        # 未找到有创操作记录
        if not med_record:
            ans = {'vote': '无否决', 'reason': '', 'score':0}
            return ans

        res1 = pattern.search(med_record.get('创建时间', ''))
        res2 = pattern.search(med_record.get('操作时间', ''))

        # 匹配到对应日期进行判断
        if res1 and res2:
            date1 = datetime.datetime(int(res1.group(1)), int(res1.group(2)), int(res1.group(3)), int(res1.group(4)),
                                      int(res1.group(5)))
            date2 = datetime.datetime(int(res2.group(1)), int(res2.group(2)), int(res2.group(3)), int(res2.group(4)),
                                      int(res2.group(5)))

            if (date1 - date2).days >= 1:
                ans = {'vote': '乙级病历', 'reason': '有创操作记录创建时间超过操作时间24小时，乙级否决。\n','score':-15}
                return ans

        ans = {'vote': '无否决', 'reason': '','score':0}

        return ans
