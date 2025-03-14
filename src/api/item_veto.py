# -*- coding:utf-8 -*-

"""
二期质控规则中所有会产生单项否决的规则
1.病历记录创建日期与接诊日期相同，判断病历是否在就诊时及时完成。  初步完成
2.同一患者当次病历记录与既往3个月病历记录内容重复，主诉、现病史、既往史、辅助检查、体格检查所有文字重复率为100%。 11分
5.同一章节内存在连续字符或语段（5个字及以上）重复。 是否写在单项否决
8.主诉缺项或只有符号等无意义内容。已完成
17.现病史缺项或只有符号等无意义内容。已完成
33.体格检查缺项或只有符号等无意义内容。已完成
50.主要诊断与主诉等病史不符。 已完成  (10分，乙级)
53.无处理内容，同时无嘱托记录。已完成  (15分，乙级)
52.手术记录创建时间超过手术结束时间24小时。
55.有创操作记录创建时间超过操作时间24小时。

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

    def check(self, emr, item, parm):
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
            '_item_8',
            '_item_33',
            '_item_50',
            '_item_52',
            '_item_53',
        ]

        reason = ''
        vote = '无否决'
        score = 0

        if item not in fields:
            return {}
        ans = getattr(self, item)(emr, parm)
        return ans

        """
        field_ans = getattr(self, item)(emr, parm)
        reason += field_ans['reason']
        score += field_ans['score']
        vote = field_ans['vote']
            
        ans = {'vote': vote, 'reason':reason,'score':score}
        return ans
        """

    def _item_1(self, emr, parm):
        """
        病历记录创建日期与接诊日期相同，判断病历是否在就诊时及时完成。
        简单通过正则匹配提取日期，然后判断是否在一天即可
        """
        # 不清楚时间格式，这里使用re库做正则匹配，如果格式确定的话可以更简单一点，
        pattern = re.compile(r'([0-9]{4})-([0-9]{2})-([0-9]{2})')
        date_list = parm['Parameter0']
        res1 = pattern.search(emr[date_list[0]])
        res2 = pattern.search(emr[date_list[1]])

        # 匹配到对应日期进行判断
        if res1 and res2:
            for i in range(3):
                # print(res1.group(i+1))
                # print(res2.group(i+1))
                if res1.group(i + 1) != res2.group(i + 1):
                    ans = {
                        # 当前规则的编号
                        "item": 1,
                        "是否扣分": "是",
                        "vote": "丙级病历",
                        "score": -100,
                        "reason": date_list[0]+"与"+date_list[1]+"不相同。",
                        }
                    return ans

        ans = {"item": 1,"是否扣分": "否", "vote": "无否决", "score": 0, "reason": ""}
        return ans

    def _item_2(self, emr, parm):
        """
        同一患者当次病历记录与既往3个月病历记录内容重复，主诉、现病史、既往史、辅助检查、体格检查所有文字重复率为100%。
        100%重复可以用简单的==判断，或者其他方法
        """
        past_record = parm['Parameter0']
        checklist = parm['Parameter1']

        # 未找到既往病历
        if not past_record or past_record == []:
            ans = {"item": 2, "是否扣分": "否", "vote": "无否决", "score": 0, "reason": ""}
            return ans

        for record in past_record:

            if record == {}:
                break
            bj = 1

            result_string = ""

            for k in checklist:
                symbol = re.compile(r'[\u4e00-\u9fffA-Za-z0-9]')
                present_string = symbol.findall(emr[k])

                past_string = symbol.findall(record[k])
                if present_string != past_string:
                    # print(past_string)
                    bj = 0
                else:
                    result_string += k
                    result_string += ' '

            if bj == 1:
                ans = {
                        "item": 2,
                        "是否扣分": "是",
                        'vote': '无否决',
                        'score': 0,
                        'reason': result_string + '文字重复率为100%。\n',
                }
                return ans

        ans = {"item": 2, "是否扣分": "否", "vote": "无否决", "score": 0, "reason": ""}
        return ans


    def _item_8(self, emr, parm):
        """
        主诉、现病史、初步诊断缺项或只有符号等无意义内容，乙级
        基于之前的规则
        """
        score = 0
        result_string = ""
        bj = "否"
        checklist = parm['Parameter0']
        for k in checklist:
            keyword = emr[k]
            if keyword == '':
                bj = "是"
                score -= 10
                result_string += k+" "
                vote = "乙级病历"

            elif not(re.search(r"[\u4e00-\u9fff]", keyword)): #判断是否有中文 有就不全是符号、数字，字母等字符
                bj = "是"
                score -= 10
                result_string += k
                vote = "乙级病历"
        if bj == "是":
            ans = {"item": 8, "是否扣分": bj, "vote": vote, "score":score, "reason": result_string+"缺项或只有符号等无意义内容。\n"}
        else:
            ans = {"item": 8, "是否扣分": "否", "vote": "无否决", "score": 0, "reason": ""}
        return ans

    def _item_33(self, emr, parm):
        """
        体格检查缺项或只有符号等无意义内容，乙级
        体格检查（同时有字母和数字）或者中文或者包含（-）（全角和半角都算），则不列入1.6.1规则；只写了无，也进行1.6.1扣分
        """
        keyword = emr[parm['Parameter0'][0]]

        exclude_list = ['(-)',  '(+)', 'T', 'Bp', '℃']

        if keyword == '':
            ans = {
                "item": 33,
                "是否扣分": "是",
                'vote': '乙级病历',
                'score': -10,
                'reason': '体格检查缺项或只有符号等无意义内容。\n'
            }
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
                    ans = {
                        "item": 33,
                        "是否扣分": "是",
                        'vote': '乙级病历',
                        'score': -10,
                        'reason': '体格检查缺项或只有符号等无意义内容。\n'
                    }
                    return ans

        ans = {"item": 33, "是否扣分": "否", "vote": "无否决", "score": 0, "reason": ""}
        return ans

    def _item_50(self, emr, parm):
        """
        主要诊断与主诉等病史不符。
        基于知识图谱/诊断模型判断？
        """
        predict_ans = parm.get('predict_ans', [])
        # print(predict_ans)
        checklist = parm['Parameter1']

        def string_similar(s1, s2):
            return difflib.SequenceMatcher(None, s1, s2).quick_ratio()

        entity_type_list = ['疾病', '症状', '症状39', '疾病39', '症状a', '疾病a', '疾病icd10']
        entities_1_list = []
        entities_2_list = []

        entities_1 = self.match_model.find_entities(emr[checklist[1]])['pos']  # 主诉
        entities_2 = self.match_model.find_entities(emr[checklist[0]])['pos']  # 初步诊断
        # print(entities_1)
        for entity in entities_1:
            if entity[1] in entity_type_list:
                entities_1_list.append(entity[0])
        for entity in entities_2:
            if entity[1] in entity_type_list:
                entities_2_list.append(entity[0])
        # print(entities_1_list)
        # print(entities_2_list)

        # 实体间关系匹配
        if entities_2_list != []:
            entity_y = entities_2_list[0]
            for entity_x in entities_1_list:
                if string_similar(entity_x, entity_y) >= 0.5:
                    ans = {"item": 50,"是否扣分": "否", "vote": "", "score": 0, "reason": ""}
                    return ans

                else:
                    attr = self.graph_model.search_link_paths(entity_y, entity_x, 1)
                    if attr != []:

                        ans = {"item": 50, "是否扣分": "否", "vote": "", "score": 0, "reason": attr}
                        return ans

        # cfg = ChineseALLConfig(data_dir=self.diagnosis_path)
        # p = Predictor(cfg)
        # data = [
        #     {
        #         "emr_id": '123456',
        #         "doc": emr.get("主诉", "") + " " + \
        #                emr.get("现病史", "") + " " + \
        #                emr.get("既往史及其他病史", "") + " " + \
        #                emr.get("体格检查", "") + " " + \
        #                emr.get("辅助检查", ""),
        #         "label": []
        #     }
        # ]
        # raw_ans = p.predict(data)  # 调用一次执行一次
        predict_ans = []

        # for key in raw_ans[0]['predict_score']:
        #     predict_ans.append((key, raw_ans[0]['predict_score'][key]))
        #     predict_ans.sort(key=lambda x: x[1], reverse=True)
        # predict_ans = predict_ans[:5]

        if entities_2_list != []:
            for key in predict_ans:
                entity_y = entities_2_list[0]
                # print(key[0])
                if entity_y in key[0]:

                    ans = {"item": 50, "是否扣分": "否", "vote": "", "score": 0, "reason": ""}
                    return ans

        if entities_2_list == [] or '复诊' in emr['主诉']:
            ans = {"item": 50, "是否扣分": "否", "vote": "", "score": 0, "reason": ""}
        else:
            ans = {
                # 当前规则的编号
                "item": 50,
                "是否扣分": "是",
                "vote": "乙级病历",
                "score": -10,
                "reason": "主要诊断与主诉等病史不符。",
                "knowledge_graph": {
                    "主诉": entities_1_list,
                    "第一诊断": entities_2_list[0]
                }
            }
        return ans




    def _item_52(self, emr, parm):
        """
        手术记录创建时间超过手术结束时间24小时。
        有创操作记录创建时间超过操作时间24小时。
        """
        surgery_records = parm['Parameter0']
        checklist = parm['Parameter1']
        # 同上，正则匹配
        pattern = re.compile(r'([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})')
        pattern2 = re.compile(r'([0-9]{4})-([0-9]{2})-([0-9]{2})')

        # 未找到手术记录
        if not surgery_records or surgery_records == []:
            ans = {"item": 52, "是否扣分": "否", "vote": "无否决", "score": 0, "reason": ""}
            return ans

        for record in surgery_records:
            # res1 = pattern.search(record[checklist[0]])
            res1 = pattern.search(record[checklist[0]])  # 记录创建时间
            res2 = pattern2.search(record[checklist[1]])  # 手术结束时间

            # 匹配到对应日期进行判断
            if res1 and res2:
                # date1 = datetime.datetime(int(res1.group(1)), int(res1.group(2)), int(res1.group(3)), int(res1.group(4)),
                                  #         int(res1.group(5)), int(res1.group(6)))
                date1 = datetime.datetime(int(res1.group(1)), int(res1.group(2)), int(res1.group(3)), int(res1.group(4)),
                                          int(res1.group(5)), int(res1.group(6)))
                date2 = datetime.datetime(int(res2.group(1)), int(res2.group(2)), int(res2.group(3)))

                if (date1 - date2).days >= 1:
                    ans = {
                            "item": 52,
                            "是否扣分": "是",
                            "vote": "乙级病历",
                            "score": -15,
                            "reason": checklist[1]+"超过"+checklist[0]+"24小时。\n"
                    }
                    return ans

        ans = {"item": 52, "是否扣分": "否", "vote": "无否决", "score": 0, "reason": ""}

        return ans

    def _item_53(self, emr, parm):
        """
        无处理内容，同时无嘱托记录。乙级
        可基于之前的指控规则

        假设【1】是嘱托，【0】是处理
        """
        checklist = parm['Parameter0']
        treatment = emr[checklist[0]]
        entrust = emr[checklist[1]] # 嘱托

        if entrust != '': # 有嘱托
            ans = {"item": 53, "是否扣分": "否", "vote": "", "score": 0, "reason": ""}
            return ans
        elif '请录入处方'in treatment or treatment == '': # 无处理
            ans = {"item": 53, "是否扣分": "是", "vote": "乙级病历", "score": -15, "reason": "无处理内容，同时无嘱托记录。\n"}
            return ans

        else: # 处理无内容
            one_site = treatment.find("1  ")
            two_site = treatment.find("2  ")

            if one_site != -1 and two_site == -1:
                num = len(re.findall(r'\d', treatment)) - len(re.findall(r'\d\s\s', emr['处理']))
                if '诊查费' in treatment and treatment.count('诊查费') == treatment.count('*') and treatment.count('诊查费') == num:
                    ans = {"item": 53, "是否扣分": "是", "vote": "乙级病历", "score": -15, "reason": "无处理内容，同时无嘱托记录。\n"}
                    return ans
            elif one_site == -1:
                ans = {"item": 53, "是否扣分": "是", "vote": "乙级病历", "score": -15,
                       "reason": "无处理内容，同时无嘱托记录。\n"}
                return ans

        ans = {"item": 53, "是否扣分": "否", "vote": "", "score": 0, "reason": ""}
        return ans
