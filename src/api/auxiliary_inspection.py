# -*- coding:utf-8 -*-

"""
辅助检查
41.不合理复制病历记录，辅助检查复制主诉内容。（41 1.7.1 未给出实例）
42.不合理复制病历记录，辅助检查复制体格检查内容。（42 1.7.2 实例：体格检查，辅助检查均为：全腹软，无压痛及反跳痛，双肾区无明显叩击痛，双输尿管行程无压痛，膀胱区无压痛。）
43.本次就诊有开具检验检查，出具报告时间在当班，无记录结果。（43 1.7.3  实例待测试：处理：2 X线胸部正位* *1 常规放射诊断组  辅助检查：“-”）
44.上一次在该科室就诊有开具检验检查，报告未在当班出具，未记录结果。（44 1.7.4  初步完成，模拟数据在test_algorithm里的更改怎么统一  实例待测试：）
45.主诉中提及体检发现辅助检查结果异常的，现病史和辅助检查未作异常结果的具体情况记录 （45 1.7.5  实例待测试：）
46.辅助检查结果未写检查时间。 初步完成，待探讨（46 1.7.6  实例可通过：辅助检查：磁共振左肩关节、肩锁关节退行性变。左肱骨头骨髓轻度水肿。左肩袖损伤（冈上肌、肩胛下肌肌腱）。左侧喙突下滑囊少量积液。）
47.外院的检查结果未写明医疗机构名称 未完成（47 1.7.7  已完成，实例：外院超声提示前列腺炎、前列腺增生症）
48.多项检验检查未按时间顺序记录 未完成（48 1.7.8  有实例）
"""
from ..extract_entity import BaseExtract
from ..knowledge_graph import BaseGraph
import re
import difflib
import json
import os
import pandas as pd
import datetime
import math
import random


class AuxiliaryInspection:
    def __init__(self, match_model: BaseExtract, ner_model: BaseExtract, spacy_model: BaseExtract,
                 graph_model: BaseGraph) -> None:
        self.match_model = match_model
        self.ner_model = ner_model
        self.spacy_model = spacy_model
        self.graph_model = graph_model

    def check(self, emr, item, parm):
        """
        Args:
            emr:{'姓名':'***', '性别':'*', '接诊时间':'*****',...}
        Return:
            { 'score' : int 该部分扣除的分数,
                'reason': str 扣分原因
            }
        """
        # print(self.match_model.find_entities(emr['职业']))
        # print(self.ner_model.find_entities(emr['职业'])) # 有高血压病史多年，多年无法识别
        # print(self.spacy_model.find_entities(emr['职业']))

        # print(self.graph_model.search_paths('麻风病',num_hop=2))
        # print(self.graph_model.search_link_paths('麻风病','四弯风',num_hop=2))
        fields = [
            '_item_42',
            '_item_43',
            '_item_44',
            '_item_45',
            '_item_46',
            '_item_47',
            '_item_48',
        ]
        if item not in fields:
            return {}
        ans = getattr(self, item)(emr, parm)
        return ans

    def _item_42(self, emr, parm):
        """
        辅助检查只有符号、数字、字母，扣2分。
        """
        keyword = parm['Parameter0'][0]
        if not (re.search(r"[\u4e00-\u9fff]", emr[keyword])):  # 判断是否有中文 有就不全是字母、符号、数字等字符 (这里未将字母看成符号，单独判断了一下)
            if not (re.search(r"[A-Za-z]", emr[keyword]) and re.search(r"[0-9]", emr[keyword])):
                ans = {"item": 42, "是否扣分": "是", "score": -2, "reason": "辅助检查只有符号、数字、字母，扣2分。\n"}
                return ans

        ans = {"item": 42, "是否扣分": "否", "score": 0, "reason": ""}
        return ans

    def _item_43(self, emr, parm):
        """
        本次就诊有开具检验检查，出具报告时间在当班，无记录结果。
        初诊病历
        """
        reason = ''
        score = 0

        checklist = parm['Parameter0']  # 其中0是就诊时间，1是处理，2是辅助检查
        # 检验报告和检查报告组成的元组列表
        inspect_records_list = [(parm['Parameter1'], parm['Parameter2']), (parm['Parameter3'], parm['Parameter4'])]

        visit_time = emr[checklist[0]]  # 就诊时间
        auxi_inspection = emr[checklist[1]] # 辅助检查

        for record_tuple in inspect_records_list:
            if record_tuple[0]:
                for record in record_tuple[0]:
                    inspect_checklist = record_tuple[1]
                    report_time = record[inspect_checklist[0]]  # 报告出具时间
                    pattern = re.compile(r'([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})')

                    res1 = pattern.search(visit_time)  # 就诊时间
                    res2 = pattern.search(report_time)  # 报告时间

                    if res1 and res2:
                        date1 = datetime.datetime(int(res1.group(1)), int(res1.group(2)), int(res1.group(3)),
                                                  int(res1.group(4)),
                                                  int(res1.group(5)), int(res1.group(6)))
                        date2 = datetime.datetime(int(res2.group(1)), int(res2.group(2)), int(res2.group(3)),
                                                  int(res2.group(4)),
                                                  int(res2.group(5)), int(res2.group(6)))

                        if (date2 - date1).days == 0:
                            # 判断病历出具在当天
                            if (8 <= date1.hour <= 12 and 8 <= date2.hour <= 12) or (
                                    14 <= date1.hour <= 17 and 14 <= date2.hour <= 17 and date2.hour <= 30):  # 判断报告出具时间在当班

                                x1 = self.match_model.find_entities(record[inspect_checklist[1]])['pos']
                                x2 = self.match_model.find_entities(auxi_inspection)['pos']

                                entity_type_list = ['疾病', '症状', '症状39', '疾病39', '症状a', '疾病a', '疾病icd10',
                                                    '检查', '检查结果']
                                process_list = []
                                auxiliary_list = []

                                for entity in x1:
                                    if entity[1] == '检查':
                                        process_list.append(entity[0])
                                for entity in x2:
                                    if entity[1] in entity_type_list:
                                        auxiliary_list.append(entity[0])

                                bj = 0

                                # 方法1：如果检查名就在辅助检查中，就算记录
                                if (record[inspect_checklist[1]]) in auxi_inspection:
                                    bj = 1

                                # 方法2：如果有相关实体就算记录
                                if process_list != []:
                                    for entity_process in process_list:
                                        for entity_auxiliary in auxiliary_list:
                                            attr = self.graph_model.search_link_paths(entity_process, entity_auxiliary)
                                            if attr != [] or entity_process == entity_auxiliary:
                                                bj = 1

                                # 方法3：如果有抽取的检查结果片段就算记录
                                inspect_result = record[inspect_checklist[2]]
                                length_i = len(inspect_result)
                                lim_len = min(math.ceil(length_i / 3), 5)  # 阈值：截取的随机片段的字符串长度
                                for k in range(length_i - lim_len):
                                    # start = random.randint(0, length_i - lim_len)
                                    sample_string = inspect_result[k:k + lim_len]
                                    if sample_string in auxi_inspection:
                                        bj = 1

                                if bj == 0:
                                    ans = {"item": 43, "是否扣分": "是", "score": -5,
                                                "reason": "本次就诊有开具检验/检查 [" + record[inspect_checklist[1]] + "] 出具报告时间在当班，无记录结果。\n"}
                                    return ans
                ans = {"item": 43, "是否扣分": "否", "score": 0, "reason": ""}
                return ans

        ans = {"item": 43, "是否扣分": "否", "score": 0, "reason": ""}
        return ans

    def _item_44(self, emr, parm):
        """
        上一次在该科室就诊有开具检验检查，报告未在当班出具，未记录结果。
        复诊病历
        """

        past_record = ['Parameter0']
        # 未找到既往病历
        if not past_record or past_record[0] == {}:
            ans = {'score': 0, 'reason': ''}
            return ans

        emr_last = past_record[0]  # 模拟初诊病历数据


        checklist_last = parm['Parameter1']  # 其中0是就诊时间，1是处理，2是辅助检查
        # 检验报告和检查报告组成的元组列表
        inspect_records_list = [(parm['Parameter2'], parm['Parameter3']), (parm['Parameter4'], parm['Parameter5'])]
        checklist = parm['Parameter6']
        visit_time = emr[checklist_last[0]]  # 就诊时间
        auxi_inspection = emr[checklist[0]]  # 辅助检查

        for record_tuple in inspect_records_list:
            if record_tuple[0]:
                for record in record_tuple[0]:
                    print(record)
                    inspect_checklist = record_tuple[1]
                    report_time = record[inspect_checklist[0]]  # 报告出具时间
                    pattern = re.compile(r'([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})')

                    res1 = pattern.search(visit_time)  # 就诊时间
                    res2 = pattern.search(report_time)  # 报告时间
                    print(res1,res2)
                    if res1 and res2:
                        date1 = datetime.datetime(int(res1.group(1)), int(res1.group(2)), int(res1.group(3)),
                                                  int(res1.group(4)),
                                                  int(res1.group(5)), int(res1.group(6)))
                        date2 = datetime.datetime(int(res2.group(1)), int(res2.group(2)), int(res2.group(3)),
                                                  int(res2.group(4)),
                                                  int(res2.group(5)), int(res2.group(6)))

                        if (date2 - date1).days > 0 or not ((8 <= date1.hour <= 12 and 8 <= date2.hour <= 12) or (
                                    14 <= date1.hour <= 17 and 14 <= date2.hour <= 17 and date2.hour <= 30)):  # 判断报告出具时间在当班
                            # 判断病历出具不在当天
                            x1 = self.match_model.find_entities(record[inspect_checklist[1]])['pos']
                            x2 = self.match_model.find_entities(auxi_inspection)['pos']

                            entity_type_list = ['疾病', '症状', '症状39', '疾病39', '症状a', '疾病a', '疾病icd10',
                                                    '检查', '检查结果']
                            process_list = []
                            auxiliary_list = []

                            for entity in x1:
                                if entity[1] == '检查':
                                    process_list.append(entity[0])
                            for entity in x2:
                                if entity[1] in entity_type_list:
                                    auxiliary_list.append(entity[0])

                            bj = 0

                            # 方法1：如果检查名就在辅助检查中，就算记录
                            if (record[inspect_checklist[1]]) in auxi_inspection:
                                bj = 1

                            # 方法2：如果有相关实体就算记录
                            if process_list != []:
                                for entity_process in process_list:
                                    for entity_auxiliary in auxiliary_list:
                                        attr = self.graph_model.search_link_paths(entity_process, entity_auxiliary)
                                        if attr != [] or entity_process == entity_auxiliary:
                                            bj = 1

                            # 方法3：如果有随机抽取的检查结果片段就算记录
                            inspect_result = record[inspect_checklist[2]]
                            length_i = len(inspect_result)
                            lim_len = min(math.ceil(length_i / 3), 5)  # 阈值：截取的随机片段的字符串长度
                            for k in range(length_i - lim_len):
                                # start = random.randint(0, length_i - lim_len)
                                sample_string = inspect_result[k:k + lim_len]
                                if sample_string in auxi_inspection:
                                    bj = 1

                            if bj == 0:
                                ans = {"item": 44, "是否扣分": "是", "score": -5,
                                        "reason": "上一次在该科室就诊有开具检验/检查 ["+record[inspect_checklist[1]]+"] 报告未在当班出具，未记录结果。\n"}
                                return ans

                ans = {"item": 44, "是否扣分": "否", "score": 0, "reason": ""}
                return ans

        ans = {"item": 44, "是否扣分": "否", "score": 0, "reason": ""}
        return ans

    def _item_45(self, emr, parm):
        """
        主诉中提及体检发现辅助检查结果异常的，现病史和辅助检查未作异常结果的具体情况记录
        """

        reason = ''
        score = 0
        cheif = emr[parm['Parameter0'][0]] # 主诉
        present = emr[parm['Parameter1'][0]] # 现病史
        auxiliary = emr[parm['Parameter1'][1]] # 辅助检查
        physical = emr[parm['Parameter1'][2]] # 体格检查
        def string_similar(s1, s2):
            return difflib.SequenceMatcher(None, s1, s2).quick_ratio()

        # 读取json文件并创立字典
        relative_path = 'E:\qualitycontrol_service2\data_and_models\knowledge-graph\disease_classify.json'
        current_dir = os.getcwd()
        json_adr = os.path.join(current_dir, relative_path)
        # df = pd.DataFrame(json_adr)
        # jslist = []
        # for i in range(len(df)):
        #         for j in range(len(df.columns)):
        #             element = df.at[i, df.columns[j]]
        #             my_dict = json.loads(element)
        #             jslist.append(my_dict)
        # my_dict = {}
        # for data_string in jslist:
        #     data_dict = json.loads(data_string)
        #     my_dict.update(data_dict)
        df = pd.read_json(json_adr)
        jslist = []
        for i in range(len(df)):
            for j in range(len(df.columns)):
                element = df.at[i, df.columns[j]]
                my_dict = json.loads(element)
                jslist.append(my_dict)
        my_dict = {}
        for data_string in jslist:
            data_dict = json.loads(data_string)
            my_dict.update(data_dict)

        entities_2 = self.match_model.find_entities(present)['pos']
        his_auxiliary_physi_list = []
        his_auxiliary_list = []
        for entity in entities_2:
            his_auxiliary_physi_list.append(entity[0])
            his_auxiliary_list.append(entity[0])
        entities_3 = self.match_model.find_entities(auxiliary)['pos']
        for entity in entities_3:
            his_auxiliary_physi_list.append(entity[0])
            his_auxiliary_list.append(entity[0])
        entities_4 = self.match_model.find_entities(physical)['pos']
        for entity in entities_4:
            his_auxiliary_physi_list.append(entity[0])
        # print(his_auxiliary_list)

        entities_1 = self.match_model.find_entities(cheif)['pos']
        # print(entities_1)
        chief_list = []
        for entity in entities_1:
            if entity[1] == '疾病' or '症状':
                chief_list.append(entity[0])
        if "体检" in chief_list:
            chief_list.remove("体检")
            # print(chief_list)
            for entity in chief_list:
                if entity in my_dict.keys():
                    his_auxiliary = my_dict[entity]
                    # print(his_auxiliary)
                    if his_auxiliary == '辅助检查':
                        # print("辅助检查实体为" + entity)
                        deduction = 0
                        for entity_2 in his_auxiliary_list:
                            if string_similar(entity, entity_2) >= 0.5:
                                # print(string_similar(entity, entity_2))
                                deduction = 1
                                break
                            else:
                                attr = self.graph_model.search_link_paths(entity, entity_2, 1)
                                if attr != []:
                                    deduction = 1
                                    break

                        if deduction == 0:
                            ans = {"item": 45, "是否扣分": "是", "score": -10, "reason":'未记录与本病相关的检验检查结果,扣10分\n'}
                            return ans
                else:
                    deduction = 0
                    for entity_2 in his_auxiliary_physi_list:
                        if string_similar(entity, entity_2) >= 0.5:
                            deduction = 1
                            # print(entity_2)
                            # print(string_similar(entity, entity_2))
                            break
                        else:
                            attr = self.graph_model.search_link_paths(entity, entity_2, 1)
                            if attr != []:
                                # print(x)
                                deduction = 1
                                break

                    if deduction == 0:
                        ans = {"item": 45, "是否扣分": "是", "score": -10, "reason":'未记录与本病相关的检验检查结果,扣10分\n'}
                        return ans

        ans = {"item": 45, "是否扣分": "否", "score": 0, "reason": ""}
        return ans

    def _item_46(self, emr, parm):
        """
        辅助检查结果未写检查时间。利用ner_model 识别是不是包含时间
        """

        keyword = parm['Parameter0'][0]

        if len(emr[keyword]) <= 4 or  emr[keyword] == '无' or emr[keyword] == '暂无' or '未见异常' in emr[keyword] or '暂缺' in emr[keyword] or not (
                re.search(r"[\u4e00-\u9fff]", emr[keyword])):
            ans = {"item": 46, "是否扣分": "否", "score": 0, "reason": ""}
            return ans

        x1 = self.ner_model.find_entities(emr[keyword])
        entities_list = []
        for entity in x1:
            if entity[1] == 'DATE':
                entities_list.append(entity[0])

        # 补充识别 XXXX-XX-XX XXXX/XX/XX  XXXX.XX.XX格式的时间信息
        pattern_sup = re.compile('(\d{4})([-,.,/,年])(\d{1,2})([-,.,/,月])(\d{1,2})')
        pattern_sup2 = re.compile('(\d{1,2})([月])(\d{1,2})')

        x2 = pattern_sup.search(emr[keyword])
        x3 = pattern_sup2.search(emr[keyword])
        print(entities_list)
        if entities_list != [] or x2 != None or x3 != None:
            ans = {"item": 46, "是否扣分": "否", "score": 0, "reason": ""}

        else:
            ans = {"item": 46, "是否扣分": "是", "score": -2,
                   "reason": '辅助检查结果记录有缺陷，检查中未检测到检查时间，扣2分。\n'}
        return ans

    def _item_47(self, emr, parm):
        """
        外院的检查结果未写明医疗机构名称
        """
        keyword = parm['Parameter0'][0]  # 辅助检查

        x = self.ner_model.find_entities(emr[keyword])
        if '外院' in emr[keyword]:
            for entity in x:
                if entity[1] == 'ORG':
                    ans = {"item": 47, "是否扣分": "否", "score": 0, "reason": ""}
                    return ans

            ans = {"item": 47, "是否扣分": "是", "score": -2, "reason": '外院的检查结果未写明医疗机构名称,扣2分。\n'}
            return ans

        ans = {"item": 47, "是否扣分": "否", "score": 0, "reason": ""}
        return ans

    def _item_48(self, emr, parm):
        """
        多项检验检查未按时间顺序记录

        一共能够识别出XXXX-XX-XX XXXX/XX/XX  XXXX.XX.XX XXXX年XX月XX日
        其中按照
        """

        # 识别 XXXX-XX-XX XXXX/XX/XX  XXXX.XX.XX XXXX年XX月XX日格式的时间信息
        keyword = parm['Parameter0'][0]
        pattern_sup = re.compile('(\d{4})([-,.,/,年])(\d{1,2})([-,.,/,月])(\d{1,2})')

        date_result = pattern_sup.findall(emr[keyword])

        last_date = []

        for date in date_result:
            if last_date != []:
                if date[0] < last_date[0]:
                    ans = {"item": 48, "是否扣分": "是", "score": -2, "reason": "多项检验检查未按时间顺序记录"}
                    return ans
                elif date[0] == last_date[0]:
                    if date[2] < last_date[2]:
                        ans = {"item": 48, "是否扣分": "是", "score": -2, "reason": "多项检验检查未按时间顺序记录"}
                        return ans
                    elif date[2] == last_date[2] and len(date) > 5:
                        if date[4] < last_date[4]:
                            ans = {"item": 48, "是否扣分": "是", "score": -2, "reason": "多项检验检查未按时间顺序记录"}
                            return ans

            last_date = date


        ans = {"item": 48, "是否扣分": "否", "score": 0, "reason": ""}
        return ans


