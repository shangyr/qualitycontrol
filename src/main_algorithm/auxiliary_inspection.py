# -*- coding:utf-8 -*-

"""
辅助检查
41.不合理复制病历记录，辅助检查复制主诉内容。(已删除符号，保留中文，字母，数字)
42.不合理复制病历记录，辅助检查复制体格检查内容。(已删除符号，保留中文，字母，数字)
43.本次就诊有开具检验检查，出具报告时间在当班，无记录结果。(s)
44.上一次在该科室就诊有开具检验检查，报告未在当班出具，未记录结果。（44 1.7.4  初步完成，模拟数据在test_algorithm里的更改怎么统一  实例待测试：）
45.主诉中提及体检发现辅助检查结果异常的，现病史和辅助检查未作异常结果的具体情况记录 （45 1.7.5  已完成，肝钙化、转氨酶升高）
46.辅助检查结果未写检查时间。 初步完成，待探讨（46 1.7.6  实例可通过：辅助检查：磁共振左肩关节、肩锁关节退行性变。左肱骨头骨髓轻度水肿。左肩袖损伤（冈上肌、肩胛下肌肌腱）。左侧喙突下滑囊少量积液。）
47.外院的检查结果未写明医疗机构名称 未完成（47 1.7.7  已完成，实例：外院超声提示前列腺炎、前列腺增生症）
48.多项检验检查未按时间顺序记录 未完成（48 1.7.8  有实例）

501:辅助检查只有符号、数字、字母。
"""

from ..extract_entity import BaseExtract
from ..knowledge_graph import BaseGraph
import re
import datetime as dt
import pandas as pd
import json
import os
import difflib

class AuxiliaryInspection:
    def __init__(self, match_model: BaseExtract, ner_model: BaseExtract, spacy_model: BaseExtract,
                 graph_model: BaseGraph) -> None:
        self.match_model = match_model
        self.ner_model = ner_model
        self.spacy_model = spacy_model
        self.graph_model = graph_model

    def check(self, emr):
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
            '_item_41',
            '_item_43',
            '_item_44',
            '_item_45',
            '_item_46',
            '_item_47',
            '_item_48',
            '_item_501'
        ]
        reason = ''
        score = 0
        for field in fields:
            field_ans = getattr(self, field)(emr)
            reason += field_ans['reason']
            score += field_ans['score']

        ans = {'score': score, 'reason': reason}
        return ans

    def _item_41(self, emr):
        """
            不合理复制病历记录，辅助检查复制主诉内容。（扣10分）
            抽取辅助检查和主诉中的中文汉字、英文字母和数字，判断是否完全一样
        """
        reason = ''
        score = 0

        pattern = re.compile(r'[\u4e00-\u9fffA-Za-z0-9]')
        text_auxiliary = pattern.findall(emr['辅助检查'])
        text_chief = pattern.findall(emr['主诉'])

        if text_auxiliary == text_chief:
            score -= 10
            reason += "不合理复制病历记录，辅助检查复制主诉内容，扣10分\n"

        ans = {'score': score, 'reason': reason}
        return ans


    def _item_43(self, emr):
        """
        本次就诊有开具检验检查，出具报告时间在当班，无记录结果。
        初诊病历
        """
        reason = ''
        score = 0

        report_time = '2024-05-09'  # 模拟出具报告时间
        # report_time = dt.strptime(report_time_string, '%Y%m%d%H%M')

        visit_time = emr['就诊时间']

        x1 = self.match_model.find_entities(emr['处理'])['pos']
        process_list = []


        for entity in x1:
            if entity[1] == '检查':
                process_list.append(entity[0])

        if process_list !=[]:
            # 判断检查检验结果是报告是否在当班出具，如果在当天出具，就检查当天病历
            if report_time == visit_time:
                # 提取辅助检查中的检查实体
                x2 = self.match_model.find_entities(emr['辅助检查'])['pos']
                auxiliary_list = []
                for entity in x2:
                    if entity[1] == '检查' or entity[1] == '检查结果' or entity[1] == '症状':
                        auxiliary_list.append(entity[0])

                # 将处理中的检查名在辅助检查中进行查找
                for entity_process in process_list :
                    bj = 0
                    for entity_auxiliary in auxiliary_list:
                        attr = self.graph_model.search_link_paths(entity_process, entity_auxiliary)
                        if attr != [] or entity_process == entity_auxiliary:
                            bj = 1
                    if bj == 0:
                        score = -5
                        reason = '本次就诊有开具检验检查，出具报告时间在当班，无记录结果。\n'
                        ans = {'score': score, 'reason': reason}
                        return ans

        ans = {'score': score, 'reason': reason}
        return ans

    def _item_44(self, emr):
        """
        上一次在该科室就诊有开具检验检查，报告未在当班出具，未记录结果。
        复诊病历
        """
        reason = ''
        score = 0
        past_record = emr.get('既往病历', '')

        # 未找到既往病历
        if not past_record or past_record[0]=={}:
            ans = {'score': 0, 'reason': ''}
            return ans

        emr_last = past_record[0] # 模拟初诊病历数据
        report_time = '2024-01-01'  # 模拟出具报告时间
        visit_time = emr['就诊时间']

        # 提取处理中的检查
        x1 = self.match_model.find_entities(emr_last['处理'])['pos']
        process_list = []
        for entity in x1:
            if entity[1] == '检查':
                process_list.append(entity[0])

        if process_list != []:
            # 判断检查检验结果是报告是否在当班出具，如果在当天出具，就检查当天病历
            if report_time == visit_time:
                # 提取辅助检查中的检查实体
                x2 = self.match_model.find_entities(emr['辅助检查'])['pos']
                auxiliary_list = []
                for entity in x2:
                    if entity[1] == '检查' or entity[1] == '检查结果':
                        auxiliary_list.append(entity[0])

                # 将处理中的检查名在辅助检查中进行查找
                for entity_process in process_list:
                    bj = 0
                    for entity_auxiliary in auxiliary_list:
                        attr = self.graph_model.search_link_paths(entity_process, entity_auxiliary)
                        if attr != [] or entity_process == entity_auxiliary:
                            bj = 1
                    if bj == 0:
                        score = -5
                        reason = '本次就诊有开具检验检查，出具报告时间在当班，无记录结果。\n'
                        ans = {'score': score, 'reason': reason}
                        return ans

        ans = {'score': score, 'reason': reason}
        return ans

    def _item_45(self, emr):
        """
        主诉中提及体检发现辅助检查结果异常的，现病史和辅助检查未作异常结果的具体情况记录
        """
        reason = ''
        score = 0

        def string_similar(s1, s2):
            return difflib.SequenceMatcher(None, s1, s2).quick_ratio()

        # 读取json文件并创立字典
        relative_path = r'C:\Users\huawei\Desktop\qualitycontrol\data_and_models\knowledge-graph\disease_classify.json'
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

        entities_2 = self.match_model.find_entities(emr['现病史'])['pos']
        his_auxiliary_physi_list = []
        his_auxiliary_list = []
        for entity in entities_2:
            his_auxiliary_physi_list.append(entity[0])
            his_auxiliary_list.append(entity[0])
        entities_3 = self.match_model.find_entities(emr['辅助检查'])['pos']
        for entity in entities_3:
            his_auxiliary_physi_list.append(entity[0])
            his_auxiliary_list.append(entity[0])
        entities_4 = self.match_model.find_entities(emr['体格检查'])['pos']
        for entity in entities_4:
            his_auxiliary_physi_list.append(entity[0])
        # print(his_auxiliary_list)

          
        entities_1 = self.match_model.find_entities(emr['主诉'])['pos']
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
                                attr = self.graph_model.search_link_paths(entity, entity_2,1)
                                if attr != []:
                                    deduction = 1
                                    break

                        if deduction == 0:
                            score = -10
                            reason = '未记录与本病相关的检验检查结果,扣10分\n'
                else:
                    deduction = 0
                    for entity_2 in his_auxiliary_physi_list:
                        if string_similar(entity, entity_2) >= 0.5:
                            deduction = 1
                            # print(entity_2)
                            # print(string_similar(entity, entity_2))
                            break
                        else:
                            attr = self.graph_model.search_link_paths(entity, entity_2,1)
                            if attr != []:
                                # print(x)
                                deduction = 1
                                break

                    if deduction == 0:
                        score = -10
                        reason = '未记录与本病相关的检验检查结果,扣10分\n'

        ans = {'score': score, 'reason': reason}
        return ans

    def _item_46(self, emr):
        """
        辅助检查结果未写检查时间。利用ner_model 识别是不是包含时间
        """
        if len(emr['辅助检查']) < 4 or '无' in emr['辅助检查'] or '暂缺' in emr['辅助检查'] or not (re.search(r"[\u4e00-\u9fff]",emr['辅助检查'])):
            ans = {'score': 0, 'reason': ''}
            return ans


        x1 = self.ner_model.find_entities(emr['辅助检查'])
        entities_list = []
        for entity in x1:
            if entity[1] == 'DATE':
                entities_list.append(entity[0])

        # 补充识别 XXXX-XX-XX XXXX/XX/XX  XXXX.XX.XX格式的时间信息
        pattern_sup = re.compile('(\d{4})([-,.,/,年])(\d{1,2})([-,.,/,月])(\d{1,2})')
        pattern_sup2 = re.compile('(\d{1,2})([-,.,/,月])(\d{1,2})')

        x2 = pattern_sup.search(emr['辅助检查'])
        x3 = pattern_sup2.search(emr['辅助检查'])

        if entities_list != [] or x2 != None or x3 != None:
            ans = {'score': 0, 'reason': ''}
        else:
            ans = {'score': -2, 'reason': '辅助检查结果记录有缺陷，检查中未检测到检查时间，扣2分。\n'}

        return ans


    def _item_47(self, emr):
        """
        外院的检查结果未写明医疗机构名称
        """
        reason = ''
        score = 0

        x = self.ner_model.find_entities(emr['辅助检查'])
        if '外院' in emr['辅助检查']:
            for entity in x:
                if entity[1] == 'ORG':
                    ans = {'score':0,'reason': ''}
                    return ans
            score = -2
            reason = '外院的检查结果未写明医疗机构名称'
        ans = {'score': score, 'reason': reason}

        return ans

    def _item_48(self, emr):
        """
        多项检验检查未按时间顺序记录
        """

        # 识别 XXXX-XX-XX XXXX/XX/XX  XXXX.XX.XX XXXX年XX月XX日格式的时间信息
        pattern_sup = re.compile('(\d{4})([-,.,/,年])(\d{1,2})([-,.,/,月])(\d{1,2})')

        date_result = pattern_sup.findall(emr['辅助检查'])
        last_date = []

        for date in date_result:
            if last_date != [] :
                if date[0] < last_date[0]:
                    ans = {'score': -2, 'reason': '多项检验检查未按时间顺序记录'}
                    return ans
                elif date[0] == last_date[0]:
                    if date[2] < last_date[2]:
                        ans = {'score': -2, 'reason': '多项检验检查未按时间顺序记录'}
                        return ans
                    elif date[2] == last_date[2] :
                        if date[4] < last_date[4]:
                            ans = {'score': -2, 'reason': '多项检验检查未按时间顺序记录'}
                            return ans
            last_date = date

        ans = {'score': 0, 'reason': ''}
        return ans

    def _item_501(self, emr):
        """
        辅助检查只有符号、数字、字母，扣2分。
        """
        score = 0
        reason = ''

        keyword = emr['辅助检查']
        if not (re.search(r"[\u4e00-\u9fff]", keyword)):  # 判断是否有中文 有就不全是字母、符号、数字等字符 (这里未将字母看成符号，单独判断了一下)
            if not(re.search(r"[A-Za-z]", keyword) and re.search(r"[0-9]", keyword)):
                score = -2
                reason = '辅助检查只有符号、数字、字母，扣2分。\n'

        ans = {'score': score, 'reason': reason}
        return ans
