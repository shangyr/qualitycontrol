# -*- coding:utf-8 -*-

"""
# 关于手术记录、有创操作记录和诊断证明模块
53.手术时间、术前诊断、术中诊断、手术名称、手术医生、麻醉方式、手术经过等缺项。
54.手术记录，手术记录与病历记录的一致性检测。
56.操作名称、操作时间、操作步骤、操作者等缺项。
57.有创操作记录与病历记录的一致性检测。
58.急性病、病情平稳、无处理内容开具7天及以上休息时间。

"""
from ..extract_entity import BaseExtract
from ..knowledge_graph import BaseGraph
import re
import datetime

class MedicalRecord:
    def __init__(self, match_model: BaseExtract, ner_model: BaseExtract, spacy_model: BaseExtract,
                 graph_model: BaseGraph, acute_illness_path, body_path) -> None:
        self.match_model = match_model
        self.ner_model = ner_model
        self.spacy_model = spacy_model
        self.graph_model = graph_model
        self.illness_path=acute_illness_path
        self.body_path = body_path

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
            '_item_53',
            
            '_item_57',
            '_item_56',
            '_item_58',
        ]
        reason = ''
        score = 0
        for field in fields:
            field_ans = getattr(self, field)(emr)
            reason += field_ans['reason']
            score += field_ans['score']

        ans = {'score': score, 'reason': reason}
        return ans

    def _item_53(self, emr):
        """
        手术时间、术前诊断、术中诊断、手术名称、手术医生、麻醉方式、手术经过等缺项。
        """
        med_record = emr.get('手术记录', '')
        score = 0
        reason = ''
        # 无数据或者为空
        if not med_record:
            ans = {'score': 0, 'reason': ''}
            return ans
        item = ["手术开始时间", "手术结束时间", "术前诊断", "术中诊断", "手术名称", "手术医生", "麻醉方式", "手术经过"]
        for item_ in item:
            res = med_record.get(item_, None)
            if res == '':
                score -= 2
                reason += '手术记录:{}缺项，扣2分。\n'.format(item_)
        ans = {'score': score, 'reason': reason}
        return ans

    def _item_54(self, emr):
        """
        手术记录，手术记录与病历记录的一致性检测。
        """
        reason = ''
        score = 0
        match_pair = [["术前诊断", "初步诊断"], ['术中诊断', '初步诊断']]
        # 获取手术记录
        surgery_record = emr.get('手术记录', {})

        # 遍历
        for pair in match_pair:
            # 获取每个模块的内容，若不存在则返回空字符串
            if pair[0] in surgery_record:
                module1_content = surgery_record.get(pair[0], '')
            else:
                module1_content = emr.get(pair[0], '')
            module2_content = emr.get(pair[1], '')

            i = open(self.body_path, 'r', encoding='utf-8')
            body_list = []
            for line in i:
                body_list.append(line.strip())

            # 提取人体部位
            body1 = [part for part in body_list if part in module1_content]
            body2 = [part for part in body_list if part in module2_content]

            # 去重
            body1 = sorted(body1, key=len, reverse=True)
            bodyy1 = []
            for part in body1:
                if not any(part in longer for longer in bodyy1):
                    bodyy1.append(part)

            body2 = sorted(body2, key=len, reverse=True)
            bodyy2 = []
            for part in body2:
                if not any(part in longer for longer in bodyy2):
                    bodyy2.append(part)

            # 打印或返回提取到的部位名称
            #print(f"从 {pair[0]} 提取的人体部位: {bodyy1}")
            #print(f"从 {pair[1]} 提取的人体部位: {bodyy2}")

            if len(module1_content) == 0 or len(module2_content) == 0:
                ans = {'score': score, 'reason': reason}
                # 若未匹配出实体，直接判断不一致，后续可进行修改
                return ans

            flag=0
            # 进行遍历，查找各个实体之间是否匹配
            for entity1 in bodyy1:
                for entity2 in bodyy2:
                    # 存在路径 跳出循环
                    res=0
                    if entity1 in entity2:
                        res = 1

                    if res == 1:
                        # 一致性检验
                        # reason +='一致性检查：{}-{}间存在一致性；路径为{}\n'.format(pair[0], pair[1], str(res[:2]))
                        flag = 1
                        break
                # 跳出循环
                if flag:
                    break
            # 两模块之间存在路径，证明存在一致性，进行下一pair的比较
            if flag:
                continue
            # 不存在路径,进行扣分
            score -= 2
            temp1 = ''
            temp1 += pair[0] + '模块实体如下:' + str([item for item in bodyy1])
            temp1 += pair[1] + '模块实体如下:' + str([item for item in bodyy2])
            reason += '一致性检查：{}-{}间不存在一致性，扣2分；原因:身体部位不匹配,{}\n'.format(pair[0], pair[1], temp1)
            break
        ans = {'score': score, 'reason': reason}
        return ans

    def _item_56(self, emr):
        """
        操作名称、操作时间、操作步骤、操作者等缺项。
        """
        med_record = emr.get('有创操作记录', '')
        score = 0
        reason = ''
        # 无数据或者为空
        if not med_record:
            ans = {'score': 0, 'reason': ''}
            return ans
        item = ["操作时间", "操作医师", "操作名称", "操作步骤"]
        for item_ in item:
            res = med_record.get(item_, None)
            if res == '':
                score -= 2
                reason += '有创操作记录:{}缺项，扣2分。\n'.format(item_)
        ans = {'score': score, 'reason': reason}
        return ans

    def _item_57(self, emr):
        """
        有创操作记录与病历记录的一致性检测。
        """
        reason = ''
        score = 0
        match_pair = ["操作名称", "初步诊断"]
        # 获取有创操作记录
        surgery_record = emr.get('有创操作记录', {})

        # 获取每个模块的内容，若不存在则返回空字符串
        if match_pair[0] in surgery_record:
            module1_content = surgery_record.get(match_pair[0], '')
        else:
            module1_content = emr.get(match_pair[0], '')
        module2_content = emr.get(match_pair[1], '')

        i = open(self.body_path, 'r', encoding='utf-8')
        body_list = []
        for line in i:
            body_list.append(line.strip())

        # 提取人体部位
        body1 = [part for part in body_list if part in module1_content]
        body2 = [part for part in body_list if part in module2_content]

        # 去重
        body1 = sorted(body1, key=len, reverse=True)
        bodyy1 = []
        for part in body1:
            if not any(part in longer for longer in bodyy1):
                bodyy1.append(part)

        body2 = sorted(body2, key=len, reverse=True)
        bodyy2 = []
        for part in body2:
            if not any(part in longer for longer in bodyy2):
                bodyy2.append(part)

        # 打印或返回提取到的部位名称
        #print(bodyy1)
        #print(bodyy2)

        # 若未匹配出实体，直接判断未不一致
        if len(module1_content) == 0 or len(module2_content) == 0:
            ans = {'score': score, 'reason': reason}
            return ans
        else:
            # 进行遍历，查找各个实体之间是否有路径
            flag = False
            for entity1 in bodyy1:
                for entity2 in bodyy2:
                    res = 0
                    if entity1 in entity2:
                        res = 1

                    # 存在路径，标记为True
                    if res == 1:
                        flag = True
                        break
                if flag:
                    break

            if not flag:
                score -= 2
                temp1 = ''
                temp1 += match_pair[0] + '模块实体如下:' + str([item for item in bodyy1]) + '\n'
                temp1 += match_pair[1] + '模块实体如下:' + str([item for item in bodyy2])
                reason += '一致性检查：{}-{}间不存在一致性，扣2分；原因:身体部位不匹配,{}\n'.format(match_pair[0], match_pair[1], temp1)
        ans = {'score': score, 'reason': reason}
        return ans

    def _item_58(self, emr):
        """
        急性病、病情平稳、无处理内容开具7天及以上休息时间。
        """
        reason = ''
        score = 0
        diagnosis = emr.get('初步诊断', '')
        stable_list = ['病情稳定', '状况平稳', '病情平稳','病情无恶化', '病情控制良好', '健康状况稳定', '病情未波动','病情保持原状', '病情未加剧', '病情无异常', '病情缓和', '疾病稳定期', '病情不活动', '症状稳定','病情未恶化', '病情未进展', '病情受控', '病情缓解期', '病情平稳期', '病情无变化']
        i=open(self.illness_path,'r',encoding='utf-8')
        diagnosis_list = []
        for line in i:
            diagnosis_list.append(line.strip())
        # 提取疾病实体
        matched_entities = self.match_model.find_entities(emr['初步诊断'])['pos']
        dis = [item[0] for item in matched_entities if '疾病' in item[1]]
        # 提取诊断字段
        chinese_pattern = re.compile(r'[\u4e00-\u9fa5]+')
        chinese_chars = chinese_pattern.findall(diagnosis)
        # 判断是否是急性病
        has_acute_disease = any(disease in dis for disease in diagnosis_list)
        # 判断处理项是否有中文内容
        keyword = emr['处理']
        if_keyword = re.search(r"[\u4e00-\u9fa5]", keyword)
        # 判断病情是否有类似于病情平稳字样
        is_stable = any(term in chinese_chars for term in stable_list)

        days_int1 = 15  # 替换为数据接口
        # 检查是否满足扣分条件
        if has_acute_disease and days_int1 >= 7:
            score -= 5  # 扣分
            reason += "急性病开具了7天及以上休息时间。\n"
        if is_stable and days_int1 >= 7:
            score -= 5  # 扣分
            reason += "病情平稳开具了7天及以上休息时间。\n"
        if (not if_keyword) and days_int1 >= 7:
            score -= 5  # 扣分
            reason += "无处理内容开具了7天及以上休息时间。"
        ans = {'score': score, 'reason': reason}
        return ans