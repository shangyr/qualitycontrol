# -*- coding:utf-8 -*-

"""
现病史 和 现病史、既往史及其他病史
18.现病史复制主诉内容。
19.现病史存在连续字符或语段（3个字及以上）重复。
20.起病情况：起病时间与主诉矛盾
21.现病史内容与既往3个月内主要诊断相同的初诊现病史记录重复率为100%。
22.现病史要素描述不全，如缺少病情变化、治疗反应等。缺少治疗后好转/效果不佳/病情反复/病情加重/病情稳定继续治疗等描述。
24.现病史描述有缺陷。
25.疾病史：次要诊断的疾病情况（开具了处方）未记录
28.疾病史：次要诊断的疾病情况（未开具处方）未记录
30.手术史：因“术后”相关情况就诊的未在现病史或既往史中描述
"""
from ..extract_entity import BaseExtract
from ..knowledge_graph import BaseGraph
from fuzzywuzzy import fuzz
import re
import jieba
import difflib


class PresentIllness:
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
            '_item_18',
            '_item_19',
            '_item_20',
            '_item_22',
            '_item_30',
        ]
        if item not in fields:
            return {}

        ans = getattr(self, item)(emr, parm)
        return ans

    def _item_18(self, emr,parm):
        """
        18.现病史复制主诉内容。
        """
        reason = ''
        score = 0

        item1 = parm.get("Parameter0", [''])[0]

        symbol = re.compile(r'[\u4e00-\u9fffA-Za-z0-9]')
        text = emr.get(item1, '')
        text = symbol.findall(text)

        for item in parm.get("Parameter1", []):
            text_pre = symbol.findall(emr.get(item, ''))

            if text_pre == text and text_pre != '':
                score -= 15
                reason += f"{item}复制{item1}内容。\n"

        if score < 0:
            return {"item": 18, "是否扣分": "是", "score": score, "reason": reason, "knowledge_graph": {}}
        else:
            return {"item": 18, "是否扣分": "否", "score": 0, "reason": "", "knowledge_graph": {}}

    def _item_19(self, emr, parm):
        """
        现病史存在连续字符或语段（3个字及以上）重复。
        """
        reason = ''
        score = 0
        for item in parm.get('Parameter0',[]):
            removed_text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', emr.get(item, ''))

            total_length = len(removed_text)
            repeated_substrings = set()

            for i in range(total_length - 1):
                # 检查从当前位置开始的所有可能的子串
                for length in range(3, total_length - i + 1):  # 子串长度至少为3
                    substring = removed_text[i:i + length]
                    # print(substring)
                    # 检查这个子串是否直接在后面连续重复出现
                    if substring == removed_text[i + length:i + 2 * length]:
                        repeated_substrings.add(substring)
                        break  # 找到重复子串后，跳出内层循环

            a = list(repeated_substrings)
            print(a)
            if len(a) != 0:
                score -= 15
                reason += f'{item}存在连续字符或语段（3个字及以上）重复，重复语段为{a}。\n'

        if score < 0:
            return {"item": 19, "是否扣分": "是", "score": score, "reason": reason, "knowledge_graph": {}}
        else:
            return {"item": 19, "是否扣分": "否", "score": score, "reason": reason, "knowledge_graph": {}}

    def _item_20(self, emr, parm):
        """
        起病情况：起病时间与主诉矛盾
        """
        score = 0
        reason = ''
        kg = {}

        item1 = parm.get('Parameter0', [''])[0]

        text = emr.get(item1, '')

        for item in parm.get('Parameter1', []):
            present = emr.get(item, '')
            type_list = ['疾病', '症状']
            # 提取主诉中的症状和疾病实体
            entities = [item for item in self.match_model.find_entities(text)['pos'] if item[1] in type_list]
            if len(entities):
                if entities[0][1] not in kg:
                    kg[entities[0][1]] = []
                kg[entities[0][1]].append(entities[0][0])
            for entities_ in entities:
                if entities_ and entities_[0]:
                    pattern = r"(.*?){}".format(entities_[0])
                    # 匹配现病史中该症状所在的句子
                    r1 = re.search(pattern, present)
                    r2 = re.search(pattern, text)
                    if r1 and r2:
                        # print(r.group(0),r2.group(0))
                        time1 = re.findall(r"(\d+)\s*(周|天|年|月|星期|小时|时|分|秒|日)", r1.group(0))
                        # time2 = re.findall(r"(\d+)\s*(周|天|年|月|星期|小时|时|分|秒|日)", emr['主诉']) # ?
                        time2 = re.findall(r"(\d+)\s*(周|天|年|月|星期|小时|时|分|秒|日)", r2.group(0))
                        if time1 and time2:
                            judge = []
                            for time in time2:
                                for time2_ in time1:
                                    if time2_ == time:
                                        judge.append(1)
                                        break
                            if len(judge) == 0:
                                score -= 2
                                reason += f'{item}起病时间与{item1}矛盾。\n'
                                break
        if score < 0:
            return {"item": 20, "是否扣分": "是", "score": score, "reason": reason, "knowledge_graph": kg}
        else:
            return {"item": 20, "是否扣分": "否", "score": 0, "reason": "", "knowledge_graph": kg}


    def _item_22(self, emr, parm):
        """
        现病史要素描述不全，如缺少病情变化、治疗反应等。缺少治疗后好转/效果不佳/病情反复/病情加重/病情稳定继续治疗等描述。
        """
        reason = ''
        score = 0
        for item in parm.get('Parameter0', []):

            symbol = 0
            present = emr.get(item, '')
            word_list = ['好转', '效果不佳', '目前', '较前', '病情反复', '病情加重', '仍', '病情稳定', '继续治疗',
                         '无',
                         '无不适', '增加', '减少', '减轻', '增多', '症状反复', '缓解']

            for word in word_list:
                if word in present:
                    # print('文本中有列表里的词：',word)
                    symbol += 1
                    break
                else:
                    for word2 in jieba.lcut(present):
                        similarity = difflib.SequenceMatcher(None, word, word2).quick_ratio()
                        if similarity > 0.6:
                            # print('对于列表里面的词{}找到的相似度大于0.6的词{}'.format(word,word2))
                            symbol += 1
                            break  # 获取两个词语的词向量
            entity = self.match_model.find_entities(present)['pos']
            entity = [item[0] for item in entity]

            for keyword in entity:
                match = re.search(keyword + r"\u4e00-\u9fff", present)
                if match:
                    words = jieba.lcut(match.group(0))
                    # print(words)
                    for word in words:
                        if self.spacy_model.find_entities(word)[0][1] == 'adj' or 'verb':
                            # print('找到是形容词或动词的症状',match.group(0))
                            symbol += 1
                            break
                    cixing = self.spacy_model.find_entities(match.group(0))[0][1]
                    # print('找到是形容词或动词的症状描述',match.group(0))
                    if cixing:
                        # print('cixing',cixing)
                        if cixing == 'VERB' and 'ADJ':
                            symbol += 1
                            break
                    else:
                        # 获取紧随其后的词
                        next_word = match.group(1)
                        if next_word:

                            # 检查这个词是否是形容词或动词，是就不扣分
                            if self.spacy_model.find_entities(next_word)[0][1] == 'VERB' and 'ADJ':
                                # print('找到是形容词或动词的症状描述',next_word)
                                symbol += 1
                                break
            if symbol == 0:
                score -= 5
                reason += f'{item}未描述病情变化、治疗反应等。\n'

                
        if score < 0:
            return {"item": 22, "是否扣分": "是", "score": score, "reason": reason, "knowledge_graph": {}}
        else:
            return {"item": 22, "是否扣分": "否", "score": 0, "reason": "", "knowledge_graph": {}}


    def _item_30(self, emr, parm):
        """
        手术史：因“术后”相关情况就诊的未在现病史或既往史中描述
        """
        reason = ''
        score = 0
        keyword = '术后'

        item = parm.get('Parameter0', [''])[0]
        text = emr.get(item, '')

        if not text:
            return {"item": 30, "是否扣分": "否", "score": 0, "reason": "", "knowledge_graph": {}}

        items = parm.get('Parameter1', [])
        history_emr = ''
        for item in items:
            history_emr += emr.get(item, '')

        pattern1 = r"(.*?术)"
        pattern2 = r".*?行(.*?术)"
        pattern3 = r"[A-Za-z-]+手?术"
        # pattern4 = r"[A-Z]+"

        # 匹配
        match_chief = re.search(pattern1, text)
        match_history = re.search(pattern1, history_emr)

        surgery_verb = False

        # 现病史和既往史中含有“术后”
        if match_history:

            # 第一种情况：主诉中手术为英文，不判断是否含有动词
            match_chief_english = re.findall(pattern3, text)
            if match_chief_english:
                surgery_verb = True
                # print(match_chief_english)

            # 第二种情况：（因症状）行手术，以“行”作为分隔依据
            if '行' in match_history.group(1):
                match_history = re.search(pattern2, history_emr)

            # 多数情况：现病史及既往史中为中文手术名称
            match_surgery = re.findall(pattern3, match_history.group(1))
            if not match_surgery:
                surgery = self.match_model.find_entities(match_history.group(1))

                if surgery['pos']:
                    # 识别手术名称（识别医疗实体，将该实体以后的字符作为手术名称）
                    position = surgery['pos'][0][2]
                    surgery = match_history.group(1)[position:]
                    # print(surgery)

                    entities = self.spacy_model.find_entities(surgery)
                    # print(entities)
                    # 手术名称中含有动词
                    for entity in entities:
                        if entity[1] == 'VERB':
                            surgery_verb = True
                            break

            # 第四种情况：现病史及既往史中的手术名称为英文，不检测是否含动词
            else:
                surgery_verb = True
                # print('英文手术（含“术”）：')
                # print(match3)


        pattern_time01 = r"(\d+个?)\s*(周|天|年|月|星期|小时|时|分|秒|日)"
        pattern_time02 = r"([零一二三四五六七八九十]+个?)\s*(周|天|年|月|星期|小时|时|分|秒|日)"
        pattern_time03 = re.compile("(\d{4})?[-./年]?(\d{1,2})[-./月]?(\d{1,2})?")

        match_time01 = re.findall(pattern_time01, history_emr)
        match_time02 = re.findall(pattern_time02, history_emr)
        match_time03 = pattern_time03.search(history_emr)
        # 年、月、日三个时间要素中至少包含两个，（可以匹配年-月，月-日，年-月-日三种格式）
        count = 0
        if match_time03:
            for index in range(4):
                if match_time03.group(index):
                    count += 1
        # print(match_time03)

        if match_chief:
            if surgery_verb:
                if not (match_time01 or match_time02 or (count >= 3)):
                    score -= 2
                    reason += f"因“术后”相关情况就诊的未在{','.join(items)}中描述时间。\n"
            else:
                score -= 2
                reason += f"因“术后”相关情况就诊的未在{','.join(items)}描述手术名称。\n"

        if score < 0:
            return{"item": 30, "是否扣分": "是", "score": score, "reason": reason, "knowledge_graph": {}}
        else:
            return{"item": 30, "是否扣分": "否", "score":0, "reason":"", "knowledge_graph": {}}
        

 