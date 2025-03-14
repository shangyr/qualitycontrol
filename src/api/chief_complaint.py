# -*- coding:utf-8 -*-

"""
主诉（10分）
9.主诉要素描述不全，缺少症状。 (9 1.3.2 未给出实例)
10.主诉要素描述不全，缺少时间。(1.3.3 实例：主诉：发现颈部肿物)
11.主诉超过20个字。(1.3.4 实例：主诉：患者减肥中，使用司美格鲁肽药物中，自诉有腰痛，担心肾脏问题。)
12.主诉的时间描述不准确，如“数天/数月/数年”。(1.3.5 实例：主诉：反复心悸不适数月)
13.主诉内容与既往3个月内主要诊断相同的初诊主诉重复率为100%。(1.3.6 未给出实例)
14.主诉时间与既往3个月内病历记录不一致。
15.主诉记录有错别字。
16.主诉描述有缺陷。

"""
import re
import datetime

from ..extract_entity import BaseExtract
from ..knowledge_graph import BaseGraph

class ChiefComplaint:
    def __init__(self, match_model:BaseExtract, ner_model:BaseExtract, spacy_model:BaseExtract, graph_model:BaseGraph, corrector_model) -> None:
        self.match_model = match_model
        self.ner_model = ner_model
        self.spacy_model = spacy_model
        self.graph_model = graph_model
        self.corrector = corrector_model

    def check(self, emr, item, parm):
        """
        Args:
            emr:{'姓名':'***', '性别':'*', '接诊时间':'*****',...}
        Return:
            { 'score' : int 该部分扣除的分数,
                'reason': str 扣分原因
            }
        """
        fields = [
            '_item_9',
            '_item_10',
            '_item_14',
            '_item_15',
        ]

        if item not in fields:
            return {}

        ans = getattr(self, item)(emr, parm)
        return ans

    def _item_9(self, emr, parm):
        """
        初诊：9.主诉要素描述不全，缺少症状。
        初诊：10.主诉要素描述不全，缺少时间。
        初诊：12.主诉的时间描述不准确，如“数天/数月/数年”。
        主诉要素描述不全，如缺少症状、时间。
        要注意复诊直接写复诊就可以
        判断是否在同一个句子里面出现时间和症状

        例如查找同一句有没有 年/月/日/天/时 等，然后判断前面是不是可以转化为数字

        输入输出格式同check函数
        """

        # 初值
        score = 0
        reason = ''
        kg = {}
        # 检查项目
        for item in parm.get("Parameter0",[]):
            content = emr.get(item, '')
            if "要求检查" in emr['主诉'] or "孕检" in emr['主诉']:
                continue

            entities = [item for item in self.match_model.find_entities(content)['pos'] if
                        (item[1]=='疾病') or (item[1]=='症状')]

            # 未匹配到症状
            if len(entities) == 0:
                score -= 5
                reason += f'{item}要素描述不全，缺少症状。\n'
            # 抽取到的症状/疾病
            if len(entities):
                if entities[0][1] not in kg:
                    kg[entities[0][1]] = []
                kg[entities[0][1]].append(entities[0][0])

            # 匹配规则，症状/疾病 + .. + 约数 + 时间
            # pattern = re.compile(r'{}\S*?(数|多)(时|天|日|周|星期|月|年)'.format(entities[0][0]))
            pattern = re.compile(r'(数|多)(时|天|日|周|星期|月|年)')
            r = pattern.search(content)
            # 匹配到持续时间，跳出循环
            if r:
                score -= 2
                reason += f'{item}要素描述有缺陷，未精确描述症状持续时间。\n'
            else:
                pattern = re.compile(
                    r'[0-9零半一二两俩三四五六七八九]+[个|余]?(时|天|日|周|星期|月|年)',
                    flags=re.IGNORECASE)
                r = pattern.search(content)
                # 未匹配到准确时间
                if not r:
                    score -= 5
                    reason += f'{item}要素描述不全，缺少持续时间。\n'
            if score < 0:
                return {"item": 9, "是否扣分": "是", "score": score, "reason": reason, "knowledge_graph": {}}
            else:
                return {"item": 9, "是否扣分": "否", "score": 0, "reason": "", "knowledge_graph": kg}



    def _item_10(self, emr, parm):
        """
        主诉长度超过20个字
        """
        score = 0
        reason = ''


        length = parm.get('Parameter0', 20)
        for item in parm.get("Parameter1",[]):
            if len(emr[item]) > length:
                score -= 5
                reason += f"{item}长度超过{length}个字。\n"

        if score < 0:
            return {"item": 10, "是否扣分": "是", "score": score, "reason": reason, "knowledge_graph": {}}
        else:
            return {"item": 10, "是否扣分": "否", "score": 0, "reason": "", "knowledge_graph": {}}

        return ans

    def _item_14(self, emr, parm):
        """
        主诉时间与既往3个月内病历记录不一致。
        """
        # 初值
        score = 0
        reason = ''


        # 既往三个月没有病历记录
        past_emrs = parm.get('anamnesis', [])
        if not past_emrs:
            return {"item": 14, "是否扣分": "否", "score": 0, "reason": "", "knowledge_graph": {}}

        date_item = parm.get('Parameter0', '就诊时间')

        for item in parm.get("Parameter1",[]):
            entities = self.match_model.find_entities(emr[item])['pos']
            entities = [item for item in entities if '疾病' in item[1] or '症状' in item[1]]
            if len(entities) == 0:
                continue
            for past_emr in past_emrs:
                entities_ = self.match_model.find_entities(past_emr["主诉"])['pos']
                entities_ = [item for item in entities_ if '疾病' in item[1] or '症状' in item[1]]
                # 求出交集
                if len(set(entities).intersection(set(entities_))) > 0:
                    # 匹配规则，匹配主诉中出现的数字， + (天|日|周|星期|月) 疾病时间
                    pattern = re.compile(r'([0-9]{1,2})(天|日|周|星期|月)')
                    r1 = pattern.search(emr[item])
                    r2 = pattern.search(past_emr[item])
                    if r1 and r2:
                        date1 = datetime.datetime.strptime(emr[date_item][:10], "%Y-%m-%d")
                        date2 = datetime.datetime.strptime(past_emr[date_item][:10], "%Y-%m-%d")
                        days = (date1 - date2).days
                        if r1.group(2) == '天' and r2.group(2) == '天':
                            if int(r2.group(1)) - int(r1.group(1)) != days:
                                score -= 2
                                reason += f'{item}时间与既往3个月内病历记录不一致。\n'
            if score < 0:
                return {"item": 14, "是否扣分": "是", "score": score, "reason": reason, "knowledge_graph": {}}
            else:
                return {"item": 14, "是否扣分": "否", "score": 0, "reason": "", "knowledge_graph": {}}


    def _item_15(self, emr, parm):
        """
        主诉记录有错别字。
        """

        # 初值
        score = 0
        reason = ''
        for item in parm.get("Parameter0",[]):
            ans = self.corrector.correct(emr.get(item, ''))
            if ans.get('error',''):
                # print(ans)
                score -= 2
                e = ans['error'][0][0]
                reason += f'{item}记录有错别字,错别字 {e}。'

        if score < 0:
            return {"item": 15, "是否扣分": "是", "score": score, "reason": reason, "knowledge_graph": {}}
        else:
            return {"item": 15, "是否扣分": "否", "score": 0, "reason": "", "knowledge_graph": {}}
