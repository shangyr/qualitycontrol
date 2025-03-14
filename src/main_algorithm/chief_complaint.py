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
            '_item_9',
            '_item_10',
            '_item_11',
            '_item_12',
            '_item_13',
            '_item_14',
            '_item_15',
            '_item_16',
        ]

        reason = ''

        score = 0
        for field in fields:
            field_ans = getattr(self, field)(emr)
            reason += field_ans['reason']
            score += field_ans['score']
            
        ans = {'score': score, 'reason': reason}
        return ans

    def _item_9(self, emr):
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
        # 初诊为 0，复诊为 1
        flag = 0 if '是否初诊' not in emr.keys() else emr['是否初诊']
        # 初值
        score = 0
        reason = ''
        # 初诊
        if not flag:
            if "要求检查" in emr['主诉'] or "孕检" in emr['主诉']:
                return {'score': score, 'reason': reason}

            entities = [item for item in self.match_model.find_entities(emr["主诉"])['pos'] if
                        ('疾病' in item[1]) or ('症状' in item[1])]

            # 未匹配到症状
            if len(entities) == 0:
                score -= 5
                reason += '主诉要素描述不全，缺少症状，扣5分。\n'

                # 匹配规则，症状/疾病 + .. + 约数 + 时间
            # pattern = re.compile(r'{}\S*?(数|多)(时|天|日|周|星期|月|年)'.format(entities[0][0]))
            pattern = re.compile(r'(数|多)(时|天|日|周|星期|月|年)')
            r = pattern.search(emr["主诉"])
            # 匹配到持续时间，跳出循环
            if r:
                score -= 2
                reason += '主诉要素描述有缺陷，未精确描述症状持续时间，扣2分。\n'
            else:
                pattern = re.compile(
                    r'[0-9零半一二两俩三四五六七八九]+[个|余]?(时|天|日|周|星期|月|年)',
                    flags=re.IGNORECASE)
                r = pattern.search(emr["主诉"])
                # 未匹配到准确时间
                if not r:
                    score -= 5
                    reason += '主诉要素描述不全，缺少持续时间，扣5分。\n'

        ans = {'score': score, 'reason': reason}
        return ans


    def _item_10(self, emr):
        """
        主诉要素描述不全，缺少时间。
        同item_9
        """

        flag = 0 if '是否初诊' not in emr.keys() else emr['是否初诊']
        # 初值
        score = 0
        reason = ''

        ans = {'score': score, 'reason': reason}
        return ans

    def _item_11(self, emr):
        """
        初诊：11.主诉超过20个字。

        统计字符数量即可
        """
        # 初诊为 0，复诊为 1
        flag = 0 if '是否初诊' not in emr.keys() else emr['是否初诊']

        score = 0
        reason = ''

        if not flag:
            if len(emr['主诉']) > 20:
                score -= 2
                reason += "初诊: 主诉超过20个字，扣2分。\n"

        ans = {'score': score, 'reason': reason}
        return ans

    def _item_12(self, emr):
        """
        初诊：12.主诉的时间描述不准确，如“数天/数月/数年”。
        查找症状同一句有没有 年/月/日/天/时 等，然后判断前面是不是可以转化为数字
        见item_9
        """
        score = 0
        reason = ''

        ans = {'score': score, 'reason':reason}
        return ans

    def _item_13(self, emr):
        """
        主诉内容与既往3个月内主要诊断相同的初诊主诉重复率为100%。
        """
        # 初值
        score = 0
        reason = ''
        # 如果当前病历为复诊，则执行如下规则判断
        if emr.get('是否初诊', 0):
            past_emrs = emr['既往病历']
            for past_emr in past_emrs:
                if not past_emr.get('是否初诊', 0):
                    # 既往病历主诉并不为空
                    if past_emr.get('主诉', ''):
                        # 既往病历和当前病历主诉重复
                        if emr.get('主诉', '') == past_emr.get('主诉', ''):
                            score = -5
                            reason = '不合理复制主诉内容，与近三月病历记录雷同，扣5分。'
                            # 正则匹配
                            pattern = re.compile(r'([0-9]{4}).([0-9]{2}).([0-9]{2}).([0-9]{2}).([0-9]{2})')
                            res = pattern.search(emr.get('流水号', ''))
                            if res:
                                reason = f'不合理复制主诉内容，与{res.group(1)}年{res.group(2)}月{res.group(3)}日病历记录雷同，扣5分。'
                            # 重复
                            return {'score': score, 'reason': reason}

        # 未重复
        ans = {'score': score, 'reason': reason}
        return ans

    def _item_14(self, emr):
        """
        主诉时间与既往3个月内病历记录不一致。
        """
        # 初值
        score = 0
        reason = ''
        # 既往三个月没有病历记录
        if emr.get('既往病历',[]):
            return {'score': score, 'reason': reason}
        # 当前病历为初诊病历
        if not emr.get('是否初诊', 0):
            return {'score': score, 'reason': reason}

        entities = self.match_model.find_entities(emr["主诉"])['pos']
        entities = [item for item in entities if '疾病' in item[1] or '症状' in item[1]]
        if len(entities) == 0:
            return {'score': score, 'reason': reason}

        # 遍历既往病历
        for past_emr in emr['既往病历']:
            entities_ = self.match_model.find_entities(past_emr["主诉"])['pos']
            entities_ = [item for item in entities_ if '疾病' in item[1] or '症状' in item[1]]
            # 求出交集
            if len(set(entities).intersection(set(entities_))) > 0:

                # 匹配规则，匹配主诉中出现的数字， + (天|日|周|星期|月) 疾病时间
                pattern = re.compile(r'([0-9]{1,2})(天|日|周|星期|月)')
                r1 = pattern.search(emr["主诉"])
                r2 = pattern.search(past_emr["主诉"])
                if r1 and r2:
                    # 判断emr["就诊时间"](str) 是否为时间格式，xxxx-xx-xxxx xx:xx
                    try:
                        # 计算日期
                        date1 = datetime.datetime.strptime(emr["就诊时间"], "%Y-%m-%d")
                        date2 = datetime.datetime.strptime(past_emr["就诊时间"], "%Y-%m-%d")
                        days = (date1 - date2).days

                    except Exception as e:
                        # 匹配失败
                        return {'score': score, 'reason': reason}
                    # r1 和r2 将周 | 星期|月 转为天
                    if r1.group(2) == '天' and r2.group(2) == '天':
                        if int(r2.group(1))-int(r1.group(1)) != days:
                            score -= 2
                            reason += f'主诉时间与既往3个月内病历记录不一致，扣2分。\n'

        ans = {'score': score, 'reason': reason}
        return ans

    def _item_15(self, emr):
        """
        主诉记录有错别字。
        """

        # 初值
        score = 0
        reason = ''
        if True:
            return {'score': score, 'reason': reason}
        ans = self.corrector.correct(emr.get('主诉', ''))

        # print(ans)
        if ans['errors']:
            score -= 2
            e = ans['errors'][0][1]
            reason += f'主诉记录有错别字,错别字 {e}。'

        ans = {'score': score, 'reason': reason}
        return ans

    def _item_16(self, emr):
        """
        主诉描述有缺陷。
        人工质控规则兜底
        """

        flag = 0 if '是否初诊' not in emr.keys() else emr['是否初诊']
        # 初值
        score = 0
        reason = ''

        ans = {'score': score, 'reason': reason}
        return ans
