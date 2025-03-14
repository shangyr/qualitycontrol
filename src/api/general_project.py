# -*- coding:utf-8 -*-
"""
一般项目
3.对于特定的性别，出现不合理的描述（男性患者病历记录中不应有月经史、子宫附件的描述，女性患者不应有前列腺等描述）。
5.姓名、性别、年龄、婚姻、民族、职业、地址缺项。
6.就诊日期、科别缺项。
7.过敏史缺项。

"""
import re

class GeneralProject:
    def __init__(self) -> None:
        pass

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
            '_item_3',
            '_item_5',
        ]

        if item not in fields:
            return {}

        ans = getattr(self, item)(emr,parm)
        return ans

    def _item_3(self, emr, parm):
        """
        3.对于特定的性别，出现不合理的描述（男性患者病历记录中不应有月经史、子宫附件的描述，女性患者不应有前列腺等描述）。
         男性病例，字符串查找有没有月经、子宫
         女性病例，字符串查找有没有前列腺
        """
        # 初值
        score = 0
        reason = ''

        sex = parm.get('Parameter0')
        sex = emr.get(sex[0], '')

        # 不合理描述
        male = ['月经', '子宫', '卵巢', '子宫颈', '阴道', '怀孕', '经期', '妊娠', '绝经']
        female = ['阴茎', '睾丸', '阴囊', '前列腺', '射精', '精子', '前列腺液']

        for item in parm.get('Parameter1',[]):
            text = emr.get(item, '')
            if sex == '男':
                for wds in male:
                    if (text.find(wds) != -1):
                        score -= 2
                        reason += f'{item}存在不符合该患者性别的描述，（关键词:{wds}）。\n'
                        # print(reason)
                        break
            elif sex == "女":
                for wds in female:
                    if (text.find(wds) != -1):
                        score -= 2
                        reason += f'{item}存在不符合该患者性别的描述，（关键词:{wds}。\n'
                        break

        if score<0:
            return {"item": 3,"是否扣分": "是", "score": score, "reason": reason, "knowledge_graph": {} }
        else:
            return {"item": 3, "是否扣分": "否", "score": score, "reason": reason, "knowledge_graph": {}}


    def _item_5(self, emr, parm):
        """
        患者姓名、性别、年龄、婚姻、民族、职业、地址等一般项目信息不完整。
        """
        score = 0
        reason = ''

        general_proj = parm.get('Parameter0',[])
        for item in general_proj:
            if not emr.get(item, ''):
                score -= 0
                reason +=  f'{item}信息不完整。\n'

        ans = {
        "item": 5,
        "是否扣分": "是" if reason else "否",
        "score": score,
        "reason": reason,
        "knowledge_graph": {}
        }

        return ans

