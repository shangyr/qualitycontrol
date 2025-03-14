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
            '_item_3',
            '_item_5',
            '_item_6',
            '_item_7',
        ]

        reason = ''
        score = 0
        for field in fields:
            field_ans = getattr(self, field)(emr)

            reason += field_ans['reason']
            score += field_ans['score']
            
        ans = {'score': score, 'reason':reason}
        return ans

    def _item_3(self, emr):
        """
        3.对于特定的性别，出现不合理的描述（男性患者病历记录中不应有月经史、子宫附件的描述，女性患者不应有前列腺等描述）。
         男性病例，字符串查找有没有月经、子宫
         女性病例，字符串查找有没有前列腺
        """
        # 初值
        score = 0
        reason = ''
        # 不合理描述
        male = ['月经', '子宫', '卵巢', '子宫颈', '阴道', '怀孕', '经期', '绝经']
        female = ['阴茎', '睾丸', '阴囊', '前列腺', '射精', '精子', '前列腺液']
        # print('text')
        text = emr["主诉"] + emr["现病史"] + emr["既往史及其他病史"] + emr["体格检查"] + emr["辅助检查"]
        sex = emr['性别']
        if sex == '男':
            for wds in male:
                if (text.find(wds) != -1):
                    score -= 2
                    reason += '存在不符合该患者性别的描述，（关键词:{}），扣2分。\n'.format(wds)
                    # print(reason)
                    break
            else:
                pass
        elif sex == "女":
            for wds in female:
                if (text.find(wds) != -1):
                    score -= 2
                    reason += '存在不符合该患者性别的描述，（关键词:{}），扣2分。\n'.format(wds)
                    break
            else:
                pass
        ans = {'score': score, 'reason': reason}

        return ans

    def _item_5(self, emr):
        """
        患者姓名、性别、年龄、婚姻、民族、职业、地址等一般项目信息不完整。
        """
        score = 0
        reason = ''

        general_proj = ['姓名', '性别', '年龄', '婚姻', '民族', '职业', '地址']
        for item in general_proj:
            if not emr.get(item, ''):
                reason +=  f'{item}信息不完整。\n'

        ans = {'score': score, 'reason': reason}

        return ans

    def _item_6(self, emr):
        """
        就诊日期、科别不完整。
        """

        score = 0
        reason = ''
        res = emr.get("就诊时间", "")
        # 就诊日期为空
        if len(res) == 0:
            score -= 0
            reason += '就诊日期不完整。\n'

        res = emr.get("科室", "")
        # 过敏史为空则判为
        if len(res) == 0:
            score -= 0
            reason += '科别不完整。\n'

        ans = {'score': score, 'reason': reason}

        return ans

    def _item_7(self, emr):
        """
        7.检查过敏史是否缺项。
        """

        score = 0
        reason = ''
        res = emr.get("药物过敏史", "")
        # 过敏史为空则判为
        if len(res) == 0:
            score -= 2
            reason += '过敏史未填写，扣2分。\n'

        ans = {'score': score, 'reason': reason}

        return ans

    '''
    下列做好迁移后删除
    '''

    def _item_9(self, emr):
        """
        检查过敏史与既往3个月记录不一致。
        """
        # 现在的过敏史
        score = 0
        reason = ''
        curr_res = emr.get('药物过敏史', '')
        # 既往过敏史
        past_res = [item.get('药物过敏史', '') for item in emr['既往病历']]
        # 简单匹配规则，后续可以更加精细
        pattern = re.compile(r"(无|没有)?(.*)过敏")
        for past_res_ in past_res:
            res1 = pattern.search(past_res_)
            # 匹配到相关规则
            if res1:
                # 匹配到无/没有等名词，--->无相关过敏史
                if res1.group(1):
                    continue
                # 匹配到过敏史
                if res1.group(2):
                    pattern1 = re.compile(r"(无|没有)?({})过敏".format(res1.group(2)))
                    # 搜寻现在的病历中有无相关记录
                    res2 = pattern1.search(curr_res)
                    # 匹配成功
                    if res2:
                        # 但是匹配到无/没有等字样，和前述病历冲突
                        if res2.group(1):
                            score -= 5
                            reason += '过敏史与既往3个月记录不一致,既往史中记录{}过敏史，现病历中记载无该过敏史，扣5分。'.format(res1.group(2))
                    # 未匹配到，证明病历冲突
                    else:
                        score -= 5
                        reason += '过敏史与既往3个月记录不一致,既往史中记录{}过敏史，现病历中未记载，扣5分。'.format(res1.group(2))

        # print(past_res)
        ans = {'score': score, 'reason': reason}

        return ans

    
