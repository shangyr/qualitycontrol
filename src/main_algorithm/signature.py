# -*- coding:utf-8 -*-
"""
签名（5分)

20.经治医师签全名。
无经治医师签名扣5分。

"""

class Signature:
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
        ans = {'score': 0, 'reason':''}
        return ans