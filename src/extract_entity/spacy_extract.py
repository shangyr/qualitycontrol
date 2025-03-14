
# https://spacy.io/models

from .base_extract import BaseExtract
import spacy
"""
spacy网站提供模型得到句法分析 辅助判断
如果efficiency效果不太好的话可以上
python -m spacy download zh_core_web_trf
model = spacy.load("zh_core_web_trf")
"""

class SpacyExtract(BaseExtract):
    def __init__(self) -> None:
        super().__init__()
        self.model = spacy.load("zh_core_web_sm")
        self.distinguish_pos_neg = False
    
    def _find_entities(self, sentence):
        doc = self.model(sentence)
        ans = []
        cur_len = 0
        for word in doc:
            ans.append((word.text, word.pos_, cur_len, cur_len + len(word.text)))
            # ans.append((word.text, word.ent_type_ + '-' + word.pos_, cur_len, cur_len + len(word.text)))
            cur_len += len(word.text)
        return ans
    