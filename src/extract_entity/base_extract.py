from abc import ABC,abstractclassmethod
import re

class BaseExtract(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    def _remove_neg_entities(self,entities:list,sentence:str) -> list:
        """
        去除否定实体
        Args:
            entities:实体列表
            sentence:句子
        Return:
            filter_entities:去否定实体之后的实体列表
        """

        neg_entities = re.findall(r"((否认)|(无))(.*?)(，|。|；|,|$|(\[SEP\]))",sentence)
        # 否认“肝炎，伤寒，结核”
        # neg_entities.extend(re.findall(r'((否认)|(无))“(.*?)”',sentence))
        neg_entities = ''.join([''.join(list(item)) for item in neg_entities]) # 电子病历中的否定实体

        # 也有一些正确的
        pos_entities = re.findall(r"无明显诱因(.*?)(，|。|；|,|$|(\[SEP\]))",neg_entities)
        pos_entities = ''.join([''.join(list(item)) for item in pos_entities])
        filter_entities = []
        for entity in entities:
            if entity not in neg_entities:
                filter_entities.append(entity)
            if entity in pos_entities:
                filter_entities.append(entity)
        return filter_entities
    
    @abstractclassmethod
    def _find_entities(self, sentence:str):
        """
        获取实体列表
        Args:
            sentence:句子/文档
        Return:
            entities:匹配实体列表 
            [(实体名, 实体类型, 实体开始位置, 实体结束位置),...]
        """

    def find_entities(self,sentence:str) -> list:
        """
        获取实体列表
        Args:
            sentence:句子/文档
        Return:
            entities:匹配实体列表 
            distinguish_pos_neg:
            {'pos': [(实体名, 实体类型, 实体开始位置, 实体结束位置),...] ,'neg':[]}
            not distinguish_pos_neg
            [(实体名, 实体类型, 实体开始位置, 实体结束位置),...]
        """
        entities = self._find_entities(sentence)
        if not self.distinguish_pos_neg:
            return entities

        entities = [entity for entity in entities if len(entity[0]) > 1]

        include_entities = []
        for entity1 in entities:
            for entity2 in entities:
                if entity1[0] == entity2[0]:
                    continue
                if entity1[2] >= entity2[2] and entity1[3] <= entity2[3]:
                    include_entities.append(entity1[0])
                    break
        entities = [entity for entity in entities if entity[0] not in include_entities]

        entity_names = [entity[0] for entity in entities]
        pos_entities = self._remove_neg_entities(entity_names,sentence)
        neg_entities = [entity for entity in entities if entity[0] not in pos_entities]
        pos_entities = [entity for entity in entities if entity[0] in pos_entities]
        
        return {
            'pos':pos_entities,
            'neg':neg_entities
        }