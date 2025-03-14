#!/usr/bin/python
# -*- coding: UTF-8 -*-
import xml.etree.ElementTree as ET
import re

from ..extract_entity import MatchExtract

class Document:
    # 效果不太好 基于规则提取难度比较大
    def __init__(self, entity_dict_path, document_info) -> None:
        self.match_model = MatchExtract(entity_dict_path)
        self._document = {}
        self.start_page = document_info['start_page']
        # self.start_page = 32 # 诊断学的start_page 为 32
        self.end_page = document_info['end_page']
        self.document_path = document_info['path']
        # self.end_page = 649 # 诊断学的end_page 为 649
        self.headers = [
            [ (r"^第([一|二|三|四|五|六|七|八|九|十]+?)篇",'>20', 'all') ], # 篇标题
            [ (r"^第([一|二|三|四|五|六|七|八|九|十]+?)章(.{1,15})$",'>14', '-1') ], # 节标题
            [ (r"^第([一|二|三|四|五|六|七|八|九|十]+?)节(.{1,15})$",'>14', '-1') ], # 节标题—
            [ (r"(^|(\\n)|(\n))([—|一|二|三|四|五|六|七|八|九|十]+?)、(.{1,30})$",'>10', '-1') ], # 节标题
            [ (r"(^|(\\n)|(\n))[［|【|\[](.*?)[］|】|\]]",'', '-1') ] # 蓝色标题
        ]
        self.cur_h = ['' for _ in range(len(self.headers))]
        self.cur_text = ''
        self.prefix = '-'.join(['h'+str(i+1)+'-{}' for i in range(len(self.cur_h))]) + '##'
    
    @staticmethod
    def split_text(text):
        text = text.replace('\n','')
        texts = re.split(r'[。|;]',text)
        texts = list(filter(lambda x:len(x)>5,texts))
        return texts

    @staticmethod
    def is_vaild_text(text):
        new_text = re.findall(r'[\u4e00-\u9fa5]+',text)
        if len(new_text)!=0 and new_text[0]!='八':
            return True
        else:
            return False

    @staticmethod
    def find_entities(text):
        raw_entities = re.findall(r'如(.*?)等',text)
        raw_entities += re.findall(r'见于(.*?)[等|。]',text)
        raw_entities += re.findall(r'有(.*?)[等|。]',text)
        raw_entities += re.findall(r'表现为(.*?)[，|等|。]',text)
        raw_entities += re.findall(r'引起(.*?)[。|等|，]',text)
        
        entities = []
        for _entities in raw_entities:
            _entities = re.split(r'[、|和|或]',_entities)
            _entities = list(filter(lambda x:len(x)>1, _entities))
            entities = entities + _entities
        return entities
    
    @staticmethod
    def match_font(font_size, rule):
        if rule == '': return True
        if rule[0] == '>' and font_size > float(rule[1:]): return True
        if rule[0] == '<' and font_size < float(rule[1:]): return True
        return False

    def get_prefix(self, sentence, font_size):
        """
        获取标题
        """
        for i, header in enumerate(self.headers):
            for match_rule in header:
                match_groups = re.findall(match_rule[0],sentence)
                match_font_size = Document.match_font(font_size, match_rule[1])
                if len(match_groups) == 0 or not match_font_size:
                    continue
                if match_rule[2] == 'all':
                    self.cur_h[i] = sentence
                else:
                    self.cur_h[i] = match_groups[0][int(match_rule[2])]
                self.cur_h[i] = self.cur_h[i].replace('\n','')
                for j in range(i+1, len(self.cur_h)):
                    self.cur_h[j] = ''
                print(self.prefix.format(*self.cur_h))
                if match_rule[2] == 'all':
                    return sentence
                else:
                    return match_groups[0][int(match_rule[2])]

        return ''


    def make_structured_representation(self):
        """
        结构化表示文档
        """
        all_texts = [] # for debug
        tree = ET.parse(self.document_path)
        for page in tree.getroot().iter('page'): # 遍历每一页
            page_id = int(page.attrib["id"])
            if page_id < self.start_page: continue
            if page_id > self.end_page: continue

            for _,element in enumerate(page.iter('textbox')): # 遍历每个box
                text = ''
                _text_font_size = []
                for i, text_element in enumerate(element.iter('text')): # 遍历每个字
                    text = text + text_element.text
                    if 'size' in text_element.attrib:
                        _text_font_size.append(float(text_element.attrib['size']))

                if len(_text_font_size) == 0: # 防止报错
                    _text_font_size.append(0)
                
                text_font_size = sum(_text_font_size) / len(_text_font_size)
                if text_font_size < 10: continue
                if not Document.is_vaild_text(text): continue
                
                text = text.strip().replace(' ','')
                pre_prefix = self.prefix.format(*self.cur_h)
                change_prefix = self.get_prefix(text,text_font_size)

                if len(change_prefix) > 0:
                    pre_text = text.split(change_prefix)[0]
                    next_text = text.split(change_prefix)[1]
                    self.cur_text = self.cur_text + pre_text
                    self._document[pre_prefix] = self.cur_text
                    self.cur_text = next_text
                else:
                    self.cur_text = self.cur_text + text

        self._document[self.prefix.format(*self.cur_h)] = self.cur_text

        for header in self._document:
            self._document[header] = Document.split_text(self._document[header])
    
    def make_triples(self):
        triples = [] # (实体1, 实体2, header)
        for header in self._document:
            for _text in self._document[header]:
                text = header + _text
                ans = self.match_model.find_entities(text)
                entities = ans['pos'] + ans['neg']
                entities = [e[0] for e in entities]
                entities = list(set(entities))

                for e1 in entities:
                    for e2 in entities:
                        if e1 == e2: continue
                        triples.append([e1,e2,header])
        return triples
    
    def make_triples_llm(self):
        triples = [] # (实体1, 实体2, header)
        for header in self._document:
            for _text in self._document[header]:
                text = header + _text
                
        return triples

