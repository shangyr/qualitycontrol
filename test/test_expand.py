# coding:utf-8 -*-
import sys
import os

PROJECT_DIR = 'C:\\Users\\asndj\\Desktop\\2023-12-29\\病历质检\\qualitycontrol_service'
sys.path.append(PROJECT_DIR)
from src.expand_graph import Document

document_info = {'path': 'C:\\Users\\asndj\Desktop\\2023-12-29\\病历质检\\extract_knowledge\\diagnostics.xml','start_page':32 ,'end_page':649}
entity_dict_path = os.path.join(PROJECT_DIR, 'data_and_models\\entity-dict\\entities.txt')

doc = Document(entity_dict_path, document_info)
doc.make_structured_representation()
triples = doc.make_triples()
print(triples)