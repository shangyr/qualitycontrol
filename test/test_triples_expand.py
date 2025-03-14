# coding:utf-8 -*-
import sys
import os
import time

PROJECT_DIR = 'C:\\Users\\asndj\\Desktop\\2023-12-29\\病历质检\\qualitycontrol_service'
sys.path.append(PROJECT_DIR)
from src.expand_graph import ExcelExtract, expand_entity
if __name__ == '__main__':
    # expand_entity(
    #     icd10_file='data_and_models\\medical_knowledge\\icd10.xlsx',
    #     symptom_file='data_and_models\\medical_knowledge\\symptom.dic',
    #     disease_file='data_and_models\\medical_knowledge\\disease.dic',
    #     jbk39jb_file='data_and_models\\medical_knowledge\\jbk39jb.xlsx',
    #     all_file='data_and_models\\medical_knowledge\\disease汇总全部数据.xlsx',
    #     entities_file='data_and_models\\entity-dict\\entities.txt',
    #     out_fil='data_and_models\\entity-dict\\entities-v2.txt',
    # )
    e = ExcelExtract(
        excel_file1='data_and_models\\medical_knowledge\\jbk39jb.xlsx',
        excel_file2='data_and_models\\medical_knowledge\\disease汇总全部数据.xlsx',
        entity_dict_path='data_and_models\\entity-dict\\entities-v2.txt',
        triple_path="data_and_models\\knowledge-graph\\triples.txt"
    )
    e.pipeline('data_and_models\\knowledge-graph\\triples-v2.txt')
