#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   expand_entity.py
@Author  :   lixin 
@Version :   1.0
@Desc    :   合并实体名称文件
'''

from tqdm import tqdm
import pandas as pd
import re

ignore_entity_set = {
    '疾病':{'急性','急','严重','先天性','静脉','血管','家族型','白细胞','消失',
            '轻度','对于','b型'},
    '症状':{'干扰','扩大','增多','异常','加重','融合','质','体征','言语',
          '缺乏','注意','感觉','主观','缩小','大量','破坏','抽','四肢','缓慢',
          '困难','降低','抑制','其他症状','医生'}
}

def count_chinese_characters(text):
    """
    统计汉字数量
    """
    pattern = re.compile(r'[\u4e00-\u9fff]')  # 匹配Unicode编码范围内的汉字字符
    chinese_chars = re.findall(pattern, text)
    return len(chinese_chars)

def text_process(text):
    return re.sub(r'(\(|（|\[).*?(\)|）|\]|$)','',text.lower().strip('.,， \n、').lstrip('有会'))

def valid_text_length(text):
    if len(text) >= 1 and len(text) <= 20:
        return True
    return False

def valid_correct_entity(text, text_type):
    text_type = re.sub(r'[a-zA-Z]','',text_type)
    if text_type in ignore_entity_set and text in ignore_entity_set[text_type]:
        return False
    if text_type != '检查' and count_chinese_characters(text) == 0:
        return False

    return True

def valid_text(text, text_type):
    if not valid_text_length(text): return False
    if not valid_correct_entity(text,text_type): return False
    return True

def expand_entity(icd10_file,symptom_file,disease_file,
                  jbk39jb_file,all_file,entities_file,out_file):
    entityname2type = {}
    entitytype2names = {}
    entitytype2count = {}

    df = pd.read_excel(icd10_file)
    data_size = len(df['Disease'])
    for i in tqdm(range(data_size)):
        disease3 = df['Disease'][i].strip()
        disease2 = df['SecondType'][i].strip()
        disease1 = df['FirstType'][i].strip()
        entityname2type[text_process(disease3)] = '疾病icd10'
        entityname2type[text_process(disease2)] = '疾病icd10'
        entityname2type[text_process(disease1)] = '疾病icd10'

    with open(symptom_file,'r',encoding='utf-8') as f:
        next(f) # head
        for line in tqdm(f):
            entityname = line.strip().split(',')[1]
            entityname2type[text_process(entityname)] = '症状39'

    with open(disease_file,'r',encoding='utf-8') as f:
        next(f) # head
        for line in tqdm(f):
            entityname = line.strip().split(',')[1]
            entityname2type[text_process(entityname)] = '疾病39'

    df = pd.read_excel(jbk39jb_file)
    data_size = len(df['名称'])
    for i in tqdm(range(data_size)):
        disease = df['名称'][i].strip()
        other_disease = re.split(r'[,|，|；|;]',df['别名'][i].strip())
        diseases = set(other_disease)
        diseases.add(disease)
        medicals = []
        if not pd.isnull(df['常用药物'][i]):
            medicals = re.split(r'[,|，|；|;]',df['常用药物'][i].strip())
        for disease in diseases:
            entityname2type[text_process(disease)] = '疾病39'
        for medical in medicals:
            medical = text_process(medical)
            if len(medical) <= 1: continue
            entityname2type[medical] = '药品39'

    df = pd.read_excel(all_file)
    data_size = len(df['序号'])

    for i in tqdm(range(data_size)):
        disease = df['字段1_文本'][i]
        raw_rel_disease = df['并发症'][i]
        raw_symptoms = df['症状'][i]
        raw_medicals = df['常用药品'][i]
        raw_tests = df['检查'][i]

        rel_diseases = []
        if not pd.isna(raw_rel_disease):
            rel_diseases = re.findall(r" {10,100}(.*?)\n",raw_rel_disease)
            rel_diseases = list(filter(lambda x:x!='' and x!='并发症', rel_diseases))

        symptoms = []
        if not pd.isna(raw_symptoms):
            symptoms = re.findall(r" {40,100}(.*?)\n",raw_symptoms)
            symptoms = list(filter(lambda x:x!='',symptoms))

        medicals = []
        if not pd.isna(raw_medicals):
            medicals = raw_medicals.split('\n')
            medicals = [m.strip() for m in medicals]
            medicals = list(filter(lambda x:x!='' and '收费标准' not in x,medicals))

        tests = []
        if not pd.isna(raw_tests):
            _tests = re.findall(r" {20,100}(.*?)\n",raw_tests)
            if len(_tests) > 1:
                for test,next_test in zip(_tests[:-1],_tests[1:]):
                    if '元' in next_test:
                        test = re.sub(r'[\(|（].*?[$|)|）]','',test)
                        tests.append(test)

        entityname2type[text_process(disease)] = '疾病a'
        for rel_disease in rel_diseases:
            entityname2type[text_process(rel_disease)] = '疾病a'
        for symptom in symptoms:
            entityname2type[text_process(symptom)] = '症状a'
        for medical in medicals:
            entityname2type[text_process(medical)] = '药品a'
        for test in tests:
            entityname2type[text_process(test)] = '检查a'

    with open(entities_file,'r',encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            entity_name, entity_type = line.split('..')[0],line.split('..')[1]
            entityname2type[text_process(entity_name)] = entity_type
            if entity_type not in entitytype2count:
                entitytype2count[entity_type] = 0
            entitytype2count[entity_type] += 1

    entity_types = list(entitytype2count.keys())
    entity_types.sort(key=lambda x:entitytype2count[x],reverse=True)

    entity_types = entity_types + ['症状39','疾病39','药品39','症状a','疾病a','药品a','检查a','疾病icd10']

    for entity_name,entity_type in entityname2type.items():
        if entity_type not in entitytype2names:
            entitytype2names[entity_type] = set()
        entitytype2names[entity_type].add(entity_name)

    entities = [entity_name+'..'+entity_type for entity_type in entitytype2names 
                for entity_name in entitytype2names[entity_type] if valid_text(entity_name,entity_type)]

    with open(out_file,'w',encoding='utf-8') as f:
        f.write('\n'.join(entities))