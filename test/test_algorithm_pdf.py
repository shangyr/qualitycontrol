# coding:utf-8 -*-
import sys
import os
import re

PROJECT_DIR = r"F:\广医质检\pdf\qualitycontrol_service"
sys.path.append(PROJECT_DIR)
DIAGNOSIS_DIR = r"F:\广医质检\pdf\qualitycontrol_service\src\diagnosis"
sys.path.append(DIAGNOSIS_DIR)

from src import EMRQualityInspectionAlgorithm
import time
import json
import PyPDF2

def read_pdf(pdf_path:str):
    pdf_reader = PyPDF2.PdfReader(open(pdf_path, 'rb'))
    emr_str = ''
    emr = {}
    # 读取pdf内容
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text = page.extract_text().replace('\n','')
        emr_str += text
    # 去除无效字符
    emr_str = re.sub('打印时间.*?第.*?页/共.*?页', '', emr_str)
    emr_str = re.sub('广州市第一人民医院 门（急）诊病历', '', emr_str)
    #print(emr_str)
    # print(emr_str)
    title = ['姓名', '性别', '年龄', '就诊卡号', '婚姻', '民族', '职业', '地址', '药物过敏史', '就诊时间', '科室', '流水号', '主诉',
             '现病史', '既往史及其他病史', '体格检查', '辅助检查', '初步诊断', '处理', '嘱托', '医生签名']
    for i in range(len(title)-1):
        if title[i] == '年龄':
           pattern = rf"{title[i]}：\s*(\d+\s*岁)"#rf"{title[i]}：\s*(\d+)[\s岁]*"
        else:
           pattern = rf"{title[i]}：\s*(.*?)(?=\s*{title[i+1]})"
        result = re.search(pattern, emr_str, re.DOTALL)
        emr.update({title[i]:result.group(1) if result else ""})

    result = re.search(r"医生签名：\s*(.*)", emr_str, re.DOTALL)
    emr.update({'医生签名': result.group(1).strip() if result else ""})

    return emr


if __name__ == '__main__':
    # !pip install PyPDF2
    # pdf路径，绝对地址
    pdf_path = r"F:\广医质检\通用规则演示实例\0801四种内涵质控规则的数据\复诊病历现病史病情变化\复诊病历现病史病情变化\精神-20240430-529-古智文.pdf"
    # 读取pdf
    emr = read_pdf(pdf_path)
    # 额外的模拟数据
    extend_emr ={'是否初诊': 0,
        '病历创建时间': "xxxx/xxxx",
        '手术记录': {
            '创建时间': '2023-09-14 18:45',
            '手术开始时间': '2023-09-14 14:45',
            '手术结束时间': '2023-09-14 17:27',
            '术前诊断': '肩周炎',
            '术中诊断': '肩周炎',
            '手术名称': '肩周炎松解术',
            '手术医生': '**',
            '麻醉方式': '全麻',
            '手术经过': '全麻诱导顺利...........',
        },
        '危急值记录': {
            '创建时间': '2023-09-14 18:43',
            '报告时间': '2023-09-14 14:52',
            '报告人': '**',
            '测定项目与结果': 'K 3.5mmol/L；Na 140mmol/L；Urea<30mmol/L'
        },
        '有创操作记录': {
            '创建时间': '2023-09-15 12:33',
            '操作时间': '2023/09/14 14:46',
            '操作医师': '***',
            '操作名称': '腹腔穿刺',
            '操作步骤': '嘱病人平卧、半卧、稍左侧卧位，取左下腹.........',
        },
        '既往病历': [{
        },{
        }]}
    emr.update(extend_emr)
    print(emr)
    entity_dict_path = os.path.join(PROJECT_DIR, 'data_and_models\\entity-dict\\entities-v2.txt')
    ner_model_path = os.path.join(PROJECT_DIR, 'data_and_models\\bert-base-chinese-ner')
    graph_model_path = os.path.join(PROJECT_DIR, 'data_and_models\\knowledge-graph\\triples-v2.txt')
    diagnosis_path = os.path.join(PROJECT_DIR, 'data_and_models\\diagnosis')
    acute_illness_path = os.path.join(PROJECT_DIR, 'data_and_models\\diagnosis\\acute_illness.txt')  # 急性病字典
    corrector_path = os.path.join(PROJECT_DIR, 'data_and_models\\bert_correct')
    body_path = os.path.join(PROJECT_DIR, 'data_and_models\\diagnosis\\body.txt')#身体部位字典
    emr_inspection = EMRQualityInspectionAlgorithm(entity_dict_path, ner_model_path, graph_model_path, diagnosis_path, corrector_path,acute_illness_path, body_path)
    t = time.time()
    ans = emr_inspection.emr_quality_inspection(emr)
    print(json.dumps(ans, indent=4, ensure_ascii=False))
    # print(ans)
    print('质检时间为 %.2f s'%(time.time() - t))
