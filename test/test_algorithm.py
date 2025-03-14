# coding:utf-8 -*-
import sys
import os

PROJECT_DIR = r"C:\Users\huawei\Desktop\qualitycontrol"
sys.path.append(PROJECT_DIR)
DIAGNOSIS_DIR = r"C:\Users\huawei\Desktop\qualitycontrol\src\diagnosis"
sys.path.append(DIAGNOSIS_DIR)

from src import EMRQualityInspectionAlgorithm
import time
import json

if __name__ == '__main__':
    # '医生签名'后数据为模拟数据，用病历进行测试只替换前半部分即可
    emr = {
        "姓名": "***",
        "性别": "女",
        "年龄": "58岁",
        "门诊号": "50103801",
        "婚姻": "",
        "民族": "",
        "职业": "",
        "地址": "",
        "就诊时间": "_急诊",
        "科室": "内科门诊",
        "流水号": "2023-09-21-8719",
        "药物过敏史": "无",
        "主诉": "不适复诊",
        "现病史": "于2023-11-16在我院行脑血管造影，术中见右侧颈内动脉中度狭窄（C7，60%）",
        "既往史及其他病史": "否认急性传染病史；否认肝炎史；否认结核病史，否认疟疾病；幼年时曾接种疫苗名称不详，否认肺部疾病史；否认外伤史；否认手术史；否认过敏史。长期居留于广州，无疫水接触史；否认吸烟嗜好；否认饮酒嗜好。",
        "体格检查": "腹部平坦，腹壁静脉无显露，未见胃肠型，未见蠕动波，无异常搏动，腹壁柔软，全腹无压痛，无反跳痛，无包块，肝脾肋下未触及。胸廓对称，胸骨无压痛，双侧语颤对称；无胸膜摩擦感，叩诊正常清音，双肺呼吸音清晰，双肺未闻及干、湿性罗音，未闻及胸膜摩擦音。心尖搏动正常，心前区无震颤或异常搏动，无心包摩擦感；心浊音界正常；心律齐，心音正常，心脏各瓣膜听诊区未闻及杂音。",
        "辅助检查": "无",
        "初步诊断": "1、腹泻；2、腹胀；3、头痛；4、睡眠障碍",
        "处理": "1常规心电图（十五导联）**1心电图室20.1心可舒片（国基）0.31g×144片用法：口服每日三次，每次1.24g0.2铝镁匹林片(II)(国谈)1片×30片用法：口服每日一次，每次1片",
        "嘱托": "适当休息，适度活动，合理饮食，按时服药",
        "医生签名": "",
        '是否初诊': 0,

# 模拟数据，不用进行修改
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
        '既往病历': [
            {

        },
            {

        }]
    }
    entity_dict_path = os.path.join(PROJECT_DIR, 'data_and_models\\entity-dict\\entities-v2.txt')
    ner_model_path = os.path.join(PROJECT_DIR, 'data_and_models\\bert-base-chinese-ner')
    graph_model_path = os.path.join(PROJECT_DIR, 'data_and_models\\knowledge-graph\\triples-v2.txt')
    diagnosis_path = os.path.join(PROJECT_DIR, 'data_and_models\\diagnosis')
    corrector_path = os.path.join(PROJECT_DIR, 'data_and_models\\bert_correct')
    acute_illness_path=os.path.join(PROJECT_DIR, 'data_and_models\\diagnosis\\acute_illness.txt')#急性病字典
    body_path = os.path.join(PROJECT_DIR, 'data_and_models\\diagnosis\\body.txt')#身体部位字典
    emr_inspection = EMRQualityInspectionAlgorithm(entity_dict_path, ner_model_path, graph_model_path, diagnosis_path, corrector_path, acute_illness_path, body_path)
    t = time.time()
    ans = emr_inspection.emr_quality_inspection(emr)
    print(json.dumps(ans, indent=4, ensure_ascii=False))
    # print(ans)
    print('质检时间为 %.2f s'%(time.time() - t))