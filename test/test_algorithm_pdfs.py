# coding:utf-8 -*-
import sys
import os
import re

PROJECT_DIR = r'E:\qualitycontrol_service2'
sys.path.append(PROJECT_DIR)
DIAGNOSIS_DIR = r'E:\qualitycontrol_service2\src\diagnosis'
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
    # print(emr_str)
    title = ['姓名', '性别', '年龄', '就诊卡号', '婚姻', '民族', '职业', '地址', '就诊时间', '科室', '流水号', '药物过敏史', '主诉',
             '现病史', '既往史及其他病史', '体格检查', '辅助检查', '初步诊断', '处理', '嘱托', '医生签名']
    for i in range(len(title)-1):
        pattern = rf"{title[i]}：\s*(.*?)(?=\s*{title[i+1]})"
        result = re.search(pattern, emr_str, re.DOTALL)
        emr.update({title[i]:result.group(1) if result else ""})

    result = re.search(r"医生签名：\s*(.*)", emr_str, re.DOTALL)
    emr.update({'医生签名': result.group(1).strip() if result else ""})

    return emr

def list_tran_json(list):
    str_json = json.dumps(list, ensure_ascii=False, indent=2)
    return str_json

if __name__ == '__main__':

    entity_dict_path = os.path.join(PROJECT_DIR, 'data_and_models\\entity-dict\\entities-v2.txt')
    ner_model_path = os.path.join(PROJECT_DIR, 'data_and_models\\bert-base-chinese-ner')
    graph_model_path = os.path.join(PROJECT_DIR, 'data_and_models\\knowledge-graph\\triples-v2.txt')
    diagnosis_path = os.path.join(PROJECT_DIR, 'data_and_models\\diagnosis')
    corrector_path = os.path.join(PROJECT_DIR, 'data_and_models\\bert_correct')
    body_path = os.path.join(PROJECT_DIR, 'data_and_models\\diagnosis\\body.txt')  # 身体部位字典
    acute_illness_path = os.path.join(PROJECT_DIR, 'data_and_models\\diagnosis\\acute_illness.txt')  # 急性病字典

    emr_inspection = EMRQualityInspectionAlgorithm(entity_dict_path, ner_model_path, graph_model_path, diagnosis_path,
                                                   corrector_path, acute_illness_path, body_path)

    # 额外的模拟数据
    extend_emr = {'是否初诊': 0,
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
                  }, {
                  }]}

    file_path = r'E:\qualitycontrol_service2\2024年第2季度抽查病历案例（+初复诊）'  # 放病历的文件夹路径
    json_result = r'E:\qualitycontrol_service2\test_result（2024年第2季度抽查病历案例（+初复诊））.json'  # 存储结果的路径
    files = os.listdir(file_path)

    # 批量操作pdf文件
    for file_name in files:
        # 读取单个文件内容
        pdf_path = (file_path + "\\" + file_name)
        # 读取pdf
        emr = read_pdf(pdf_path)
        emr.update(extend_emr)
        if '初诊' in file_name:
            emr['是否初诊'] = 0
        elif '复诊' in file_name:
            emr['是否初诊'] = 1

        print(emr)

        t = time.time()
        try:
            ans = emr_inspection.emr_quality_inspection(emr)
            # print(json.dumps(ans, indent=4, ensure_ascii=False))
        except Exception as e:
            ans = {'filename': file_name, 'quality': {'result': '质检失败', 'reason': str(e)}}

        # print('质检病历：', file_name)
        # print('质检时间为 %.2f s' % (time.time() - t))


        ans_all = {}
        ans_all['filename'] = file_name
        ans_all['quality'] = ans

        print(ans_all)

        # 输出存储为json文件，注意这里每一次运行之前调整json文件路径，才能保存新的json文件
        if os.path.exists(json_result) == True and os.path.getsize(json_result) != 0:
            f = open(json_result, "r", encoding='gbk')
            json_str = f.read()
            f.close()
            result_list = json.loads(json_str)
            result_list.append(ans_all)
            f = open(json_result, 'w')
            f.write(list_tran_json(result_list))
            f.close()
        else:
            result_list = []
            result_list.append(ans_all)
            f = open(json_result, 'w')
            f.write(list_tran_json(result_list))
            f.close()
