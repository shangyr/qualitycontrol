# coding:utf-8 -*-
import sys
import os

PROJECT_DIR = r'D:\qualitycontrol_service'
sys.path.append(PROJECT_DIR)
DIAGNOSIS_DIR = r'D:\qualitycontrol_service\src\diagnosis'
sys.path.append(DIAGNOSIS_DIR)

from src import EMRQualityInspectionAlgorithmapi
import time
import json

if __name__ == '__main__':
    # 接口测试
    # 单条质控规则api测试
    entity_dict_path = os.path.join(PROJECT_DIR, 'data_and_models\\entity-dict\\entities-v2.txt')
    ner_model_path = os.path.join(PROJECT_DIR, 'data_and_models\\bert-base-chinese-ner')
    graph_model_path = os.path.join(PROJECT_DIR, 'data_and_models\\knowledge-graph\\triples-v2.txt')
    diagnosis_path = os.path.join(PROJECT_DIR, 'data_and_models\\diagnosis')
    corrector_path = os.path.join(PROJECT_DIR, 'data_and_models\\bert_correct')
    body_path = os.path.join(PROJECT_DIR, 'data_and_models\\diagnosis\\body.txt')  # 身体部位字典
    acute_illness_path = os.path.join(PROJECT_DIR, 'data_and_models\\diagnosis\\acute_illness.txt')  # 急性病字典

    emr_inspection = EMRQualityInspectionAlgorithmapi(entity_dict_path, ner_model_path, graph_model_path,
                                                      diagnosis_path,
                                                      corrector_path, acute_illness_path, body_path)

    # 读取输入
    inputs = json.load(open(r'数据.json', 'r', encoding='utf-8'))
    # 当前门诊病历数据
    emr = inputs['Medical_history']

    # 既往三月病历
    past_emr = inputs.get('anamnesis',[{}])

    # 需要处理的质控规则
    items = ['_item_'+item for item in inputs['Rule_numbers'].split(',')]

    # 每条规则需要的额外参数
    parms = inputs['data']

    # 单项否决的规则,这个之后看情况，可能会去掉这个判断
    votes = ['_item_1', '_item_2',  '_item_8', '_item_33', '_item_50', '_item_52', '_item_53']

    # 输出的病历结果
    outs = []
    # 逐条规则判断
    for item, parm in zip(items,parms):
        # 既往病历放在参数中，需要时可从中取出
        parm['anamnesis'] = past_emr
        # 判断当前规则是否为单项否决规则
        if item in votes:
            ans = emr_inspection.emr_quality_inspection(emr, item,parm, is_veto=True)
            outs.append(ans)
        # 否则为单条质控规则
        else:
            ans = emr_inspection.emr_quality_inspection(emr, item, parm)
            outs.append(ans)
        # print(item, parm)
    print(outs)



    # # 演示实例
    # # 演示示例1：单项否决规则，输入为emr:string, item:str, is_veto:bool
    # # item:'_item_17' 为质控规则序号，从_item_1-_item_58中选择，单项否决规则需满足 is_veto=True & item为单否规则
    # ans = emr_inspection.emr_quality_inspection(emr,item='_item_17',is_veto=True)
    # print(ans)
    #
    # # 演示示例2：单项规则，输入为emr:string, item:str,
    # # item:'_item_9' 为指控规则序号，从_item_1-_item_58中选择
    # ans = emr_inspection.emr_quality_inspection(emr, item='_item_9')
    # print(ans)
    #
    # # 演示示例3：疾病辅助诊断模型
    # # item: 'predict' 为疾病辅助诊断模型序号，
    # ans = emr_inspection.emr_quality_inspection(emr, item='predict')
    # print(ans)