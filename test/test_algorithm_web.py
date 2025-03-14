# coding:utf-8 -*-
import sys
import os

PROJECT_DIR = r'D:\qualitycontrol_service'
sys.path.append(PROJECT_DIR)
DIAGNOSIS_DIR = r'D:\qualitycontrol_service\src\diagnosis'
sys.path.append(DIAGNOSIS_DIR)

from src import EMRQualityInspectionAlgorithmapi
import json
from flask import Flask, request, jsonify

# 创建Flask应用
app = Flask(__name__)

# 模型加载和实例化
entity_dict_path = os.path.join(PROJECT_DIR, 'data_and_models\\entity-dict\\entities-v2.txt')
ner_model_path = os.path.join(PROJECT_DIR, 'data_and_models\\bert-base-chinese-ner')
graph_model_path = os.path.join(PROJECT_DIR, 'data_and_models\\knowledge-graph\\triples-v2.txt')
diagnosis_path = os.path.join(PROJECT_DIR, 'data_and_models\\diagnosis')
corrector_path = os.path.join(PROJECT_DIR, 'data_and_models\\bert_correct')
body_path = os.path.join(PROJECT_DIR, 'data_and_models\\diagnosis\\body.txt')  # 身体部位字典
acute_illness_path = os.path.join(PROJECT_DIR, 'data_and_models\\diagnosis\\acute_illness.txt')  # 急性病字典

emr_inspection = EMRQualityInspectionAlgorithmapi(entity_dict_path, ner_model_path, graph_model_path, diagnosis_path,
                                               corrector_path, acute_illness_path, body_path)

# 定义路由，post请求
@app.route('/process', methods=['POST'])
def process_data():
    # 获取JSON数据
    inputs = request.get_json()

    # 当前门诊病历数据
    emr = inputs['Medical_history']

    # 既往三月病历
    past_emr = inputs.get('anamnesis',[{}])

    # 需要处理的质控规则
    items = [item for item in inputs['Rule_numbers'].split(',')]

    # 每条规则需要的额外参数
    parms = inputs['data']

    # 单项否决的规则,这个之后看情况，可能会去掉这个判断
    votes = ['_item_1', '_item_2', '_item_8', '_item_33', '_item_50', '_item_52', '_item_53']

    # 输出的病历结果
    outs = []

    # 逐条规则判断
    for item, parm in zip(items, parms):
        try:
            # 既往病历放在参数中，需要时可从中取出
            parm['anamnesis'] = past_emr
            # 判断当前规则是否为单项否决规则
            if "_item_"+item in votes:
                ans = emr_inspection.emr_quality_inspection(emr, '_item_'+item, parm, is_veto=True)
                outs.append({"code": 0, "item": item, "data": json.dumps(ans, ensure_ascii=False), "message": ""})
            # 否则为单条质控规则
            else:
                ans = emr_inspection.emr_quality_inspection(emr, '_item_'+item, parm)
                outs.append({"code": 0, "item": item, "data": json.dumps(ans, ensure_ascii=False), "message": ""})
        except Exception as e:
            outs.append({"code": 1, "item": item, "data": {}, "message": str(e)})

    return outs


if __name__ == '__main__':
    app.run("0.0.0.0",8080, debug=True)

    """
    请求示例
    import requests
    import json
    
    data = json.load(open("数据.json", "r", encoding="utf-8"))
    headers = {
        "Content-Type": "application/json"
    }
    url = "http://127.0.0.1:8080/process"
    data = json.dumps(data)
    response = requests.post(url, headers=headers, data=data)
    print(json.loads(response.text))
    """
