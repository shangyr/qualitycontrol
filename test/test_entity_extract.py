# coding:utf-8 -*-
import sys
import os
import json

PROJECT_DIR = 'C:\\Users\\asndj\\Desktop\\2023-12-29\\病历质检\\qualitycontrol_service'
sys.path.append(PROJECT_DIR)

from src.extract_entity import MatchExtract
import time

entity_dict_path = os.path.join(PROJECT_DIR, 'data_and_models\\entity-dict\\entities-v2.txt')

# match_model = MatchExtract(entity_dict_path)
# emr_file = 'C:\\Users\\asndj\\Desktop\\2023-12-29\\chinese\\train2.json'
# with open(emr_file,'r',encoding='utf-8') as f:
#     emrs = json.load(f)
# entity_count = {}
# for emr in emrs:
#     text = emr['主诉'] + emr['既往史'] + emr['现病史']
#     ans = match_model.find_entities(text)
#     for key in ans:
#         for entity in ans[key]:
#             if entity[0] not in entity_count:
#                 entity_count[entity[0]] = 0
#             entity_count[entity[0]] += 1
# entities = [(e,n) for e,n in entity_count.items()]
# entities.sort(key=lambda x:x[1],reverse=True)
text = '患者14年前无明显诱因下出现口干多饮，每日饮开水2瓶，夜间小便增多至2'
# ans = match_model.find_entities(text)
# print()

from src.extract_entity.ner_cmed_extract import NERCmedExtract
model_path = os.path.join(PROJECT_DIR, 'data_and_models\\cmed-ner')

model = NERCmedExtract(model_path)

text = '1.意识丧失、抽搐，即Adams-Stokes综合征。2.面色苍白或青紫，脉搏消失，心音听不到，血压为零。3.如不及时抢救，随之呼吸、心跳停止。/min以上的患者，也可见于反复型患者。有报道小儿患者约半数有呕吐、腹痛等症状，原因不清。,现悲观厌世、绝望、幻觉妄想、身体功能减退、并伴有严重的自杀企图，甚至自杀行为。'
text = text.replace(' ','')
ans = model.find_entities(text)
print()