from utils.graph import Graph
import os
import json

data_path = '/home/lixin/work/diagnoisev3/data/electronic-medical-record-42'


graph_path1 = os.path.join(data_path, "anss.txt")
graph_path2 = os.path.join(data_path, "triples.txt")
entity_path = os.path.join(data_path, "entities.txt")
label_idx_path = os.path.join(data_path, "label2id.txt")

train_path = os.path.join(data_path, f"train.json")
dev_path = os.path.join(data_path, f"dev.json")
test_path = os.path.join(data_path, f"test.json")

graph = Graph(graph_path1,graph_path2,entity_path,label_idx_path)

with open(dev_path,'r',encoding='utf-8') as f:
    data = json.load(f)
all = 0
acc = 0
for i in range(len(data)):
    dic = data[i]
    if '主诉' not in dic["主诉"]:
        dic['主诉'] = '主诉：'+dic['主诉']
    if '现病史' not in dic["现病史"]:
        dic['现病史'] = '现病史：'+dic['现病史']
    if '既往史' not in dic["既往史"]:
        dic['既往史'] = '既往史：'+dic['既往史']
    chief_complaint = '性别：' + dic['性别'] + '；年龄：'+ dic['年龄'] + '；'+ dic["主诉"]
    now_history, past_history = dic["现病史"], dic["既往史"]
    raw_doc = chief_complaint + '[SEP]' + now_history + '[SEP]' + past_history

    nodes, edges, node_types, edge_types, edge_prompts, num_direct_edges, num_single_nodes = graph.get_graph(raw_doc)
    nodes_name = [graph.match_model.id2entity[e] for e in nodes if type(e) is int]
    # print(nodes_name)
    
    for label in dic['label']:
        if label.split('..')[0] in nodes_name:
            acc += 1
        all += 1
    # print(i)
    # print(acc / all)

print(acc / all)


with open(test_path,'r',encoding='utf-8') as f:
    data = json.load(f)
all = 0
acc = 0
for i in range(len(data)):
    dic = data[i]
    if '主诉' not in dic["主诉"]:
        dic['主诉'] = '主诉：'+dic['主诉']
    if '现病史' not in dic["现病史"]:
        dic['现病史'] = '现病史：'+dic['现病史']
    if '既往史' not in dic["既往史"]:
        dic['既往史'] = '既往史：'+dic['既往史']
    chief_complaint = '性别：' + dic['性别'] + '；年龄：'+ dic['年龄'] + '；'+ dic["主诉"]
    now_history, past_history = dic["现病史"], dic["既往史"]
    raw_doc = chief_complaint + '[SEP]' + now_history + '[SEP]' + past_history

    nodes, edges, node_types, edge_types, edge_prompts, num_direct_edges, num_single_nodes = graph.get_graph(raw_doc)
    nodes_name = [graph.match_model.id2entity[e] for e in nodes if type(e) is int]
    # print(nodes_name)
    
    for label in dic['label']:
        if label.split('..')[0] in nodes_name:
            acc += 1
        all += 1
print(acc / all)