import re
from .base_match import TrieTreeMatch
from .base import BaseGraph
from tqdm import tqdm


class ChineseMatch(TrieTreeMatch):
    def __init__(self, entity_file_path, entity_cache_path):
        super().__init__(entity_file_path, entity_cache_path)
    
    def _remove_neg_entities(self,entities:list,sentence:str) -> list:
        """
        去除否定实体
        Args:
            entities:实体列表
            sentence:句子
        Return:
            filter_entities:去否定实体之后的实体列表
        """

        neg_entities = re.findall(r"((否认)|(无))(.*?)(，|。|；|,|$|(\[SEP\]))",sentence)
        # 否认“肝炎，伤寒，结核”
        # neg_entities.extend(re.findall(r'((否认)|(无))“(.*?)”',sentence))
        neg_entities = ''.join([''.join(list(item)) for item in neg_entities]) # 电子病历中的否定实体

        # 也有一些正确的
        pos_entities = re.findall(r"无明显诱因(.*?)(，|。|；|,|$|(\[SEP\]))",neg_entities)
        pos_entities = ''.join([''.join(list(item)) for item in pos_entities])
        filter_entities = []
        for entity in entities:
            if entity not in neg_entities:
                filter_entities.append(entity)
            if entity in pos_entities:
                filter_entities.append(entity)
        return filter_entities

class ChineseGraph(BaseGraph):

    def __init__(self,kg_graph_path,entity_path, label_path,emr2kg_path = None, max_hop=3, entity_cache_path=None) -> None:
        """
        初始化一些数据
        Args:
            kg_graph_path:str
            内部格式如下
            普鲁斯病..疾病\t别名\t波型热..疾病
            慢性肺原性心脏病..疾病\t伴随疾病\t室性期前收缩..疾病
            ...

            logic_graph_path:str
            内部格式如下
            id1\t疾病1\t症状1\t症状2\t症状3...
            id2\t疾病2\t症状1\t症状2\t症状3...
            ...
            idn\t症状1\t疾病1\t疾病2\t疾病3...
            ...
            
            entity_path:str
            内部格式如下
            纳食减少..症状
            纳食减少..症状	4
            骨髓移植及外周血干细胞移植..治疗	4
            所有人群，年龄多为20～50岁..特定人群	4
            IgG2、IgG4缺陷..症状	4
            ACTH试验..检查	4

            label_path:str
            内部格式如下
            胆囊炎
            肺炎
            高血压
            关节炎

        Return:
            None
        """
        # super().__init__(graph_path, entity_path, emr2kg_path, label_path, max_hop)

        # h2t = {} # headid:[tail_id1,tail_id2,..]
        # t2h = {} # 
        # ht2rel = {} # {(h,t):relation(str),...}
        # rel2id = {} # {'伴随':0,...}

        self.match_model = ChineseMatch(entity_path, entity_cache_path)
        self.max_hop = max_hop

        self.entityname2type = self.match_model.entityname2type
        self.id2entity = self.match_model.id2entity
        self.h2t_prune_cache_path = re.sub(r'triples.*?.txt','h2t_prune.pkl',kg_graph_path)
        self.ignore_set = {'精神','饮食','SE','体重', '睡眠', '大小',
                         '受限','加重', '药物', '持续', '休息', '体温',
                         '血压','进食', '不适','右侧','药物治','血压病',
                         '急性','反复发作','全身', '手术', '困难','手术治','伴疼痛',
                         '预防接种史','接种史','预防接种'}
        self.emr_name = '电子病历'
        self.emr_co_name = '出现'
        self.yes_token = '对'
        self.no_token = '错'
        self.reverse_word = '反向'

        """
        1. 知识图谱
        """
        self._read_kg_file(kg_graph_path)
        
        # """
        # 2. 逻辑图
        # """
        # self._read_logic_file(logic_graph_path)
        """
        分类节点
        """
        self._read_label_file(label_path)

        self._pruning_h2t()

        self.split_sign = {'simple':'','reverse':'反向','diLsy_logic':['伴随','是','临床表现'],
                           'diZsy_logic':['伴随','是','诊断依据'],
                           'logic_reverse':['具有症状','伴随']}
        
        print('实体数量:',self.match_model.entity_size)
        print('二元关系数量:',len(self.ht2rel))
        print('二元关系平均节点度:',len(self.ht2rel)/self.match_model.entity_size)

    def generate_entity_name_by_text(self,sentence,ner_entities=None):

        origin_entity_names = self.match_model.find_entities(sentence)
        origin_entity_names = list(filter(lambda x:x not in self.ignore_set,origin_entity_names))

        # nodes,edges,nodes_type,edges_type,edges_prompt = self._genrate_graph_by_entity(origin_entity_names)

        return origin_entity_names
        # return nodes,edges,nodes_type,edges_type,edges_prompt,len(origin_entity_names)


def function(nodes,edges,nodes_type,edges_type,edges_prompt,graph):
    node_set = set()
    kg_node_num = 0
    for node in nodes:
        if type(node) is int:
            node_set.add(node)
            kg_node_num += 1
        else:
            for _node in node:
                node_set.add(_node)
    logic_type_id = 2 * len(graph.rel2id) + 2
    logic_count = edges_type.count(logic_type_id)


    _max_logic_num = 0
    for node in nodes:
        if type(node) is int:
            continue
        _max_logic_num = max(_max_logic_num,len(node))

    nodes_num = len(node_set)
    edges_num = len(edges_prompt)//2-logic_count
    logic_edges_num = logic_count
    ave_degree = edges_num / kg_node_num
    max_logic_num = _max_logic_num
    return nodes_num,edges_num,logic_edges_num,ave_degree,max_logic_num

import json
if __name__ == '__main__':
    kg_graph_path = '/home/lixin/work/diagnoisev3/data/electronic-medical-record-42/triplesv2.txt'
    logic_graph_path = '/home/lixin/work/diagnoisev3/data/electronic-medical-record-42/anss.txt'
    entity_path = '/home/lixin/work/diagnoisev3/data/electronic-medical-record-42/entities.txt'
    emr2kg_path = '/home/lixin/work/diagnoisev3/data/electronic-medical-record-42/emr2kg.pkl'
    label_path = '/home/lixin/work/diagnoisev3/data/electronic-medical-record-42/label2id.txt'
    train_path = '/home/lixin/work/diagnoisev3/data/electronic-medical-record-42/train.json'
    dev_path = '/home/lixin/work/diagnoisev3/data/electronic-medical-record-42/dev.json'
    test_path = '/home/lixin/work/diagnoisev3/data/electronic-medical-record-42/test.json'
    bert_path = 'bert-base-chinese'
    graph = Chinese42Graph(
        kg_graph_path=kg_graph_path,
        logic_graph_path=logic_graph_path,
        entity_path=entity_path,
        label_path=label_path,
        logic_node_num=2,
        max_hop=2
    )

    nodes_num = []
    edges_num = []
    logic_edges_num = []
    ave_degree = []
    max_logic_num = []
    ave_path = []
    ave_acc = []
    ave_path_num = []

    with open(test_path,'r',encoding='utf-8') as f:
        test = json.load(f)
    for item in tqdm(test):
        labels_id = [graph.match_model.entity2id[label.split('..')[0]] for label in item['label']]
        text = item['主诉'] + item['现病史']+item['既往史']
        nodes,edges,nodes_type,edges_type,edges_prompt,_ave_path,_ave_acc,path_num = graph.generate_graph_by_text(text,labels_id=labels_id)
        
        _nodes_num,_edge_num,_logic_edges_num,_ave_degree,_max_logic_num = function(nodes,edges,nodes_type,edges_type,edges_prompt,graph)
        nodes_num.append(_nodes_num)
        edges_num.append(_edge_num)
        logic_edges_num.append(_logic_edges_num)
        ave_degree.append(_ave_degree)
        max_logic_num.append(_max_logic_num)
        ave_path.append(_ave_path)
        ave_acc.append(_ave_acc)
        ave_path_num.append(path_num)

    # with open(dev_path,'r',encoding='utf-8') as f:
    #     test = json.load(f)
    # for item in tqdm(test):
    #     labels_id = [graph.match_model.entity2id[label.split('..')[0]] for label in item['label']]
    #     text = item['主诉'] + item['现病史']+item['既往史']
    #     nodes,edges,nodes_type,edges_type,edges_prompt,_ave_path,_ave_acc,path_num = graph.generate_graph_by_text(text,labels_id=labels_id)
        
    #     _nodes_num,_edge_num,_logic_edges_num,_ave_degree,_max_logic_num = function(nodes,edges,nodes_type,edges_type,edges_prompt,graph)
    #     nodes_num.append(_nodes_num)
    #     edges_num.append(_edge_num)
    #     logic_edges_num.append(_logic_edges_num)
    #     ave_degree.append(_ave_degree)
    #     max_logic_num.append(_max_logic_num)
    #     ave_path.append(_ave_path)
    #     ave_acc.append(_ave_acc)
    #     ave_path_num.append(path_num)

    # with open(train_path,'r',encoding='utf-8') as f:
    #     test = json.load(f)
    # for item in tqdm(test):
    #     labels_id = [graph.match_model.entity2id[label.split('..')[0]] for label in item['label']]
    #     text = item['主诉'] + item['现病史']+item['既往史']
    #     nodes,edges,nodes_type,edges_type,edges_prompt,_ave_path,_ave_acc,path_num = graph.generate_graph_by_text(text,labels_id=labels_id)
        
    #     _nodes_num,_edge_num,_logic_edges_num,_ave_degree,_max_logic_num = function(nodes,edges,nodes_type,edges_type,edges_prompt,graph)
    #     nodes_num.append(_nodes_num)
    #     edges_num.append(_edge_num)
    #     logic_edges_num.append(_logic_edges_num)
    #     ave_degree.append(_ave_degree)
    #     max_logic_num.append(_max_logic_num)
    #     ave_path.append(_ave_path)
    #     ave_acc.append(_ave_acc)
    #     ave_path_num.append(path_num)
    print(sum(nodes_num)/len(nodes_num))
    print(sum(edges_num)/len(edges_num))
    print(sum(logic_edges_num)/len(logic_edges_num))
    print(sum(ave_degree)/len(ave_degree))
    print(sum(max_logic_num)/len(max_logic_num))
    print(sum(ave_path)/len(ave_path))
    print(sum(ave_acc)/len(ave_acc))
    print(sum(ave_path_num)/len(ave_path_num))

