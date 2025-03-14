# 通过excel表提取，难度会一小些
import pandas as pd
import re
from tqdm import tqdm

from ..extract_entity import MatchExtract
from .expand_entity import text_process,valid_text

class UnionFind:
    def __init__(self, n):
        # 初始化并查集，每个元素各自为一个集合，父节点指向自己
        self.parent = [i for i in range(n)]
        self.rank = [0] * n  # 记录树的深度

    def find(self, x):
        # 查找根节点
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 路径压缩
        return self.parent[x]

    def union(self, x, y):
        # 合并两个集合
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x != root_y:
            # 将深度较小的树合并到深度较大的树上
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_x] = root_y
                self.rank[root_y] += 1

    def get_sets(self):
        # 获取每个集合中的元素
        sets_dict = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            if root not in sets_dict:
                sets_dict[root] = [i]
            else:
                sets_dict[root].append(i)
        return list(sets_dict.values())

class ExcelExtract:
    def __init__(self, excel_file1, excel_file2, entity_dict_path, triple_path) -> None:
        self.excel_file = excel_file1
        self.excel_file2 = excel_file2
        self.triple_path = triple_path
        self._data = pd.read_excel(self.excel_file)
        self._data2 = pd.read_excel(self.excel_file2)
        self.match_model = MatchExtract(entity_dict_path)

    def extract_knowledge(self):
        print('processing',self.excel_file)
        entity2count = {}
        keys = self._data.keys()
        data_size = len(self._data[keys[0]])
        triples = set()
        for i in tqdm(range(data_size)):

            disease = self._data['名称'][i].strip()
            other_diseases = re.split(r'[,|，|；|;]',self._data['别名'][i].strip())
            diseases = set(other_diseases + [disease])

            medicals = []
            if not pd.isnull(self._data['常用药物'][i]):
                medicals = re.split(r'[,|，|；|;]',self._data['常用药物'][i].strip())

            _description_text = ''
            if not pd.isna(self._data['症状'][i]):
                _description_text = self._data['症状'][i]
            
            if not pd.isna(self._data['病因'][i]):
                _description_text += self._data['病因'][i]
            
            if not pd.isna(self._data['并发症'][i]):
                _description_text += self._data['并发症'][i]

            entities = self.match_model.find_entities(_description_text)
            
            symptoms = set()
            rel_diseases = set()
            for entity in entities['pos']:
                if entity[1] == '疾病' or entity[1] == '疾病39':
                    rel_diseases.add(entity)
                if entity[1] == '症状' or entity[1] == '症状39':
                    symptoms.add(entity)
                if entity[0] + '\t' +entity[1] not in entity2count:
                    entity2count[entity[0] + '\t' +entity[1]] = 0
                entity2count[entity[0] + '\t' +entity[1]] += 1
            
            # for entity in entities['neg']:
            #     if entity[1] == '疾病' or entity[1] == '疾病39':
            #         diseases.add(entity[0])
            #     if entity[1] == '症状' or entity[1] == '症状39':
            #         symptoms.add(entity[0])

            # 添加三元组
            # 症状 - 疾病 疾病 - 药物 疾病 - 相关疾病
            e2t = self.match_model.entityname2type
            for disease in diseases:
                for symptom in symptoms:
                    disease = text_process(disease)
                    # symptom[0] = text_process(symptom[0])
                    if not valid_text(disease, '疾病'):
                        continue
                    triples.add((disease, e2t[disease], '具有症状', symptom[0], symptom[1]))

            for disease in diseases:
                for rel_disease in rel_diseases:
                    disease = text_process(disease)
                    # rel_disease[0] = text_process(rel_disease[0])
                    if not valid_text(disease, '疾病'):
                        continue
                    triples.add((disease, e2t[disease], '伴随疾病', rel_disease[0], rel_disease[1]))
            
            for disease in diseases:
                for medical in medicals:
                    disease = text_process(disease)
                    medical = text_process(medical)
                    if not valid_text(disease,'疾病') or not valid_text(medical,'药品'):
                        continue

                    if len(medical) <= 1: continue 
                    triples.add((disease, e2t[disease], '推荐药品', medical, e2t[medical]))

            for d1 in diseases:
                for d2 in diseases:
                    d1 = text_process(d1)
                    d2 = text_process(d2)
                    if d1 == d2: continue
                    if not valid_text(d1,'疾病') or not valid_text(d2,'疾病'):
                        continue
                    triples.add((d1, e2t[d1], '别名', d2, e2t[d2]))

        entities = list(entity2count.keys())
        entities.sort(key=lambda x:entity2count[x],reverse=True)
        entities = [e + '\t' + str(entity2count[e]) for e in entities]
        with open('output.txt','w',encoding='utf-8') as f:
            f.write('\n'.join(entities))
        return triples

    def extract_knowledge2(self,):
        print('processing',self.excel_file2)
        entity2count = {}
        keys = self._data2.keys()
        data_size = len(self._data2[keys[0]])
        triples = set()
        for i in tqdm(range(data_size)):

            disease = self._data2['字段1_文本'][i]
            raw_rel_disease = self._data2['并发症'][i]
            raw_symptoms = self._data2['症状'][i]
            raw_medicals = self._data2['常用药品'][i]
            raw_tests = self._data2['检查'][i]

            rel_diseases = []
            if not pd.isna(raw_rel_disease):
                rel_diseases = re.findall(r" {10,100}(.*?)\n",raw_rel_disease)
                rel_diseases = list(filter(lambda x:x!='' and x!='并发症', rel_diseases))

            symptoms,match_symptoms = [],[]
            if not pd.isna(raw_symptoms):
                symptoms = re.findall(r" {40,100}(.*?)\n",raw_symptoms)
                symptoms = list(filter(lambda x:x!='',symptoms))
                match_symptoms = self.match_model.find_entities(raw_symptoms)
                match_symptoms = [(s[0],s[1]) for s in match_symptoms['pos'] if '症状' in s[1]]

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
            
            # 添加三元组
            e2t = self.match_model.entityname2type
            disease = text_process(disease)
            if not valid_text(disease,'疾病'):
                continue

            for rel_disease in rel_diseases:
                rel_disease = text_process(rel_disease)
                if not valid_text(rel_disease,'疾病'):
                    continue
                triples.add((disease, e2t[disease], '伴随疾病', rel_disease,e2t[rel_disease]))
            # 症状
            for symptom in symptoms:
                symptom = text_process(symptom)
                if not valid_text(symptom,'症状'):
                    continue
                triples.add((disease, e2t[disease] , '具有症状', symptom, e2t[symptom]))
            for symptom in match_symptoms:
                triples.add((disease, e2t[disease] , '具有症状', symptom[0], symptom[1]))
                if symptom[0] + '\t' +symptom[1] not in entity2count:
                    entity2count[symptom[0] + '\t' +symptom[1]] = 0
                entity2count[symptom[0] + '\t' +symptom[1]] += 1
            # 药品
            for medical in medicals:
                medical = text_process(medical)
                if not valid_text(medical,'药品'):
                    continue
                triples.add((disease, e2t[disease] , '推荐药品', medical, e2t[medical]))
            # 检查
            for test in tests:
                test = text_process(test)
                if not valid_text(test,'检查'):
                    continue
                triples.add((disease, e2t[disease] , '采取检查', test, e2t[test]))
        entities = list(entity2count.keys())
        entities.sort(key=lambda x:entity2count[x],reverse=True)
        entities = [e + '\t' + str(entity2count[e]) for e in entities]
        with open('output2.txt','w',encoding='utf-8') as f:
            f.write('\n'.join(entities))
        return triples

    def origin_triples(self,):
        """
        合并三元组
        """
        print('processing',self.triple_path)
        triples = set()
        with open(self.triple_path,'r',encoding='utf-8') as f:
            for line in tqdm(f):
                line = line.strip()
                if not line: continue
                triple = line.split('\t')
                head_name = triple[0].split('..')[0].lower()
                head_type = triple[0].split('..')[1]
                relation  = triple[1]
                tail_name = triple[2].split('..')[0].lower()
                tail_type = triple[2].split('..')[1]
                if head_name not in self.match_model.entity2id:
                    continue
                if tail_name not in self.match_model.entity2id:
                    continue
                head_type = self.match_model.entityname2type[head_name]
                tail_type = self.match_model.entityname2type[tail_name]
                triples.add((head_name, head_type, relation, tail_name, tail_type))

        return triples
    
    def expand_triples(self, triples):
        """
        基于逻辑扩充知识图谱
        """
        print('logit_triples')
        h2t = {}
        t2h = {}
        uf = UnionFind(self.match_model.entity_size)
        for triple in triples:
            if triple[0] not in h2t:
                h2t[triple[0]] = set()
            if triple[3] not in t2h:
                t2h[triple[3]] = set()
            h2t[triple[0]].add((triple[2], triple[3]))
            t2h[triple[3]].add((triple[2], triple[0]))

            if triple[2] == '别名':
                uf.union(self.match_model.entity2id[triple[0]], self.match_model.entity2id[triple[3]])
        sets = uf.get_sets()

        expand_triples = set()
        triples_num = 0
        h2t_new = {}
        t2h_new = {}
        for _set in tqdm(sets):
            heads = set()
            tails = set()
            for entity_id in _set:
                entity_name = self.match_model.id2entity[entity_id]
                if entity_name in t2h:
                    for h in t2h[entity_name]:
                        if h[0] == '别名': continue
                        heads.add(h)
                if entity_name in h2t:
                    for t in h2t[entity_name]:
                        if t[0] == '别名': continue
                        tails.add(t)

            for entity_id in _set:
                entity_name = self.match_model.id2entity[entity_id]
                h2t_new[entity_name] = tails
                t2h_new[entity_name] = heads
                triples_num += len(tails)

        e2t = self.match_model.entityname2type
        for h in tqdm(h2t_new):
            for t in h2t_new[h]:
                expand_triples.add((h, e2t[h], t[0], t[1], e2t[t[1]]))
                if len(expand_triples) % 1_000_000 == 0:
                    print(len(expand_triples), 'triples.')
        for t in tqdm(t2h_new):
            for h in t2h_new[t]:
                expand_triples.add((h[1], e2t[h[1]], h[0], t, e2t[t]))
                if len(expand_triples) % 1_000_000 == 0:
                    print(len(expand_triples), 'triples.')
        triples = triples | expand_triples
        return expand_triples
    
    def pipeline(self, outpath):
        triples = self.extract_knowledge()
        triples2 = self.extract_knowledge2()
        origin_triples = self.origin_triples()
        triples = triples | triples2 | origin_triples
        # triples = self.expand_triples(triples)
        triples = list(triples)
        triples.sort(key=lambda x:x[2]+x[1]+x[4])
        triples = [f"{t[0]}..{t[1]}\t{t[2]}\t{t[3]}..{t[4]}" for t in triples]
        with open(outpath,'w',encoding='utf-8') as f:
            f.write('\n'.join(triples))