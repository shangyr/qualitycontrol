# 将多个知识图谱合并为一个知识图谱


import glob
import os

data_path = '/home/lixin/work/prompt-gnn/data/traditional_chinese_medicine'

entity_files = glob.glob(os.path.join(data_path,'entities*.txt'))
triple_files = glob.glob(os.path.join(data_path,'triples*.txt'))

entity_files.sort()
triple_files.sort()

assert len(entity_files) == len(triple_files)

entities = []
entity2type = {}
triples_set = set()
for entity_file,triple_file in zip(entity_files, triple_files):
    with open(entity_file,'r',encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            line = line.split('\t')[0]
            entity_name = line.split('..')[0]
            entity_type = line.split('..')[1]
            if entity_name not in entity2type:
                entity2type[entity_name] = entity_type

    with open(triple_file,'r',encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            head_entity = line.split('\t')[0].split('..')[0]
            head_entity_type = line.split('\t')[0].split('..')[1]
            relation = line.split('\t')[1]
            tail_entity = line.split('\t')[2].split('..')[0]
            tail_entity_type = line.split('\t')[2].split('..')[1]
            if head_entity not in entity2type:
                entity2type[head_entity] = head_entity_type
            if tail_entity not in entity2type:
                entity2type[tail_entity] = tail_entity_type
            
            triples_set.add(head_entity + '..' + entity2type[head_entity] + '\t' + \
                            relation + '\t' + tail_entity + '..' + entity2type[tail_entity])

entities = [entity + '..' + entity_type for entity, entity_type in entity2type.items()]
triples = list(triples_set)

with open(os.path.join(data_path,'triples.txt') ,'w',encoding='utf-8') as f:
    f.write('\n'.join(triples))
with open(os.path.join(data_path,'entities.txt'),'w',encoding='utf-8') as f:
    f.write('\n'.join(entities))
    

