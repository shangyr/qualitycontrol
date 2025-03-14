from abc import ABC,abstractmethod
import pickle as pkl
import os
import random
import copy
import time
class BaseGraph(ABC):
    """
    图
    """
    @abstractmethod
    def __init__(self,**args) -> None:
        pass

    def _search(self,cur_node,cur_trie_tree,cur_level):
        if cur_node in self.label_nodes and cur_level == 0 and cur_trie_tree is not None:
            if cur_node not in cur_trie_tree:
                cur_trie_tree[cur_node] = None
            return
        if cur_node not in self.h2t_prune:
            return
        if cur_trie_tree is None:
            return
        if cur_node not in cur_trie_tree:
            cur_trie_tree[cur_node] = {}
        
        for i in range(1,cur_level+1):
            if i not in self.h2t_prune[cur_node]:
                continue
            for node in self.h2t_prune[cur_node][i]:
                self._search(node,cur_trie_tree[cur_node],i-1)
    
    def _search_trie_tree(self,trie_tree):
        all_paths = []
        def _bfs_search(path,cur_trie_tree):
            if cur_trie_tree is None:
                all_paths.append(path[:])
                return
            for node in cur_trie_tree:
                if node not in path:
                    path.append(node)
                    _bfs_search(path,cur_trie_tree[node])
                    path.pop()

        for node in trie_tree:
            _bfs_search([node],trie_tree[node])
        return all_paths

    def entity_num(self):
        return len(self.match_model.entity2id)

    def print_path(self,path):
        for p in path:
            print(self.match_model.id2entity[p],end=" ")
        print()

    def path_to_str(self,path):
        ans = ''
        for p in path:
            ans += self.match_model.id2entity[p] + ' '
        return ans

    def _pruning_h2t(self):
        # if os.path.exists(self.h2t_prune_cache_path):
        if False:
            with open(self.h2t_prune_cache_path,'rb') as f:
                data = pkl.load(f)
                self.node_prune = data['node_prune']
                self.entity_step2label = data['entity_step2label']
        else:
            self.h2t_prune = {}
            visited_node = set()
            def _loop_layer(cur_node,step_num):
                if step_num > self.max_hop-2:
                    return
                flag = f"{cur_node}-{step_num + 1}"
                if cur_node in self.t2h and flag not in visited_node:
                    visited_node.add(flag)
                    for next_node in self.t2h[cur_node]:
                        if next_node not in self.h2t_prune:
                            self.h2t_prune[next_node] = {}
                        if step_num + 1 not in self.h2t_prune[next_node]:
                            self.h2t_prune[next_node][step_num + 1] = set()
                        self.h2t_prune[next_node][step_num + 1].add(cur_node)
                        _loop_layer(next_node,step_num + 1)

            for label_node in self.label_nodes:
                _loop_layer(label_node,0)
            data = {'h2t_prune':self.h2t_prune}
            with open(self.h2t_prune_cache_path,'wb') as f:
                pkl.dump(data,f)
    @abstractmethod
    def generate_entity_name_by_text(self,sentence,ner_entities=None):
            
        pass

    def generate_graph_by_text(self,sentence,ner_entities=None,return_logic = True,labels_id=None):
        """
        通过症状实体得到相应的图
        Args:
            sentence: 电子病历文本
        Return:
            nodes: 实体转化为数字 [0,1,2,3] 有数字有元组，元组表示超图 emr节点为第一个0
            edges: 边 [[1,2,0,1],[2,3,1,3]] 下标按照node下标
            nodes_type: 节点类型 [0,1,2,3,4]  emr节点为第一个0
            edges_type: 边类型 [1,2,3,4] 边类型
            edges_prompt: ['实体1关系实体2',...] 边对应的prompt
        """
        origin_entity_names = self.generate_entity_name_by_text(sentence,ner_entities)
        if labels_id is not None:
            nodes,edges,nodes_type,edges_type,edges_prompt,ave_path,ave_acc,path_num = self._generate_graph_by_entity(origin_entity_names,
                                                                                        return_logic=return_logic,
                                                                                        labels_id=labels_id)
            return nodes,edges,nodes_type,edges_type,edges_prompt,ave_path,ave_acc,path_num
        
        else:
            nodes,edges,nodes_type,edges_type,edges_prompt = self._generate_graph_by_entity(origin_entity_names,
                                                                                        return_logic=return_logic,
                                                                                        labels_id=labels_id)
            return nodes,edges,nodes_type,edges_type,edges_prompt
    
    def _generate_graph_by_entity(self,origin_entity_names,return_logic=True,labels_id=None):
        """
        根据实体名字和知识图谱构建出个性化电子病历子图,由于剪枝只能找通往标签节点的路径
        Args:
            origin_entity_names: ['咳嗽','感冒',...]
        Return:
            nodes:节点 [[1],[2],[3],[4,5],...] 只有1个元素的列表是图谱知识 多个是逻辑知识
            edges:边 [[1,2,3,...],[0,1,2,...]]
            nodes_type:节点类型 [0,1,2,3,1] len(nodes_types)=len(nodes)+1
            edges_type:边类型 [2,3,4,2]
            edges_prompt:边对应的prompt['高血压导致糖尿病',...]
        """
        threshold_num = self.logic_node_num
        entity_ids = []
        for entity_name in origin_entity_names:
            if entity_name not in self.match_model.entity2id:
                print(entity_name)
            else:
                entity_ids.append(self.match_model.entity2id[entity_name])
        # entity_ids = [self.match_model.entity2id[entity] for entity in entity_names]
        all_paths = []

        edges_with_prompt = [[],[]]     # 每条边[[1,2,3,4],[2,3,4,5]]
        edges_without_prompt = [[],[]]     # 每条边[[1,2,3,4],[2,3,4,5]]
        edges_type_with_prompt = []     # 每条边[[1,2,3,4],[2,3,4,5]]
        edges_type_without_prompt = []     # 每条边[[1,2,3,4],[2,3,4,5]]

        edges_prompt = []   # 每条边的prompt
        nodes_type = [0]    # [1,2,3,4]
        entity2edge = {}    # 实体id到edge中实体id 0为emr在图中的表示

        rel2id = self.rel2id
        type2id = self.match_model.type2id.copy()
        for entity_type in type2id:
            type2id[entity_type] += 1 # 0 留给emr电子病历

        rel_type1 = 2 * len(rel2id)     # 病历->症状
        rel_type2 = 2 * len(rel2id) + 1 # 症状->病历
        rel_type3 = 2 * len(rel2id) + 2 # 逻辑症状->疾病
        rel_type4 = 2 * len(rel2id) + 3 # 疾病 -> 逻辑症状
        rel_type5 = 2 * len(rel2id) + 4 # 病历->其他节点
        rel_type6 = 2 * len(rel2id) + 5 # 其他节点->病历
        rel_type7 = 2 * len(rel2id) + 6 # 电子病历->标签
        rel_type8 = 2 * len(rel2id) + 7 # 标签->电子病历

        """
        加入所有疾病节点
        """
        for label_id in self.label_nodes:
            entity2edge[label_id] = len(entity2edge) + 1
            nodes_type.append(type2id[self.entityname2type[self.id2entity[label_id]]])
            # 加入边
            edges_without_prompt[0].append(0)
            edges_without_prompt[1].append(entity2edge[label_id])
            edges_without_prompt[0].append(entity2edge[label_id])
            edges_without_prompt[1].append(0)
            edges_type_without_prompt.append(rel_type7) # 边的类型
            edges_type_without_prompt.append(rel_type8) # 边的类型

        """
        处理图谱知识
        """
        # 找出所有路径
        # print('start',time.time())
        trie_tree = {}
        for entity_id in entity_ids:
            self._search(entity_id,trie_tree,self.max_hop-1)
            
        # print('mid',time.time())
        all_paths = self._search_trie_tree(trie_tree)
        # print('end',time.time())

        if labels_id is not None:
            acc_labels_id = set()
            for path in all_paths:
                if path[-1] in labels_id:
                    acc_labels_id.add(path[-1])
            ave_path_length = sum([len(path) for path in all_paths]) / (len(all_paths) + 1e-3)
        print('all_paths:',len(all_paths))
        all_path_str = []
        for path in all_paths:
            all_path_str.append(self.path_to_str(path))
        print()

        # 添加所有电子病历实体
        for path in all_paths:
            entity_id = path[0]
            if entity_id not in entity2edge:
                entity2edge[entity_id] = len(entity2edge) + 1
                nodes_type.append(type2id[self.entityname2type[self.id2entity[entity_id]]])
                # 加入边
                edges_without_prompt[0].append(0)
                edges_without_prompt[1].append(entity2edge[entity_id])
                edges_without_prompt[0].append(entity2edge[entity_id])
                edges_without_prompt[1].append(0)

                edges_type_without_prompt.append(rel_type1) # 边的类型
                edges_type_without_prompt.append(rel_type2) # 边的类型

        # 添加不相关实体
        for path in all_paths:
            # self.print_path(path)
            for i,entity_id in enumerate(path[1:]):
                if entity_id not in entity2edge:
                    # print(self.id2entity[entity_id])
                    entity2edge[entity_id] = len(entity2edge) + 1
                    nodes_type.append(type2id[self.entityname2type[self.id2entity[entity_id]]])
                    # 加入边
                    edges_without_prompt[0].append(0)
                    edges_without_prompt[1].append(entity2edge[entity_id])
                    edges_without_prompt[0].append(entity2edge[entity_id])
                    edges_without_prompt[1].append(0)

                    edges_type_without_prompt.append(rel_type5) # 边的类型
                    edges_type_without_prompt.append(rel_type6) # 边的类型

        edges_set = set()
        # 添加所有路径
        for path in all_paths:
            for h,t in zip(path[:-1],path[1:]):
                if (h,t) not in edges_set:
                    edges_set.add((h,t))
                    edges_with_prompt[0].append(entity2edge[h])
                    edges_with_prompt[1].append(entity2edge[t])
                    edges_type_with_prompt.append(rel2id[self.ht2rel[(h,t)]])
                    edges_prompt.append(self.id2entity[h]+ self.split_sign['simple'] + self.ht2rel[(h,t)]
                                    + self.split_sign['simple'] + self.id2entity[t])
                if (t,h) not in edges_set:
                    edges_set.add((t,h))
                    # 这块报错删除缓存即可 或者_pruning_h2t 第一行改为 if False: 
                    edges_with_prompt[0].append(entity2edge[t])
                    edges_with_prompt[1].append(entity2edge[h])
                    if (t,h) in self.ht2rel:
                        edges_prompt.append(self.id2entity[t] + self.split_sign['simple'] + self.ht2rel[(t,h)]
                                            + self.split_sign['simple'] + self.id2entity[h])
                        edges_type_with_prompt.append(rel2id[self.ht2rel[(t,h)]])
                    else:
                        edges_prompt.append(self.id2entity[t] + self.split_sign['reverse'] + self.ht2rel[(h,t)]
                                            + self.split_sign['simple'] + self.id2entity[h])
                        edges_type_with_prompt.append(len(rel2id)+rel2id[self.ht2rel[(h,t)]])

        if return_logic:
            """
            处理逻辑知识
            """
            disease2symptoms = {}
            # {'1-diLsy':[1,2,3,4],...}
            for entity_id in entity_ids:
                if entity_id not in self.logic_dict['sy2di']:
                    continue
                if entity_id in self.label_nodes: # 不加入标签
                    continue
                disease_ids = self.logic_dict['sy2di'][entity_id] # []
                for disease_id_with_type in disease_ids:
                    if disease_id_with_type not in disease2symptoms:
                        disease2symptoms[disease_id_with_type] = []
                    disease2symptoms[disease_id_with_type].append(entity_id)

            # 加入所有疾病
            # 去重
            disease_symptoms_set = set()
            keys = list(disease2symptoms.keys())
            for dkey in keys:
                key = dkey.split('-')[0] + dkey.split('-')[1]
                symptom_ids = disease2symptoms[dkey]
                symptom_ids = [str(i) for i in symptom_ids]
                key += '-'.join(symptom_ids)
                if key not in disease_symptoms_set:
                    disease_symptoms_set.add(key)
                else:
                    del disease2symptoms[dkey]

            for disease_id_with_type in disease2symptoms:
                disease2symptoms[disease_id_with_type] = list(set(disease2symptoms[disease_id_with_type]))
                disease_id = int(disease_id_with_type.split('-')[0])
                if disease_id not in self.label_nodes:
                    continue
                if labels_id is not None and disease_id in labels_id:
                    acc_labels_id.add(disease_id)

            # 加入所有症状
            for disease_id_with_type in disease2symptoms:
                disease2symptoms[disease_id_with_type] = list(set(disease2symptoms[disease_id_with_type]))
                disease_id = int(disease_id_with_type.split('-')[0])
                if disease_id not in self.label_nodes:
                    continue
                symptom_ids = disease2symptoms[disease_id_with_type]
                if len(symptom_ids) >= threshold_num and tuple(symptom_ids) not in entity2edge:

                    # 加入症状节点
                    entity2edge[tuple(symptom_ids)] = len(entity2edge) + 1
                    nodes_type.append(len(type2id) + 1) # 逻辑实体编号
                    edges_without_prompt[0].append(0)
                    edges_without_prompt[1].append(entity2edge[tuple(symptom_ids)])
                    edges_without_prompt[0].append(entity2edge[tuple(symptom_ids)])
                    edges_without_prompt[1].append(0)
                    edges_type_without_prompt.append(rel_type1)
                    edges_type_without_prompt.append(rel_type2)

            # 加入所有边
            for disease_id_with_type in disease2symptoms:
                if len(disease2symptoms[disease_id_with_type]) >= threshold_num:

                    disease_id = int(disease_id_with_type.split('-')[0])
                    symptom_ids = disease2symptoms[disease_id_with_type]
                    if disease_id not in self.label_nodes:
                        continue
                    disease_type = disease_id_with_type.split('-')[1]
                    edges_with_prompt[0].append(entity2edge[tuple(symptom_ids)])
                    edges_with_prompt[1].append(entity2edge[disease_id])
                    edges_with_prompt[0].append(entity2edge[disease_id])
                    edges_with_prompt[1].append(entity2edge[tuple(symptom_ids)])
                    edges_type_with_prompt.append(rel_type3)
                    edges_type_with_prompt.append(rel_type4)

                    if disease_type == 'diLsy':
                        entity_ids = symptom_ids + [disease_id]
                        entity_names = [self.id2entity[entity_id] for entity_id in entity_ids]
                        edges_prompt.append(self.split_sign['diLsy_logic'][0].join(entity_names[:-1]) + \
                                            self.split_sign['diLsy_logic'][1] + entity_names[-1]+ \
                                            self.split_sign['diLsy_logic'][2])
                        edges_prompt.append(entity_names[-1]+\
                                            self.split_sign['logic_reverse'][0]+\
                                            self.split_sign['logic_reverse'][1].join(entity_names[:-1]))
                    if disease_type == 'diZsy':
                        entity_ids = symptom_ids + [disease_id]
                        entity_names = [self.id2entity[entity_id] for entity_id in entity_ids]
                        edges_prompt.append(self.split_sign['diZsy_logic'][0].join(entity_names[:-1]) + \
                                            self.split_sign['diZsy_logic'][1] + entity_names[-1]+ \
                                            self.split_sign['diZsy_logic'][2])
                        edges_prompt.append(entity_names[-1]+\
                                            self.split_sign['logic_reverse'][0]+\
                                            self.split_sign['logic_reverse'][1].join(entity_names[:-1]))

        edges = [edges_without_prompt[0] + edges_with_prompt[0], edges_without_prompt[1] + edges_with_prompt[1]]
        edges_type = edges_type_without_prompt + edges_type_with_prompt
        nodes = [entity_id for entity_id in entity2edge]


        assert len(nodes) + 1 == len(nodes_type) and len(edges[0]) == len(edges_type)
        if labels_id is not None:
            return nodes,edges,nodes_type,edges_type,edges_prompt,ave_path_length,len(acc_labels_id) / len(labels_id),len(all_paths)
        else:
            return nodes,edges,nodes_type,edges_type,edges_prompt


    def generate_kg_text_by_text(self,sentence,ner_entities=None):
        """
        KG2Text的实现接口(sentence -(ner or 字符串匹配)-> graph -> text)
        Args:
            sentence: emr文本 str
            ner_entities: 命名实体识别得到的实体
        Return:
            kg_text: 文本,知识图谱直接转化为文本
        """

        origin_entity_names = self.generate_entity_name_by_text(sentence,ner_entities)
        entity_ids = []
        for entity_name in origin_entity_names:
            if entity_name not in self.match_model.entity2id:
                print(entity_name)
            else:
                entity_ids.append(self.match_model.entity2id[entity_name])

        all_paths = []

        """
        处理图谱知识
        """
        # 找出所有路径
        for entity_id in entity_ids:
            path = [entity_id]
            self._search(path,all_paths,self.max_hop,self.max_hop-1)

        # 添加所有路径
        edges_prompt = []
        for path in all_paths:
            _edges_prompt = []
            if len(path) == 1:
                _edges_prompt.append(self.id2entity[path[0]])
            else:
                for h,t in zip(path[:-1],path[1:]):
                    _edges_prompt.append(self.id2entity[h]+ self.split_sign['simple'] + self.ht2rel[(h,t)]
                                        + self.split_sign['simple'] + self.id2entity[t])
            edges_prompt.append('#'.join(_edges_prompt))
        return '*'.join(edges_prompt)
    
    def generate_graph_by_co_occurrence(self,A,sentence,ner_entities=None,threshold=30):
        """
        构建出实体-实体、实体-疾病 共现次数字典构建出图
        Args:
            A : {(id1,id2):共现次数,...} dict
            sentence: emr文档 str
            ner_entities: 命名实体识别得到的实体 list
            threshold: 共现阈值，只有超过阈值才作为边 int
        Return:
            nodes: [1,2,3] 实体id list
            edges: [[1,2,0,...],[2,1,2,...]] list
        """
        
        origin_entity_names = self.generate_entity_name_by_text(sentence,ner_entities)
        class_num = len(self.label_nodes)

        entity_ids = []
        for entity_name in origin_entity_names:
            if entity_name not in self.match_model.entity2id:
                print(entity_name)
            else:
                entity_ids.append(self.match_model.entity2id[entity_name])
        
        edges = [[],[]]
        entity2edge = {}
        for label_id in range(class_num):
            entity2edge[label_id] = label_id
        
        # Symptom -> Disease
        for label_id in range(class_num):
            for entity_id in entity_ids:
                key = (label_id,entity_id + class_num)
                if key in A and A[key] > threshold:
                    if entity_id+class_num not in entity2edge:
                        entity2edge[entity_id+class_num]=len(entity2edge)
                    edges[0].append(entity2edge[entity_id+class_num])
                    edges[1].append(label_id)
        # Symptom <-> Symptom
        for entity_id1 in entity_ids:
            for entity_id2 in entity_ids:
                key = (entity_id1+class_num,entity_id2+class_num)
                if key in A and A[key] > threshold:
                    if entity_id1+class_num not in entity2edge:
                        entity2edge[entity_id1+class_num]=len(entity2edge)
                    if entity_id2+class_num not in entity2edge:
                        entity2edge[entity_id2+class_num]=len(entity2edge)
                    edges[0].append(entity2edge[entity_id1+class_num])
                    edges[1].append(entity2edge[entity_id2+class_num])
        nodes = [node_id for node_id in entity2edge]
        return nodes,edges
    
    def filter_edges(self,edges,edge_types,edge_prompts,max_num = 1e6):
        """
        过滤边
        """
        num_direct_edges = len(edges[0]) - len(edge_prompts)
        if len(edge_prompts) > max_num:
            raw_random_choice = random.sample(list(range(len(edge_prompts)//2)),max_num//2)
            raw_random_choice.sort()
            random_choice = []
            for i in raw_random_choice:
                random_choice.append(i*2   + num_direct_edges)
                random_choice.append(i*2+1 + num_direct_edges)

            filter_edges = []
            filter_edges.append(edges[0][:num_direct_edges] + [edges[0][i] for i in random_choice])
            filter_edges.append(edges[1][:num_direct_edges] + [edges[1][i] for i in random_choice])
            filter_edges_types =edge_types[:num_direct_edges] + [edge_types[i] for i in random_choice]

            edge_prompts_choice = [j - num_direct_edges for j in random_choice if j - i>=0]
            if len(edge_prompts_choice) > 0:
                filter_edge_prompts = [edge_prompts[i] for i in edge_prompts_choice]
            return filter_edges,filter_edges_types,filter_edge_prompts
        else:
            return edges,edge_types,edge_prompts

    def generate_prompt_by_attention(self,direct_nodes,node_score,alpha,nodes,node_mask,
                                     edges,edge_types,labels,tokenizer,raw_text,
                                     max_length, num_path = 6, sample_type='attention'):
        """
        根据注意力获取prompt
        Args:
            alpha: 注意力list len(alpha) == len(edges[0])
            nodes: [[1,0],[1,2]]
            node_mask: [[1,0],[1,1]]
            edges: [[1,2,3,0],[0,1,2,3]]
        Return:
            prompt_text: list
            prompt_mask_label: list
        """
        assert sample_type in {'random','attention'}
        yes_token_id = tokenizer.convert_tokens_to_ids(self.yes_token)
        no_token_id = tokenizer.convert_tokens_to_ids(self.no_token)

        # log 文本
        print(raw_text)
        # log 实体名字/注意力分数
        nodes_name = []
        for node_idx,entity_id in enumerate(nodes):
            node_name = []
            for i,_entity_id in enumerate(entity_id):
                if node_mask[node_idx][i]:
                    node_name.append(self.id2entity[_entity_id])
            nodes_name.append([tuple(node_name),node_score[node_idx +1]])
        print(nodes_name)
        rel2id = self.rel2id
        rel_type1 = 2 * len(rel2id)     # 病历->症状
        rel_type2 = 2 * len(rel2id) + 1 # 症状->病历
        rel_type3 = 2 * len(rel2id) + 2 # 逻辑症状->疾病
        rel_type4 = 2 * len(rel2id) + 3 # 疾病 -> 逻辑症状
        rel_type5 = 2 * len(rel2id) + 4 # 病历->其他节点
        rel_type6 = 2 * len(rel2id) + 5 # 其他节点->病历
        rel_type7 = 2 * len(rel2id) + 6 # 电子病历->标签
        rel_type8 = 2 * len(rel2id) + 7 # 标签->电子病历

        for i in range(0,len(edges[0])):
            head_name = ''
            if edges[0][i] == 0:
                head_name = '电子病历'
                tail_name = nodes_name[edges[1][i]-1][0]
            elif edges[1][i] == 0:
                head_name = nodes_name[edges[0][i]-1][0]
                tail_name = '电子病历'
            else:
                head_name = nodes_name[edges[0][i]-1][0]
                tail_name = nodes_name[edges[1][i]-1][0]
            if edge_types[i] == rel_type1 or edge_types[i] == rel_type2:
                rel_name = '共现'
            elif edge_types[i] == rel_type3:
                rel_name = '逻辑症状->疾病'
            elif edge_types[i] == rel_type4:
                rel_name = '疾病->逻辑症状'
            elif edge_types[i] == rel_type5:
                rel_name = '病历->其他节点'
            elif edge_types[i] == rel_type6:
                rel_name = '其他节点->病历'
            elif edge_types[i] == rel_type7:
                rel_name = '病历->标签'
            elif edge_types[i] == rel_type8:
                rel_name = '标签->病历'
            else:
                rel_name = self.id2rel.get(edge_types[i],'')
            print(head_name,rel_name,tail_name,alpha[i])

        def _search(path):
            if nodes[path[-1]][0] in self.label_nodes:
                return
            if path[-1] not in head2tail:
                return

            head_node_idx = path[-1]
            
            next_nodes_with_score = []
            for tail_node in head2tail[head_node_idx]:
                tail_node_idx = tail_node[0]
                if tail_node_idx in path:
                    continue
                next_nodes_with_score.append([tail_node_idx,tail_node[1],
                                             node_score[tail_node_idx+1]])
            
            if len(next_nodes_with_score) == 0:
                return

            next_nodes_with_score.sort(key=lambda x:x[1],reverse=True)
            print(f'第{len(path)}层节点')
            for node_with_score in next_nodes_with_score:
                tail_node_idx = node_with_score[0]
                key = (nodes[head_node_idx][0],nodes[tail_node_idx][0])
                if len(nodes_name[tail_node_idx][0]) > 1 or len(nodes_name[head_node_idx][0]) > 1:
                    print(nodes_name[tail_node_idx][0],'逻辑知识','节点注意力分数:',node_with_score[1],'节点分数:',node_with_score[2])
                elif key in self.ht2rel:
                    print(nodes_name[tail_node_idx][0],self.ht2rel[key],'节点注意力分数:',node_with_score[1],'节点分数:',node_with_score[2])
                else:
                    print(nodes_name[tail_node_idx][0],'','节点注意力分数:',node_with_score[1],'节点分数:',node_with_score[2])

            if sample_type == 'attention':
                if len(path) < self.max_hop:
                    path.append(next_nodes_with_score[0][0])
                else:
                    # 超过最大跳数选择疾病节点
                    for next_node in next_nodes_with_score:
                        if nodes[next_node[0]][0] in self.label_nodes:
                            break
                    if nodes[next_node[0]][0] in self.label_nodes:
                        path.append(next_node[0])
                    else:
                        path.append(next_nodes_with_score[0][0])
            else:
                if len(path) < self.max_hop:
                    path.append(random.choice(head2tail[head_node_idx])[0])
                else:
                    next_label_nodes = []
                    for node in head2tail[head_node_idx]:
                        if nodes[node[0]][0] in self.label_nodes:
                            next_label_nodes.append(node[0])
                    path.append(random.choice(next_label_nodes))

            print(f'被选择节点{nodes_name[path[-1]][0]}')
            _search(path)
            
        head2tail = {}
        for i in range(len(edges[0])):
            head_id = edges[1][i]-1 # 要反过来
            tail_id = edges[0][i]-1
            if head_id < 0 or tail_id <0:
                continue
            if head_id not in head2tail:
                head2tail[head_id] = []
            head2tail[head_id].append((tail_id,alpha[i]))
        
        node_scores = [(i-1,node_score[i]) for i in range(1,len(node_score))]
        if sample_type=='attention':
            node_scores.sort(key = lambda x:x[1],reverse=True)
        prompts = []
        input_ids = []
        attention_mask = []
        token_labels = []
        num_logic_path = 0

        for node_with_score in node_scores:
            head_node_idx = node_with_score[0]
            if head_node_idx not in direct_nodes:
                continue
            # 强制保留一条逻辑知识
            if num_logic_path == 0 and len(prompts) == num_path - 1 and sum(node_mask[head_node_idx]) == 1:
                continue
            path = [head_node_idx]
            print(f'头节点: {nodes_name[head_node_idx][0]},头结点分数{node_with_score[1]}')
            _search(path)
            if nodes[path[-1]][0] in self.label_nodes:
                label_id = self.label_nodes.index(nodes[path[-1]][0])
                input_ids += [tokenizer.mask_token_id]
                attention_mask += [1]
                if labels is None:
                    token_labels += [-100]
                elif label_id in labels:
                    token_labels += [yes_token_id]
                else:
                    token_labels += [no_token_id]

                if len(path) == 1:
                    head_name = self.emr_name
                    tail_name = self.id2entity[nodes[path[0]][0]]
                    rel_name = self.emr_co_name
                    head_name = tokenizer(head_name,add_special_tokens=False)
                    tail_name = tokenizer(tail_name,add_special_tokens=False)
                    rel_name = tokenizer(rel_name,add_special_tokens=False)

                    _input_ids = head_name['input_ids'] + rel_name['input_ids'] + tail_name['input_ids']
                    _attention_mask = head_name['attention_mask'] + rel_name['attention_mask'] + tail_name['attention_mask']
                    _labels = [-100] * len(_input_ids)

                    input_ids += _input_ids
                    attention_mask += _attention_mask
                    token_labels += _labels
                
                for h,t in zip(path[:-1],path[1:]):
                    # 逻辑知识
                    if sum(node_mask[h]) > 1:
                        head_name = ''
                        num_logic_path += 1
                        for node_id,mask in zip(nodes[h],node_mask[h]):
                            if head_name != '' and mask == 1:
                                head_name += self.split_sign['diLsy_logic'][0]
                            if mask == 1:
                                head_name += self.id2entity[node_id]
                        
                        tail_name = self.id2entity[nodes[t][0]]
                        head_name = tokenizer(head_name,add_special_tokens=False)
                        tail_name = tokenizer(tail_name,add_special_tokens=False)
                        rel_name1 = tokenizer(self.split_sign['diZsy_logic'][1],add_special_tokens=False)
                        rel_name2 = tokenizer(self.split_sign['diZsy_logic'][2],add_special_tokens=False)
                        
                        _input_ids = head_name['input_ids'] + rel_name1['input_ids'] + tail_name['input_ids'] + rel_name2['input_ids']
                        _attention_mask = head_name['attention_mask'] + rel_name1['attention_mask'] + tail_name['attention_mask'] + rel_name2['attention_mask']
                        _labels = [-100] * len(_input_ids)
                        
                    # 图谱知识
                    else:
                        head_name = self.id2entity[nodes[h][0]]
                        tail_name = self.id2entity[nodes[t][0]]
                        head_name = tokenizer(head_name,add_special_tokens=False)
                        tail_name = tokenizer(tail_name,add_special_tokens=False)
                        key = (nodes[t][0],nodes[h][0])
                        if key in self.ht2rel:
                            rel_name = self.ht2rel[key]
                        else:
                            key = (nodes[h][0],nodes[t][0])
                            rel_name = self.ht2rel.get(key,'')

                        rel_name = tokenizer(rel_name,add_special_tokens=False)
                        
                        _input_ids = head_name['input_ids'] + rel_name['input_ids'] + tail_name['input_ids']
                        _attention_mask = head_name['attention_mask'] + rel_name['attention_mask'] + tail_name['attention_mask']
                        _labels = [-100] * len(_input_ids)
                        
                    input_ids += _input_ids
                    attention_mask += _attention_mask
                    token_labels += _labels
                prompts.append(path)
            if len(prompts) >= num_path:
                break

        doc = tokenizer(raw_text,add_special_tokens=False)

        input_ids = [tokenizer.cls_token_id] + input_ids + doc['input_ids'] + [tokenizer.sep_token_id]
        attention_mask = [1] + attention_mask + doc['attention_mask'] + [1]
        token_labels = [-100] + token_labels + [-100] * len(doc['input_ids']) + [-100]

        # assert sum(labels) != -100 * len(labels)
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        token_labels = token_labels[:max_length]

        sentence = tokenizer.decode(input_ids)

        # assert sum(labels) != -100 * len(labels)

        ans = {'input_ids':input_ids,'attention_mask':attention_mask,'labels':token_labels}

        return ans

    def _read_kg_file(self,kg_graph_path):
        h2t = {} # headid:[tail_id1,tail_id2,..]
        t2h = {} # 
        ht2rel = {} # {(h,t):relation(str),...}
        rel2id = {} # {'伴随':0,...}
        with open(kg_graph_path,'r',encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                triple = line.split('\t')
                head_name = self.match_model._process_entity_name(triple[0].split('..')[0])
                tail_name = self.match_model._process_entity_name(triple[2].split('..')[0])
                if head_name not in self.match_model.entity2id or tail_name not in self.match_model.entity2id:
                    continue
                headid = self.match_model.entity2id[head_name]
                tailid = self.match_model.entity2id[tail_name]
                # h -> t
                if headid not in h2t:
                    h2t[headid] = set()
                if tailid not in h2t:
                    h2t[tailid] = set()
                h2t[headid].add(tailid)
                h2t[tailid].add(headid)
                
                # t -> h
                if tailid not in t2h:
                    t2h[tailid] = set()
                if headid not in t2h:
                    t2h[headid] = set()
                t2h[tailid].add(headid)
                t2h[headid].add(tailid)

                ht2rel[(headid,tailid)] = triple[1]
                # ht2rel[(tailid,headid)] = triple[1]
                if triple[1] not in rel2id:
                    rel2id[triple[1]] = len(rel2id)
        ht2rel_tmp = ht2rel.copy()
        for key in ht2rel_tmp:
            reverse_key = (key[1],key[0])
            if reverse_key not in ht2rel:
                ht2rel[reverse_key] = self.reverse_word + ht2rel[key]
                if self.reverse_word + ht2rel[key] not in rel2id:
                    rel2id[self.reverse_word + ht2rel[key]] = len(rel2id)

        self.ht2rel = ht2rel
        self.rel2id = rel2id
        self.id2rel = {}
        for rel in self.rel2id:
            self.id2rel[self.rel2id[rel]] = rel
        self.h2t = h2t
        self.t2h = t2h

    def _read_logic_file(self,logic_graph_path):
        logic_dict = {'di2sy':{},'sy2di':{}} # 逻辑知识
        logic_num = 0
        max_logic_num = 0
        with open(logic_graph_path,'r',encoding='utf-8') as f:
            for i,line in enumerate(f):
                line = line.strip()
                if not line: continue
                logic_item = line.split('\t')
                if logic_item[0] == 'diLsy' or logic_item[0] == 'diZsy':
                    logic_num += 1
                    entities = [self.match_model.entity2id[self.match_model._process_entity_name(_.split('..')[0])]\
                                 for _ in logic_item[1:]]
                    disease = entities[0]
                    symptoms = entities[1:]
                    max_logic_num = max(max_logic_num,len(symptoms))
                    if str(disease)+'-'+logic_item[0] not in logic_dict['di2sy']:
                        logic_dict['di2sy'][str(disease)+'-'+logic_item[0]+'-'+str(i)] = []
                        # logic_dict['di2sy'][str(disease)+'-'+logic_item[0]] = []
                    for symptom in symptoms:
                        if symptom not in logic_dict['sy2di']:
                            logic_dict['sy2di'][symptom] = []
                        logic_dict['sy2di'][symptom].append(str(disease)+'-'+logic_item[0]+'-'+str(i))
                        # logic_dict['sy2di'][symptom].append(str(disease)+'-'+logic_item[0])
                        logic_dict['di2sy'][str(disease)+'-'+logic_item[0]+'-'+str(i)].append(symptom)
                        # logic_dict['di2sy'][str(disease)+'-'+logic_item[0]].append(symptom)
        self.logic_dict = logic_dict
        self.logic_num = logic_num
        self.max_logic_num = max_logic_num
    
    def _read_label_file(self,label_path):
        label_nodes = []
        with open(label_path,'r',encoding='utf-8') as f:
            for line in f:
                label = line.strip()
                if not label: continue
                if '..疾病' in label:
                    label_name = label.split('..')[0]
                else:
                    label_name = self.match_model._process_entity_name(label)
                label_nodes.append(self.match_model.entity2id[label_name])
        self.label_nodes = label_nodes