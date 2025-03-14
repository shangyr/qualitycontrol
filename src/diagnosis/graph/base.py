from abc import ABC,abstractmethod
import pickle as pkl
import os
import random
import copy
import time
import queue

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

    def relation_num(self):
        return len(self.rel2id)

    def print_path(self,path):
        for p in path:
            print(self.match_model.id2entity[p],end=" ")
        print()

    # def path_to_str(self,path):
    #     ans = []
    #     for p in path:
    #         ans.append(self.match_model.id2entity[p])
    #     return ans
    def path_to_str(self,path):

        ans = ''
        for head_id,tail_id in zip(path[:-1],path[1:]):
            head = self.match_model.id2entity[head_id]
            tail = self.match_model.id2entity[tail_id]
            if (head_id,tail_id) in self.ht2rel:
                rel = self.ht2rel[(head_id,tail_id)]
            else:
                rel = ''
            if ans == '':
                ans += head
            ans += f"-[{rel}]->{tail}"

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

    def generate_graph_by_text(self,sentence,ner_entities=None,path_type = 'v2'):
        """
        通过症状实体得到相应的图
        Args:
            sentence: 电子病历文本 str
            ner_entities: 可选，命名实体识别的结果 ['咳嗦','肺炎',...]
            path_type: 'v1' 标准路径
                       'v2' 只保留路径中的头结点(电子病历中的实体)和尾结点(标签节点)
        Return:
            nodes: 实体转化为数字 [0,1,2,3] 有数字有元组，元组表示超图 emr节点为第一个0
            edges: 边 [[1,2,0,1],[2,3,1,3]] 下标按照node下标
            nodes_type: 节点类型 [0,1,2,3,4]  emr节点为第一个0
            edges_type: 边类型 [1,2,3,4] 边类型
            paths: 路径字符串列表 [['咳嗦','肺炎'],...]
        """
        entity_names = self.generate_entity_name_by_text(sentence,ner_entities)
        uniq_entity_names = list(set(entity_names))

        nodes,edges,nodes_type,edges_type,paths = self._generate_graph_by_entity(uniq_entity_names,path_type = path_type)
        return nodes,edges,nodes_type,edges_type,paths,entity_names
    
    def _generate_graph_by_entity(self,entity_names,path_type = 'v1', filter_node=False):
        """
        根据实体名字和知识图谱构建出个性化电子病历子图,由于剪枝只能找通往标签节点的路径
        Args:
            entity_names: ['咳嗽','感冒',...]
        Return:
            nodes:节点 [2121,213,...] 实体节点id
            edges:边 [[1,2,3,...],[0,1,2,...]]
            nodes_type:节点类型 [0,1,2,3,1] len(nodes_types)=len(nodes)+1
            edges_type:边类型 [2,3,4,2]
            paths: 路径字符串列表 [['咳嗦','肺炎'],...]
        """
        class_num = len(self.label_nodes)
        entity_ids = []
        for entity_name in entity_names:
            if entity_name in self.match_model.entity2id:
                entity_ids.append(self.match_model.entity2id[entity_name])
        # entity_ids = [self.match_model.entity2id[entity] for entity in entity_names]
        all_paths = []

        edges = [[],[]]     # 每条边[[1,2,3,4],[2,3,4,5]]
        edges_type = []     # 每条边[[1,2,3,4],[2,3,4,5]]

        nodes_type = []    # [1,2,3,4]
        entity2edge = {}    # 实体id到edge中实体id 0为emr在图中的表示

        rel2id = self.rel2id
        type2id = self.match_model.type2id.copy()
        
        """
        加入所有疾病节点
        """
        for label_id in self.label_nodes:
            entity2edge[label_id] = len(entity2edge)
            nodes_type.append(type2id[self.entityname2type[self.id2entity[label_id]]])

        """
        处理图谱知识
        step 1. 构建字典树
        所有 上下文实体 到 标签实体 形成一颗树 trie_tree
        step 2. 遍历树
        遍历该树，得到所有的路径
        """
        trie_tree = {}
        for entity_id in entity_ids:
            self._search(entity_id,trie_tree,self.max_hop-1)
            
        _all_paths = self._search_trie_tree(trie_tree)
        random.shuffle(_all_paths)
        all_paths = []
        label_count = {}
        for path in _all_paths:
            if path[-1] not in label_count:
                label_count[path[-1]] = 0
            label_count[path[-1]] += 1
            if label_count[path[-1]] <= 200:
                all_paths.append(path)
    
        all_path_str = []
        edges_set = set()
        node_count = {}
        for path in all_paths:
            if path_type == 'v1':
                all_path_str.append(self.path_to_str(path))
            elif path_type == 'v2':
                edge = (path[0],path[-1])
                if edge not in edges_set and len(path) > 1:
                    all_path_str.append(self.path_to_str([path[0],path[-1]]))
                    edges_set.add(edge)
                    if path[0] not in node_count:
                        node_count[path[0]] = 0
                    node_count[path[0]] += 1
        pruning_nodes = set()
        if filter_node:
            for node in node_count:
                if node_count[node] > class_num // 2:
                    pruning_nodes.add(node)

        # if path_type == 'v2':
        #     all_paths = [[item[0],item[-1]] for item in all_paths]
        # print()

        # 添加所有实体
        for path in all_paths:
            for entity_id in path:
                if entity_id not in entity2edge:
                    entity2edge[entity_id] = len(entity2edge)
                    entity_name = self.entityname2type[self.id2entity[entity_id]]
                    nodes_type.append(type2id[entity_name])

        nodes = [entity_id for entity_id in entity2edge]

        edges_set = set()
        # 添加所有路径
        for path in all_paths:
            if path_type == 'v1':
                for h,t in zip(path[:-1],path[1:]):
                    if (h,t) not in edges_set:
                        edges_set.add((h,t))
                        edges[0].append(entity2edge[h])
                        edges[1].append(entity2edge[t])
                        edges_type.append(rel2id[self.ht2rel[(h,t)]])

            elif path_type == 'v2':
                h,t = path[0],path[-1]
                if h in pruning_nodes and len(path) >= 3: # 过滤一部分
                    continue
                if (h,t) not in edges_set and len(path) >= 2:
                    edges_set.add((h,t))
                    edges[0].append(entity2edge[h])
                    edges[1].append(entity2edge[t])
                    rel_id = 0
                    for h,t in zip(path[:-1],path[1:]):
                        rel_id *= self.relation_num() + 1
                        _rel_id = rel2id[self.ht2rel[(h,t)]]
                        rel_id += _rel_id
                    if len(path) < self.max_hop:
                        for _ in range(self.max_hop - len(path)):
                            rel_id *= self.relation_num() + 1
                            rel_id += self.relation_num()

                    edges_type.append(rel_id)
            else:
                raise NotImplementedError

        # assert len(nodes) == len(nodes_type) and len(edges[0]) == len(edges_type)
        if len(edges[0]) == 0:
            edges[0].append(0)
            edges[1].append(0)
            edges_type.append(0)
        return nodes,edges,nodes_type,edges_type,all_paths

    def generate_kg_text_by_text(self,sentence,ner_entities=None):
        """
        KG2Text的实现接口(sentence -(ner or 字符串匹配)-> graph -> text)
        Args:
            sentence: emr文本 str
            ner_entities: 命名实体识别得到的实体
        Return:
            kg_text: 文本,知识图谱直接转化为文本
        """

        entity_names = self.generate_entity_name_by_text(sentence,ner_entities)
        entity_ids = []
        for entity_name in entity_names:
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
        
        entity_names = self.generate_entity_name_by_text(sentence,ner_entities)
        class_num = len(self.label_nodes)

        entity_ids = []
        for entity_name in entity_names:
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
                        num_logic_path += 1
                        head_name = ''
                        for node_id,mask in zip(nodes[h],node_mask[h]):
                            if head_name != '' and mask == 1:
                                head_name += self.split_sign['diZsy_logic'][0]
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

        # sentence = tokenizer.decode(input_ids)

        # assert sum(labels) != -100 * len(labels)

        ans = {'input_ids':input_ids,'attention_mask':attention_mask,'labels':token_labels}

        return ans

    
    def pruning_h2t_data(self):
        self._pruning_h2t()
        node_queue = queue.Queue()

        edges = set()
        visit_nodes = set()
        for node in self.h2t_prune:
            visit_nodes.add(node)
            for num_hop in self.h2t_prune[node]:
                for tail_node in self.h2t_prune[node][num_hop]:
                    node_queue.put(tail_node)
                    edges.add((node,self.rel2id[self.ht2rel[(node,tail_node)]],tail_node))
        
        while not node_queue.empty():
            node = node_queue.get()
            if node in visit_nodes:
                continue
            visit_nodes.add(node)
            for num_hop in self.h2t_prune[node]:
                for tail_node in self.h2t_prune[node][num_hop]:
                    node_queue.put(tail_node)
                    edges.add((node,self.rel2id[self.ht2rel[(node,tail_node)]],tail_node))
        node_count = {}
        relation_count = {}
        for edge in edges:
            if edge[0] not in node_count:
                node_count[edge[0]] = 0
            if edge[1] not in relation_count:
                relation_count[edge[1]] = 0
            if edge[2] not in node_count:
                node_count[edge[2]] = 0
            
            node_count[edge[0]] += 1
            relation_count[edge[1]] += 1
            node_count[edge[2]] += 1

        nodes,nodes_prob = [],[]
        for node in node_count:
            nodes.append(node)
            nodes_prob.append(node_count[node])
        sum_node_prob = sum(nodes_prob)
        nodes_prob = [i/sum_node_prob for i in nodes_prob]

        relations,relations_prob = [],[]
        for relation in relation_count:
            relations.append(relation)
            relations_prob.append(relation_count[relation])
        sum_relation_prob = sum(relations_prob)
        relations_prob = [i/sum_relation_prob for i in relations_prob]

        return edges,nodes,nodes_prob,relations,relations_prob

    def _read_kg_file(self,kg_graph_path):
        h2t = {} # headid:[tail_id1,tail_id2,..]
        t2h = {} # 
        ht2rel = {} # {(h,t):relation(str),...}
        rel2id = {} # {'伴随':0,...}
        def read_kg_file(h2t,t2h,ht2rel,rel2id,kg_graph_path):
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

        read_kg_file(h2t,t2h,ht2rel,rel2id,kg_graph_path)
        if os.path.exists(kg_graph_path.replace('triples.txt','triples_expand.txt')):
            read_kg_file(h2t,t2h,ht2rel,rel2id,kg_graph_path.replace('triples.txt','triples_expand.txt'))

        
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

