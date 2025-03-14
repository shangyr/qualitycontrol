from .base_trainer import BaseTrainer
import torch
from transformers import AutoTokenizer

class TextKGNNTrainer(BaseTrainer):
    def __init__(self, model, config,loss_func,strategy):
        super().__init__(model, config, strategy)
        self.loss_func = loss_func
        self.prompt_ids = []

    def _fit_batch(self,data):
        """
        计算一个 batch 的损失
        Args:
            data: 输入 dict{key:str : value:Tensor}
        Return:
            loss: Tensor
        """
        if torch.cuda.is_available():
            data = self._to_gpu(data)
        logits = self.model(data['input_ids'],data['attention_mask'],
                            data['nodes'],data['node_mask'],data['nodes_type'],
                            data['edges'],data['edges_type'],
                            data['prompts'])
        loss = self.loss_func(logits,data['label'])
        return loss
    
    def _inference_batch(self,data):
        """
        计算一个 batch 的预测值
        Args:
            data: 输入 dict { key:str : value:Tensor }
        Return:
            logits: Tensor
        """
        if torch.cuda.is_available():
            data = self._to_gpu(data)
        logits = self.model(data['input_ids'],data['attention_mask'],
                            data['nodes'],data['node_mask'],data['nodes_type'],
                            data['edges'],data['edges_type'],
                            data['prompts'])
        # self.prompt_ids.append(promot_ids)
        return logits

    def _to_gpu(self,data):
        """
        把数据放在gpu上面
        """

        data['input_ids'] = data['input_ids'].to(torch.cuda.current_device())
        data['attention_mask'] = data['attention_mask'].to(torch.cuda.current_device())
        data['label'] = data['label'].to(torch.cuda.current_device())

        for i in range(len(data['edges'])):
            data['edges'][i] = data['edges'][i].to(torch.cuda.current_device())
            data['nodes'][i] = data['nodes'][i].to(torch.cuda.current_device())
            data['node_mask'][i] = data['node_mask'][i].to(torch.cuda.current_device())

            data['nodes_type'][i] = data['nodes_type'][i].to(torch.cuda.current_device())
            data['edges_type'][i] = data['edges_type'][i].to(torch.cuda.current_device())

        return data

    def _make_dict(self):
        
        self.entity2id = self.dataset.graph.match_model.entity2id
        self.rel2id = self.dataset.graph.rel2id
        self.type2id = self.dataset.graph.match_model.type2id

        self.id2entity = []
        for entity in self.entity2id:
            self.id2entity.append(entity)
        
        self.id2rel = []
        for rel in self.rel2id:
            self.id2rel.append(rel)
        for rel in self.rel2id:
            self.id2rel.append('反向'+rel)
        self.id2rel.append('病历-实体共现')
        self.id2rel.append('实体-病历共现')
        self.id2rel.append('逻辑症状->疾病')
        self.id2rel.append('疾病->逻辑症状')
        self.id2rel.append('病历->其他节点')
        self.id2rel.append('其他节点->病历')
        self.id2rel.append('病历->标签')
        self.id2rel.append('标签->病历')
        self.id2type = ['病历节点']
        for ent_type in self.type2id:
            self.id2type.append(ent_type)
        self.id2type.append('逻辑知识节点')
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.bert_path)

    def _decode_data(self, data, predicts):
        """
        将batch_data解码为可视化需要的形式
        Args:
            data: 每个batch的输入数据,dict
        return: 
            ans: 每个batch被转化为list [{'predict':[],
                'nodes':[],'nodes_type':[],'paths':[]},...]
            其中,predict是模型的预测结果,里面每一项是字符串(疾病名字),
            node是电子病历中的节点名称,node_type是电子病历的节点类型
            path是路径,每一项是一个三元组, [头实体名字,关系名字,尾实体名字]
        """
        if not hasattr(self,'id2entity'):
            self._make_dict()

        batch_size = predicts.size(0)

        ans = []
        for i in range(batch_size):
            item = {}
            item['nodes'] = []
            item['nodes_type'] = []
            item['paths'] = []
            predict = torch.where(predicts[i]>0)[0]
            item['predict'] = [self.id2label[int(p)] for p in predict]
            item['predict_score'] = {}
            for j,label in enumerate(self.id2label):
                item['predict_score'][label] = float(predicts[i][j])
            # if self.prompt_ids[i] == -1:
            #     item['prompt'] = ''
            # else:
            #     item['prompt'] = data['prompts'][i]['edge_prompt'][self.prompt_ids[i]] + data['prompts'][i]['raw_doc']
            
            nodes = data['nodes'][i] # [node_num,pre_node_name_num]
            node_mask = data['node_mask'][i] # [node_num,pre_node_name_num]
            nodes_type = data['nodes_type'][i] # [node_num+1]
            edges = data['edges'][i] # [2, edge_num]
            edges_type = data['edges_type'][i] # [edge_num]
            item['nodes'].append('病历')
            for j in range(nodes.size(0)):
                node = ''
                for k in range(nodes[j].size(0)):
                    if node_mask[j,k] == 0:
                        break
                    else:
                        if len(node) > 0: node += '##'
                        node += self.id2entity[int(nodes[j,k])]
                item['nodes'].append(node)
                item['nodes_type'].append(self.id2type[int(nodes_type[j])])
            
            for head,rel,tail in zip(edges[0],edges_type,edges[1]):
                # 三元组关系
                item['paths'].append((item['nodes'][int(head)],
                                     self.id2rel[int(rel)],
                                     item['nodes'][int(tail)]))
            
            ans.append(item)
        return ans
