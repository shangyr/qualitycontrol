from .base_trainer import BaseTrainer
import torch
from transformers import BertTokenizer
import math

class PromptGNNTrainer(BaseTrainer):
    def __init__(self, model, config,loss_func,strategy):
        super().__init__(model, config, strategy)
        self.loss_func = loss_func
        
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
        if 'edges_types' not in data:
            logits = self.model(data['input_ids'],data['attention_mask'],
                                data['nodes'],data['edges'])
        else:
            logits = self.model(data['input_ids'],data['attention_mask'],
                                data['nodes'],data['edges'],data['edges_types'])
        loss = self.loss_func(logits,data['label'].float())
        return loss
    
    def _inference_batch(self,data):
        """
        计算一个 batch 的预测值
        Args:
            data: 输入 dict{key:str : value:Tensor}
        Return:
            logits: Tensor
        """
        if torch.cuda.is_available():
            data = self._to_gpu(data)
        
        if 'edges_types' not in data:
            logits,text_attentions,graph_attentions = self.model(data['input_ids'],data['attention_mask'],
                                data['nodes'],data['edges'],True)
        else:
            logits,text_attentions,graph_attentions = self.model(data['input_ids'],data['attention_mask'],
                                data['nodes'],data['edges'],data['edges_types'],True)
        return logits,text_attentions,graph_attentions

    def _to_gpu(self,data):
        """
        把数据放在gpu上面
        """
        # data['input_ids'] = data['input_ids'].to(self.config.gpu)
        # data['attention_mask'] = data['attention_mask'].to(self.config.gpu)
        # data['label'] = data['label'].to(self.config.gpu)
        
        data['input_ids'] = data['input_ids'].to(torch.cuda.current_device())
        data['attention_mask'] = data['attention_mask'].to(torch.cuda.current_device())

        for i in range(len(data['edges'])):
            data['nodes'][i] = data['nodes'][i].to(torch.cuda.current_device())
            data['edges'][i] = data['edges'][i].to(torch.cuda.current_device())
            if 'edges_types' in data:
                data['edges_types'][i] = data['edges_types'][i].to(torch.cuda.current_device())
        data['label'] = data['label'].to(torch.cuda.current_device())
        return data
    
    # def _make_dict(self):
    #     self.id2entity = []
    #     for entity in self.entity2id:
    #         self.id2entity.append(entity)

    #     self.tokenizer = AutoTokenizer.from_pretrained(self.config.bert_path)

    def _decode_data(self, data, predicts):
        """
        将batch_data解码为可视化需要的形式
        Args:
            data: 每个batch的输入数据,dict
            predicts: 模型输出结果，包括每个疾病的概率和attention
        return: 
            ans: 每个batch被转化为list [{'predict':[],
                'predict_score':[]},...]
            其中,predict是模型的预测结果,里面每一项是字符串(疾病名字),
            node是电子病历中的节点名称,node_type是电子病历的节点类型
            path是路径,每一项是一个三元组, [头实体名字,关系名字,尾实体名字]
        """
        # if not hasattr(self,'id2entity'):
        #     self._make_dict()
        logits, text_attention, graph_attention = predicts
        # text_attention[-1].size() = [batch_size, num_head, seq_len, seq_len]
        # graph_attention[0].size() = [edge_num, num_head]
        batch_size = logits.size(0)
        text_attention = text_attention[-1].mean(dim = 1)

        ans = []
        for i in range(batch_size):
            item = {}
            # predict = torch.where(logits[i]>0)[0]
            # item['predict'] = [self.id2label[int(p)] for p in predict]
            predict = logits[i].argmax(dim=0)
            item['predict'] = [self.id2label[int(predict)]]
            item['predict_score'] = {}
            item['text_attention_score'] = {}
            if graph_attention is not None:
                item['graph_attention_score'] = []
            _text_tokens = self.dataset.tokenizer.convert_ids_to_tokens(data['input_ids'][i].tolist())
            
            for j,label in enumerate(self.id2label):
                item['predict_score'][label] = float(logits[i][j])
                _text_attention_score = torch.cat((text_attention[i,j+1,:1],\
                                                   text_attention[i,j+1,1+self.config.class_num:]),dim=0).tolist()
                # text_attention
                text_tokens,text_attention_score = [],[]
                max_score = 0.001
                for token,score in zip(_text_tokens, _text_attention_score):
                    if token not in self.ignore_tokens_set:
                        text_tokens.append(token)
                        text_attention_score.append(score)
                        max_score = max(max_score,score)
                text_attention_score = [math.pow(score / max_score,0.3) for score in text_attention_score]
                text_attention_score = [(token,score) for token,score in zip(text_tokens, text_attention_score)]
                item['text_attention_score'][label] = text_attention_score
            
            if graph_attention is not None:
                # graph attention
                _graph_attention = graph_attention[i].mean(dim = 1).tolist()
                _edges = data['edges'][i].T.tolist()
                _nodes = data['nodes'][i].tolist()
                _edges_types = data['edges_types'][i].tolist()
                # paths = data['paths'][i]

                if len(_edges) == 1 and sum(_edges[0]) == 0:
                    _edges,_edges_types = [],[]
                for j,(edge,edge_type) in enumerate(zip(_edges, _edges_types)):
                    _head = self.dataset.id2entity[_nodes[edge[0]]]
                    _tail = self.dataset.id2entity[_nodes[edge[1]]]
                    if self.config.path_type == 'v1':
                        _rel = self.dataset.id2rel[edge_type]
                    else:
                        _rel = ''
                        for _ in range(self.config.num_hop):
                            cur_edge_type = edge_type % (self.config.relation_num + 1)
                            edge_type //= self.config.relation_num + 1
                            if cur_edge_type == self.config.relation_num:
                                continue
                            if len(_rel) > 0: _rel = '->' + _rel
                            _rel = self.dataset.id2rel[cur_edge_type] + _rel

                    item['graph_attention_score'].append([_head, _rel, _tail, _graph_attention[j]])

            ans.append(item)
        return ans
