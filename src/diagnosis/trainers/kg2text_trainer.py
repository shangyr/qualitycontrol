from .base_trainer import BaseTrainer
import torch

class KG2TextTrainer(BaseTrainer):
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
        logits = self.model(data['input_ids'],data['attention_mask'], 
                            data['kg_input_ids'],data['kg_attention_mask'])
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
        logits = self.model(data['input_ids'],data['attention_mask'], 
                            data['kg_input_ids'],data['kg_attention_mask'])
        return logits

    def _to_gpu(self,data):
        """
        把数据放在gpu上面
        """
        # data['input_ids'] = data['input_ids'].to(self.config.gpu)
        # data['attention_mask'] = data['attention_mask'].to(self.config.gpu)
        # data['label'] = data['label'].to(self.config.gpu)
        
        data['input_ids'] = data['input_ids'].to(torch.cuda.current_device())
        data['attention_mask'] = data['attention_mask'].to(torch.cuda.current_device())
        data['kg_input_ids'] = data['kg_input_ids'].to(torch.cuda.current_device())
        data['kg_attention_mask'] = data['kg_attention_mask'].to(torch.cuda.current_device())
        data['label'] = data['label'].to(torch.cuda.current_device())
        return data
    
    # def _make_dict(self):
    #     self.id2entity = []
    #     for entity in self.entity2id:
    #         self.id2entity.append(entity)
    
        # self.tokenizer = AutoTokenizer.from_pretrained(self.config.bert_path)

    def _decode_data(self, data, predicts):
        """
        将batch_data解码为可视化需要的形式
        Args:
            data: 每个batch的输入数据,dict
        return: 
            ans: 每个batch被转化为list [{'predict':[],
                'predict_score':[]},...]
            其中,predict是模型的预测结果,里面每一项是字符串(疾病名字),
            node是电子病历中的节点名称,node_type是电子病历的节点类型
            path是路径,每一项是一个三元组, [头实体名字,关系名字,尾实体名字]
        """
        # if not hasattr(self,'id2entity'):
        #     self._make_dict()

        batch_size = predicts.size(0)

        ans = []
        for i in range(batch_size):
            item = {}
            predict = torch.where(predicts[i]>0)[0]
            item['predict'] = [self.id2label[int(p)] for p in predict]
            item['predict_score'] = {}
            for j,label in enumerate(self.id2label):
                item['predict_score'][label] = float(predicts[i][j])
            
            ans.append(item)
        return ans
