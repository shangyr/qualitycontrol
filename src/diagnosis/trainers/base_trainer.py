from abc import ABC,abstractmethod
import os
# os.environ['NCCL_BACKEND'] = 'SOCKET'  # 或者 'SOCKET'
import torch
from tqdm import tqdm
from sklearn import metrics
from utils.metrics import all_metrics, print_metrics,topk_accuracy
from strategies import NaiveStrategy
import transformers

from utils.distribute import is_rank_0

from utils.optimizer import model_optimizer
from utils.tools import find_threshold_micro
import wandb
import re
import json
from datetime import datetime
import numpy as np
from transformers import AutoTokenizer
from utils.tokenizer import Tokenizer
import torch.nn as nn
import torch.distributed as dist

class BaseTrainer(ABC):
    def __init__(self,model,config,strategy:NaiveStrategy):
        # self.model = model.cuda(config.gpu)
        self.strategy = strategy
        self.optimizer = model_optimizer(model,config)
        self.model = self.strategy.setup_model(model)
        self.config = config
    
        self.cur_epoch = 0
        self.best_f1 = 0
        self.best_threshold_micro = 0


        # if '.embeds' in self.config.bert_path:
        #     self.tokenizer = Tokenizer.from_pretrained(self.config.bert_path)
        # else:
        #     self.tokenizer = AutoTokenizer.from_pretrained(self.config.bert_path)

        self.id2label = []
        with open(self.config.label_idx_path, "r", encoding="utf-8") as f:
            for line in f:
                lin = line.strip().split()
                self.id2label.append(lin[0])

        # 加载模型
        if os.path.exists(self.config.save_model_path):
            self.load_checkpoint()

    def setup_dataset(self, datasets):
        if len(datasets) == 3:
            # train dev test
            self.dataset = datasets[0]
            self.train_dataloader = self.strategy.setup_dataloader(datasets[0],pin_memory=True)
            self.dev_dataloader = self.strategy.setup_dataloader(datasets[1],pin_memory=True)
            self.test_dataloader = self.strategy.setup_dataloader(datasets[2],pin_memory=True)

            updates_total = len(self.train_dataloader) // (self.config.accumulation_steps) * self.config.epochs
            self.scheduler = transformers.get_cosine_schedule_with_warmup(self.optimizer,
                                                                    num_warmup_steps=self.config.warmup_rate * updates_total,
                                                                    num_training_steps=updates_total)
        if len(datasets) == 2:
            # train dev
            self.dataset = datasets[0]
            self.train_dataloader = self.strategy.setup_dataloader(datasets[0],pin_memory=True)
            self.dev_dataloader = self.strategy.setup_dataloader(datasets[1],pin_memory=True)
            self.test_dataloader = self.strategy.setup_dataloader(datasets[1],pin_memory=True)
            
            updates_total = len(self.train_dataloader) // (self.config.accumulation_steps) * self.config.epochs
            self.scheduler = transformers.get_cosine_schedule_with_warmup(self.optimizer,
                                                                    num_warmup_steps=self.config.warmup_rate * updates_total,
                                                                    num_training_steps=updates_total)
        if len(datasets) == 1:
            # test
            self.dataset = datasets[0]
            self.test_dataloader = self.strategy.setup_dataloader(datasets[0],pin_memory=True)
        
        if hasattr(self.dataset.tokenizer,'cls_token'):
            self.ignore_tokens_set = {
                self.dataset.tokenizer.pad_token,
                self.dataset.tokenizer.cls_token,
                self.dataset.tokenizer.sep_token
            }


    def fit(self):
        # valid_micro_f1, valid_report = self.inference(self.dev_dataloader,epoch=self.cur_epoch)
        while self.cur_epoch < self.config.epochs:
        
            if is_rank_0():
                print("\n=== Epoch %d train ===" % self.cur_epoch)
                print(datetime.now())
            step_bar = tqdm(range(self.train_dataloader.__len__()),
                            desc='Train step of epoch %d' % self.cur_epoch,
                            disable=not is_rank_0())
            
            all_loss = 0

            self.model.train()
            for i, data in enumerate(self.train_dataloader):
                if dist.is_initialized() and (i+1) % self.config.accumulation_steps != 0:
                    with self.model.no_sync():
                        loss = self._fit_batch(data)
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                else:
                    loss = self._fit_batch(data)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                step_bar.update()
                step_bar.set_postfix({'loss':loss.item()})
                if is_rank_0() and self.config.use_wandb:
                    wandb.log({'loss': loss.item()})
                all_loss += loss.item()
                if (i+1) % self.config.accumulation_steps == 0: 
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                if (i+1) % 10 == 0 and torch.distributed.is_initialized():
                    torch.distributed.barrier()
            if (self.cur_epoch + 1) % self.config.test_freq == 0:
                # valid_micro_f1, valid_report = self.inference(self.dev_dataloader,epoch=self.cur_epoch)
                test_micro_f1, test_report = self.inference(self.test_dataloader,epoch=self.cur_epoch)
                if test_micro_f1 > self.best_f1:
                    self.best_f1 = test_micro_f1
                    if is_rank_0():
                        self.save_checkpoint()

            if is_rank_0():
                print("目前最优验证集结果:{:.5f}".format(self.best_f1))
                print("\n=== Epoch %d end ===" % self.cur_epoch)
            step_bar.set_postfix({'loss':all_loss/len(self.train_dataloader)})
            step_bar.close()
            self.cur_epoch += 1
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

    @abstractmethod
    def _fit_batch(self,data):
        pass
    
    @abstractmethod
    def _inference_batch(self,data):
        pass

    @abstractmethod
    def _to_gpu(self,data):
        pass

    @abstractmethod
    def _decode_data(self,data):
        pass

    def inference(self,data_loader=None,epoch=0,save_result_file=False):
        """
        模型测试
        """
        if data_loader is None:
            data_loader = self.test_dataloader
        data_type = ''
        if data_loader is self.dev_dataloader:
            data_type = 'dev'
            # with open(self.config.dev_path,'r',encoding='utf-8') as f:
            #     origin_data = json.load(f)
        else:
            data_type = 'test'
            with open(self.config.test_path,'r',encoding='utf-8') as f:
                origin_data = json.load(f)
        step_bar = tqdm(range(data_loader.__len__()),
                            desc='Test step',
                            disable=not is_rank_0())
        self.model.eval()
        all_predicts = []
        all_labels = []
        all_items = []
        with torch.no_grad():
            for ii, data in enumerate(data_loader):
                predicts = self._inference_batch(data) # []
                if type(predicts) is tuple:
                    all_predicts.append(predicts[0])
                else:
                    all_predicts.append(predicts)

                step_bar.update()
                # step_bar.set_postfix({'loss':loss.item()})
                all_labels.append(data['label'])
                if data_type == 'test' and save_result_file:
                    all_items.extend(self._decode_data(data,predicts))

                if (ii+1) % 10 == 0 and torch.distributed.is_initialized():
                    torch.distributed.barrier()

        # step_bar.set_postfix({'loss':all_loss/len(self.train_dataloader)})
        step_bar.close()

        all_predicts = torch.cat(all_predicts,dim = 0).data.cpu().numpy()
        all_labels = torch.cat(all_labels,dim = 0).data.cpu().numpy()
        
        all_labels[all_labels>0.5] = 1
        all_labels[all_labels<=0.5] = 0

        max_score_class = np.argmax(all_predicts,axis=1) # [batch_size]
        y_hat = all_predicts.copy()

        # if data_type == 'dev':
        #     self.best_threshold_micro = find_threshold_micro(y_hat,all_labels)

        # y_hat[y_hat>self.best_threshold_micro] = 1
        # y_hat[y_hat<=self.best_threshold_micro] = 0
        y_hat[y_hat>0] = 1
        y_hat[y_hat<=0] = 0
        # 分数最大的类别被预测出来
        # y_hat[np.arange(max_score_class.shape[0]),max_score_class] = 1

        metrics_test = all_metrics(y_hat, all_labels)
        report = metrics.classification_report(all_labels, y_hat, digits=4,target_names = self.id2label)
        topk = topk_accuracy(all_predicts,all_labels)
        if is_rank_0():
            print_metrics(metrics_test)
            print(report)
            print(topk)
        if is_rank_0() and self.config.use_wandb:
            wandb.log(metrics_test)
            wandb.log({'top1':topk[0],'top5':topk[1],'top10':topk[2]})
            #  macro avg     0.5322    0.4711    0.4857      4555
            macro = re.findall(r'macro avg     (\d.\d\d\d\d)    (\d.\d\d\d\d)    (\d.\d\d\d\d)',report)[-1]
            wandb.log({'macro_p_sklearn':float(macro[0]),
                       'macro_r_sklearn':float(macro[1]),
                       'macro_f1_sklearn':float(macro[2])})
        if is_rank_0() and data_type == 'test' and save_result_file:
            data = []
            del_list = ['上下文实体','候选答案实体','path','实体','节点','边']
            for item1,item2 in zip(origin_data,all_items):
                for key in del_list:
                    if key in item1:
                        del item1[key]
                data.append({**item1 , **item2})

            save_result_path = f"{self.config.result_path}/{data_type}-{epoch}.json"
            with open(save_result_path,'w',encoding='utf-8') as f:
                json.dump(data,f,ensure_ascii=False,indent=4)
            
        return metrics_test['f1_micro'], report

    def predict(self,dataset,origin_data):
        self.model.eval()
        all_predicts = []
        all_items = []
        with torch.no_grad():
            for ii, data in enumerate(dataset):
                data = dataset.collate_fn([data])
                predicts = self._inference_batch(data) # []
                if type(predicts) is tuple:
                    all_predicts.append(predicts[0])
                else:
                    all_predicts.append(predicts)
                # all_labels.append(data['label'])
                all_items.extend(self._decode_data(data,predicts))
        
        data = []
        del_list = ['上下文实体','候选答案实体','path','实体','节点','边']
        for item1,item2 in zip(origin_data,all_items):
            for key in del_list:
                if key in item1:
                    del item1[key]
            data.append({**item1 , **item2})

        return data

    def save_checkpoint(self):
        save_dict = {}
        origin_dict = self.model.state_dict()
        for key,value in origin_dict.items():
            save_dict[key] = value.data.cpu()

        torch.save({'dict':save_dict,
                    'optimizer':self.optimizer,
                    'cur_epoch':self.cur_epoch,
                    'best_f1':self.best_f1},self.config.save_model_path)

    def load_checkpoint(self):
        save_dict = torch.load(self.config.save_model_path, map_location=torch.device('cpu'))
        # print(list(save_dict['dict'].keys()))
        # for k,v in self.model.named_parameters():
        #     print(k)
        new_dict = {}
        for k,v in save_dict['dict'].items():
            new_k = k.replace('module.', '')
            if 'position_ids' in new_k:
                continue
            new_dict[new_k] = v

        self.model.load_state_dict(new_dict)
        # del save_dict['optimizer']
        # del save_dict['cur_epoch']
        # del save_dict['best_f1']
        # torch.save(save_dict, self.config.save_model_path)
        # self.optimizer = save_dict['optimizer']
        # self.scheduler = save_dict['scheduler']
        # self.cur_epoch = save_dict['cur_epoch']
        # self.best_f1 = save_dict['best_f1']
