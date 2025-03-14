import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from tqdm import tqdm

from utils.tools import create_alias_table,alias_smaple

class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(TransE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        
        # 实体和关系的嵌入向量
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim)
        self.relation_embedding = nn.Embedding(num_relations, embedding_dim)
        
        # 初始化嵌入向量
        self.init_weights()
    
    def init_weights(self):
        nn.init.xavier_uniform_(self.entity_embedding.weight.data)
        nn.init.xavier_uniform_(self.relation_embedding.weight.data)
        
    def forward(self, positive_samples, negative_samples, mode='single'):
        # 正样本和负样本的嵌入向量
        pos_heads, pos_relations, pos_tails = torch.chunk(positive_samples, chunks=3, dim=1)
        neg_heads, neg_relations, neg_tails = torch.chunk(negative_samples, chunks=3, dim=1)
        
        pos_heads = self.entity_embedding(pos_heads)
        pos_relations = self.relation_embedding(pos_relations)
        pos_tails = self.entity_embedding(pos_tails)
        
        neg_heads = self.entity_embedding(neg_heads)
        neg_relations = self.relation_embedding(neg_relations)
        neg_tails = self.entity_embedding(neg_tails)
        
        # 用L1范数或L2范数来计算损失
        if mode == 'single':
            pos_distances = torch.norm(pos_heads + pos_relations - pos_tails, p=1, dim=1)
            neg_distances = torch.norm(neg_heads + neg_relations - neg_tails, p=1, dim=1)
        elif mode == 'double':
            pos_distances = torch.norm(pos_heads + pos_relations - pos_tails, p=2, dim=1)
            neg_distances = torch.norm(neg_heads + neg_relations - neg_tails, p=2, dim=1)
        else:
            raise ValueError("Mode should be 'single' or 'double'.")
        
        return pos_distances, neg_distances
    

class KGETrainer:
    def __init__(self,graph,entity2id=None,language=None,train_path=None) -> None:
        # from graph import ChineseGraph
        # self.graph = ChineseGraph('/home/lixin/prompt-gnn/data/chinese-small/triplesv2.txt',
        #                           '/home/lixin/prompt-gnn/data/chinese-small/entities.txt',
        #                           '/home/lixin/prompt-gnn/data/chinese-small/label2id.txt')
        
        self.graph = graph
        
        num_entities = self.graph.entity_num()
        num_relations = len(self.graph.rel2id)

        self.batch_size = 1000
        self.epochs = 10

        edges,nodes,nodes_prob,relations,relations_prob = self.graph.pruning_h2t_data()

        self.nodes = nodes
        self.relations = relations

        self.nodes_accept_prob, self.nodes_alias_index = create_alias_table(nodes_prob)
        self.relations_accept_prob, self.relations_alias_index = create_alias_table(relations_prob)

        edges = list(edges)

        self.model = TransE(num_entities,num_relations,768).to(torch.cuda.current_device())
        
        batch_size = self.batch_size
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        print(f'start training kge(transe),batch_size={batch_size},epoch={self.epochs}')
        for epoch in range(self.epochs):
            random.shuffle(edges)
            # 做正例
            # step_bar = tqdm(range(len(edges) // batch_size))
            print(f'##epoch##:{epoch}')
            for i in range(len(edges) // batch_size):

                positive_samples = edges[i*batch_size:(i+1)*batch_size]
                negative_samples = self.generate_fake_data(batch_size,edges)

                positive_samples = self.to_tensor(positive_samples)
                negative_samples = self.to_tensor(negative_samples)

                pos_distances, neg_distances = self.model(positive_samples, negative_samples, mode='double')
                loss = F.relu(pos_distances - neg_distances + 1.0).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if i % 100 == 0:
                    print(f'loss:{float(loss.item())}')
            #     step_bar.set_postfix({'loss':float(loss.item())})
            #     step_bar.update()
            # step_bar.close()
    
    def export_model(self,output_path):
        torch.save({
            'entity_embedding':self.model.entity_embedding.weight.data.cpu(),
            'relation_embedding':self.model.relation_embedding.weight.data.cpu()
        },output_path)

    def generate_fake_data(self,batch_size,edges):
        fake_edges = []
        for _ in range(batch_size):
            edge = random.choice(edges)
            if random.random() > 0.5:
                head = self.nodes[alias_smaple(self.nodes_accept_prob,self.nodes_alias_index)]
                fake_edge = (head,edge[1],edge[2])
            else:
                tail = self.nodes[alias_smaple(self.nodes_accept_prob,self.nodes_alias_index)]
                fake_edge = (edge[0],edge[1],tail)
            fake_edges.append(fake_edge)
        return fake_edges

    def to_tensor(self,edges):
        edges = torch.tensor(edges).to(torch.cuda.current_device())
        return edges
