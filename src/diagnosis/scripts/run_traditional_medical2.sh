
dt=`date '+%Y%m%d_%H%M%S'`

# CUDA_VISIBLE_DEVICES=2 python main.py --dataset PromptGNNDataset --trainer PromptGNNTrainer --model_name PromptGNNEdge --bert_path /root/prompt-gnn/PLM/Erlangshen-Longformer-110M > ../logs/logs/Longformer-MSRPS-GCN-${dt}.log
# CUDA_VISIBLE_DEVICES=2 python main.py --dataset PromptGNNDataset --trainer PromptGNNTrainer --model_name PromptGNNEdge --bert_path /root/prompt-gnn/PLM/bert-base-chinese > ../logs/logs/bert-base-chinese-MSRPS-GCN-${dt}.log
CUDA_VISIBLE_DEVICES=2 python main.py --dataset PromptGNNDataset --trainer PromptGNNTrainer --model_name PromptGNNEdge --bert_path /root/prompt-gnn/PLM/Smed-bert > ../logs/logs/Smed-bert-MSRPS-GCN-${dt}.log

# CUDA_VISIBLE_DEVICES=2 python main.py --dataset PromptGNNDataset --trainer PromptGNNTrainer --model_name PromptGATEdge --bert_path /root/prompt-gnn/PLM/Erlangshen-Longformer-110M --path_type v1 > ../logs/logs/Longformer-MSRPS-ALL-${dt}.log
# CUDA_VISIBLE_DEVICES=2 python main.py --dataset PromptGNNDataset --trainer PromptGNNTrainer --model_name PromptGATEdge --bert_path /root/prompt-gnn/PLM/bert-base-chinese --path_type v1 > ../logs/logs/bert-base-chinese-MSRPS-ALL-${dt}.log
CUDA_VISIBLE_DEVICES=2 python main.py --dataset PromptGNNDataset --trainer PromptGNNTrainer --model_name PromptGATEdge --bert_path /root/prompt-gnn/PLM/Smed-bert --path_type v1 > ../logs/logs/Smed-bert-MSRPS-ALL-${dt}.log
