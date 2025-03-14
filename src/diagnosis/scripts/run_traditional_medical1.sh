
dt=`date '+%Y%m%d_%H%M%S'`

# CUDA_VISIBLE_DEVICES=1 python main.py --dataset AutoDataset --trainer AutoTrainer --model_name AutoModel --bert_path /root/prompt-gnn/PLM/Erlangshen-Longformer-110M > ../logs/logs/Longformer-${dt}.log
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset AutoDataset --trainer AutoTrainer --model_name AutoModel --bert_path /root/prompt-gnn/PLM/bert-base-chinese > ../logs/logs/bert-${dt}.log
CUDA_VISIBLE_DEVICES=1 python main.py --dataset AutoDataset --trainer AutoTrainer --model_name AutoModel --bert_path /root/prompt-gnn/PLM/Smed-bert > ../logs/logs/Smed-bert-${dt}.log

# CUDA_VISIBLE_DEVICES=1 python main.py --dataset PromptGNNDataset --trainer PromptGNNTrainer --model_name PromptGATEdge --bert_path /root/prompt-gnn/PLM/Erlangshen-Longformer-110M > ../logs/logs/Longformer-MSRPS-${dt}.log
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset PromptGNNDataset --trainer PromptGNNTrainer --model_name PromptGATEdge --bert_path /root/prompt-gnn/PLM/bert-base-chinese > ../logs/logs/bert-base-chinese-MSRPS-${dt}.log
CUDA_VISIBLE_DEVICES=1 python main.py --dataset PromptGNNDataset --trainer PromptGNNTrainer --model_name PromptGATEdge --bert_path /root/prompt-gnn/PLM/Smed-bert > ../logs/logs/Smed-bert-MSRPS-${dt}.log

