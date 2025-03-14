
dt=`date '+%Y%m%d_%H%M%S'`

# CUDA_VISIBLE_DEVICES=3 python main.py --dataset PromptGNNDataset --trainer PromptGNNTrainer --model_name PromptGATEdge --bert_path /root/prompt-gnn/PLM/Erlangshen-Longformer-110M --kge_trainer BertKGETrainer > ../logs/logs/Longformer-MSRPS-Node-bert-${dt}.log
# CUDA_VISIBLE_DEVICES=3 python main.py --dataset PromptGNNDataset --trainer PromptGNNTrainer --model_name PromptGATEdge --bert_path /root/prompt-gnn/PLM/bert-base-chinese --kge_trainer BertKGETrainer > ../logs/logs/bert-base-chinese-MSRPS-Node-bert-${dt}.log
CUDA_VISIBLE_DEVICES=3 python main.py --dataset PromptGNNDataset --trainer PromptGNNTrainer --model_name PromptGATEdge --bert_path /root/prompt-gnn/PLM/Smed-bert --kge_trainer BertKGETrainer > ../logs/logs/Smed-bert-MSRPS-Node-bert-${dt}.log

# CUDA_VISIBLE_DEVICES=3 python main.py --dataset PromptGNNDataset --trainer PromptGNNTrainer --model_name PromptGATEdge --bert_path /root/prompt-gnn/PLM/Erlangshen-Longformer-110M --kge_trainer KGETrainer > ../logs/logs/Longformer-MSRPS-Node-transE-${dt}.log
# CUDA_VISIBLE_DEVICES=3 python main.py --dataset PromptGNNDataset --trainer PromptGNNTrainer --model_name PromptGATEdge --bert_path /root/prompt-gnn/PLM/bert-base-chinese --kge_trainer KGETrainer > ../logs/logs/bert-base-chinese-MSRPS-Node-transE-${dt}.log
CUDA_VISIBLE_DEVICES=3 python main.py --dataset PromptGNNDataset --trainer PromptGNNTrainer --model_name PromptGATEdge --bert_path /root/prompt-gnn/PLM/Smed-bert --kge_trainer KGETrainer > ../logs/logs/Smed-bert-MSRPS-Node-transE-${dt}.log

