#!/bin/bash

dt=`date '+%Y%m%d_%H%M%S'`

python main.py --dataset AutoDataset --trainer AutoTrainer --model_name TextRNN --bert_path ../data/traditional_chinese_medicine/embeds/128_0_10_cb_5n_5w.embeds --embedding_dim 128 > ../logs/logs/TextRNN-${dt}.log
python main.py --dataset AutoDataset --trainer AutoTrainer --model_name RNNAttn --bert_path ../data/traditional_chinese_medicine/embeds/128_0_10_cb_5n_5w.embeds --embedding_dim 128 > ../logs/logs/RNNAttn-${dt}.log
python main.py --dataset AutoDataset --trainer AutoTrainer --model_name RCNN --bert_path ../data/traditional_chinese_medicine/embeds/128_0_10_cb_5n_5w.embeds --embedding_dim 128 > ../logs/logs/RCNN-${dt}.log
python main.py --dataset AutoDataset --trainer AutoTrainer --model_name DPCNN --bert_path ../data/traditional_chinese_medicine/embeds/128_0_10_cb_5n_5w.embeds --embedding_dim 128 > ../logs/logs/DPCNN-${dt}.log
python main.py --dataset AutoDataset --trainer AutoTrainer --model_name CAML --bert_path ../data/traditional_chinese_medicine/embeds/128_0_10_cb_5n_5w.embeds --embedding_dim 128 > ../logs/logs/CAML-${dt}.log
python main.py --dataset AutoDataset --trainer AutoTrainer --model_name MultiResCNN --bert_path ../data/traditional_chinese_medicine/embeds/128_0_10_cb_5n_5w.embeds --embedding_dim 128 > ../logs/logs/MultiResCNN-${dt}.log
python main.py --dataset AutoDataset --trainer AutoTrainer --model_name TextCNN --bert_path ../data/traditional_chinese_medicine/embeds/128_0_10_cb_5n_5w.embeds --embedding_dim 128 > ../logs/logs/TextCNN-${dt}.log
python main.py --dataset AutoDataset --trainer AutoTrainer --model_name LAAT --bert_path ../data/traditional_chinese_medicine/embeds/128_0_10_cb_5n_5w.embeds --embedding_dim 128 > ../logs/logs/LAAT-${dt}.log
python main.py --dataset AutoDataset --trainer AutoTrainer --model_name AutoModel --bert_path /root/prompt-gnn/PLM/Erlangshen-Longformer-110M > ../logs/logs/Longformer-${dt}.log
python main.py --dataset AutoDataset --trainer AutoTrainer --model_name AutoModel --bert_path /root/prompt-gnn/PLM/bert-base-chinese > ../logs/logs/bert-${dt}.log
python main.py --dataset AutoDataset --trainer AutoTrainer --model_name AutoModel --bert_path /root/prompt-gnn/PLM/Smed-bert > ../logs/logs/Smed-bert-${dt}.log

python main.py --dataset PromptGNNDataset --trainer PromptGNNTrainer --model_name PromptGATEdge --bert_path /root/prompt-gnn/PLM/Erlangshen-Longformer-110M > ../logs/logs/Longformer-MSRPS-${dt}.log
python main.py --dataset PromptGNNDataset --trainer PromptGNNTrainer --model_name PromptGATEdge --bert_path /root/prompt-gnn/PLM/bert-base-chinese > ../logs/logs/bert-base-chinese-MSRPS-${dt}.log
python main.py --dataset PromptGNNDataset --trainer PromptGNNTrainer --model_name PromptGATEdge --bert_path /root/prompt-gnn/PLM/Smed-bert > ../logs/logs/Smed-bert-MSRPS-${dt}.log

python main.py --dataset PromptGNNDataset --trainer PromptGNNTrainer --model_name PromptGNNEdge --bert_path /root/prompt-gnn/PLM/Erlangshen-Longformer-110M > ../logs/logs/Longformer-MSRPS-GCN-${dt}.log
python main.py --dataset PromptGNNDataset --trainer PromptGNNTrainer --model_name PromptGNNEdge --bert_path /root/prompt-gnn/PLM/bert-base-chinese > ../logs/logs/bert-base-chinese-MSRPS-GCN-${dt}.log
python main.py --dataset PromptGNNDataset --trainer PromptGNNTrainer --model_name PromptGNNEdge --bert_path /root/prompt-gnn/PLM/Smed-bert > ../logs/logs/Smed-bert-MSRPS-GCN-${dt}.log

python main.py --dataset PromptGNNDataset --trainer PromptGNNTrainer --model_name PromptGATEdge --bert_path /root/prompt-gnn/PLM/Erlangshen-Longformer-110M --path_type v1 > ../logs/logs/Longformer-MSRPS-ALL-${dt}.log
python main.py --dataset PromptGNNDataset --trainer PromptGNNTrainer --model_name PromptGATEdge --bert_path /root/prompt-gnn/PLM/bert-base-chinese --path_type v1 > ../logs/logs/bert-base-chinese-MSRPS-ALL-${dt}.log
python main.py --dataset PromptGNNDataset --trainer PromptGNNTrainer --model_name PromptGATEdge --bert_path /root/prompt-gnn/PLM/Smed-bert --path_type v1 > ../logs/logs/Smed-bert-MSRPS-ALL-${dt}.log

python main.py --dataset PromptGNNDataset --trainer PromptGNNTrainer --model_name PromptGATEdge --bert_path /root/prompt-gnn/PLM/Erlangshen-Longformer-110M --kge_trainer BertKGETrainer > ../logs/logs/Longformer-MSRPS-Node-bert-${dt}.log
python main.py --dataset PromptGNNDataset --trainer PromptGNNTrainer --model_name PromptGATEdge --bert_path /root/prompt-gnn/PLM/bert-base-chinese --kge_trainer BertKGETrainer > ../logs/logs/bert-base-chinese-MSRPS-Node-bert-${dt}.log
python main.py --dataset PromptGNNDataset --trainer PromptGNNTrainer --model_name PromptGATEdge --bert_path /root/prompt-gnn/PLM/Smed-bert --kge_trainer BertKGETrainer > ../logs/logs/Smed-bert-MSRPS-Node-bert-${dt}.log

python main.py --dataset PromptGNNDataset --trainer PromptGNNTrainer --model_name PromptGATEdge --bert_path /root/prompt-gnn/PLM/Erlangshen-Longformer-110M --kge_trainer KGETrainer > ../logs/logs/Longformer-MSRPS-Node-transE-${dt}.log
python main.py --dataset PromptGNNDataset --trainer PromptGNNTrainer --model_name PromptGATEdge --bert_path /root/prompt-gnn/PLM/bert-base-chinese --kge_trainer KGETrainer > ../logs/logs/bert-base-chinese-MSRPS-Node-transE-${dt}.log
python main.py --dataset PromptGNNDataset --trainer PromptGNNTrainer --model_name PromptGATEdge --bert_path /root/prompt-gnn/PLM/Smed-bert --kge_trainer KGETrainer > ../logs/logs/Smed-bert-MSRPS-Node-transE-${dt}.log

