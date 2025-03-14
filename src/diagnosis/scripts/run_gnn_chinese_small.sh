#!/bin/bash

dt=`date '+%Y%m%d_%H%M%S'`

# python main.py --config Chinese50SmallV2Config --dataset PromptGNNDataset --trainer PromptGNNTrainer --model_name PromptGAT --bert_path bert-base-chinese > ../logs/logs/bert-base-chinese-chinesesmall-gat-${dt}.log
# python main.py --config Chinese50SmallV2Config --dataset PromptGNNDataset --trainer PromptGNNTrainer --model_name PromptGAT --bert_path IDEA-CCNL/Erlangshen-Longformer-110M > ../logs/logs/IDEA-CCNL-Erlangshen-Longformer-110M-chinesesmall-gat-${dt}.log
# python main.py --config Chinese50SmallV2Config --dataset PromptGNNDataset --trainer PromptGNNTrainer --model_name PromptGAT --bert_path nghuyong/ernie-health-zh > ../logs/logs/nghuyong-ernie-health-zh-chinesesmall-gat-${dt}.log

python main.py --config Chinese50Config --dataset PromptGNNDataset --trainer PromptGNNTrainer --model_name PromptGATEdge --bert_path bert-base-chinese > ../logs/logs/bert-base-chinese-chinese50-gatedge-${dt}.log
python main.py --config Chinese50Config --dataset PromptGNNDataset --trainer PromptGNNTrainer --model_name PromptGATEdge --bert_path IDEA-CCNL/Erlangshen-Longformer-110M > ../logs/logs/IDEA-CCNL-Erlangshen-Longformer-110M-chinese50-gatedge-${dt}.log
python main.py --config Chinese50Config --dataset PromptGNNDataset --trainer PromptGNNTrainer --model_name PromptGATEdge --bert_path nghuyong/ernie-health-zh > ../logs/logs/nghuyong-ernie-health-zh-chinese50-gatedge-${dt}.log


# python main.py --config Chinese50SmallConfig --dataset PromptGNNDataset --trainer PromptGNNTrainer --model_name PromptGNN --bert_path emilyalsentzer/Bio_ClinicalBERT > ../logs/logs/emilyalsentzer-Bio_ClinicalBERT-chinesesmall-gnn-${dt}.log
# python main.py --config Chinese50SmallConfig --dataset PromptGNNDataset --trainer PromptGNNTrainer --model_name PromptGNN --bert_path dmis-lab/biobert-base-cased-v1.2 > ../logs/logs/dmis-lab-biobert-base-cased-v1.2-chinesesmall-gnn-${dt}.log
# python main.py --config Chinese50SmallConfig --dataset PromptGNNDataset --trainer PromptGNNTrainer --model_name PromptGNN --bert_path freedomking/mc-bert > ../logs/logs/freedomking-mc-bert-chinesesmall-gnn-${dt}.log
# python main.py --config Chinese50SmallConfig --dataset PromptGNNDataset --trainer PromptGNNTrainer --model_name PromptGNN --bert_path /home/lixin/PLM/SMedbert > ../logs/logs/SMedbert-chinesesmall-gnn-${dt}.log
