#!/bin/bash
python main.py --config Chinese50SmallConfig --dataset PromptGNNDataset --trainer PromptGNNTrainer --model_name PromptGAT --bert_path bert-base-chinese > ../logs/logs/bert-base-chinese-chinesesmall-gat.log
python main.py --config Chinese50SmallConfig --dataset PromptGNNDataset --trainer PromptGNNTrainer --model_name PromptGAT --bert_path IDEA-CCNL/Erlangshen-Longformer-110M > ../logs/logs/IDEA-CCNL-Erlangshen-Longformer-110M-chinesesmall-gat.log
python main.py --config Chinese50SmallConfig --dataset PromptGNNDataset --trainer PromptGNNTrainer --model_name PromptGAT --bert_path nghuyong/ernie-health-zh > ../logs/logs/nghuyong-ernie-health-zh-chinesesmall-gat.log
python main.py --config Chinese50SmallConfig --dataset PromptGNNDataset --trainer PromptGNNTrainer --model_name PromptGAT --bert_path emilyalsentzer/Bio_ClinicalBERT > ../logs/logs/emilyalsentzer-Bio_ClinicalBERT-chinesesmall-gat.log
python main.py --config Chinese50SmallConfig --dataset PromptGNNDataset --trainer PromptGNNTrainer --model_name PromptGAT --bert_path dmis-lab/biobert-base-cased-v1.2 > ../logs/logs/dmis-lab-biobert-base-cased-v1.2-chinesesmall-gat.log
python main.py --config Chinese50SmallConfig --dataset PromptGNNDataset --trainer PromptGNNTrainer --model_name PromptGAT --bert_path freedomking/mc-bert > ../logs/logs/freedomking-mc-bert-chinesesmall-gat.log
python main.py --config Chinese50SmallConfig --dataset PromptGNNDataset --trainer PromptGNNTrainer --model_name PromptGAT --bert_path /home/lixin/PLM/SMedbert > ../logs/logs/SMedbert-chinesesmall-gat.log
