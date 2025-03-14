
dt=`date '+%Y%m%d_%H%M%S'`

CUDA_VISIBLE_DEVICES=0 python main.py --dataset AutoDataset --trainer AutoTrainer --model_name TextRNN --bert_path ../data/traditional_chinese_medicine/embeds/128_0_10_cb_5n_5w.embeds --embedding_dim 128 > ../logs/logs/TextRNN-${dt}.log
CUDA_VISIBLE_DEVICES=0 python main.py --dataset AutoDataset --trainer AutoTrainer --model_name RNNAttn --bert_path ../data/traditional_chinese_medicine/embeds/128_0_10_cb_5n_5w.embeds --embedding_dim 128 > ../logs/logs/RNNAttn-${dt}.log
CUDA_VISIBLE_DEVICES=0 python main.py --dataset AutoDataset --trainer AutoTrainer --model_name RCNN --bert_path ../data/traditional_chinese_medicine/embeds/128_0_10_cb_5n_5w.embeds --embedding_dim 128 > ../logs/logs/RCNN-${dt}.log
CUDA_VISIBLE_DEVICES=0 python main.py --dataset AutoDataset --trainer AutoTrainer --model_name DPCNN --bert_path ../data/traditional_chinese_medicine/embeds/128_0_10_cb_5n_5w.embeds --embedding_dim 128 > ../logs/logs/DPCNN-${dt}.log
CUDA_VISIBLE_DEVICES=0 python main.py --dataset AutoDataset --trainer AutoTrainer --model_name CAML --bert_path ../data/traditional_chinese_medicine/embeds/128_0_10_cb_5n_5w.embeds --embedding_dim 128 > ../logs/logs/CAML-${dt}.log
CUDA_VISIBLE_DEVICES=0 python main.py --dataset AutoDataset --trainer AutoTrainer --model_name MultiResCNN --bert_path ../data/traditional_chinese_medicine/embeds/128_0_10_cb_5n_5w.embeds --embedding_dim 128 > ../logs/logs/MultiResCNN-${dt}.log
CUDA_VISIBLE_DEVICES=0 python main.py --dataset AutoDataset --trainer AutoTrainer --model_name TextCNN --bert_path ../data/traditional_chinese_medicine/embeds/128_0_10_cb_5n_5w.embeds --embedding_dim 128 > ../logs/logs/TextCNN-${dt}.log
CUDA_VISIBLE_DEVICES=0 python main.py --dataset AutoDataset --trainer AutoTrainer --model_name LAAT --bert_path ../data/traditional_chinese_medicine/embeds/128_0_10_cb_5n_5w.embeds --embedding_dim 128 > ../logs/logs/LAAT-${dt}.log
