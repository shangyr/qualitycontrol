

current_time=$(date "+%Y-%m-%d %H:%M:%S")
echo "The current time is: $current_time"
export CUDA_VISIBLE_DEVICES=2,3

nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port=1235 main.py > run_$current_time.out &

