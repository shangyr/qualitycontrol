# python -m torch.distributed.launch --nproc_per_node=3 main.py
# python -m torch.distributed.launch --nproc_per_node=3 main.py --other_lr 1e-3
# python -m torch.distributed.launch --nproc_per_node=3 main.py --other_lr 5e-4
# python -m torch.distributed.launch --nproc_per_node=3 main.py --label_smooth_lambda 0.0
# python -m torch.distributed.launch --nproc_per_node=3 main.py --label_smooth_lambda 0.0 --other_lr 1e-3

#!/bin/bash


# 嵌套的for循环生成所有可能的超参数组合
for lr in 5e-4 2e-4 1e-3; do
  for bs in 16 8; do
    for fn in BCE; do
      # 在此处调用你的深度学习程序，并传递超参数
      python main.py --other_lr $lr --batch_size $bs --loss_fn $fn
    done
  done
done
