!/usr/bin/env bash
#set -x
export PYTHONWARNINGS="ignore"

save_path=$2


python -m torch.distributed.launch --nproc_per_node=$1 main.py  \
    --using_apex --model mynet --logdir $save_path  ${@:3} | tee -a  $save_path/log.txt