!/usr/bin/env bash
#set -x
export PYTHONWARNINGS="ignore"

save_path=$2

#DATAPATH="/home/jiaxiaogang/research/GANet-Vision/dataset/kitti2015/"

python -m torch.distributed.launch --nproc_per_node=$1 main.py \
    --using_apex --model mynet --logdir $save_path  ${@:3} | tee -a  $save_path/log.txt \
    # --dataset kitti \
    # --datapath $DATAPATH --trainlist ./filenames/kitti15_train.txt --testlist ./filenames/kitti15_val.txt \
    # --test_datapath $DATAPATH --test_dataset kitti \
    # --epochs 300 --lrepochs "200:10" \
    # --crop_width 528  --crop_height 240 --test_crop_width 1248  --test_crop_height 384 \
    # --dlossw "0.5,2.0" \
    #--model mynet --logdir $save_path  ${@:3} | tee -a  $save_path/log.txt \
