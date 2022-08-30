#!/usr/bin/env bash
#set -x
export PYTHONWARNINGS="ignore"

save_path="/output/stage1"

if [ ! -d $save_path ];then
    mkdir -p $save_path
fi

DATAPATH="/data/laniakea/GAData/dataset/kitti2012/"
python save_disp.py --test_dataset kitti  --test_datapath $DATAPATH --testlist ./filenames/kitti12_test.txt \
               --model mynet  --loadckpt ./checkpoints/checkpoint_001999.ckpt  \
               --logdir $save_path \
               --test_crop_width 1248  --test_crop_height 384 \
               --dlossw "0,1.0" --using_ns --ns_size 3 \
               ${@:3} | tee -a  $save_path/log.txt