#!/usr/bin/env bash

export PYTHONWARNINGS="ignore"

save_path="/output"

DATAPATH="/data/laniakea/GAData/dataset/scene_flow/"

python -m torch.distributed.launch --nproc_per_node=1 main.py --dataset sceneflow \
    --datapath $DATAPATH --trainlist ./filenames/sceneflow_train.txt --testlist ./filenames/sceneflow_test.txt \
    --test_datapath $DATAPATH --test_dataset sceneflow \
    --epochs 16 --lr 0.001 --lrepochs "10,14:2" \
    --crop_width 528  --crop_height 240 --test_crop_width 960  --test_crop_height 528 --using_ns --ns_size 3 --using_apex\
    --dlossw "0.5, 2.0" --batch_size 6 --test_batch_size 2 --eval_freq 3 \
    --model mynet --logdir $save_path  ${@:3} | tee -a  $save_path/log.txt \
    # --loadckpt "./checkpoints/checkpoint_000015.ckpt" \
    
     
