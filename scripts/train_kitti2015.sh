loadckpt="./checkpoints/checkpoint_best.ckpt"
save_path="/output"

save_path_stage1=$save_path
# save_path_stage2=$save_path/stage2
# save_path_stage12=$save_path/stage12

# bash scripts/kitti15.sh 8  $save_path \
#                       --dataset kitti --datapath /data/laniakea/GAData/dataset/ --crop_height 240 --crop_width 528 \
#                       --test_dataset kitti --test_datapath /data/laniakea/GAData/dataset/kitti2015/   --test_crop_height 384  --test_crop_width 1248 \
#                       --trainlist ./filenames/kitti15_train_more.txt  --testlist ./filenames/kitti15_val.txt \
#                       --dlossw "0,1.0"  --using_apex \
#                       --save_freq 100 --eval_freq 50 --batch_size 6 --test_batch_size 2  --epochs 2000 --lr 0.0005 --lrepochs "1000:2" \
#                       --using_ns --ns_size 3 --model mynet \
#                       --loadckpt ./checkpoints/checkpoint_001999.ckpt 

bash scripts/kitti12.sh 8  $save_path \
                      --dataset kitti --datapath /data/laniakea/GAData/dataset/kitti2012/ --crop_height 240 --crop_width  528 \
                      --test_dataset kitti --test_datapath /data/laniakea/GAData/dataset/kitti2012/   --test_crop_height 384  --test_crop_width 1248 \
                      --trainlist ./filenames/kitti12_train.txt  --testlist ./filenames/kitti12_val.txt \
                      --dlossw "0,1.0"   --using_apex \
                      --save_freq 100 --eval_freq 50 --batch_size 6 --test_batch_size 2  --epochs 2000 --lr 0.0001 --lrepochs "1000,1500:2" \
                      --using_ns --ns_size 3  --model mynet \
                      --loadckpt ./checkpoints/checkpoint_001999.ckpt \
#                       --sync_bn

# ./scripts/kitti15.sh 2  $save_path_stage2\
#                        --dataset kitti --datapath /home/jiaxiaogang/research/GANet-Vision/dataset/kitti/ --crop_height 336 --crop_width 1200 \
#                        --test_dataset sceneflow --test_datapath /home/jiaxiaogang/research/GANet-Vision/dataset/scene_flow/  --test_crop_height 480 --test_crop_width 960 \
#                        --trainlist ./filenames/kitti15_train_more.txt  --testlist ./filenames/sceneflow_test_select.txt \
#                        --dlossw "0.5,2.0" \
#                        --save_freq 1 --eval_freq 1 --batch_size 4 --test_batch_size 2 --epochs 8 --lrepochs "400:10"  \
#                        --loadckpt $save_path_stage1/checkpoint_000799.ckpt


# ./scripts/kitti15.sh 2  $save_path_stage12 \
#                        --dataset kitti --datapath /home/jiaxiaogang/research/GANet-Vision/dataset/kitti/ --crop_height 240 --crop_width 1224 \
#                        --test_dataset kitti --test_datapath /home/jiaxiaogang/research/GANet-Vision/dataset/kitti/  --test_crop_height 384 --test_crop_width 1248 \
#                        --trainlist ./filenames/kitti15_trainval.txt  --testlist ./filenames/kitti15_trainval.txt \
#                        --dlossw "0.5,2.0"  \
#                        --save_freq 1 --eval_freq 1 --batch_size 4 --test_batch_size 2 --epochs 8 --lrepochs "400:10"  \
#                        --loadckpt $save_path_stage2/checkpoint_best.ckpt \
#                        --mode test3