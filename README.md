# Data Preparation
Download [Scene Flow Datasets](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html), [KITTI 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo), [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)


# Scene Flow Datasets
Please update `DATAPATH` in the ``sceneflow.sh`` file as your training data path.
## Training
Multi-GPU training
```
export NGPUS=8
export save_results_dir="./checkpoints"
./scripts/sceneflow.sh $NGPUS $save_results_dir  --dlossw "1.0,0.5"  --batch_size 2 --eval_freq 3  --model mynet
```


## Evaluation
* Set ``CKPT_FILE`` as your checkpoint file
```
export NGPUS=8
export save_results_dir="./checkpoints"
./scripts/sceneflow.sh $NGPUS $save_results_dir --loadckpt $CKPT_FILE-  "4,1"   --batch_size 2 --mode test  --model mynet
```



# KITTI 2012 / 2015
Please update `DATAPATH` in the ``kitti15.sh`` file as your training data path.

## Training
```
export NGPUS=8
export save_results_dir="./checkpoints"
./scripts/kitti15.sh $NGPUS $save_results_dir   --dlossw "0.5,1.0"  --batch_size 2 --eval_freq 3  --model mynet
```

## Evaluation
* Set ``CKPT_FILE`` as your checkpoint file
```
export NGPUS=8
export save_results_dir="./checkpoints"
./scripts/kitti15.sh $NGPUS $save_results_dir  --loadckpt $CKPT_FILE-   --batch_size 2 --mode test  --model mynet
```

## Save Disps
```
export save_path="./outputs/predictions"
./scripts/kitti15_save.sh $save_results_dir  --loadckpt $CKPT_FILE  --batch_size 2 --model mynet
```