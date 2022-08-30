from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
from models import __models__
from utils import *
from torch.utils.data import DataLoader
import gc
import skimage
from skimage import io

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Cascade Stereo Network (CasStereoNet)')
parser.add_argument('--model', default='gwcnet-c', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')

parser.add_argument('--test_dataset', required=True, help='dataset name', choices=__datasets__.keys())
parser.add_argument('--test_datapath', required=True, help='data path')
parser.add_argument('--testlist', required=True, help='testing list')

parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')

parser.add_argument('--logdir', required=True, help='the directory to save logs and checkpoints')
parser.add_argument('--loadckpt', help='load the weights from a specific checkpoint')

parser.add_argument("--local_rank", type=int, default=0)

parser.add_argument('--dlossw', type=str, default="0.5,2.0", help='depth loss weight for different stage')
parser.add_argument('--cr_base_chs', type=str, default="32,32,16", help='cost regularization base channels')
parser.add_argument('--grad_method', type=str, default="detach", choices=["detach", "undetach"], help='predicted disp detach, undetach')

parser.add_argument('--using_ns', action='store_true', help='using neighbor search')
parser.add_argument('--ns_size', type=int, default=3, help='nb_size')

parser.add_argument('--test_crop_height', type=int, required=True, help="crop height")
parser.add_argument('--test_crop_width', type=int, required=True, help="crop width")


# parse arguments
args = parser.parse_args()

# dataset, dataloader
Test_StereoDataset = __datasets__[args.test_dataset]
test_dataset = Test_StereoDataset(args.test_datapath, args.testlist, False,
                                  crop_height=args.test_crop_height, crop_width=args.test_crop_width,
                                  test_crop_height=args.test_crop_height, test_crop_width=args.test_crop_width)

TestImgLoader = DataLoader(test_dataset, 1, shuffle=False, num_workers=4, drop_last=False)

# model, optimizer
model = __models__[args.model](
                                maxdisp=args.maxdisp)

model.cuda()

# load parameters
print("loading model {}".format(args.loadckpt))
state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict['model'])

model = nn.DataParallel(model)

num_stage = 2

def test():
    total_time = 0
    num = 0
    for batch_idx, sample in enumerate(TestImgLoader):
        # start_time = time.time()
        disp_est_np = test_sample(sample)
        # print('time = {:3f}'.format(time.time() - start_time))
        
        # if time.time() - start_time < 1:
        #     #print("LONGTIME")
        # #else:
        #     num += 1
        #     total_time += time.time() - start_time

        disp_est_np = tensor2numpy(disp_est_np)
        top_pad_np = tensor2numpy(sample["top_pad"])
        right_pad_np = tensor2numpy(sample["right_pad"])
        left_filenames = sample["left_filename"]
        

        for disp_est, top_pad, right_pad, fn in zip(disp_est_np, top_pad_np, right_pad_np, left_filenames):
            assert len(disp_est.shape) == 2
            disp_est = np.array(disp_est[top_pad:, :-right_pad], dtype=np.float32)
            fn = os.path.join(args.logdir, fn.split('/')[-1])
            print("saving to", fn, disp_est.shape)
            disp_est_uint = np.round(disp_est * 256).astype(np.uint16)
            skimage.io.imsave(fn, disp_est_uint)
    # print('total_time = {:3f}'.format(total_time))
    # print('num = {:3f}'.format(num))


# test one sample
@make_nograd_func
def test_sample(sample):
    model.eval()
    
    outputs = model(sample['left'].cuda(), sample['right'].cuda())
    
    outputs_stage = outputs["stage{}".format(num_stage)]
    disp_ests = [outputs_stage["pred"]]

    return disp_ests[-1]


if __name__ == '__main__':
    test()
