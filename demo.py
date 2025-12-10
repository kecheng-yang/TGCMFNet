# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str("0")
import argparse
import torch
import torch.utils.data as Data
import numpy as np
import time

from utils import train_patch, show_calaError, get_dataset
from TGCMFNet import train_network


# torch.cuda.set_device(0)
# -------------------------------------------------------------------------------XT
# Parameter Setting
parser = argparse.ArgumentParser("TGCMFNet")
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=0, help='number of seed')
parser.add_argument('--epoches', type=int, default=300, help='epoch number')
# parser.add_argument('--epoches2', type=int, default=0, help='epoch number')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--factor_lambda', type=float, default=0.1, help='theta')
parser.add_argument('--dataset', choices=['muufl', 'trento', 'houston2013', 'augsburg', "berlin"], default='houston2013', help='dataset')
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
parser.add_argument('--patch_size', type=int, default=14, help='number1 of patches')
parser.add_argument('--num_labelled', type=int, default=20, help='number of sampling from unlabeled samples')
parser.add_argument('--depth', type=int, default=2, help='depth')
parser.add_argument('--train', default=True, help='train or test')

args = parser.parse_args()


def train():
    print("----training----")
    # -------------------------------------------------------------------------------
    # prepare data
    Data1, Data2, TrLabel, TsLabel, num_class, label_values, label_queue = get_dataset(args.dataset, args.num_labelled)
    args.num_class = num_class


    Data1 = Data1.astype(np.float32)
    Data2 = Data2.astype(np.float32)
    [m1, n1, l1] = np.shape(Data1)
    Data2 = Data2.reshape([m1, n1, -1])  # when lidar is one band, this is used
    height1, width1, band1 = Data1.shape
    height2, width2, band2 = Data2.shape
    # data size
    print("height1={0},width1={1},band1={2}".format(height1, width1, band1))
    print("height2={0},width2={1},band2={2}".format(height2, width2, band2))



    patchsize = args.patch_size  # input spatial size for 2D-CNN
    pad_width = np.floor(patchsize / 2)
    pad_width = int(pad_width)  # 8
    TrainPatch1, TrainPatch2, TrainLabel = train_patch(Data1, Data2, patchsize, pad_width, TrLabel)
    # TrainPatch1, TrainPatch2, TrainLabel = RandomAugment(TrainPatch1, TrainPatch2, TrainLabel)
    TestPatch1, TestPatch2, TestLabel = train_patch(Data1, Data2, patchsize, pad_width, TsLabel)
    train_dataset = Data.TensorDataset(TrainPatch1, TrainPatch2, TrainLabel)
    train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    print('Data1 Training size and testing size are:', TrainPatch1.shape, 'and', TestPatch1.shape)
    print('Data2 Training size and testing size are:', TrainPatch2.shape, 'and', TestPatch2.shape)
    tic1 = time.time()
    pred_y, val_acc = train_network(train_loader, TestPatch1, TestPatch2, TestLabel,
                                    LR=args.learning_rate, EPOCH=args.epoches, SEED=args.seed, NC=band1, NCLidar=band2, Classes=args.num_class,
                                    patchsize=args.patch_size, batchsize=args.batch_size ,num_labelled=args.num_labelled, factor_lambda=args.factor_lambda, dataset_name=args.dataset,
                                    label_values=label_values, label_queue=label_queue, depth=args.depth)
    pred_y.type(torch.FloatTensor)
    TestLabel.type(torch.FloatTensor)
    # print("***********************Train raw***************************")
    # print("Maxmial Accuracy: %f, index: %i" % (max(val_acc), val_acc.index(max(val_acc))))
    toc1 = time.time()
    time_1 = toc1 - tic1
    print('1st training complete in {:.0f}m {:.0f}s'.format(time_1 / 60, time_1 % 60))
    OA, Kappa, CA, AA = show_calaError(pred_y, TestLabel)
    toc = time.time()
    time_all = toc - tic1
    print('All process complete in {:.0f}m {:.0f}s'.format(time_all / 60, time_all % 60))

    print("\n")
    return OA.item(), Kappa.item(), CA, AA.item()



if __name__ == '__main__':
    train()




