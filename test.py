import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
import torchvision
import numpy as np
from torchviz import make_dot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from tensorboardX import SummaryWriter

from lib.network.model import ResnetGenerator_our, fixedBranchedCIMNetWithDepthAngle
from lib.engine.onlineSimulation import onlineSimulationWithNetwork as onlineSimulator
from lib.dataset.dataset import AlignDataSetDaggerAug
from lib.utils import get_gpu_mem_info, get_transform

np.random.seed(0)


def get_args():
    parser = argparse.ArgumentParser(description='Train the SCNet on images and target landmarks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--load', dest='load', type=bool, default=True, help='Load model from a .pth file')
    parser.add_argument('-d', '--dataset-dir', dest='dataset_dir', type=str, default="train_set", help='Path of dataset for training and validation')
    parser.add_argument('-m', '--model-dir', dest='model_dir', type=str, default="checkpoints", help='Path of trained model for saving')
    parser.add_argument('--human', action='store_true', help='AI co-pilot control with human, default: Artificial Expert Agent')

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = fixedBranchedCIMNetWithDepthAngle()
    if args.load:
        pretrained_dict = torch.load(os.path.join(args.model_dir, "policy_model.pth"), map_location=device)
        model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # 不必要的键去除掉
        model_dict.update(pretrained_dict)  # 覆盖现有的字典里的条目
        net.load_state_dict(model_dict)
    else:
        args.load_epoch = -1
    net.to(device=device)

    online_test_centerline_names_list = ['siliconmodel3 Centerline model'] + ['siliconmodel3 Centerline model_{}'.format(x) for x in range(1, 60)]

    transforms_eval = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    global_batch_count = 0
    count = 0

    with torch.no_grad():
        net.eval()
        for online_test_centerline_name in online_test_centerline_names_list:
            simulator = onlineSimulator(args.dataset_dir, online_test_centerline_name, renderer='pyrender', training=False)
            path_centerline_pred_position_list, path_centerline_error_list, path_centerline_ratio_list, path_centerline_length_list, safe_distance \
                = simulator.run(args, net, model_dir=args.model_dir, net_transfer=None, transform_func=transforms_eval, training=False)
            print(np.mean(path_centerline_error_list))
            print(path_centerline_error_list)
            print(path_centerline_ratio_list)
            print(path_centerline_length_list)

