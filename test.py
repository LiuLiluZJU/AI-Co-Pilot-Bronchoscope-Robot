import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
from torchviz import make_dot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
from tensorboardX import SummaryWriter

from model import BranchedCIMNet, CIMNet, ResnetGenerator_our, fixedBranchedCIMNetWithDepthAngle, fixedBranchedCIMNetWithDepthAngleMultiFrame
from onlineSimulation_with_depth_transfer_angle_disctrl_inertia import onlineSimulationWithNetwork as onlineSimulator
from onlineSimulation_with_depth_transfer_angle_disctrl_inertia_save3d import onlineSimulationWithNetwork as onlineSimulator_save3d
from dataset import AlignDataSet, AlignDataSetDagger, AlignDataSetDaggerAug, AlignDataSetSplit
from utils import get_gpu_mem_info
from utils import get_transform, tensor2im
np.random.seed(0)


def get_args():
    parser = argparse.ArgumentParser(description='Train the SCNet on images and target landmarks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', dest='epochs', metavar='E', type=int, default=800, help='Number of epochs')
    parser.add_argument('-de', '--decay-epochs', dest='decay_epochs', metavar='DE', type=int, default=800, help='Number of decay epochs')
    parser.add_argument('-b', '--batch-size', dest='batchsize', metavar='B', type=int, default=128, help='Batch size')
    parser.add_argument('-l', '--learning-rate', dest='lr', metavar='LR', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('-f', '--load', dest='load', type=bool, default=True, help='Load model from a .pth file')
    parser.add_argument('-fe', '--load-epoch', dest='load_epoch', type=int, default=269, help='Load model from which epoch')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('-t', '--tensorboard', dest='tensorboard', type=bool, default=True, help='Record data in tensorboard')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=0.2, help='Percent of the data that is used as validation (0-1)')
    parser.add_argument('-d', '--dataset-dir', dest='dataset_dir', type=str, default="E:/pybullet_test/train_set",
                        help='Path of dataset for training and validation')
    parser.add_argument('-vald', '--val-dataset', dest='val_dataset', type=str, default="centerlines_siliconmodel3_video", 
                        help='Name of dataset for validation')
    parser.add_argument('-m', '--model-dir', dest='model_dir', type=str, default="F:/conditional_imitation_learning_checkpoints/sliliconmodel1&2_pink_hypercontrol(10-30)_randstart_CILRS_fixebranched_fixcor_aug70_transfer30_with_depth_angle_advd_distctrl_randroll_randintensity", 
                        help='Path of trained model for saving')
    parser.add_argument('-rs', '--results-saving-dir', dest='results_saving_dir', type=str, default="results/2023-04-04", 
                        help='Path of trained model for saving')
    # Style transfer parameters
    parser.add_argument('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
    parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
    parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
    parser.add_argument('--num_threads', default=0, type=int, help='# threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--load_size', type=int, default=200, help='scale images to this size')
    parser.add_argument('--crop_size', type=int, default=200, help='then crop to this size')
    parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
    parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
    parser.add_argument('--no_flip', type=bool, default=True, help='if specified, do not flip the images for data augmentation')
    parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # net = CIMNet()
    # net = BranchedCIMNet()
    net = fixedBranchedCIMNetWithDepthAngle()
    # net = fixedBranchedCIMNetWithDepthAngleMultiFrame()
    if args.load:
        pretrained_dict = torch.load(os.path.join(args.model_dir, "regular_{}.pth".format(args.load_epoch)), map_location=device)
        model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # 不必要的键去除掉
        model_dict.update(pretrained_dict)  # 覆盖现有的字典里的条目
        net.load_state_dict(model_dict)
    else:
        args.load_epoch = -1
    net.to(device=device)

    # Style tranfer network
    transform_func = get_transform(args)
    # net_transfer = ResnetGenerator(input_nc=3, output_nc=3, ngf=64, norm_layer=functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False), use_dropout=False, n_blocks=9)
    net_transfer = ResnetGenerator_our(input_nc=3, output_nc=3, ngf=64, n_blocks=9)
    pretrained_dict = torch.load("E:/AttentionGAN-master/checkpoints/bronchus5_attentiongan/latest_net_G_A.pth", map_location=device)
    # pretrained_dict = torch.load("E:/AttentionGAN-master/checkpoints/bronchus5_policy_attentiongan_union/5_net_G_A.pth", map_location=device)
    model_dict = net_transfer.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # 不必要的键去除掉
    model_dict.update(pretrained_dict)  # 覆盖现有的字典里的条目
    net_transfer.load_state_dict(model_dict)
    net_transfer.to(device=device)
    net_transfer.eval()

    online_test_centerline_names_list = os.listdir(os.path.join(args.dataset_dir, args.val_dataset))

    # dataset = AlignDataSetDagger(args.dataset_dir)
    dataset = AlignDataSetDaggerAug(args.dataset_dir)
    # dataset.updateDataSet()
    # dataset.updateSpecificCenterlineDataSet(online_test_centerline_names_list[0])

    optimizer = torch.optim.Adam([{'params': net.parameters()}], lr=1e-4, weight_decay=1e-8)
    mse_loss = nn.MSELoss()

    test_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)  # num_workers=0 in Windows

    if args.tensorboard:
        writer = SummaryWriter()

    global_batch_count = 0
    count = 0

    with torch.no_grad():
        net.eval()
        results = {}
        results_error_list = {}
        total_error = 0
        total_ratio = 0
        centerlines_position_pred = []
        centerlines_position_gt = []
        # sns.set(style="white", font_scale=1.5)
        # plt.gcf().subplots_adjust(bottom=0.2)
        for online_test_centerline_name in online_test_centerline_names_list:
            simulator = onlineSimulator(args.dataset_dir, online_test_centerline_name, renderer='pyrender', training=False)
            path_centerline_pred_position_list, path_centerline_error_list, path_centerline_ratio_list, path_centerline_length_list, safe_distance \
                = simulator.run(net, model_dir=args.model_dir, epoch=args.load_epoch, net_transfer=None, transform_func=dataset.transforms_eval, training=False)
            print(np.mean(path_centerline_error_list))
            print(path_centerline_error_list)
            print(path_centerline_ratio_list)
            print(path_centerline_length_list)
            path_centerline_length_list = [int(x * 100) for x in path_centerline_length_list]
            path_centerline_error_list = [x * 100 for x in path_centerline_error_list]
            total_error += np.mean(path_centerline_error_list)
            total_ratio += path_centerline_ratio_list[-1]
            results[online_test_centerline_name] = [np.mean(path_centerline_error_list), path_centerline_ratio_list[-1]]
            results_error_list[online_test_centerline_name] = path_centerline_error_list
            # plt.plot(path_centerline_length_list, path_centerline_error_list, label=online_test_centerline_name)
            count+=1
            # if count > 16:
            #     break
            # fig = plt.figure()
            # ax = Axes3D(fig)
            # ax.plot(simulator.centerlineArray[:, 0], simulator.centerlineArray[:, 1], simulator.centerlineArray[:, 2], c=(0 / 255, 255 / 255, 0 / 255))
            # path_centerline_pred_position_array = np.array(path_centerline_pred_position_list)
            # ax.plot(path_centerline_pred_position_array[:, 0], path_centerline_pred_position_array[:, 1], path_centerline_pred_position_array[:, 2])
            # plt.show()
            # saving ground truth and predicted centerlines
            centerlines_position_pred.append(np.array(path_centerline_pred_position_list))
            centerlines_position_gt.append(simulator.centerlineArray)

            # # Generate and save 3d map
            # simulator = onlineSimulator_save3d(args.dataset_dir, online_test_centerline_name, renderer='pyrender', training=False)
            # simulator.run(net, epoch=args.load_epoch, net_transfer=None, transform_func=dataset.transforms_eval, training=False)

        # error_list = []
        # ratio_list = []
        # for key in results.keys():
        #     error_list.append(results[key][0])
        #     ratio_list.append(results[key][1])
        # completed_path_rate = np.sum(np.abs(np.array(ratio_list) - 1) < 1e-5) / len(ratio_list)
        # print(results)
        # print(error_list, ratio_list)
        # print("mean_error, mean_ratio:", total_error / len(online_test_centerline_names_list), total_ratio / len(online_test_centerline_names_list))
        # print("mean_error, std_error, mean_ratio, std_ratio:", np.mean(error_list), np.std(error_list), np.mean(ratio_list), np.std(ratio_list))
        # print("completed_path_rate:", completed_path_rate)
        # # plt.xlim((0, 230))
        # # plt.xlim((0, 160))
        # # plt.ylim((0, 8))
        # plt.xlabel("Distance(mm)")
        # plt.ylabel("Error(mm)")
        # # plt.legend()
        # plt.savefig(os.path.join(args.results_saving_dir, args.val_dataset + "-" + args.model_dir.split("/")[-1] + "-" + str(args.load_epoch) + ".png"))
        # plt.clf()
        # # plt.show()

        # # Show gt and pred centerline paths
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # for index in range(len(online_test_centerline_names_list)):
        #     ax.plot(centerlines_position_gt[index][:, 0], centerlines_position_gt[index][:, 1], centerlines_position_gt[index][:, 2], c=(0 / 255, 255 / 255, 0 / 255))
        #     ax.plot(centerlines_position_pred[index][:, 0], centerlines_position_pred[index][:, 1], centerlines_position_pred[index][:, 2])
        # plt.savefig(os.path.join(args.results_saving_dir, args.val_dataset + "-" + args.model_dir.split("/")[-1] + "-" + str(args.load_epoch) + "-" + "3d-traj" + ".png"))
        # plt.clf()
        # # plt.show()

        # # Show gt centerline paths
        # sns.set(style="white", font_scale=1.5)
        # plt.gcf().subplots_adjust(bottom=0.2)
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # for index in range(len(online_test_centerline_names_list)):
        #     ax.plot(centerlines_position_gt[index][:, 0], centerlines_position_gt[index][:, 1], centerlines_position_gt[index][:, 2], c=(0 / 255, 255 / 255, 0 / 255))
        # plt.savefig(os.path.join(args.results_saving_dir, args.val_dataset + "-" + args.model_dir.split("/")[-1] + "-" + str(args.load_epoch) + "-" + "3d-traj-gt" + ".png"))
        # plt.clf()

        # # Save results
        # df = pd.DataFrame(results, index=['avg error', 'comp ratio'])
        # df2 = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
        # df2.to_excel(os.path.join(args.results_saving_dir, args.val_dataset + "-" + args.model_dir.split("/")[-1] + "-" + str(args.load_epoch) + ".xlsx"), sheet_name="sheet1", index=True)
        # f = open(os.path.join(args.results_saving_dir, args.val_dataset + "-" + args.model_dir.split("/")[-1] + "-" + str(args.load_epoch) + ".txt"), 'w')
        # for key in results_error_list.keys():
        #     # f.write(key + ' ' + str(results[key][0]) + ' ' + str(results[key][1]) + '\n')
        #     f.write(key)
        #     for i in results_error_list[key]:
        #         f.write(',' + str(i))
        #     f.write('\n')
        # f.write('mean_error, std_error, mean_ratio, std_ratio: ' + str(np.mean(error_list)) + ' ' + str(np.std(error_list)) + ' ' + str(np.mean(ratio_list)) + ' ' + str(np.std(ratio_list)) + '\n')
        # f.write('completed_path_rate: ' + str(completed_path_rate))
        # f_3d_gt = open(os.path.join(args.results_saving_dir, args.val_dataset + "-" + args.model_dir.split("/")[-1] + "-" + str(args.load_epoch) + "-" + "3d-traj-gt" + ".txt"), 'w')
        # for index in range(len(online_test_centerline_names_list)):
        #     for point in centerlines_position_gt[index]:
        #         f_3d_gt.write("({},{},{});".format(point[0], point[1], point[2]))
        #     f_3d_gt.write("\n")
        # f_3d_pred = open(os.path.join(args.results_saving_dir, args.val_dataset + "-" + args.model_dir.split("/")[-1] + "-" + str(args.load_epoch) + "-" + "3d-traj-pred" + ".txt"), 'w')
        # for index in range(len(online_test_centerline_names_list)):
        #     for point in centerlines_position_pred[index]:
        #         f_3d_pred.write("({},{},{});".format(point[0], point[1], point[2]))
        #     f_3d_pred.write("\n")