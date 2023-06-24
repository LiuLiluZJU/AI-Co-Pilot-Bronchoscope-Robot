import os
import sys
import argparse
from cv2 import transform
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

from lib.network.model import ResnetGenerator_our, fixedBranchedCIMNetWithDepthAngle
from lib.engine.onlineSimulation import onlineSimulationWithNetwork
from lib.dataset.dataset import AlignDataSetDaggerWithDepthAugAngle
from lib.utils import get_gpu_mem_info, get_transform


def get_args():
    parser = argparse.ArgumentParser(description='Train the SCNet on images and target landmarks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', dest='epochs', metavar='E', type=int, default=2000, help='Number of epochs')
    parser.add_argument('-de', '--decay-epochs', dest='decay_epochs', metavar='DE', type=int, default=2000, help='Number of decay epochs')
    parser.add_argument('-b', '--batch-size', dest='batchsize', metavar='B', type=int, default=64, help='Batch size')
    parser.add_argument('-l', '--learning-rate', dest='lr', metavar='LR', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('-f', '--load', dest='load', type=bool, default=False, help='Load model from a .pth file')
    parser.add_argument('-fe', '--load-epoch', dest='load_epoch', type=int, default=229, help='Load model from which epoch')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('-t', '--tensorboard', dest='tensorboard', type=bool, default=True, help='Record data in tensorboard')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=0.2, help='Percent of the data that is used as validation (0-1)')
    parser.add_argument('-d', '--dataset-dir', dest='dataset_dir', type=str, default="train_set",
                        help='Path of dataset for training and validation')
    parser.add_argument('-m', '--model-dir', dest='model_dir', type=str, default="E:/cond_imitation_learning/checkpoints", 
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
    parser.add_argument('--transfer_model_dir', dest='transfer_model_dir', type=str, default="E:/policy-attention-gan-copy/checkpoints/bronchus9_attentiongan_AtoB_add_depth2_219lab/25_net_G_A.pth",
                        help='Path of dataset for style transfer model')

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # net = CIMNet()
    # net = BranchedCIMNet()
    net = fixedBranchedCIMNetWithDepthAngle()
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
    transform_func_transfer = get_transform(args)
    # net_transfer = ResnetGenerator(input_nc=3, output_nc=3, ngf=64, norm_layer=functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False), use_dropout=False, n_blocks=9)
    net_transfer = ResnetGenerator_our(input_nc=3, output_nc=3, ngf=64, n_blocks=9)
    pretrained_dict = torch.load(args.transfer_model_dir, map_location=device)
    model_dict = net_transfer.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # 不必要的键去除掉
    model_dict.update(pretrained_dict)  # 覆盖现有的字典里的条目
    net_transfer.load_state_dict(model_dict)
    net_transfer.to(device=device)
    net_transfer.eval()

    online_test_centerline_names_list = os.listdir(os.path.join(args.dataset_dir, "centerlines"))

    # dataset = AlignDataSetDagger(args.dataset_dir)
    dataset = AlignDataSetDaggerWithDepthAugAngle(args.dataset_dir, train_flag=True)
    # dataset.updateDataSet()
    # dataset.updateSpecificCenterlineDataSet(online_test_centerline_names_list[0])

    optimizer = torch.optim.Adam([{'params': net.parameters()}], lr=1e-4, weight_decay=1e-8)
    mse_loss = nn.MSELoss()
    L1_loss = nn.L1Loss()

    train_loader = DataLoader(dataset, batch_size=args.batchsize, shuffle=True, num_workers=0, pin_memory=True)  # num_workers=0 in Windows

    if args.tensorboard:
        writer = SummaryWriter()

    global_batch_count = 0

    for epoch in range(args.load_epoch + 1, args.epochs + args.decay_epochs + 1):

        print(f"Start {epoch}th epoch!")
        net.train()
        batch_count = 0
        for batch in train_loader:
            batch_count += 1
            global_batch_count += 1
            print("Start {}th epoch, {}th batch!".format(epoch, batch_count))
            image = batch[0].to(device=device, dtype=torch.float32)
            command = batch[1].to(device=device, dtype=torch.float32)
            action = batch[2].to(device=device, dtype=torch.float32)
            depth = batch[3].to(device=device, dtype=torch.float32)
            predicted_action, predicted_depth = net(image, command)
            # print("action, pred_action:", action, predicted_action)
            # loss = mse_loss(action * 100, predicted_action * 100)  # scale value 100 for fast convergence
            # loss_action = L1_loss(action * torch.tensor([100, 100, 100]).to(device=device, dtype=torch.float32), \
            #                 predicted_action * torch.tensor([100, 100, 100]).to(device=device, dtype=torch.float32))  # scale value 100 for fast convergence
            loss_action = L1_loss(action * 100, predicted_action * (np.pi / 2) * 100)  # scale value 100 for fast convergence, pred action belongs to (-1, 1) and needs scaling to (-pi / 2, pi / 2)
            loss_depth = mse_loss(predicted_depth, depth)
            loss = loss_action + loss_depth * 100
            # g = make_dot(output)
            # g.view()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("Loss:", loss)
            print("Loss action:", loss_action)
            print("Loss depth:", loss_depth)
            print("dataset length:", len(dataset))

            if args.tensorboard:
                writer.add_scalar('loss/total loss', loss.item(), global_step=global_batch_count)
                writer.add_scalar('loss/action loss', loss_action.item(), global_step=global_batch_count)
                writer.add_scalar('loss/depth loss', loss_depth.item(), global_step=global_batch_count)

            # Get GPU memory
            total_mem, used_mem, left_mem = get_gpu_mem_info()
            print("total, used, left memory:", total_mem, used_mem, left_mem)

        if (epoch + 1) % 1 == 0:  # online validation and update
            with torch.no_grad():
                random_index = np.random.choice([1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 39, 41, 43, 44])
                net_transfer.to(device=torch.device('cpu'))
                args.transfer_model_dir = "E:/policy-attention-gan-copy/checkpoints/bronchus14_attentiongan_AtoB_add_depth2/{}_net_G_A.pth".format(random_index)
                pretrained_dict = torch.load(args.transfer_model_dir, map_location=device)
                model_dict = net_transfer.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # 不必要的键去除掉
                model_dict.update(pretrained_dict)  # 覆盖现有的字典里的条目
                net_transfer.load_state_dict(model_dict)
                net_transfer.to(device=device)
                net_transfer.eval()
                net.eval()
                # for online_test_centerline_name in online_test_centerline_names_list:
                #     simulator = onlineSimulationWithNetwork(args.dataset_dir, online_test_centerline_name, renderer='pyrender')
                #     path_centerline_error_list, complete_ratio = simulator.run(net, epoch)
                #     if args.tensorboard:
                #         for index, error in enumerate(path_centerline_error_list):
                #             writer.add_scalar('paths/{}'.format(online_test_centerline_name), error, global_step=index)
                #         writer.add_scalar('mean_error/{}'.format(online_test_centerline_name), np.mean(path_centerline_error_list), global_step=epoch)
                #         writer.add_scalar('complete_ratio/{}'.format(online_test_centerline_name), complete_ratio, global_step=epoch)
                dataset_size = len(dataset)
                while dataset_size == len(dataset):
                    online_test_centerline_index = epoch % len(online_test_centerline_names_list)
                    # online_test_centerline_index = 0
                    online_test_centerline_name = online_test_centerline_names_list[online_test_centerline_index]
                    simulator = onlineSimulationWithNetwork(args.dataset_dir, online_test_centerline_name, renderer='pyrender')
                    _, path_centerline_error_list, path_centerline_ratio_list, _, _ = simulator.run(args, net, epoch, net_transfer=net_transfer, transform_func=dataset.transforms_eval, transform_func_transfer=transform_func_transfer)
                    if args.tensorboard:
                        for index, error in enumerate(path_centerline_error_list):
                            writer.add_scalars('paths/{}'.format(online_test_centerline_name), {'{} epoch'.format(epoch): error}, global_step=int(path_centerline_ratio_list[index] * 1000))
                        writer.add_scalar('mean_error/{}'.format(online_test_centerline_name), np.mean(path_centerline_error_list), global_step=epoch)
                        writer.add_scalar('complete_ratio/{}'.format(online_test_centerline_name), path_centerline_ratio_list[-1], global_step=epoch)
                    dataset.updateDataSet()
                # dataset.updateSpecificCenterlineDataSet(online_test_centerline_name)
                train_loader = DataLoader(dataset, batch_size=args.batchsize, shuffle=True, num_workers=0, pin_memory=True)

        if (epoch + 1) % 10 == 0:
            if not os.path.exists(args.model_dir):
                os.mkdir(args.model_dir)
            torch.save(net.state_dict(), os.path.join(args.model_dir, "regular_{}.pth".format(epoch)))
        