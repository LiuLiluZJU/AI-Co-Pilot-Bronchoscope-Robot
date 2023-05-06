import os
import sys
import h5py
import matplotlib.pyplot as plt

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
from PIL import Image


class AlignDataSet(Dataset):

    def __init__(self, dataset_dir):
        super(AlignDataSet, self).__init__()
        self.dataset_root = dataset_dir
        self.file_name = os.path.join(self.dataset_root, "actions.txt")
        self.rgb_root = os.path.join(self.dataset_root, "rgb_images")
        with open(self.file_name, 'r') as f:
            data_list = f.readlines()
        self.data_list = data_list
        print(self.data_list)
        self.dataset_size = len(self.data_list)
    
    def __len__(self):
        return self.dataset_size

    def get_data_path(self, root, index_name):
        pass

    def load_file(self, rgb_root, data):
        data_split = data.rstrip('\n').split(' ')
        image_name = data_split[0]
        action = np.array([float(data_split[1]), float(data_split[2]), float(data_split[3])]) * 1000
        command = np.array([float(data_split[4]), float(data_split[5]), float(data_split[6]), float(data_split[7]), float(data_split[8])])
        image = cv2.imread(os.path.join(rgb_root, image_name))
        return image, command, action
        # h5_file.close()
        # return image, landmarks, transformation

    def preprocess(self, image, max_value=255, min_value=0):
        pass

    def __getitem__(self, item):
        image, command, action = self.load_file(self.rgb_root, self.data_list[item])
        image = cv2.resize(image, (200, 200))
        image = np.transpose(image, axes=(2, 0, 1))
        # image = np.expand_dims(image, axis=0)
        
        return image, command, action, self.data_list[item]


class AlignDataSetSplit(Dataset):

    def __init__(self, dataset_dir):
        super(AlignDataSetSplit, self).__init__()
        self.trajectory_list = os.listdir(dataset_dir)
        self.data_full_list = []
        for trajectory_name in self.trajectory_list:
            file_name = os.path.join(dataset_dir, trajectory_name, "actions.txt")
            with open(file_name, 'r') as f:
                data_list = f.readlines()
                for data in data_list:
                    data = data.rstrip('\n').split(' ')
                    data[0] = os.path.join(dataset_dir, trajectory_name, "rgb_images", data[0])
                    self.data_full_list.append(data)
        print(self.data_full_list)
        self.dataset_size = len(self.data_full_list)
    
    def __len__(self):
        return self.dataset_size

    def get_data_path(self, root, index_name):
        pass

    def load_file(self, data):
        rgb_path = data[0]
        action = np.array([float(data[1]), float(data[2]), float(data[3])])
        command = np.array([float(data[7]), float(data[8]), float(data[9]), float(data[10]), float(data[11])])
        image = cv2.imread(rgb_path)
        return image, command, action
        # h5_file.close()
        # return image, landmarks, transformation

    def preprocess(self, image, max_value=255, min_value=0):
        pass

    def __getitem__(self, item):
        image, command, action = self.load_file(self.data_full_list[item])
        image = cv2.resize(image, (200, 200))
        image = np.transpose(image, axes=(2, 0, 1))
        # image = np.expand_dims(image, axis=0)
        
        return image, command, action, self.data_full_list[item][0]


class AlignDataSetDagger(Dataset):

    def __init__(self, dataset_dir):
        super(AlignDataSetDagger, self).__init__()
        self.dataset_dir = dataset_dir
        centerlines_dir = os.path.join(dataset_dir, "centerlines")
        self.data_centerlines_list = self.readCenterlineData(centerlines_dir)
        self.data_full_list = self.data_centerlines_list
        self.dataset_size = len(self.data_full_list)
    
    def __len__(self):
        return self.dataset_size

    def sort_and_clip_dirs(self, dir_list, clip_length=None):
        dir_list.sort(key=lambda x: -int(x.split("dagger")[-1]))  # from big to small number
        if clip_length:
            if clip_length > len(dir_list):
                return dir_list
            else:
                return dir_list[:clip_length]

    def readCenterlineData(self, centerlines_dir, dagger_set_flag=False):
        trajectory_list = os.listdir(centerlines_dir)
        if dagger_set_flag:
            trajectory_list = self.sort_and_clip_dirs(trajectory_list, clip_length=592)
        data_full_list = []
        for trajectory_name in trajectory_list:
            file_name = os.path.join(centerlines_dir, trajectory_name, "actions.txt")
            with open(file_name, 'r') as f:
                data_list = f.readlines()
                for data in data_list:
                    data = data.rstrip('\n').split(' ')
                    data[0] = os.path.join(centerlines_dir, trajectory_name, "rgb_images", data[0])
                    data_full_list.append(data)
        return data_full_list

    def readSpecificCenterlineData(self, centerlines_dir, trajectory_list, dagger_set_flag=False):
        if dagger_set_flag:
            trajectory_list = self.sort_and_clip_dirs(trajectory_list, clip_length=128)
        data_full_list = []
        for trajectory_name in trajectory_list:
            file_name = os.path.join(centerlines_dir, trajectory_name, "actions.txt")
            with open(file_name, 'r') as f:
                data_list = f.readlines()
                for data in data_list:
                    data = data.rstrip('\n').split(' ')
                    data[0] = os.path.join(centerlines_dir, trajectory_name, "rgb_images", data[0])
                    data_full_list.append(data)
        return data_full_list

    def updateDataSet(self):
        if os.path.exists(os.path.join(self.dataset_dir, "centerlines_with_dagger")):
            centerlines_dir = os.path.join(self.dataset_dir, "centerlines_with_dagger")  # add centerlines with dagger
            self.data_centerlines_dagger_list = self.readCenterlineData(centerlines_dir, dagger_set_flag=True)
            self.data_full_list = self.data_centerlines_list + self.data_centerlines_dagger_list
            # self.data_full_list += self.data_centerlines_dagger_list
            self.dataset_size = len(self.data_full_list)
    
    def updateSpecificCenterlineDataSet(self, centerline_name):
        if os.path.exists(os.path.join(self.dataset_dir, "centerlines_with_dagger")):
            centerlines_dir = os.path.join(self.dataset_dir, "centerlines_with_dagger")  # add centerlines with dagger
            trajectory_list = os.listdir(centerlines_dir)
            trajectory_list_new = []
            for trajectory_name in trajectory_list:
                if trajectory_name.split("-")[0] == centerline_name:
                    trajectory_list_new.append(trajectory_name)
            self.data_centerlines_dagger_list = self.readSpecificCenterlineData(centerlines_dir, trajectory_list_new, dagger_set_flag=False)
            self.data_full_list = self.data_centerlines_list + self.data_centerlines_dagger_list
            # self.data_full_list += self.data_centerlines_dagger_list
            self.dataset_size = len(self.data_full_list)
    
    def load_file(self, data):
        rgb_path = data[0]
        action = np.array([float(data[1]), float(data[2]), float(data[3])])
        command = np.array([float(data[7]), float(data[8]), float(data[9]), float(data[10]), float(data[11])])
        image = cv2.imread(rgb_path)
        image = image[:, :, ::-1]  # BGR to RGB
        return image, command, action

    def __getitem__(self, item):
        image, command, action = self.load_file(self.data_full_list[item])
        image = cv2.resize(image, (200, 200))
        image = np.transpose(image, axes=(2, 0, 1))
        # image = np.expand_dims(image, axis=0)
        
        return image, command, action, self.data_full_list[item][0]


class AlignDataSetDaggerAug(Dataset):

    def __init__(self, dataset_dir, train_flag=True):
        super(AlignDataSetDaggerAug, self).__init__()
        self.dataset_dir = dataset_dir
        centerlines_dir = os.path.join(dataset_dir, "centerlines")
        self.data_centerlines_list = self.readCenterlineData(centerlines_dir)
        self.data_full_list = self.data_centerlines_list
        self.dataset_size = len(self.data_full_list)
        self.train_flag = train_flag
        self.transforms_train = torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            torchvision.transforms.ToTensor()
        ])
        self.transforms_eval = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
    
    def __len__(self):
        return self.dataset_size

    def sort_and_clip_dirs(self, dir_list, clip_length=None):
        dir_list.sort(key=lambda x: -int(x.split("dagger")[-1]))  # from big to small number
        if clip_length:
            if clip_length > len(dir_list):
                return dir_list
            else:
                return dir_list[:clip_length]

    def readCenterlineData(self, centerlines_dir, dagger_set_flag=False):
        trajectory_list = os.listdir(centerlines_dir)
        if dagger_set_flag:
            trajectory_list = self.sort_and_clip_dirs(trajectory_list, clip_length=592)
        data_full_list = []
        for trajectory_name in trajectory_list:
            file_name = os.path.join(centerlines_dir, trajectory_name, "actions.txt")
            with open(file_name, 'r') as f:
                data_list = f.readlines()
                for data in data_list:
                    data = data.rstrip('\n').split(' ')
                    data[0] = os.path.join(centerlines_dir, trajectory_name, "rgb_images", data[0])
                    data_full_list.append(data)
        return data_full_list

    def readSpecificCenterlineData(self, centerlines_dir, trajectory_list, dagger_set_flag=False):
        if dagger_set_flag:
            trajectory_list = self.sort_and_clip_dirs(trajectory_list, clip_length=128)
        data_full_list = []
        for trajectory_name in trajectory_list:
            file_name = os.path.join(centerlines_dir, trajectory_name, "actions.txt")
            with open(file_name, 'r') as f:
                data_list = f.readlines()
                for data in data_list:
                    data = data.rstrip('\n').split(' ')
                    data[0] = os.path.join(centerlines_dir, trajectory_name, "rgb_images", data[0])
                    data_full_list.append(data)
        return data_full_list

    def updateDataSet(self):
        if os.path.exists(os.path.join(self.dataset_dir, "centerlines_with_dagger")):
            centerlines_dir = os.path.join(self.dataset_dir, "centerlines_with_dagger")  # add centerlines with dagger
            self.data_centerlines_dagger_list = self.readCenterlineData(centerlines_dir, dagger_set_flag=True)
            self.data_full_list = self.data_centerlines_list + self.data_centerlines_dagger_list
            # self.data_full_list += self.data_centerlines_dagger_list
            self.dataset_size = len(self.data_full_list)
    
    def updateSpecificCenterlineDataSet(self, centerline_name):
        if os.path.exists(os.path.join(self.dataset_dir, "centerlines_with_dagger")):
            centerlines_dir = os.path.join(self.dataset_dir, "centerlines_with_dagger")  # add centerlines with dagger
            trajectory_list = os.listdir(centerlines_dir)
            trajectory_list_new = []
            for trajectory_name in trajectory_list:
                if trajectory_name.split("-")[0] == centerline_name:
                    trajectory_list_new.append(trajectory_name)
            self.data_centerlines_dagger_list = self.readSpecificCenterlineData(centerlines_dir, trajectory_list_new, dagger_set_flag=False)
            self.data_full_list = self.data_centerlines_list + self.data_centerlines_dagger_list
            # self.data_full_list += self.data_centerlines_dagger_list
            self.dataset_size = len(self.data_full_list)
    
    def load_file(self, data):
        rgb_path = data[0]
        action = np.array([float(data[1]), float(data[2]), float(data[3])])
        command = np.array([float(data[7]), float(data[8]), float(data[9]), float(data[10]), float(data[11])])
        image = cv2.imread(rgb_path)
        image = image[:, :, ::-1]  # BGR to RGB
        return image, command, action

    def show_image_tensor(self, image_tensor):
        image_array = image_tensor.cpu().data.numpy()
        image_array = np.transpose(image_array, axes=(1, 2, 0))
        plt.imshow(image_array)
        plt.show()

    def __getitem__(self, item):
        image, command, action = self.load_file(self.data_full_list[item])
        image = cv2.resize(image, (200, 200))
        # image = np.transpose(image, axes=(2, 0, 1))
        image_PIL = Image.fromarray(image)
        if self.train_flag:
            image_tensor = self.transforms_train(image_PIL)
            # print(image_tensor.max(), image_tensor.min())
            # self.show_image_tensor(image_tensor)
        else:
            image_tensor = self.transforms_eval(image_PIL)
        return image_tensor, command, action, self.data_full_list[item][0]


class AlignDataSetDaggerWithDepthAug(Dataset):

    def __init__(self, dataset_dir, train_flag=True):
        super(AlignDataSetDaggerWithDepthAug, self).__init__()
        self.dataset_dir = dataset_dir
        centerlines_dir = os.path.join(dataset_dir, "centerlines")
        self.data_centerlines_list = self.readCenterlineData(centerlines_dir)
        self.data_full_list = self.data_centerlines_list
        self.dataset_size = len(self.data_full_list)
        self.train_flag = train_flag
        self.transforms_train = torchvision.transforms.Compose([
            # torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            torchvision.transforms.ToTensor()
        ])
        self.transforms_eval = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        self.transforms_depth = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
    
    def __len__(self):
        return self.dataset_size

    def sort_and_clip_dirs(self, dir_list, clip_length=None):
        dir_list.sort(key=lambda x: -int(x.split("dagger")[-1]))  # from big to small number
        if clip_length:
            if clip_length > len(dir_list):
                return dir_list
            else:
                return dir_list[:clip_length]

    def readCenterlineData(self, centerlines_dir, dagger_set_flag=False):
        trajectory_list = os.listdir(centerlines_dir)
        if dagger_set_flag:
            trajectory_list = self.sort_and_clip_dirs(trajectory_list, clip_length=592)
        data_full_list = []
        for trajectory_name in trajectory_list:
            file_name = os.path.join(centerlines_dir, trajectory_name, "actions.txt")
            with open(file_name, 'r') as f:
                data_list = f.readlines()
                for data in data_list:
                    data = data.rstrip('\n').split(' ')
                    data[0] = os.path.join(centerlines_dir, trajectory_name, "rgb_images", data[0])
                    data_full_list.append(data)
        return data_full_list

    def readSpecificCenterlineData(self, centerlines_dir, trajectory_list, dagger_set_flag=False):
        if dagger_set_flag:
            trajectory_list = self.sort_and_clip_dirs(trajectory_list, clip_length=128)
        data_full_list = []
        for trajectory_name in trajectory_list:
            file_name = os.path.join(centerlines_dir, trajectory_name, "actions.txt")
            with open(file_name, 'r') as f:
                data_list = f.readlines()
                for data in data_list:
                    data = data.rstrip('\n').split(' ')
                    data[0] = os.path.join(centerlines_dir, trajectory_name, "rgb_images", data[0])
                    data_full_list.append(data)
        return data_full_list

    def updateDataSet(self):
        if os.path.exists(os.path.join(self.dataset_dir, "centerlines_with_dagger")):
            centerlines_dir = os.path.join(self.dataset_dir, "centerlines_with_dagger")  # add centerlines with dagger
            self.data_centerlines_dagger_list = self.readCenterlineData(centerlines_dir, dagger_set_flag=True)
            self.data_full_list = self.data_centerlines_list + self.data_centerlines_dagger_list
            # self.data_full_list += self.data_centerlines_dagger_list
            self.dataset_size = len(self.data_full_list)
    
    def updateSpecificCenterlineDataSet(self, centerline_name):
        if os.path.exists(os.path.join(self.dataset_dir, "centerlines_with_dagger")):
            centerlines_dir = os.path.join(self.dataset_dir, "centerlines_with_dagger")  # add centerlines with dagger
            trajectory_list = os.listdir(centerlines_dir)
            trajectory_list_new = []
            for trajectory_name in trajectory_list:
                if trajectory_name.split("-")[0] == centerline_name:
                    trajectory_list_new.append(trajectory_name)
            self.data_centerlines_dagger_list = self.readSpecificCenterlineData(centerlines_dir, trajectory_list_new, dagger_set_flag=False)
            self.data_full_list = self.data_centerlines_list + self.data_centerlines_dagger_list
            # self.data_full_list += self.data_centerlines_dagger_list
            self.dataset_size = len(self.data_full_list)
    
    def load_file(self, data):
        rgb_path = data[0]
        depth_path = rgb_path.replace("rgb_images", "depth_images")
        action = np.array([float(data[1]), float(data[2]), float(data[3])])
        command = np.array([float(data[7]), float(data[8]), float(data[9]), float(data[10]), float(data[11])])
        image = cv2.imread(rgb_path)
        image = image[:, :, ::-1]  # BGR to RGB
        depth = cv2.imread(depth_path)
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        return image, command, action, depth

    def show_image_tensor(self, image_tensor):
        image_array = image_tensor.cpu().data.numpy()
        image_array = np.transpose(image_array, axes=(1, 2, 0))
        plt.imshow(image_array)
        plt.show()

    def __getitem__(self, item):
        image, command, action, depth = self.load_file(self.data_full_list[item])
        image = cv2.resize(image, (200, 200))
        # image = np.transpose(image, axes=(2, 0, 1))
        image_PIL = Image.fromarray(image)
        if self.train_flag:
            image_tensor = self.transforms_train(image_PIL)
            # print(image_tensor.max(), image_tensor.min())
            # self.show_image_tensor(image_tensor)
        else:
            image_tensor = self.transforms_eval(image_PIL)
        depth_PIL = Image.fromarray(depth)
        depth_tensor = self.transforms_eval(depth_PIL)
        return image_tensor, command, action, depth_tensor, self.data_full_list[item][0]


class AlignDataSetDaggerWithDepthAugAngle(Dataset):

    def __init__(self, dataset_dir, train_flag=True):
        super(AlignDataSetDaggerWithDepthAugAngle, self).__init__()
        self.dataset_dir = dataset_dir
        centerlines_dir = os.path.join(dataset_dir, "centerlines")
        self.data_centerlines_list = self.readCenterlineData(centerlines_dir)
        self.data_full_list = self.data_centerlines_list
        self.dataset_size = len(self.data_full_list)
        self.train_flag = train_flag
        self.transforms_train = torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            torchvision.transforms.ToTensor()
        ])
        self.transforms_eval = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        self.transforms_depth = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
    
    def __len__(self):
        return self.dataset_size

    def sort_and_clip_dirs(self, dir_list, clip_length=None):
        dir_list.sort(key=lambda x: -int(x.split("dagger")[-1]))  # from big to small number
        if clip_length:
            if clip_length > len(dir_list):
                return dir_list
            else:
                return dir_list[:clip_length]

    def readCenterlineData(self, centerlines_dir, dagger_set_flag=False):
        trajectory_list = os.listdir(centerlines_dir)
        if dagger_set_flag:
            trajectory_list = self.sort_and_clip_dirs(trajectory_list, clip_length=592)
        data_full_list = []
        for trajectory_name in trajectory_list:
            file_name = os.path.join(centerlines_dir, trajectory_name, "actions.txt")
            with open(file_name, 'r') as f:
                data_list = f.readlines()
                for data in data_list:
                    data = data.rstrip('\n').split(' ')
                    data[0] = os.path.join(centerlines_dir, trajectory_name, "rgb_images", data[0])
                    data_full_list.append(data)
        return data_full_list

    def readSpecificCenterlineData(self, centerlines_dir, trajectory_list, dagger_set_flag=False):
        if dagger_set_flag:
            trajectory_list = self.sort_and_clip_dirs(trajectory_list, clip_length=128)
        data_full_list = []
        for trajectory_name in trajectory_list:
            file_name = os.path.join(centerlines_dir, trajectory_name, "actions.txt")
            with open(file_name, 'r') as f:
                data_list = f.readlines()
                for data in data_list:
                    data = data.rstrip('\n').split(' ')
                    data[0] = os.path.join(centerlines_dir, trajectory_name, "rgb_images", data[0])
                    data_full_list.append(data)
        return data_full_list

    def updateDataSet(self):
        if os.path.exists(os.path.join(self.dataset_dir, "centerlines_with_dagger")):
            centerlines_dir = os.path.join(self.dataset_dir, "centerlines_with_dagger")  # add centerlines with dagger
            self.data_centerlines_dagger_list = self.readCenterlineData(centerlines_dir, dagger_set_flag=True)
            self.data_full_list = self.data_centerlines_list + self.data_centerlines_dagger_list
            # self.data_full_list += self.data_centerlines_dagger_list
            self.dataset_size = len(self.data_full_list)
    
    def updateSpecificCenterlineDataSet(self, centerline_name):
        if os.path.exists(os.path.join(self.dataset_dir, "centerlines_with_dagger")):
            centerlines_dir = os.path.join(self.dataset_dir, "centerlines_with_dagger")  # add centerlines with dagger
            trajectory_list = os.listdir(centerlines_dir)
            trajectory_list_new = []
            for trajectory_name in trajectory_list:
                if trajectory_name.split("-")[0] == centerline_name:
                    trajectory_list_new.append(trajectory_name)
            self.data_centerlines_dagger_list = self.readSpecificCenterlineData(centerlines_dir, trajectory_list_new, dagger_set_flag=False)
            self.data_full_list = self.data_centerlines_list + self.data_centerlines_dagger_list
            # self.data_full_list += self.data_centerlines_dagger_list
            self.dataset_size = len(self.data_full_list)
    
    def load_file(self, data):
        rgb_path = data[0]
        depth_path = rgb_path.replace("rgb_images", "depth_images")
        action = np.array([float(data[12]), float(data[13])])
        command = np.array([float(data[7]), float(data[8]), float(data[9]), float(data[10]), float(data[11])])
        image = cv2.imread(rgb_path)
        image = image[:, :, ::-1]  # BGR to RGB
        depth = cv2.imread(depth_path)
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        return image, command, action, depth

    def show_image_tensor(self, image_tensor):
        image_array = image_tensor.cpu().data.numpy()
        image_array = np.transpose(image_array, axes=(1, 2, 0))
        plt.imshow(image_array)
        plt.show()

    def __getitem__(self, item):
        image, command, action, depth = self.load_file(self.data_full_list[item])
        image = cv2.resize(image, (200, 200))
        # image = np.transpose(image, axes=(2, 0, 1))
        image_PIL = Image.fromarray(image)
        if self.train_flag:
            image_tensor = self.transforms_train(image_PIL)
            # print(image_tensor.max(), image_tensor.min())
            # self.show_image_tensor(image_tensor)
        else:
            image_tensor = self.transforms_eval(image_PIL)
        depth_PIL = Image.fromarray(depth)
        depth_tensor = self.transforms_eval(depth_PIL)
        return image_tensor, command, action, depth_tensor, self.data_full_list[item][0]


class AlignDataSetDaggerWithDepthAugAngleMultiFrame(Dataset):

    def __init__(self, dataset_dir, train_flag=True):
        super(AlignDataSetDaggerWithDepthAugAngleMultiFrame, self).__init__()
        self.dataset_dir = dataset_dir
        centerlines_dir = os.path.join(dataset_dir, "centerlines")
        self.data_centerlines_list = self.readCenterlineData(centerlines_dir)
        self.data_full_list = self.data_centerlines_list
        self.dataset_size = len(self.data_full_list)
        self.train_flag = train_flag
        self.transforms_train = torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            torchvision.transforms.ToTensor()
        ])
        self.transforms_eval = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        self.transforms_depth = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
    
    def __len__(self):
        return self.dataset_size

    def sort_and_clip_dirs(self, dir_list, clip_length=None):
        dir_list.sort(key=lambda x: -int(x.split("dagger")[-1]))  # from big to small number
        if clip_length:
            if clip_length > len(dir_list):
                return dir_list
            else:
                return dir_list[:clip_length]

    def readCenterlineData(self, centerlines_dir, dagger_set_flag=False):
        trajectory_list = os.listdir(centerlines_dir)
        if dagger_set_flag:
            trajectory_list = self.sort_and_clip_dirs(trajectory_list, clip_length=592)
        data_full_list = []
        for trajectory_name in trajectory_list:
            file_name = os.path.join(centerlines_dir, trajectory_name, "actions.txt")
            with open(file_name, 'r') as f:
                data_list = f.readlines()
                for data in data_list:
                    data = data.rstrip('\n').split(' ')
                    data[0] = os.path.join(centerlines_dir, trajectory_name, "rgb_images", data[0])
                    data_full_list.append(data)
        return data_full_list

    def readSpecificCenterlineData(self, centerlines_dir, trajectory_list, dagger_set_flag=False):
        if dagger_set_flag:
            trajectory_list = self.sort_and_clip_dirs(trajectory_list, clip_length=128)
        data_full_list = []
        for trajectory_name in trajectory_list:
            file_name = os.path.join(centerlines_dir, trajectory_name, "actions.txt")
            with open(file_name, 'r') as f:
                data_list = f.readlines()
                for data in data_list:
                    data = data.rstrip('\n').split(' ')
                    data[0] = os.path.join(centerlines_dir, trajectory_name, "rgb_images", data[0])
                    data_full_list.append(data)
        return data_full_list

    def updateDataSet(self):
        if os.path.exists(os.path.join(self.dataset_dir, "centerlines_with_dagger")):
            centerlines_dir = os.path.join(self.dataset_dir, "centerlines_with_dagger")  # add centerlines with dagger
            self.data_centerlines_dagger_list = self.readCenterlineData(centerlines_dir, dagger_set_flag=True)
            self.data_full_list = self.data_centerlines_list + self.data_centerlines_dagger_list
            # self.data_full_list += self.data_centerlines_dagger_list
            self.dataset_size = len(self.data_full_list)
    
    def updateSpecificCenterlineDataSet(self, centerline_name):
        if os.path.exists(os.path.join(self.dataset_dir, "centerlines_with_dagger")):
            centerlines_dir = os.path.join(self.dataset_dir, "centerlines_with_dagger")  # add centerlines with dagger
            trajectory_list = os.listdir(centerlines_dir)
            trajectory_list_new = []
            for trajectory_name in trajectory_list:
                if trajectory_name.split("-")[0] == centerline_name:
                    trajectory_list_new.append(trajectory_name)
            self.data_centerlines_dagger_list = self.readSpecificCenterlineData(centerlines_dir, trajectory_list_new, dagger_set_flag=False)
            self.data_full_list = self.data_centerlines_list + self.data_centerlines_dagger_list
            # self.data_full_list += self.data_centerlines_dagger_list
            self.dataset_size = len(self.data_full_list)
    
    def load_file(self, data):
        action = np.array([float(data[12]), float(data[13])])
        command = np.array([float(data[7]), float(data[8]), float(data[9]), float(data[10]), float(data[11])])
        rgb_path = data[0]
        rgb_root_dir = rgb_path.rstrip(rgb_path.split("\\")[-1])
        rgb_suffix = "." + rgb_path.split("\\")[-1].split(".")[-1]
        rgb_index = int(rgb_path.split("\\")[-1].rstrip(rgb_suffix))
        image_tenosr_list = []
        depth_tensor_list = []
        for offset in range(5):
            rgb_index_cur = rgb_index - offset
            rgb_path_cur = os.path.join(rgb_root_dir, str(rgb_index_cur) + rgb_suffix)
            if os.path.exists(rgb_path_cur):
                rgb_path = rgb_path_cur
            depth_path = rgb_path.replace("rgb_images", "depth_images")
            image = cv2.imread(rgb_path)
            image = image[:, :, ::-1]  # BGR to RGB
            image = cv2.resize(image, (200, 200))
            image_PIL = Image.fromarray(image)
            if self.train_flag:
                image_tensor = self.transforms_train(image_PIL)
                # print(image_tensor.max(), image_tensor.min())
                # self.show_image_tensor(image_tensor)
            else:
                image_tensor = self.transforms_eval(image_PIL)
            image_tenosr_list.append(image_tensor)
            depth = cv2.imread(depth_path)
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
            depth_PIL = Image.fromarray(depth)
            depth_tensor = self.transforms_eval(depth_PIL)
            depth_tensor_list.append(depth_tensor)
        image_tensor = torch.cat(image_tenosr_list, dim=0)
        depth_tensor = torch.cat(depth_tensor_list, dim=0)
        return image_tensor, command, action, depth_tensor

    def show_image_tensor(self, image_tensor):
        image_array = image_tensor.cpu().data.numpy()
        image_array = np.transpose(image_array, axes=(1, 2, 0))
        plt.imshow(image_array)
        plt.show()

    def __getitem__(self, item):
        image_tensor, command, action, depth_tensor = self.load_file(self.data_full_list[item])
        return image_tensor, command, action, depth_tensor, self.data_full_list[item][0]
