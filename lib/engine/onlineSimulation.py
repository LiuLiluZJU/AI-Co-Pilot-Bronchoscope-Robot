from ctypes import windll
from re import X
from turtle import left
import os
from cv2 import TM_CCOEFF_NORMED
from graphviz import render
import pybullet as p
import pybullet_data
from mayavi import mlab
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import cv2
from PIL import Image
import time
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import trimesh
from pyrender import IntrinsicsCamera, PerspectiveCamera,\
                     DirectionalLight, SpotLight, PointLight,\
                     MetallicRoughnessMaterial,\
                     Primitive, Mesh, Node, Scene,\
                     Viewer, OffscreenRenderer, RenderFlags

from camera import fixedCamera
# from extractCenterline import extractCenterline, ExtractCenterline
from keyBoardEvents import getAddition, getAdditionPlain, getDirection
from model import CIMNet
from utils import get_gpu_mem_info, tensor2im


def apply_control_pad_icon(image, direction):
    # color = (0, 165, 255)
    color = (204, 0, 51)
    offset = np.array([-160, -50])
    up_arrow = np.array([[255, 90], [240, 105], [270, 105]]) + offset
    down_arrow = np.array([[255, 170], [240, 155], [270, 155]]) + offset
    left_arrow = np.array([[210, 130], [225, 115], [225, 145]]) + offset
    right_arrow = np.array([[300, 130], [285, 115], [285, 145]]) + offset
    # front_arrow = np.array([[255, 125], [245, 135], [265, 135]])
    front_rect = np.array([[245, 120], [265, 140]]) + offset
    cv2.drawContours(image, [up_arrow], 0, color, 2)
    cv2.drawContours(image, [down_arrow], 0, color, 2)
    cv2.drawContours(image, [left_arrow], 0, color, 2)
    cv2.drawContours(image, [right_arrow], 0, color, 2)
    # cv2.drawContours(image, [front_arrow], 0, (255, 255, 0), 2)
    # cv2.circle(image, [255, 130], 10, (255, 255, 0), 2)
    cv2.rectangle(image, front_rect[0], front_rect[1], color, 2)
    if direction == [1, 0, 0, 0, 0]:
        # cv2.putText(image, 'Up', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.drawContours(image, [up_arrow], 0, color, -1)
    elif direction == [0, 1, 0, 0, 0]:
        # cv2.putText(image, 'Left', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.drawContours(image, [left_arrow], 0, color, -1)
    elif direction == [0, 0, 1, 0, 0]:
        # cv2.putText(image, 'Down', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.drawContours(image, [down_arrow], 0, color, -1)
    elif direction == [0, 0, 0, 1, 0]:
        # cv2.putText(image, 'Right', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.drawContours(image, [right_arrow], 0, color, -1)
    elif direction == [0, 0, 0, 0, 1]:
        # cv2.putText(image, 'Straight', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        # cv2.drawContours(image, [front_arrow], 0, (255, 255, 0), -1)
        # cv2.circle(image, [255, 130], 10, (255, 255, 0), 2)
        cv2.rectangle(image, front_rect[0], front_rect[1], color, -1)
    else:
        raise NotImplementedError()

    return image


def dcm2quat(R):
	
    epsilon = 1e-5
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    assert trace > -1
    if np.fabs(trace + 1) < epsilon:
        if np.argmax([R[0, 0], R[1, 1], R[2, 2]]) == 0:
            t = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
            q0 = (R[2, 1] - R[1, 2]) / t
            q1 = t / 4
            q2 = (R[0, 2] + R[2, 0]) / t
            q3 = (R[0, 1] + R[1, 0]) / t
        elif np.argmax([R[0, 0], R[1, 1], R[2, 2]]) == 1:
            t = np.sqrt(1 - R[0, 0] + R[1, 1] - R[2, 2])
            q0 = (R[0, 2] - R[2, 0]) / t
            q1 = (R[0, 1] + R[1, 0]) / t
            q2 = t / 4
            q3 = (R[2, 1] + R[1, 2]) / t
        else:
            t = np.sqrt(1 - R[0, 0] - R[1, 1] + R[2, 2])
            q0 = (R[1, 0] - R[0, 1]) / t
            q1 = (R[0, 2] + R[2, 0]) / t
            q2 = (R[1, 2] - R[2, 1]) / t
            q3 = t / 4
    else:
        q0 = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
        q1 = (R[2, 1] - R[1, 2]) / (4 * q0)
        q2 = (R[0, 2] - R[2, 0]) / (4 * q0)
        q3 = (R[1, 0] - R[0, 1]) / (4 * q0)

    return np.array([q1, q2, q3, q0])


class onlineSimulationWithNetwork(object):

    def __init__(self, dataset_dir, centerline_name, renderer=None, training=True):
        
        # Create saving folder
        if not os.path.exists(os.path.join(dataset_dir, "centerlines_with_dagger")):
            os.mkdir(os.path.join(dataset_dir, "centerlines_with_dagger"))
        self.dataset_dir = dataset_dir

        # Load models
        name = centerline_name.split(" ")[0]
        self.root_dir = "E:/pybullet_test"
        self.bronchus_model_dir = os.path.join(self.root_dir, "Airways", "AirwayHollow_{}_simUV.obj".format(name))
        # self.bronchus_model_dir = os.path.join(self.root_dir, "Airways", "AirwayHollow_{}.obj".format(name))
        self.airway_model_dir = os.path.join(self.root_dir, "Airways", "AirwayModel_Peach_{}.vtk".format(name))
        self.centerline_name = centerline_name
        centerline_model_name = centerline_name.lstrip(name + " ")
        self.centerline_model_dir = os.path.join(self.root_dir, "Airways", "centerline_models_{}".format(name), centerline_model_name + ".obj")

        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(1. / 120.)
        # useMaximalCoordinates is much faster then the default reduced coordinates (Featherstone)
        p.loadURDF("plane100.urdf", useMaximalCoordinates=True)

        shift = [0, 0, 0]
        meshScale = [0.01, 0.01, 0.01]
        # meshScale = [0.0001, 0.0001, 0.0001]
        # the visual shape and collision shape can be re-used by all createMultiBody instances (instancing)
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,
                                            # fileName="C:/Users/leko/Downloads/AirwayModel_2_Peach.obj",
                                            fileName=self.bronchus_model_dir,
                                            rgbaColor=[249 / 255, 204 / 255, 226 / 255, 1],
                                            specularColor=[0, 0, 0],
                                            visualFramePosition=shift,
                                            meshScale=meshScale)
        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                                # fileName="C:/Users/leko/Downloads/AirwayModel_2_Peach.obj",
                                                fileName=self.bronchus_model_dir,
                                                collisionFramePosition=shift,
                                                meshScale=meshScale)

        # visualShapeId_2 = p.createVisualShape(shapeType=p.GEOM_MESH,
        #                                     fileName=centerline_model_dir,
        #                                     rgbaColor=[1, 1, 1, 1],
        #                                     specularColor=[0.4, .4, 0],
        #                                     visualFramePosition=shift,
        #                                     meshScale=meshScale)
        # collisionShapeId_2 = p.createCollisionShape(shapeType=p.GEOM_MESH,
        #                                           # fileName="C:/Users/leko/Downloads/AirwayModel_2_Peach.obj",
        #                                           fileName="E:/pybullet_test/CenterlineComputationModel.obj",
        #                                           collisionFramePosition=shift,
        #                                           meshScale=meshScale)

        # rangex = 5
        # rangey = 5
        # for i in range(rangex):
        #   for j in range(rangey):
        #     p.createMultiBody(baseMass=1,
        #                       baseInertialFramePosition=[0, 0, 0],
        #                       # baseCollisionShapeIndex=collisionShapeId,
        #                       baseVisualShapeIndex=visualShapeId,
        #                       basePosition=[((-rangex / 2) + i) * meshScale[0] * 2,
        #                                     (-rangey / 2 + j) * meshScale[1] * 2, 1],
        #                       useMaximalCoordinates=True)

        # Augment on roll angle
        if training:
            self.rand_roll = (np.random.rand() - 0.5) * 2 * np.pi
            # self.rand_roll = 0
        else:
            self.rand_roll = 0
        
        euler = p.getEulerFromQuaternion([np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2])
        self.quaternion_model = p.getQuaternionFromEuler([np.pi / 2, self.rand_roll, 0])
        self.matrix_model = p.getMatrixFromQuaternion(self.quaternion_model)
        self.R_model = np.reshape(self.matrix_model, (3, 3))
        self.t_model = np.array([0, 0, 5])

        airwayBodyId = p.createMultiBody(baseMass=1,
                                            baseInertialFramePosition=[0, 0, 0],
                                            baseCollisionShapeIndex=collisionShapeId,
                                            baseVisualShapeIndex=visualShapeId,
                                            basePosition=[0, 0, 5],
                                            # baseOrientation=[np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2],
                                            baseOrientation=self.quaternion_model,
                                            #   basePosition=[0, 0, 10],
                                            useMaximalCoordinates=True)

        # p.setGravity(0, 0, -10)
        p.setRealTimeSimulation(1)

        # Set camera path
        file_path = self.centerline_model_dir
        reader = vtk.vtkOBJReader()
        reader.SetFileName(file_path)
        reader.Update()

        mesh = reader.GetOutput()
        points = mesh.GetPoints()
        data = points.GetData()
        centerlineArray = vtk_to_numpy(data)
        centerlineArray = np.dot(self.R_model, centerlineArray.T).T * 0.01 + self.t_model

        # # Choose random start point
        # total_point_num = len(centerlineArray)
        # rand_start_index = np.random.randint(int(total_point_num / 2), total_point_num + 1)  # random integer in [int(total_point_num / 2), total_point_num + 1) to ensure the path is long enough
        # centerlineArray = centerlineArray[:rand_start_index]

        # Downsample or upsample the centerline to the same length/size rate
        centerline_length = 0
        for i in range(len(centerlineArray) - 1):
            length_diff = np.linalg.norm(centerlineArray[i] - centerlineArray[i + 1])
            centerline_length += length_diff
        centerline_size = len(centerlineArray)
        lenth_size_rate = 0.007  # refer to Siliconmodel1
        centerline_size_exp = int(centerline_length / lenth_size_rate)
        centerlineArray_exp = np.zeros((centerline_size_exp, 3))
        for index_exp in range(centerline_size_exp):
            index = index_exp / (centerline_size_exp - 1) * (centerline_size - 1)
            index_left_bound = int(index)
            index_right_bound = int(index) + 1
            if index_left_bound == centerline_size - 1:
                centerlineArray_exp[index_exp] = centerlineArray[index_left_bound]
            else:
                centerlineArray_exp[index_exp] = (index_right_bound - index) * centerlineArray[index_left_bound] + (index - index_left_bound) * centerlineArray[index_right_bound]
        centerlineArray = centerlineArray_exp

        # Smoothing trajectory
        self.originalCenterlineArray = centerlineArray
        centerlineArray_smoothed = np.zeros_like(centerlineArray)
        for i in range(len(centerlineArray)):
            left_bound = i - 10
            right_bound = i + 10
            # left_bound = i - 20
            # right_bound = i + 20
            if left_bound < 0: left_bound = 0
            if right_bound > len(centerlineArray): right_bound = len(centerlineArray)
            centerlineArray_smoothed[i] = np.mean(centerlineArray[left_bound : right_bound], axis=0)
        self.centerlineArray = centerlineArray_smoothed

        # Calculate trajectory length
        centerline_length = 0
        for i in range(len(self.centerlineArray) - 1):
            length_diff = np.linalg.norm(self.centerlineArray[i] - self.centerlineArray[i + 1])
            centerline_length += length_diff
        self.centerline_length = centerline_length

        # Generate new path in each step
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(self.airway_model_dir)
        reader.Update()
        self.vtkdata = reader.GetOutput()
        # self.centerlineExtractor = ExtractCenterline(self.vtkdata)
        self.targetPoint = centerlineArray[0]
        self.transformed_target = np.dot(np.linalg.inv(self.R_model), self.targetPoint - self.t_model) * 100
        self.transformed_target_vtk_cor = np.array([-self.transformed_target[0], -self.transformed_target[1], self.transformed_target[2]])  # x and y here is opposite to those in the world coordinate system

        # Collision detection
        self.pointLocator = vtk.vtkPointLocator()
        self.pointLocator.SetDataSet(self.vtkdata)
        self.pointLocator.BuildLocator()

        self.camera = fixedCamera(0.01, p)
        # # camera.lookat(0, -89.999, [0.15, -0.05, -6])
        # camera.lookat(0, -90.001, [0, 0, 0])
        # # camera.getImg()
        # count = -6

        boundingbox = p.getAABB(airwayBodyId)
        print(boundingbox)
        print(np.max(centerlineArray, axis=0))
        print(np.min(centerlineArray, axis=0))
        print(np.argmax(centerlineArray, axis=0))
        # print(centerlineArray[1350])
        position = p.getBasePositionAndOrientation(airwayBodyId)

        # Pyrender initialization
        self.renderer = renderer
        fuze_trimesh = trimesh.load(self.bronchus_model_dir)
        # material = MetallicRoughnessMaterial(
        #                 metallicFactor=1.0,
        #                 alphaMode='OPAQUE',
        #                 roughnessFactor=0.7,
        #                 baseColorFactor=[253 / 255, 149 / 255, 158 / 255, 1])
        # material = MetallicRoughnessMaterial(
        #                     metallicFactor=0.1,
        #                     alphaMode='OPAQUE',
        #                     roughnessFactor=0.7,
        #                     baseColorFactor=[206 / 255, 108 / 255, 131 / 255, 1])
        # fuze_mesh = Mesh.from_trimesh(fuze_trimesh, material=material)
        fuze_mesh = Mesh.from_trimesh(fuze_trimesh)
        spot_l = SpotLight(color=np.ones(3), intensity=0.3,
                        innerConeAngle=0, outerConeAngle=np.pi/2, range=1)
        # self.cam = IntrinsicsCamera(fx=181.9375, fy=183.2459, cx=103.0638, cy=95.4945, znear=0.000001)
        self.cam = IntrinsicsCamera(fx=175 / 1.008, fy=175 / 1.008, cx=200, cy=200, znear=0.00001)
        self.scene = Scene(bg_color=(0., 0., 0.))
        self.fuze_node = Node(mesh=fuze_mesh, scale=meshScale, rotation=self.quaternion_model, translation=self.t_model)
        self.scene.add_node(self.fuze_node)
        self.spot_l_node = self.scene.add(spot_l)
        self.cam_node = self.scene.add(self.cam)
        # self.r = OffscreenRenderer(viewport_width=200, viewport_height=200)
        self.r = OffscreenRenderer(viewport_width=400, viewport_height=400)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    def smooth_centerline(self, centerlineArray, win_width=10):
        centerlineArray_smoothed = np.zeros_like(centerlineArray)
        for i in range(len(centerlineArray)):
            left_bound = i - win_width
            right_bound = i + win_width
            if left_bound < 0: left_bound = 0
            if right_bound > len(centerlineArray): right_bound = len(centerlineArray)
            centerlineArray_smoothed[i] = np.mean(centerlineArray[left_bound : right_bound], axis=0)
        return centerlineArray_smoothed


    def random_start_point(self, rand_index=None):
        centerline_length = len(self.centerlineArray)
        if not rand_index:
            rand_index = np.random.choice(np.arange(int(2 * centerline_length / 3), centerline_length - 3), 1)[0]
        pos_vector = self.centerlineArray[rand_index - 2] - self.centerlineArray[rand_index + 2]
        pitch = np.arcsin(pos_vector[2] / np.linalg.norm(pos_vector))
        if pos_vector[0] > 0:
            yaw = -np.arccos(pos_vector[1] / np.sqrt(pos_vector[0] ** 2 + pos_vector[1] ** 2))  # 相机绕自身坐标系旋转，Y轴正前，X轴正右，Z轴正上，yaw绕Z轴，pitch绕X轴，先yaw后pitch
        else:
            yaw = np.arccos(pos_vector[1] / np.sqrt(pos_vector[0] ** 2 + pos_vector[1] ** 2))
        quat = p.getQuaternionFromEuler([pitch, 0, yaw])
        R = p.getMatrixFromQuaternion(quat)
        R = np.reshape(R, (3, 3))

        rand_start_point = self.centerlineArray[centerline_length - 1]
        inside_flag = 0
        distance = 5
        while inside_flag == 0 or distance < 0.1:
            rand_start_point_in_original_cor = np.array([(np.random.rand() - 0.5) * 20, 0, (np.random.rand() - 0.5) * 20]) / 100
            rand_start_point = np.dot(R, rand_start_point_in_original_cor) + self.centerlineArray[rand_index]

            # Collision detection (check whether a point is inside the object by vtk and use the closest vertex)
            transformed_point = np.dot(np.linalg.inv(self.R_model), rand_start_point - self.t_model) * 100
            # transformed_point_vtk_cor = np.array([-transformed_point[0], -transformed_point[1], transformed_point[2]]) # x and y here is opposite to those in the world coordinate system
            transformed_point_vtk_cor = np.array([transformed_point[0], transformed_point[1], transformed_point[2]]) # x and y here is opposite to those in the world coordinate system
            pointId_target = self.pointLocator.FindClosestPoint(transformed_point_vtk_cor)
            cloest_point_vtk_cor = np.array(self.vtkdata.GetPoint(pointId_target))
            distance = np.linalg.norm(transformed_point_vtk_cor - cloest_point_vtk_cor)
            points = vtk.vtkPoints()
            points.InsertNextPoint(transformed_point_vtk_cor)
            pdata_points = vtk.vtkPolyData()
            pdata_points.SetPoints(points)
            enclosed_points_filter = vtk.vtkSelectEnclosedPoints()
            enclosed_points_filter.SetInputData(pdata_points)
            enclosed_points_filter.SetSurfaceData(self.vtkdata)
            enclosed_points_filter.SetTolerance(0.000001)  # should not be too large
            enclosed_points_filter.Update()
            inside_flag = int(enclosed_points_filter.GetOutput().GetPointData().GetArray('SelectedPoints').GetTuple(0)[0])
        
        rand_pitch = (np.random.rand() - 0.5) * 140
        rand_yaw = (np.random.rand() - 0.5) * 140
        return rand_pitch, rand_yaw, rand_start_point[0], rand_start_point[1], rand_start_point[2]


    def indexFromDistance(self, centerlineArray, count, distance):
        centerline_size = len(centerlineArray)
        start_index = count
        cur_index = start_index
        centerline_length = 0
        if cur_index <= 0:
            return False
        while(1):
            length_diff = np.linalg.norm(centerlineArray[cur_index - 1] - centerlineArray[cur_index])
            centerline_length += length_diff
            cur_index -= 1
            if cur_index <= 0:
                return False
            if centerline_length > distance:
                return cur_index


    def run(self, net, model_dir=None, epoch=None, net_transfer=None, transform_func=None, transform_func_transfer=None, training=True):

        if not training:
            saving_root = os.path.join(self.root_dir, "train_set", "test_ineria", model_dir.split("/")[-1][78:] + "-" + str(epoch))
            if not os.path.exists(saving_root):
                os.mkdir(saving_root)
            saving_root = os.path.join(saving_root, self.centerline_name)
            if not os.path.exists(saving_root):
                os.mkdir(saving_root)
            actions_saving_dir = os.path.join(saving_root, "actions.txt")
            images_saving_root = os.path.join(saving_root, "rgb_images")
            images_ctrl_pad_saving_root = os.path.join(saving_root, "rgb_images_ctrl_pad")
            depth_saving_root = os.path.join(saving_root, "depth_images")
            three_d_map_saving_root = os.path.join(saving_root, "three_d_map_images")
            pred_depth_saving_root = os.path.join(saving_root, "pred_depth_images")
            if not os.path.exists(images_saving_root):
                os.mkdir(images_saving_root)
            if not os.path.exists(images_ctrl_pad_saving_root):
                os.mkdir(images_ctrl_pad_saving_root)
            if not os.path.exists(depth_saving_root):
                os.mkdir(depth_saving_root)
            if not os.path.exists(pred_depth_saving_root):
                os.mkdir(pred_depth_saving_root)
            if not os.path.exists(three_d_map_saving_root):
                os.mkdir(three_d_map_saving_root)
            f = open(actions_saving_dir, 'w')

        count = len(self.centerlineArray) - 1

        if training:
            pitch, yaw, x, y, z = self.random_start_point()
            # start_index = len(self.centerlineArray) - 3
            # pitch, yaw, x, y, z = self.random_start_point(rand_index=start_index)
            # yaw = 0
            # pitch = 0
        else:
            start_index = len(self.centerlineArray) - 3
            pitch, yaw, x, y, z = self.random_start_point(rand_index=start_index)
            yaw = 0
            # pitch = -89.9999
            pitch = 0
            # x = self.centerlineArray[len(self.centerlineArray) - 1, 0]
            # y = self.centerlineArray[len(self.centerlineArray) - 1, 1]
            # z = self.centerlineArray[len(self.centerlineArray) - 1, 2]

        quat_init = p.getQuaternionFromEuler([pitch, 0, yaw])
        # quat_init = np.array([35, 78, 33, 80]) / np.linalg.norm(np.array([35, 78, 33, 80]))
        R = p.getMatrixFromQuaternion(quat_init)
        R = np.reshape(R, (3, 3))
        quat = dcm2quat(R)
        t = np.array([x, y, z])
        pos_vector = self.centerlineArray[count - 1] - self.centerlineArray[count]
        pos_vector_last = pos_vector

        for i in range(len(self.centerlineArray) - 1):
            p.addUserDebugLine(self.centerlineArray[i], self.centerlineArray[i + 1], lineColorRGB=[0, 1, 0], lifeTime=0, lineWidth=3)
        
        path_length = 0
        path_centerline_error_list = []
        path_centerline_length_list = []
        path_centerline_ratio_list = []
        safe_distance_list = []
        path_centerline_pred_position_list = []

        while 1:
            
            tic = time.time()
            print("self.rand_roll:", self.rand_roll)

            # Get GPU memory
            total_mem, used_mem, left_mem = get_gpu_mem_info()
            print("total, used, left memory:", total_mem, used_mem, left_mem)

            p.stepSimulation()

            # Smooth the rest centerline and get ground truth camera path from existing path
            # self.centerlineArray[:nearest_centerline_point_sim_cor_index] = self.smooth_centerline(self.centerlineArray[:nearest_centerline_point_sim_cor_index], win_width=10)
            nearest_original_centerline_point_sim_cor_index = np.linalg.norm(self.originalCenterlineArray - t, axis=1).argmin()
            # if nearest_original_centerline_point_sim_cor_index == 0:  # reach the target point
            # if nearest_original_centerline_point_sim_cor_index <= 5:  # reach the target point
            if nearest_original_centerline_point_sim_cor_index <= 10:  # reach the target point
                # pos_vector_gt = np.array([0, 0, 0])
                path_centerline_ratio_list.append(1.0)  # complete the path
                break
            else:
                restSmoothedCenterlineArray = self.smooth_centerline(self.originalCenterlineArray[:nearest_original_centerline_point_sim_cor_index], win_width=10)
                # if len(restSmoothedCenterlineArray) < 10:
                #     # pos_vector_gt = (restSmoothedCenterlineArray[0] - t) / len(restSmoothedCenterlineArray)
                #     break
                # else:
                #     pos_vector_gt = (restSmoothedCenterlineArray[-10] - t) / 10
                # index_form_dis = self.indexFromDistance(restSmoothedCenterlineArray, len(restSmoothedCenterlineArray) - 1, 0.07 * 0.85)
                index_form_dis = self.indexFromDistance(restSmoothedCenterlineArray, len(restSmoothedCenterlineArray) - 1, 0.07)
                if not index_form_dis:
                    if training:
                        path_centerline_ratio_list.append(1.0)  # complete the path
                        break
                    else:
                        index_form_dis = len(restSmoothedCenterlineArray) - 1
                pos_vector_gt = (restSmoothedCenterlineArray[index_form_dis] - t) / 10

            # # Genterate ground truth camera path (maybe not necessary)
            # sourcePoint = t
            # transformed_source = np.dot(np.linalg.inv(self.R_model), sourcePoint - self.t_model) * 100
            # transformed_source_vtk_cor = np.array([-transformed_source[0], -transformed_source[1], transformed_source[2]])  # x and y here is opposite to those in the world coordinate system
            # centerline_points, _ = extractCenterline(self.vtkdata, transformed_source_vtk_cor, self.transformed_target_vtk_cor)
            # # centerline_points, _ = self.centerlineExtractor.process(transformed_source_vtk_cor, self.transformed_target_vtk_cor)
            # points = centerline_points.GetPoints()
            # pointsdata = points.GetData()
            # centerlineArray0 = vtk_to_numpy(pointsdata)
            # centerlineArray0 = self.smooth_centerline(centerlineArray0, win_width=10)
            # centerlineArray0[:, 0] = -centerlineArray0[:, 0]
            # centerlineArray0[:, 1] = -centerlineArray0[:, 1]
            # print("Start, end:", centerlineArray0[-1], centerlineArray0[0])
            # if len(centerlineArray0) < 10:
            #     pos_vector_gt = (np.dot(self.R_model, centerlineArray0[0]) * 0.01 + self.t_model - sourcePoint) / len(centerlineArray0)
            # else:
            #     pos_vector_gt = (np.dot(self.R_model, centerlineArray0[-10]) * 0.01 + self.t_model - sourcePoint) / 10

            # Get direction -- Keyboard control
            # keys = p.getKeyboardEvents()
            # direction = getDirection(keys)

            # # Get direction (base control) -- Nearest neighbor
            nearest_centerline_point_sim_cor_index = np.linalg.norm(self.centerlineArray - t, axis=1).argmin()
            # data_index = len(self.centerlineArray) - 1 - nearest_centerline_point_sim_cor_index
            # file_name = os.path.join(self.dataset_dir, "centerlines", self.centerline_name, "actions.txt")
            # with open(file_name, 'r') as f_actions:
            #     data_list = f_actions.readlines()
            # if data_index >= len(data_list):  # action data is less than centerline data
            #     data_index = len(data_list) - 1
            # data = data_list[data_index].split(' ')
            # direction = np.array([float(data[7]), float(data[8]), float(data[9]), float(data[10]), float(data[11])])
            # # print("Direction:", direction)

            # Get direction (hyper control) -- distance between positon and centerline
            # if nearest_original_centerline_point_sim_cor_index == 0:  # reach the target point
            # if nearest_original_centerline_point_sim_cor_index <= 5:  # reach the target point
            if nearest_original_centerline_point_sim_cor_index <= 10:  # reach the target point
                # pos_vector_current = np.array([0, 0, 0])
                path_centerline_ratio_list.append(1.0)  # complete the path
                break
            else:
                # # restSmoothedCenterlineArray = self.smooth_centerline(self.originalCenterlineArray[:nearest_original_centerline_point_sim_cor_index], win_width=10)
                # if len(restSmoothedCenterlineArray) < 30:
                #     pos_vector_current = (restSmoothedCenterlineArray[0] - t) / len(restSmoothedCenterlineArray)
                # else:
                #     pos_vector_current = (restSmoothedCenterlineArray[-30] - t) / 30
                # index_form_dis = self.indexFromDistance(restSmoothedCenterlineArray, len(restSmoothedCenterlineArray) - 1, 0.2 * 0.85)
                index_form_dis = self.indexFromDistance(restSmoothedCenterlineArray, len(restSmoothedCenterlineArray) - 1, 0.2)
                if index_form_dis:
                    pos_vector_current = (restSmoothedCenterlineArray[index_form_dis] - t) / 30
                else:
                    path_centerline_ratio_list.append(1.0)  # complete the path
                    # break
                    pos_vector_current = (restSmoothedCenterlineArray[0] - t) / len(restSmoothedCenterlineArray)
            direction = np.array([0, 0, 0, 0, 1])
            pitch_current = pitch / 180 * np.pi
            yaw_current = yaw / 180 * np.pi
            quat_current = p.getQuaternionFromEuler([pitch_current, 0, yaw_current])
            R_current = p.getMatrixFromQuaternion(quat_current)
            R_current = np.reshape(R_current, (3, 3))
            pose_next_in_current_cor = np.dot(np.linalg.inv(R_current), pos_vector_current)
            pose_gt_in_current_cor = np.dot(np.linalg.inv(R_current), pos_vector_gt)
            pose_gt_in_camera_cor = np.array([pose_gt_in_current_cor[0], -pose_gt_in_current_cor[2], pose_gt_in_current_cor[1]])
            pitch_gt_in_camera_cor = np.arcsin(-pose_gt_in_camera_cor[1] / np.linalg.norm(pose_gt_in_camera_cor))
            if pose_gt_in_camera_cor[0] > 0:
                yaw_gt_in_camera_cor = np.arccos(pose_gt_in_camera_cor[2] / np.sqrt(pose_gt_in_camera_cor[0] ** 2 + pose_gt_in_camera_cor[2] ** 2))  # 相机绕自身坐标系旋转，Z轴正前，X轴正右，Y轴正下，yaw绕Z轴，pitch绕X轴，先yaw后pitch
            else:
                yaw_gt_in_camera_cor = -np.arccos(pose_gt_in_camera_cor[2] / np.sqrt(pose_gt_in_camera_cor[0] ** 2 + pose_gt_in_camera_cor[2] ** 2))
            theta_cone = np.arccos(pose_next_in_current_cor[1] / np.linalg.norm(pose_next_in_current_cor)) / np.pi * 180
            current_cor_x = pose_next_in_current_cor[0]
            current_cor_y = pose_next_in_current_cor[2]
            if current_cor_y > 0:
                phi = np.arccos(current_cor_x / np.sqrt(current_cor_x ** 2 + current_cor_y ** 2)) / np.pi * 180
            else:
                phi = (np.arccos(-current_cor_x / np.sqrt(current_cor_x ** 2 + current_cor_y ** 2)) + np.pi) / np.pi * 180
            nearest_distance_to_centerline = np.linalg.norm(self.centerlineArray - t, axis=1).min()
            print("theta_cone:", theta_cone)
            print("phi:", phi)
            print("nearest_distance_to_centerline:", nearest_distance_to_centerline)
            index_form_dis_for_last_control = self.indexFromDistance(restSmoothedCenterlineArray, len(restSmoothedCenterlineArray) - 1, 0.05)
            if not index_form_dis_for_last_control:
                path_centerline_ratio_list.append(1.0)  # complete the path
                break
            if (nearest_distance_to_centerline > 0.05 or theta_cone > 5):
            # if theta_cone > 5:
                # if phi <= 22.5 or phi > 337.5:
                #     direction = np.array([0, 0, 0, 1, 0])
                #     print("key right")
                # elif phi <= 67.5 and phi > 22.5:
                #     direction = np.array([1, 0, 0, 1, 0])
                #     print("key right up")
                # elif phi <= 112.5 and phi > 67.5:
                #     direction = np.array([1, 0, 0, 0, 0])
                #     print("key up")
                # elif phi <= 157.5 and phi > 112.5:
                #     direction = np.array([1, 1, 0, 0, 0])
                #     print("key left up")
                # elif phi <= 202.5 and phi > 157.5:
                #     direction = np.array([0, 1, 0, 0, 0])
                #     print("key left")
                # elif phi <= 247.5 and phi > 202.5:
                #     direction = np.array([0, 1, 1, 0, 0])
                #     print("key left down")
                # elif phi <= 292.5 and phi > 247.5:
                #     direction = np.array([0, 0, 1, 0, 0])
                #     print("key down")
                # elif phi <= 337.5 and phi > 292.5:
                #     direction = np.array([0, 0, 1, 1, 0])
                #     print("key right down")
                # else:
                #     raise NotImplementedError()
                if np.abs(pose_next_in_current_cor[0]) > np.abs(pose_next_in_current_cor[2]):
                    if pose_next_in_current_cor[0] > 0:
                        direction = np.array([0, 0, 0, 1, 0])
                        print("key right")
                    else:
                        direction = np.array([0, 1, 0, 0, 0])
                        print("key left")
                else:
                    if pose_next_in_current_cor[2] > 0:
                        direction = np.array([1, 0, 0, 0, 0])
                        print("key up")
                    else:
                        direction = np.array([0, 0, 1, 0, 0])
                        print("key down")
            print("Direction:", direction)

            # # Get direction -- Keyboard control
            keys = p.getKeyboardEvents()
            direction = getDirection(keys)
            direction = np.array(direction)

            # Adversial detection (check whether a point is inside the object by vtk and use the closest vertex)
            transformed_point = np.dot(np.linalg.inv(self.R_model), t - self.t_model) * 100
            # transformed_point_vtk_cor = np.array([-transformed_point[0], -transformed_point[1], transformed_point[2]]) # x and y here is opposite to those in the world coordinate system
            transformed_point_vtk_cor = np.array([transformed_point[0], transformed_point[1], transformed_point[2]]) # x and y here is opposite to those in the world coordinate system
            pointId_target = self.pointLocator.FindClosestPoint(transformed_point_vtk_cor)
            cloest_point_vtk_cor = np.array(self.vtkdata.GetPoint(pointId_target))
            distance_adv = np.linalg.norm(transformed_point_vtk_cor - cloest_point_vtk_cor)
            if distance_adv < 1.5 and training:
                direction = np.zeros(5)
                rand_index = np.random.randint(0, 5)
                direction[rand_index] = 1
                print("Adversial situation augmentation! Direction:", direction)
            
            # Record the error between path and centerline
            if self.centerline_length < 1e-5:
                break
            path_centerline_error_list.append(np.linalg.norm(self.centerlineArray - t, axis=1).min())
            completed_centerline_length = 0
            for i in range(nearest_centerline_point_sim_cor_index, len(self.centerlineArray) - 1):
                length_diff = np.linalg.norm(self.centerlineArray[i] - self.centerlineArray[i + 1])
                completed_centerline_length += length_diff
            path_centerline_length_list.append(completed_centerline_length)
            path_centerline_ratio_list.append(completed_centerline_length / self.centerline_length)
            path_centerline_pred_position_list.append(t)

            # Get image
            # if self.renderer == 'pyrender':
            rgb_img_bullet, _, _ = self.camera.lookat(yaw, pitch, t, -pos_vector) # for visulization
            rgb_img_bullet = rgb_img_bullet[:, :, :3]
            # rgb_img = rgb_img[:, :, ::-1]
            rgb_img_bullet = cv2.resize(rgb_img_bullet, (200, 200))
            rgb_img_bullet = np.transpose(rgb_img_bullet, axes=(2, 0, 1))
            pitch = pitch / 180 * np.pi + np.pi / 2
            yaw = yaw / 180 * np.pi
            quat = p.getQuaternionFromEuler([pitch, 0, yaw])
            R = p.getMatrixFromQuaternion(quat)
            R = np.reshape(R, (3, 3))
            pose = np.identity(4)
            pose[:3, 3] = t
            pose[:3, :3] = R
            light_intensity = 0.3
            self.scene.clear()
            self.scene.add_node(self.fuze_node)
            spot_l = SpotLight(color=np.ones(3), intensity=light_intensity,
                innerConeAngle=0, outerConeAngle=np.pi/2, range=1)
            spot_l_node = self.scene.add(spot_l, pose=pose)
            cam_node = self.scene.add(self.cam, pose=pose)
            self.scene.set_pose(spot_l_node, pose)
            self.scene.set_pose(cam_node, pose)
            rgb_img, depth_img = self.r.render(self.scene)
            rgb_img_ori = rgb_img.copy()
            rgb_img = rgb_img[:, :, :3]

            # mean_intensity = np.mean(rgb_img)
            # count_AE = 0
            # while np.abs(mean_intensity - 140) > 20:
            #     if count_AE > 1000:
            #         break
            #     if mean_intensity > 140:
            #         light_intensity -= 0.01
            #     else:
            #         light_intensity += 0.01
            #     if light_intensity < 0.01:
            #         break
            #     if light_intensity > 20:
            #         break
            #     self.scene.clear()
            #     self.scene.add_node(self.fuze_node)
            #     spot_l = SpotLight(color=np.ones(3), intensity=light_intensity,
            #             innerConeAngle=0, outerConeAngle=np.pi/2, range=1)
            #     spot_l_node = self.scene.add(spot_l, pose=pose)
            #     cam_node = self.scene.add(self.cam, pose=pose)
            #     self.scene.set_pose(spot_l_node, pose)
            #     self.scene.set_pose(cam_node, pose)
            #     rgb_img, _ = self.r.render(self.scene)
            #     rgb_img = rgb_img[:, :, :3]
            #     mean_intensity = np.mean(rgb_img)
            #     count_AE += 1
            #     # print("Light intensity:", light_intensity)
            # mean_intensity = print("Mean intensity:", np.mean(rgb_img))

            mean_intensity = np.mean(rgb_img)
            count_AE = 0
            min_light_intensity = 0.001
            max_light_intensity = 20
            while np.abs(mean_intensity - 140) > 20:
                if count_AE > 1000:
                    break
                if np.abs(min_light_intensity - light_intensity) < 1e-5 or np.abs(max_light_intensity - light_intensity) < 1e-5:
                    break
                if mean_intensity > 140:
                    max_light_intensity = light_intensity
                    light_intensity = (min_light_intensity + max_light_intensity) / 2
                else:
                    min_light_intensity = light_intensity
                    light_intensity = (min_light_intensity + max_light_intensity) / 2
                self.scene.clear()
                self.scene.add_node(self.fuze_node)
                spot_l = SpotLight(color=np.ones(3), intensity=light_intensity,
                        innerConeAngle=0, outerConeAngle=np.pi/2, range=1)
                spot_l_node = self.scene.add(spot_l, pose=pose)
                cam_node = self.scene.add(self.cam, pose=pose)
                self.scene.set_pose(spot_l_node, pose)
                self.scene.set_pose(cam_node, pose)
                rgb_img, depth_img = self.r.render(self.scene)
                rgb_img_ori = rgb_img.copy()
                rgb_img = rgb_img[:, :, :3]
                mean_intensity = np.mean(rgb_img)
                count_AE += 1
                # print("Light intensity:", light_intensity)
            mean_intensity = print("Mean intensity:", np.mean(rgb_img))

            # rgb_img = rgb_img[:, :, ::-1]
            rgb_img = cv2.resize(rgb_img, (200, 200))
            rgb_img = np.transpose(rgb_img, axes=(2, 0, 1))
            if self.renderer == 'pybullet':
                rgb_img = rgb_img_bullet

            depth_img[depth_img == 0] = 0.5
            depth_img[depth_img > 0.5] = 0.5
            depth_img = depth_img / 0.5 * 255
            depth_img = depth_img.astype(np.uint8)
            depth_img = cv2.resize(depth_img, (200, 200))
            # else:
            #     rgb_img, _, _ = self.camera.lookat(yaw, pitch, t, -pos_vector)
            #     rgb_img = rgb_img[:, :, :3]
            #     # rgb_img = rgb_img[:, :, ::-1]
            #     rgb_img = cv2.resize(rgb_img, (200, 200))
            #     rgb_img = np.transpose(rgb_img, axes=(2, 0, 1))
            
            # Style transfer
            # transfer_prob = 0.3
            # transfer_prob = 0
            # if training:
            #     if net_transfer and np.random.rand() < transfer_prob:
            #         rgb_image_PIL = Image.fromarray(np.transpose(rgb_img, axes=(1, 2, 0)))
            #         rgb_image_tensor = transform_func_transfer(rgb_image_PIL).unsqueeze(0).to(device=self.device, dtype=torch.float32)
            #         # plt.subplot(121)
            #         # image_array = tensor2im(rgb_image_tensor)
            #         # plt.imshow(image_array)
            #         transfered_rgb_image_tensor,_, _, _, _, _, _, _, _, _, _, \
            #         _, _, _, _, _, _, _, _, _, _, \
            #         _, _, _, _, _, _, _, _, _ = net_transfer(rgb_image_tensor)
            #         # image_tensor, _ = net_transfer(image_tensor)
            #         # plt.subplot(122)
            #         # image_array = tensor2im(transfered_rgb_image_tensor)
            #         # plt.imshow(image_array)
            #         # plt.show()
            #         transfered_rgb_image_tensor = (transfered_rgb_image_tensor * 0.5 + 0.5) * 255  # denormalization
            #         transfered_rgb_image_tensor = transfered_rgb_image_tensor / 255
            #         rgb_img = (transfered_rgb_image_tensor * 255).cpu().data.numpy().astype(np.uint8)[0]
            #         # rgb_img = np.transpose(rgb_img, (1, 2, 0))
            #         # cv2.imshow("Transferd image", cv2.resize(rgb_image[:, :, ::-1], (400, 400)))
            #     else:
            #         # Get image with random intensity
            #         if self.renderer == 'pyrender':
            #             light_intensity = 0.3
            #             self.scene.clear()
            #             self.scene.add_node(self.fuze_node)
            #             spot_l = SpotLight(color=np.ones(3), intensity=light_intensity,
            #                 innerConeAngle=0, outerConeAngle=np.pi/2, range=1)
            #             spot_l_node = self.scene.add(spot_l, pose=pose)
            #             cam_node = self.scene.add(self.cam, pose=pose)
            #             self.scene.set_pose(spot_l_node, pose)
            #             self.scene.set_pose(cam_node, pose)
            #             rgb_img, depth_img = self.r.render(self.scene)
            #             rgb_img = rgb_img[:, :, :3]

            #             exp_mean_intensity = np.random.randint(40, 215)
            #             mean_intensity = np.mean(rgb_img)
            #             count_AE = 0
            #             min_light_intensity = 0.001
            #             max_light_intensity = 20
            #             while np.abs(mean_intensity - exp_mean_intensity) > 20:
            #                 if count_AE > 1000:
            #                     break
            #                 if np.abs(min_light_intensity - light_intensity) < 1e-5 or np.abs(max_light_intensity - light_intensity) < 1e-5:
            #                     break
            #                 if mean_intensity > exp_mean_intensity:
            #                     max_light_intensity = light_intensity
            #                     light_intensity = (min_light_intensity + max_light_intensity) / 2
            #                 else:
            #                     min_light_intensity = light_intensity
            #                     light_intensity = (min_light_intensity + max_light_intensity) / 2
            #                 self.scene.clear()
            #                 self.scene.add_node(self.fuze_node)
            #                 spot_l = SpotLight(color=np.ones(3), intensity=light_intensity,
            #                         innerConeAngle=0, outerConeAngle=np.pi/2, range=1)
            #                 spot_l_node = self.scene.add(spot_l, pose=pose)
            #                 cam_node = self.scene.add(self.cam, pose=pose)
            #                 self.scene.set_pose(spot_l_node, pose)
            #                 self.scene.set_pose(cam_node, pose)
            #                 rgb_img, depth_img = self.r.render(self.scene)
            #                 rgb_img = rgb_img[:, :, :3]
            #                 mean_intensity = np.mean(rgb_img)
            #                 count_AE += 1
            #                 # print("Light intensity:", light_intensity)
            #             mean_intensity = print("Mean intensity:", np.mean(rgb_img))

            #             # rgb_img = rgb_img[:, :, ::-1]
            #             rgb_img = cv2.resize(rgb_img, (200, 200))
            #             rgb_img = np.transpose(rgb_img, axes=(2, 0, 1))

            #             depth_img[depth_img == 0] = 0.5
            #             depth_img[depth_img > 0.5] = 0.5
            #             depth_img = depth_img / 0.5 * 255
            #             depth_img = depth_img.astype(np.uint8)
            #             depth_img = cv2.resize(depth_img, (200, 200))
            #         else:
            #             rgb_img, _, _ = self.camera.lookat(yaw, pitch, t, -pos_vector)
            #             rgb_img = rgb_img[:, :, :3]
            #             # rgb_img = rgb_img[:, :, ::-1]
            #             rgb_img = cv2.resize(rgb_img, (200, 200))
            #             rgb_img = np.transpose(rgb_img, axes=(2, 0, 1))

            # Network inference
            if transform_func:
                rgb_img_PIL = Image.fromarray(np.transpose(rgb_img, axes=(1, 2, 0)))
                rgb_img_tensor = transform_func(rgb_img_PIL).unsqueeze(0)
            else:
                rgb_img_tensor = torch.tensor(rgb_img.copy()).unsqueeze(0)
            direction_tensor = torch.tensor(direction.copy()).unsqueeze(0)
            image = rgb_img_tensor.to(device=self.device, dtype=torch.float32)
            command = direction_tensor.to(device=self.device, dtype=torch.float32)
            # print(command)
            predicted_action, predicted_depth = net(image, command)
            # predicted_action_test = net(image, torch.tensor([[0., 1., 0., 0., 0.]]).to(device=self.device))
            # print(predicted_action, predicted_action_test)
            # pos_vector = predicted_action.squeeze(0).cpu().data.numpy() / 100
            yaw_in_camera_cor = predicted_action.squeeze(0).cpu().data.numpy()[0] * (np.pi / 2)  # pred action belongs to (-1, 1) and needs scaling to (-pi / 2, pi / 2)
            pitch_in_camera_cor = predicted_action.squeeze(0).cpu().data.numpy()[1] * (np.pi / 2)  # pred action belongs to (-1, 1) and needs scaling to (-pi / 2, pi / 2)
            quat_in_camera_cor = p.getQuaternionFromEuler([pitch_in_camera_cor, yaw_in_camera_cor, 0])
            R_in_camera_cor = p.getMatrixFromQuaternion(quat_in_camera_cor)
            R_in_camera_cor = np.reshape(R_in_camera_cor, (3, 3))
            pose_in_camera_cor = np.dot(R_in_camera_cor, [0, 0, 1 / 100])
            pose_in_current_cor = np.array([pose_in_camera_cor[0], pose_in_camera_cor[2], -pose_in_camera_cor[1]])
            # pose_in_current_cor = predicted_action.squeeze(0).cpu().data.numpy() / 100
            # pos_vector = np.dot(R_current, pose_in_current_cor)
            pos_vector = np.dot(R_current, pose_in_current_cor) * 0.8 + pos_vector_last * 0.2  # introduce inertia term
            pos_vector_last = pos_vector
            # p_expert = (30 - epoch) / 30 * 0.5
            # if np.random.rand() < p_expert:
            #     pos_vector = pos_vector_gt

            p.addUserDebugLine(t, t + pos_vector_gt, lineColorRGB=[0, 0, 1], lifeTime=0.05, lineWidth=3)
            p.addUserDebugLine(t, t + pos_vector, lineColorRGB=[1, 0, 0], lifeTime=0.05, lineWidth=3)

            # pos_vector = centerlineArray[count - 1] - centerlineArray[count]
            # pos_vector = centerlineArray[count - 10] - centerlineArray[count]
            pos_vector_norm = np.linalg.norm(pos_vector)
            if pos_vector_norm < 1e-5:
                count -= 1
                continue
            # pitch = np.pi / 2 - np.arccos(pos_vector[2] / pos_vector_norm)
            # if pos_vector[1] > 0:
            #     yaw = -np.pi / 2 + np.arccos(pos_vector[0] / pos_vector_norm)
            # else:
            #     yaw = np.pi / 2 + np.arccos(-pos_vector[0] / pos_vector_norm)
            pitch = np.arcsin(pos_vector[2] / pos_vector_norm)
            if pos_vector[0] > 0:
                yaw = -np.arccos(pos_vector[1] / np.sqrt(pos_vector[0] ** 2 + pos_vector[1] ** 2))  # 相机绕自身坐标系旋转，Y轴正前，X轴正右，Z轴正上，yaw绕Z轴，pitch绕X轴，先yaw后pitch
            else:
                yaw = np.arccos(pos_vector[1] / np.sqrt(pos_vector[0] ** 2 + pos_vector[1] ** 2))
            pitch = pitch / np.pi * 180
            yaw = yaw / np.pi * 180
            # pitch = 0
            # yaw = 0
            # t = centerlineArray[count]
            t = t + pos_vector
            print("t:", t)

            # # Veering outside the lane
            lane_width = nearest_centerline_point_sim_cor_index / (len(self.centerlineArray) - 1) * 0.08 + 0.02  # minimal width is 2mm, maximal width is 10mm 
            if nearest_distance_to_centerline > lane_width:
                break

            # # Collision detection (use the closest vertex)
            # transformed_point = np.dot(np.linalg.inv(self.R_model), t - self.t_model) * 100
            # transformed_point_vtk_cor = np.array([-transformed_point[0], -transformed_point[1], transformed_point[2]]) # x and y here is opposite to those in the world coordinate system
            # pointId_target = self.pointLocator.FindClosestPoint(transformed_point_vtk_cor)
            # cloest_point_vtk_cor = np.array(self.vtkdata.GetPoint(pointId_target))
            # distance = np.linalg.norm(transformed_point_vtk_cor - cloest_point_vtk_cor)
            # cloest_point = np.array([-cloest_point_vtk_cor[0], -cloest_point_vtk_cor[1], cloest_point_vtk_cor[2]])
            # cloest_point_sim_cor = np.dot(self.R_model, cloest_point) * 0.01 + self.t_model
            # centerline_point_sim_cor = self.centerlineArray[np.linalg.norm(self.centerlineArray - t, axis=1).argmin()]
            # vector_bt_centerline_t = t - centerline_point_sim_cor
            # vector_bt_t_cloest = cloest_point_sim_cor - t
            # print("dist, pos_vec, pos_vec_gt:", distance, pos_vector, pos_vector_gt)
            # # if distance < 1.5:
            # #     break

            # Collision detection (check whether a point is inside the object by vtk and use the closest vertex)
            transformed_point = np.dot(np.linalg.inv(self.R_model), t - self.t_model) * 100
            # transformed_point_vtk_cor = np.array([-transformed_point[0], -transformed_point[1], transformed_point[2]]) # x and y here is opposite to those in the world coordinate system
            transformed_point_vtk_cor = np.array([transformed_point[0], transformed_point[1], transformed_point[2]]) # x and y here is opposite to those in the world coordinate system
            pointId_target = self.pointLocator.FindClosestPoint(transformed_point_vtk_cor)
            cloest_point_vtk_cor = np.array(self.vtkdata.GetPoint(pointId_target))
            distance = np.linalg.norm(transformed_point_vtk_cor - cloest_point_vtk_cor)
            points = vtk.vtkPoints()
            points.InsertNextPoint(transformed_point_vtk_cor)
            pdata_points = vtk.vtkPolyData()
            pdata_points.SetPoints(points)
            enclosed_points_filter = vtk.vtkSelectEnclosedPoints()
            enclosed_points_filter.SetInputData(pdata_points)
            enclosed_points_filter.SetSurfaceData(self.vtkdata)
            enclosed_points_filter.SetTolerance(0.000001)  # should not be too large
            enclosed_points_filter.Update()
            inside_flag = int(enclosed_points_filter.GetOutput().GetPointData().GetArray('SelectedPoints').GetTuple(0)[0])
            safe_distance_list.append(distance)
            if inside_flag == 0 or distance < 0.1:
                break
            
            # Caluculate path length
            path_length_diff = np.linalg.norm(pos_vector)
            path_length += path_length_diff
            if path_length > self.centerline_length and training:
                break
            
            count -= 1
            toc = time.time()
            print("Step frequency:", 1 / (toc - tic))

            # Save image
            if not training:
                predicted_depth = predicted_depth[0].squeeze(0).cpu().data.numpy()
                predicted_depth[predicted_depth > 1] = 1
                predicted_depth[predicted_depth < 0] = 0
                predicted_depth = (predicted_depth * 255).astype(np.uint8)
                # speed = pos_vector / (toc - tic)
                speed = (pose_gt_in_camera_cor / 1) * 100
                position = (t - pos_vector) * 100
                # rgb_img = np.transpose(rgb_img, axes=(1, 2, 0))
                # Show control pad
                rgb_img = rgb_img_ori[:, :, ::-1].copy()  # RGB to BGR for saving
                rgb_img_ori = rgb_img_ori[:, :, ::-1].copy()
                rgb_img = apply_control_pad_icon(rgb_img, direction.tolist())
                # Show arrows
                intrinsic_matrix = np.array([[175 / 1.008, 0, 200],
                                    [0, 175 / 1.008, 200],
                                    [0, 0, 1]])
                # quat_in_camera_cor = p.getQuaternionFromEuler([pitch_gt_in_camera_cor, yaw_gt_in_camera_cor, 0])
                # R_in_camera_cor = p.getMatrixFromQuaternion(quat_in_camera_cor)
                # R_in_camera_cor = np.reshape(R_in_camera_cor, (3, 3))
                # predicted_action_in_camera_cor = np.dot(R_in_camera_cor, [0, 0, 10 / 1000])
                # predicted_action_in_image_cor = np.dot(intrinsic_matrix, predicted_action_in_camera_cor) / predicted_action_in_camera_cor[2]
                # cv2.arrowedLine(rgb_img_ori, (200, 200), (int((predicted_action_in_image_cor[0] - 200) + 200 + 0.5), int((predicted_action_in_image_cor[1] - 200) + 200 + 0.5)), (0, 255, 0), thickness=2, line_type=8, shift=0, tipLength=0.3)
                # cv2.circle(rgb_img_ori, (200, 200), 3, (0, 0, 255), -1)
                quat_in_camera_cor = p.getQuaternionFromEuler([pitch_in_camera_cor, yaw_in_camera_cor, 0])
                R_in_camera_cor = p.getMatrixFromQuaternion(quat_in_camera_cor)
                R_in_camera_cor = np.reshape(R_in_camera_cor, (3, 3))
                predicted_action_in_camera_cor = np.dot(R_in_camera_cor, [0, 0, 10 / 1000])
                predicted_action_in_image_cor = np.dot(intrinsic_matrix, predicted_action_in_camera_cor) / predicted_action_in_camera_cor[2]
                cv2.arrowedLine(rgb_img, (200, 200), (int((predicted_action_in_image_cor[0] - 200) + 200 + 0.5), int((predicted_action_in_image_cor[1] - 200) + 200 + 0.5)), (0, 255, 0), thickness=2, line_type=8, shift=0, tipLength=0.3)
                cv2.circle(rgb_img, (200, 200), 3, (0, 0, 255), -1)
                # cv2.imshow("saved rgb image", rgb_img)
                # cv2.waitKey(5)
                cv2.imwrite(os.path.join(images_saving_root, "{}.jpg".format(len(self.centerlineArray) - 1 - count)), rgb_img_ori)
                cv2.imwrite(os.path.join(images_ctrl_pad_saving_root, "{}.jpg".format(len(self.centerlineArray) - 1 - count)), rgb_img)
                cv2.imwrite(os.path.join(depth_saving_root, "{}.jpg".format(len(self.centerlineArray) - 1 - count)), depth_img)
                cv2.imwrite(os.path.join(pred_depth_saving_root, "{}.jpg".format(len(self.centerlineArray) - 1 - count)), predicted_depth)
                f.write("{}.jpg {} {} {} {} {} {} {} {} {} {} {} {} {}\n".format(len(self.centerlineArray) - 1 - count, speed[0], speed[1], speed[2], \
                    position[0], position[1], position[2], direction[0], direction[1], direction[2], direction[3], direction[4], yaw_gt_in_camera_cor, pitch_gt_in_camera_cor))
                # # Generate 3d map
                # fig = mlab.figure(bgcolor=(1,1,1))
                # src = mlab.pipeline.add_dataset(self.vtkdata, figure=fig)
                # surf = mlab.pipeline.surface(src, opacity=0.2, color=(206 / 255, 108 / 255, 131 / 255))
                # centerlineArray_original = np.dot(np.linalg.inv(self.R_model), (self.centerlineArray - self.t_model).T).T * 100
                # path_centerline_pred_position_array = np.array(path_centerline_pred_position_list)
                # path_centerline_pred_position_array = np.dot(np.linalg.inv(self.R_model), (path_centerline_pred_position_array - self.t_model).T).T * 100
                # # centerlineArray_original[:, 0] = -centerlineArray_original[:, 0] # x and y here is opposite to those in the world coordinate system
                # # centerlineArray_original[:, 1] = -centerlineArray_original[:, 1] # x and y here is opposite to those in the world coordinate system
                # print(np.max(centerlineArray_original, axis=0))
                # print(np.min(centerlineArray_original, axis=0))
                # mlab.plot3d([p[0] for p in centerlineArray_original], [p[1] for p in centerlineArray_original], [p[2] for p in centerlineArray_original], color=(0, 1, 0), tube_radius=1, tube_sides=10, figure=fig)
                # mlab.plot3d([p[0] for p in path_centerline_pred_position_array], [p[1] for p in path_centerline_pred_position_array], [p[2] for p in path_centerline_pred_position_array], color=(1, 0, 0), tube_radius=1, tube_sides=10, figure=fig)
                # mlab.view(azimuth=90, elevation=90, distance=600, figure=fig)
                # # # mlab.show()
                # mlab.savefig(os.path.join(three_d_map_saving_root, "{}.jpg".format(len(self.centerlineArray) - 1 - count)), figure=fig, magnification=5)
                # mlab.clf()
            # Show image
            else:
                rgb_img = np.transpose(rgb_img, axes=(1, 2, 0))
                rgb_img = rgb_img[:, :, ::-1]  # RGB to BGR for showing
                cv2.imshow("saved rgb image", rgb_img)
                cv2.waitKey(5)
        
        p.disconnect()
        self.r.delete()

        return path_centerline_pred_position_list, path_centerline_error_list, path_centerline_ratio_list, path_centerline_length_list, safe_distance_list

  