from dis import dis
from email.mime import base
from re import X
from turtle import end_fill, left, pos
import os
import pybullet as p
import pybullet_data
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import cv2
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from lib.engine.camera import fixedCamera
# from extractCenterline import extractCenterline
from lib.engine.keyBoardEvents import getDirection
import trimesh
from pyrender import IntrinsicsCamera, PerspectiveCamera,\
                     DirectionalLight, SpotLight, PointLight,\
                     MetallicRoughnessMaterial,\
                     Primitive, Mesh, Node, Scene,\
                     Viewer, OffscreenRenderer, RenderFlags


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


def indexFromDistance(centerlineArray, count, distance):
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
        


name = "siliconmodel3"
root_dir = "airways"
bronchus_model_dir = os.path.join(root_dir, "AirwayHollow_{}_simUV.obj".format(name))
airway_model_dir = os.path.join(root_dir, "AirwayModel_Peach_{}.vtk".format(name))
centerline_models_root = os.path.join(root_dir, "centerline_models_{}".format(name))
centerline_model_name_list = os.listdir(centerline_models_root)
for centerline_model_name in centerline_model_name_list:
    if centerline_model_name.split(".")[-1] != "obj": continue
    centerline_model_name = centerline_model_name.rstrip(".obj")
    centerline_model_dir = os.path.join(centerline_models_root, centerline_model_name + ".obj")
    saving_root = os.path.join("train_set/centerlines", name + " " + centerline_model_name)
    if not os.path.exists(saving_root):
        os.makedirs(saving_root)
    actions_saving_dir = os.path.join(saving_root, "actions.txt")
    images_saving_root = os.path.join(saving_root, "rgb_images")
    depth_saving_root = os.path.join(saving_root, "depth_images")
    if not os.path.exists(images_saving_root):
        os.mkdir(images_saving_root)
    if not os.path.exists(depth_saving_root):
        os.mkdir(depth_saving_root)

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
                                        fileName=bronchus_model_dir,
                                        rgbaColor=[249 / 255, 204 / 255, 226 / 255, 1],
                                        specularColor=[0, 0, 0],
                                        visualFramePosition=shift,
                                        meshScale=meshScale)
    collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                            # fileName="C:/Users/leko/Downloads/AirwayModel_2_Peach.obj",
                                            fileName=bronchus_model_dir,
                                            collisionFramePosition=shift,
                                            meshScale=meshScale)

    euler = p.getEulerFromQuaternion([np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2])
    # quaternion_model = p.getQuaternionFromEuler([np.pi / 2, -np.pi / 2, 0])
    quaternion_model = p.getQuaternionFromEuler([np.pi / 2, np.pi, 0])
    # quaternion_model = p.getQuaternionFromEuler([np.pi / 2, 0, 0])
    matrix_model = p.getMatrixFromQuaternion(quaternion_model)
    R_model = np.reshape(matrix_model, (3, 3))
    t_model = [0, 0, 5]

    airwayBodyId = p.createMultiBody(baseMass=1,
                                        baseInertialFramePosition=[0, 0, 0],
                                        baseCollisionShapeIndex=collisionShapeId,
                                        baseVisualShapeIndex=visualShapeId,
                                        basePosition=[0, 0, 5],
                                        # baseOrientation=[np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2],
                                        baseOrientation=quaternion_model,
                                        #   basePosition=[0, 0, 10],
                                        useMaximalCoordinates=True)

    p.setRealTimeSimulation(1)

    # Genterate camera path
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(airway_model_dir)
    reader.Update()
    vtkdata = reader.GetOutput()

    # Set camera path
    file_path = centerline_model_dir
    reader = vtk.vtkOBJReader()
    reader.SetFileName(file_path)
    reader.Update()

    mesh = reader.GetOutput()
    points = mesh.GetPoints()
    data = points.GetData()
    centerlineArray_original = vtk_to_numpy(data)  # centimetre
    # centerlineArray = centerlineArray * 0.01
    centerlineArray = np.dot(R_model, centerlineArray_original.T).T * 0.01 + t_model  # metre

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
    centerlineArray_smoothed = np.zeros_like(centerlineArray)
    for i in range(len(centerlineArray)):
        left_bound = i - 10
        right_bound = i + 10
        if left_bound < 0: left_bound = 0
        if right_bound > len(centerlineArray): right_bound = len(centerlineArray)
        centerlineArray_smoothed[i] = np.mean(centerlineArray[left_bound : right_bound], axis=0)
    centerlineArray = centerlineArray_smoothed

    count = len(centerlineArray) - 1
    camera = fixedCamera(0.01, p)

    boundingbox = p.getAABB(airwayBodyId)
    print(boundingbox)
    print(np.max(centerlineArray, axis=0))
    print(np.min(centerlineArray, axis=0))
    print(np.argmax(centerlineArray, axis=0))
    # print(centerlineArray[1350])
    position = p.getBasePositionAndOrientation(airwayBodyId)

    yaw = 0
    roll = 0
    pitch = 0
    x = centerlineArray[len(centerlineArray) - 1, 0]
    y = centerlineArray[len(centerlineArray) - 1, 1]
    z = centerlineArray[len(centerlineArray) - 1, 2]

    quat_init = p.getQuaternionFromEuler([pitch, 0, yaw])
    # quat_init = np.array([35, 78, 33, 80]) / np.linalg.norm(np.array([35, 78, 33, 80]))
    R = p.getMatrixFromQuaternion(quat_init)
    R = np.reshape(R, (3, 3))
    quat = dcm2quat(R)
    t = np.array([x, y, z])
    pos_vector = np.array([0, 1, 0])

    f = open(actions_saving_dir, 'w')

    for i in range(len(centerlineArray) - 1):
        p.addUserDebugLine(centerlineArray[i], centerlineArray[i + 1], lineColorRGB=[0, 1, 0], lifeTime=0, lineWidth=3)

    # Fuze trimesh
    renderer = 'pyrender'
    fuze_trimesh = trimesh.load(bronchus_model_dir)
    # material = MetallicRoughnessMaterial(
    #                 metallicFactor=1.0,
    #                 alphaMode='OPAQUE',
    #                 roughnessFactor=0.7,
    #                 baseColorFactor=[253 / 255, 149 / 255, 158 / 255, 1])
    material = MetallicRoughnessMaterial(
                    metallicFactor=0.1,
                    alphaMode='OPAQUE',
                    roughnessFactor=0.7,
                    baseColorFactor=[206 / 255, 108 / 255, 131 / 255, 1])
    # material = MetallicRoughnessMaterial(
    #                 metallicFactor=1.0,
    #                 alphaMode='OPAQUE',
    #                 roughnessFactor=1.0,
    #                 baseColorFactor=[253 / 255, 149 / 255, 158 / 255, 1])
    fuze_mesh = Mesh.from_trimesh(fuze_trimesh, material=material)
    direc_l = DirectionalLight(color=np.ones(3), intensity=1.0)
    spot_l = SpotLight(color=np.ones(3), intensity=0.3,
                    innerConeAngle=0, outerConeAngle=np.pi/2, range=1)
    point_l = PointLight(color=np.ones(3), intensity=10.0)
    # cam = IntrinsicsCamera(fx=181.9375, fy=183.2459, cx=103.0638, cy=95.4945, znear=0.00001)
    cam = IntrinsicsCamera(fx=175 / 1.008, fy=175 / 1.008, cx=200, cy=200, znear=0.00001)
    cam_pose = np.identity(4)
    cam_pose[:3, :3] = R
    cam_pose[:3, 3] = t
    scene = Scene(bg_color=(0., 0., 0.))
    fuze_node = Node(mesh=fuze_mesh, scale=np.array([0.01, 0.01, 0.01]), rotation=quaternion_model, translation=t_model)
    scene.add_node(fuze_node)
    spot_l_node = scene.add(spot_l, pose=cam_pose)
    cam_node = scene.add(cam, pose=cam_pose)
    # r = OffscreenRenderer(viewport_width=200, viewport_height=200)
    r = OffscreenRenderer(viewport_width=400, viewport_height=400)
    light_intensity = 0.3

    while 1:
        print("lenth_size_rate:", lenth_size_rate)
        tic = time.time()
        if count < 1:
            count = len(centerlineArray) - 1
            break

        p.stepSimulation()

        # Position
        t = centerlineArray[count]

        # Keyboard control (basic control)
        keys = p.getKeyboardEvents()
        direction = getDirection(keys)

        # Record pose
        pitch_current = pitch / 180 * np.pi
        yaw_current = yaw / 180 * np.pi
        quat_current = p.getQuaternionFromEuler([pitch_current, 0, yaw_current])
        R_current = p.getMatrixFromQuaternion(quat_current)
        R_current = np.reshape(R_current, (3, 3))
        t_current = t
        # print(np.dot(np.linalg.inv(R_current), pos_vector))

        # Hyper control with large bias fix
        index_form_dis = indexFromDistance(centerlineArray, count, 0.07)
        # if not index_form_dis:
        #     break
        if index_form_dis:
            pos_vector_next = centerlineArray[index_form_dis] - centerlineArray[count]
        else:
            pos_vector_next = centerlineArray[0] - centerlineArray[count]
        
        # Calculate trajectory length
        centerline_length = 0
        for i in range(index_form_dis, count):
            length_diff = np.linalg.norm(centerlineArray[i] - centerlineArray[i + 1])
            centerline_length += length_diff
        print("centerline length:", centerline_length)

        direction = [0, 0, 0, 0, 1]
        p.addUserDebugLine(t, t + pos_vector_next, lineColorRGB=[1, 0, 0], lifeTime=0.05, lineWidth=3)
        pose_next_in_current_cor = np.dot(np.linalg.inv(R_current), pos_vector_next)
        # print("pos_vector_next:", pos_vector_next)
        # print("pose_next_in_current_cor:", pose_next_in_current_cor)
        theta_cone = np.arccos(pose_next_in_current_cor[1] / np.linalg.norm(pose_next_in_current_cor)) / np.pi * 180
        # print("theta_cone:", theta_cone)
        current_cor_x = pose_next_in_current_cor[0]
        current_cor_y = pose_next_in_current_cor[2]
        if current_cor_y > 0:
            phi = np.arccos(current_cor_x / np.sqrt(current_cor_x ** 2 + current_cor_y ** 2)) / np.pi * 180
        else:
            phi = (np.arccos(-current_cor_x / np.sqrt(current_cor_x ** 2 + current_cor_y ** 2)) + np.pi)  / np.pi * 180
        print("phi:", phi)
        if theta_cone > 0.5:
            if np.abs(pose_next_in_current_cor[0]) > np.abs(pose_next_in_current_cor[2]):
                if pose_next_in_current_cor[0] > 0:
                    direction = [0, 0, 0, 1, 0]
                    print("key right")
                else:
                    direction = [0, 1, 0, 0, 0]
                    print("key left")
            else:
                if pose_next_in_current_cor[2] > 0:
                    direction = [1, 0, 0, 0, 0]
                    print("key up")
                else:
                    direction = [0, 0, 1, 0, 0]
                    print("key down")

        # Get image
        # if renderer == 'pyrender':
        rgb_img_bullet, _, _ = camera.lookat(yaw, pitch, t, -pos_vector) # for visulization
        rgb_img_bullet = rgb_img_bullet[:, :, :3]
        rgb_img_bullet = rgb_img_bullet[:, :, ::-1]
        pitch = pitch / 180 * np.pi + np.pi / 2
        yaw = yaw / 180 * np.pi
        quat = p.getQuaternionFromEuler([pitch, 0, yaw])
        R = p.getMatrixFromQuaternion(quat)
        R = np.reshape(R, (3, 3))
        pose = np.identity(4)
        pose[:3, 3] = t
        pose[:3, :3] = R
        # light_intensity = 0.3
        scene.clear()
        scene.add_node(fuze_node)
        spot_l = SpotLight(color=np.ones(3), intensity=light_intensity,
                innerConeAngle=0, outerConeAngle=np.pi/2, range=1)
        spot_l_node = scene.add(spot_l, pose=cam_pose)
        cam_node = scene.add(cam, pose=cam_pose)
        scene.set_pose(spot_l_node, pose)
        scene.set_pose(cam_node, pose)
        rgb_img, depth_img = r.render(scene)
        rgb_img = rgb_img[:, :, :3]
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
            scene.clear()
            scene.add_node(fuze_node)
            spot_l = SpotLight(color=np.ones(3), intensity=light_intensity,
                    innerConeAngle=0, outerConeAngle=np.pi/2, range=1)
            spot_l_node = scene.add(spot_l, pose=cam_pose)
            cam_node = scene.add(cam, pose=cam_pose)
            scene.set_pose(spot_l_node, pose)
            scene.set_pose(cam_node, pose)
            rgb_img, depth_img = r.render(scene)
            rgb_img = rgb_img[:, :, :3]
            mean_intensity = np.mean(rgb_img)
            count_AE += 1
            # print("Light intensity:", light_intensity)
        # print("Mean intensity:", np.mean(rgb_img))
        
        rgb_img = cv2.resize(rgb_img, (200, 200))
        rgb_img = rgb_img[:, :, ::-1]
        if renderer == 'pybullet':
            rgb_img = rgb_img_bullet

        depth_img[depth_img == 0] = 0.5
        depth_img[depth_img > 0.5] = 0.5
        depth_img = depth_img / 0.5 * 255
        depth_img = depth_img.astype(np.uint8)
        depth_img = cv2.resize(depth_img, (200, 200))

        # Automatic navigation
        # pos_vector = np.mean(centerlineArray[count - 20 : count], axis=0) - centerlineArray[count]
        # pos_vector = centerlineArray[count - 1] - centerlineArray[count]
        index_form_dis = indexFromDistance(centerlineArray, count, 0.07)
        if not index_form_dis:
            break
        pos_vector = (centerlineArray[index_form_dis] - centerlineArray[count]) / 10

        pose_in_current_cor = np.dot(np.linalg.inv(R_current), pos_vector)
        pose_in_camera_cor = np.array([pose_in_current_cor[0], -pose_in_current_cor[2], pose_in_current_cor[1]])
        pitch_in_camera_cor = np.arcsin(-pose_in_camera_cor[1] / np.linalg.norm(pose_in_camera_cor))
        if pose_in_camera_cor[0] > 0:
            yaw_in_camera_cor = np.arccos(pose_in_camera_cor[2] / np.sqrt(pose_in_camera_cor[0] ** 2 + pose_in_camera_cor[2] ** 2))  # 相机绕自身坐标系旋转，Z轴正前，X轴正右，Y轴正下，yaw绕Z轴，pitch绕X轴，先yaw后pitch
        else:
            yaw_in_camera_cor = -np.arccos(pose_in_camera_cor[2] / np.sqrt(pose_in_camera_cor[0] ** 2 + pose_in_camera_cor[2] ** 2))
        quat_in_camera_cor = p.getQuaternionFromEuler([pitch_in_camera_cor, yaw_in_camera_cor, 0])
        R_in_camera_cor = p.getMatrixFromQuaternion(quat_in_camera_cor)
        R_in_camera_cor = np.reshape(R_in_camera_cor, (3, 3))

        pos_vector_norm = np.linalg.norm(pos_vector) 
        if pos_vector_norm < 1e-5:
            count -= 1
            continue

        pitch = np.arcsin(pos_vector[2] / pos_vector_norm)
        if pos_vector[0] > 0:
            yaw = -np.arccos(pos_vector[1] / np.sqrt(pos_vector[0] ** 2 + pos_vector[1] ** 2))  # 相机绕自身坐标系旋转，Y轴正前，X轴正右，Z轴正上，yaw绕Z轴，pitch绕X轴，先yaw后pitch
        else:
            yaw = np.arccos(pos_vector[1] / np.sqrt(pos_vector[0] ** 2 + pos_vector[1] ** 2))
        pitch = pitch / np.pi * 180
        yaw = yaw / np.pi * 180
        # print("pitch yaw:", pitch, yaw)
        # print(pos_vector)

        # Collision detection (use the closest vertex)
        pointLocator = vtk.vtkPointLocator()
        pointLocator.SetDataSet(vtkdata)
        pointLocator.BuildLocator()
        transformed_point = np.dot(np.linalg.inv(R_model), t - t_model) * 100
        # transformed_point_vtk_cor = np.array([-transformed_point[0], -transformed_point[1], transformed_point[2]]) # x and y here is opposite to those in the world coordinate system
        transformed_point_vtk_cor = np.array([transformed_point[0], transformed_point[1], transformed_point[2]]) # x and y here is opposite to those in the world coordinate system
        pointId_target = pointLocator.FindClosestPoint(transformed_point_vtk_cor)
        cloest_point_vtk_cor = np.array(vtkdata.GetPoint(pointId_target))
        distance = np.linalg.norm(transformed_point_vtk_cor - cloest_point_vtk_cor)
        cloest_point = np.array([-cloest_point_vtk_cor[0], -cloest_point_vtk_cor[1], cloest_point_vtk_cor[2]])
        cloest_point_sim_cor = np.dot(R_model, cloest_point) * 0.01 + t_model
        centerline_point_sim_cor = centerlineArray[np.linalg.norm(centerlineArray - t, axis=1).argmin()]
        vector_bt_centerline_t = t - centerline_point_sim_cor
        vector_bt_t_cloest = cloest_point_sim_cor - t
        # print("distance:", distance, np.dot(vector_bt_centerline_t, vector_bt_t_cloest))
        # if distance < 1.5 or np.dot(vector_bt_centerline_t, vector_bt_t_cloest) > 0:
        #     break

        # with open("distance.txt", 'a') as fd:
        #     fd.write(centerline_model_name + ' ' + str(distance) + '\n')

        # Collision detection (check whether a point is inside the object by vtk)
        points = vtk.vtkPoints()
        points.InsertNextPoint(transformed_point_vtk_cor)
        pdata_points = vtk.vtkPolyData()
        pdata_points.SetPoints(points)
        enclosed_points_filter = vtk.vtkSelectEnclosedPoints()
        enclosed_points_filter.SetInputData(pdata_points)
        enclosed_points_filter.SetSurfaceData(vtkdata)
        enclosed_points_filter.SetTolerance(0.00001)  # should not be too large
        enclosed_points_filter.Update()
        inside_flag = int(enclosed_points_filter.GetOutput().GetPointData().GetArray('SelectedPoints').GetTuple(0)[0])
        if inside_flag == 0:
            break

        # points_in_cell = [k_point for k_point in xrange(points.GetNumberOfPoints()) if enclosed_points_filter.GetOutput().GetPointData().GetArray('SelectedPoints').GetTuple(k_point)[0]]

        count -= 1
        # print(count)

        toc = time.time()

        # Save image
        # speed = pos_vector / (toc - tic)
        # speed = pos_vector / np.linalg.norm(pos_vector)
        speed = pose_in_camera_cor * 100
        position = t * 100
        cv2.imwrite(os.path.join(images_saving_root, "{}.jpg".format(len(centerlineArray) - 1 - count)), rgb_img)
        cv2.imwrite(os.path.join(depth_saving_root, "{}.jpg".format(len(centerlineArray) - 1 - count)), depth_img)
        f.write("{}.jpg {} {} {} {} {} {} {} {} {} {} {} {} {}\n".format(len(centerlineArray) - 1 - count, speed[0], speed[1], speed[2], \
            position[0], position[1], position[2], direction[0], direction[1], direction[2], direction[3], direction[4], yaw_in_camera_cor, pitch_in_camera_cor))
    p.disconnect()

  