import pybullet as p
import numpy as np

def setCameraPicAndGetPic(RPY = True,
                        cameraEyePosition=[0, 1, 1], cameraUpVector=[0,-1,0],
                        distance=0.5, yaw=0, pitch=-30, roll = 0, upAxisIndex = 2, 
                        cameraTargetPosition=[0, 0, 0], 
                        width : int = 320, height : int = 240, physicsClientId : int = 0):
    """
    给合成摄像头设置图像并返回robot_id对应的图像
    摄像头的位置为miniBox前头的位置
    """
    # basePos, baseOrientation = p.getBasePositionAndOrientation(robot_id, physicsClientId=physicsClientId)
    # 从四元数中获取变换矩阵，从中获知指向(左乘(1,0,0)，因为在原本的坐标系内，摄像机的朝向为(1,0,0))
    matrix = p.getMatrixFromQuaternion(baseOrientation, physicsClientId=physicsClientId)

    # basePos = np.array(basePos)
    # 摄像头的位置
    # cameraPos = basePos + BASE_RADIUS * tx_vec + 0.5 * BASE_THICKNESS * tz_vec
    targetPos = cameraPos + 1 * tx_vec

    viewMatrix = p.computeViewMatrix(
        cameraEyePosition=cameraPos,
        cameraTargetPosition=targetPos,
        cameraUpVector=tz_vec,
        physicsClientId=physicsClientId
    )
    projectionMatrix = p.computeProjectionMatrixFOV(
        fov=50.0,               # 摄像头的视线夹角
        aspect=1.0,
        nearVal=0.01,            # 摄像头焦距下限
        farVal=20,               # 摄像头能看上限
        physicsClientId=physicsClientId
    )

    width, height, rgbImg, depthImg, segImg = p.getCameraImage(
        width=width, height=height,
        viewMatrix=viewMatrix,
        projectionMatrix=projectionMatrix,
        physicsClientId=physicsClientId
    )
    
    return width, height, rgbImg, depthImg, segImg

class Camera(object):
    def __init__(self):
        pass
    
    def getViewMatrix(self):
        pass
    
    def getProjectionMatrix(self):
        pass

    def getImg(self):
        pass


class fixedCamera(Camera):

    def __init__(self, dis, physics_server, targetPos = [0, 0, 0], physicsClientId = 0, 
                # RPY
                yaw=0, pitch=0, roll=0, upAxisIndex = 2,
                # Intrinsics
                fov=2 * np.arctan(100 / 181.9375) / np.pi * 180, aspect=None, nearVal=0.00001, farVal=100,
                # IMG
                width=200, height=200
                ):

        self.dis = dis
        self.targetPos = targetPos
        self.physicsClientId = physicsClientId
        self.p = physics_server

        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
        self.upAxisIndex = upAxisIndex

        self.fov = fov
        self.aspect = aspect
        if aspect is None:
            self.aspect = width / height
        self.far = farVal
        self.near = nearVal

        self.width = width
        self.height = height

    def getViewMatrix(self):

        viewMatrix = self.p.computeViewMatrixFromYawPitchRoll(
            distance=self.dis, yaw=self.yaw, pitch=self.pitch, roll=self.roll, 
            upAxisIndex=self.upAxisIndex, cameraTargetPosition=self.targetPos,
            physicsClientId = self.physicsClientId
            )
        
        return viewMatrix

    def getProjectionMatrix(self):

        projectionMatrix = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=self.aspect,
            nearVal=self.near,
            farVal=self.far,
            physicsClientId=self.physicsClientId
        )

        return projectionMatrix

    def getImg(self):
        # get images
        viewMatrix = self.getViewMatrix()
        projectionMatrix = self.getProjectionMatrix()
        width, height, rgbImg, depthImg, segImg =\
             p.getCameraImage(self.width, self.height,
                            viewMatrix,
                            projectionMatrix,
                            # lightDirection=[-0.15, 0.05, 6],
                            # lightDirection=[0, 0, 10],
                            physicsClientId=self.physicsClientId)
        # postprocess
        depth = self.far * self.near / (self.far - (self.far - self.near) * depthImg)
        rgb = np.reshape(rgbImg, [height, width, 4])
        rgb = rgb[:, :, :3]
        depth = np.reshape(depth, [height, width])

        return rgb, depth

    def getIntrinsic(self):

        projectionMatrix = self.getProjectionMatrix()
        projectionMatrix = np.reshape(projectionMatrix, [4, 4])
        fx = projectionMatrix[0, 0] * self.width / 2
        fy = projectionMatrix[1, 1] * self.height / 2
        intrinsic = np.zeros([3, 3])
        intrinsic[0, 0] = fx
        intrinsic[1, 1] = fy
        intrinsic[2, 2] = 1

        return intrinsic

    def lookat(self, yaw, pitch, targetPos, lightDirection):
        
        self.yaw = yaw
        self.pitch = pitch
        self.targetPos = targetPos
        self.lightDirection = lightDirection

        return self.visualize()

    def visualize(self):

        # self.p.configureDebugVisualizer(self.p.COV_ENABLE_GUI, 1, lightPosition=[0, 0, 4])
        self.p.resetDebugVisualizerCamera(self.dis, self.yaw, self.pitch, self.targetPos)
        # print(self.targetPos)

        camera_info = self.p.getDebugVisualizerCamera(physicsClientId=self.physicsClientId)
        width = camera_info[0]
        height = camera_info[1]
        # viewMatrix = camera_info[2]
        # projectionMatrix = camera_info[3]
        viewMatrix = self.getViewMatrix()
        projectionMatrix = self.getProjectionMatrix()
        intrinsic = self.getIntrinsic()
        _, _, rgbImg, depthImg, segImg =\
             p.getCameraImage(self.width, self.height,
                            viewMatrix,
                            projectionMatrix,
                            lightDirection=self.lightDirection,
                            # lightDirection=[0, -1, 0],
                            # lightAmbientCoeff=0.5,
                            # lightDiffuseCoeff=0.5,
                            # lightDistance=0.1,
                            # shadow=1,
                            # lightSpecularCoeff=100,
                            # lightDirection=[0, 0, 10],
                            physicsClientId=self.physicsClientId)

        return rgbImg, depthImg, segImg


class movingCamera(Camera):

    def __init__(self, dis, physics_server, targetPos = [0, 0, 0], physicsClientId = 0, 
                # RPY
                yaw=0, pitch=-40, roll=0, upAxisIndex = 2,
                # Intrinsics
                fov=50, aspect=None, nearVal=0.01, farVal=10000,
                # IMG
                width=320, height=240
                ):

        self.dis = dis
        self.targetPos = targetPos
        self.physicsClientId = physicsClientId
        self.p = physics_server

        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
        self.upAxisIndex = upAxisIndex

        self.fov = fov
        self.aspect = aspect
        if aspect is None:
            self.aspect = width/height
        self.far = farVal
        self.near = nearVal

        self.width = width
        self.height = height

    def getViewMatrix(self):

        viewMatrix = self.p.computeViewMatrixFromYawPitchRoll(
            distance=self.dis, yaw=self.yaw, pitch=self.pitch, roll=self.roll, 
            upAxisIndex=self.upAxisIndex, cameraTargetPosition=self.targetPos,
            physicsClientId = self.physicsClientId
            )
        
        return viewMatrix

    def getProjectionMatrix(self):

        projectionMatrix = p.computeProjectionMatrixFOV(
            fov=self.fov, 
            aspect=self.aspect,
            nearVal=self.near,
            farVal=self.far, 
            physicsClientId=self.physicsClientId
        )

        return projectionMatrix

    def getImg(self):
        # get images
        viewMatrix = self.getViewMatrix()
        projectionMatrix = self.getProjectionMatrix()
        width, height, rgbImg, depthImg, segImg =\
             p.getCameraImage(self.width, self.height,
                            viewMatrix,
                            projectionMatrix,
                            # lightDirection=[-0.15, 0.05, 6],
                            # lightDirection=[0, 0, 10],
                            physicsClientId=self.physicsClientId)
        # postprocess
        depth = self.far * self.near / (self.far - (self.far - self.near) * depthImg)
        rgb = np.reshape(rgbImg, [height, width, 4])
        rgb = rgb[:, :, :3]
        depth = np.reshape(depth, [height, width])

        return rgb, depth

    def getIntrinsic(self):

        projectionMatrix = self.getProjectionMatrix()
        projectionMatrix = np.reshape(projectionMatrix, [4, 4])
        fx = projectionMatrix[0, 0] * self.width/2
        fy = projectionMatrix[1, 1] * self.height/2
        intrinsic = np.zeros([3, 3])
        intrinsic[0, 0] = fx
        intrinsic[1, 1] = fy
        intrinsic[2, 2] = 1

        return intrinsic

    def lookat(self, yaw, pitch, targetPos):
        
        self.yaw = yaw
        self.pitch = pitch
        self.targetPos = targetPos

        self.visualize()

    def visualize(self):

        # self.p.configureDebugVisualizer(self.p.COV_ENABLE_GUI, 1, lightPosition=[0, 0, 4])
        self.p.resetDebugVisualizerCamera(self.dis, self.yaw, self.pitch, self.targetPos)




