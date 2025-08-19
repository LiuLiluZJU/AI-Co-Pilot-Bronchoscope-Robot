import os
import sys
import argparse
import torch
import torchvision
import numpy as np
import cv2
import time
from PIL import Image

from lib.engine.onlineSimulation import onlineSimulationWithNetwork as onlineSimulator
from lib.utils import get_transform

np.random.seed(42)


def get_args():
    parser = argparse.ArgumentParser(description='Generate dataset from centerline trajectory',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dataset-dir', dest='dataset_dir', type=str, default="all_groundtruth_set", 
                       help='Path of dataset directory')
    parser.add_argument('-o', '--output-dir', dest='output_dir', type=str, default="centerline_dataset", 
                       help='Output directory for generated dataset')
    parser.add_argument('--centerline-name', dest='centerline_name', type=str, 
                       default="siliconmodel3 Centerline model", help='Name of centerline to use')
    parser.add_argument('--renderer', dest='renderer', type=str, default='pyrender', 
                       choices=['pyrender', 'pybullet'], help='Renderer to use')
    parser.add_argument('--image-size', dest='image_size', type=int, default=200, 
                       help='Size of output images')
    parser.add_argument('--step-size', dest='step_size', type=int, default=1, 
                       help='Step size for sampling points along centerline')
    parser.add_argument('--max-points', dest='max_points', type=int, default=0,
                       help='If >0, limit generation to this many points (for debugging)')
    
    # 批量处理参数
    parser.add_argument('--batch', action='store_true', 
                       help='Generate datasets for multiple centerlines in batch mode')
    parser.add_argument('--start-index', dest='start_index', type=int, default=0, 
                       help='Start from specific centerline index (batch mode)')
    parser.add_argument('--end-index', dest='end_index', type=int, default=59, 
                       help='End at specific centerline index (batch mode)')
    parser.add_argument('--output-base-dir', dest='output_base_dir', type=str, default="all_centerline_datasets", 
                       help='Base output directory for batch mode')
    
    return parser.parse_args()


def generate_batch_datasets(args):
    """
    批量生成多条中心线的数据集
    """
    from datetime import datetime
    
    # 创建基础输出目录
    if not os.path.exists(args.output_base_dir):
        os.makedirs(args.output_base_dir)
    
    # 中心线名称列表
    centerline_names = []
    centerline_names.append("siliconmodel3 Centerline model")  # 主中心线
    for i in range(1, 60):  # 分支中心线 1-59
        centerline_names.append(f"siliconmodel3 Centerline model_{i}")
    
    print(f"=== 批量数据集生成模式 ===")
    print(f"总中心线数量: {len(centerline_names)}")
    print(f"处理范围: {args.start_index} 到 {args.end_index}")
    print(f"输出基础目录: {args.output_base_dir}")
    print("=" * 60)
    
    # 记录处理结果
    success_count = 0
    failed_centerlines = []
    start_time = datetime.now()
    
    # 创建日志文件
    log_file = os.path.join(args.output_base_dir, "generation_log.txt")
    with open(log_file, 'w') as f:
        f.write(f"中心线数据集生成日志\n")
        f.write(f"开始时间: {start_time}\n")
        f.write(f"总中心线数: {len(centerline_names)}\n")
        f.write(f"处理范围: {args.start_index} 到 {args.end_index}\n")
        f.write("=" * 50 + "\n")
    
    # 处理指定范围的中心线
    for idx in range(args.start_index, min(args.end_index + 1, len(centerline_names))):
        centerline_name = centerline_names[idx]
        print(f"\n[{idx+1}/{len(centerline_names)}] 正在处理: {centerline_name}")
        
        # 创建输出目录名称
        safe_name = centerline_name.replace(" ", "_").replace("model", "model")
        output_dir = os.path.join(args.output_base_dir, f"centerline_{idx:02d}_{safe_name}")
        
        try:
            # 构建subprocess命令（单个中心线模式）
            cmd = [
                sys.executable, __file__,  # 使用当前Python解释器和脚本
                "--dataset-dir", args.dataset_dir,
                "--output-dir", output_dir,
                "--centerline-name", centerline_name,
                "--renderer", args.renderer,
                "--image-size", str(args.image_size),
                "--step-size", str(args.step_size)
                # 注意：不传 --batch 参数，这样会进入单个中心线模式
            ]
            
            print(f"执行命令: {' '.join(cmd)}")
            print(f"输出目录: {output_dir}")
            
            # 执行subprocess
            import subprocess
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  timeout=1800, encoding='utf-8', errors='replace')  # 30分钟超时
            
            if result.returncode == 0:
                # 检查是否成功生成了图像
                rgb_dir = os.path.join(output_dir, "rgb_images")
                if os.path.exists(rgb_dir):
                    image_count = len([f for f in os.listdir(rgb_dir) if f.endswith('.jpg')])
                    print(f"✓ 成功为 {centerline_name} 生成了 {image_count} 对图像")
                    success_count += 1
                    
                    # 记录成功信息
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(f"成功: {centerline_name} -> {output_dir} ({image_count} 对图像)\n")
                else:
                    print(f"✗ {centerline_name} - 输出目录不存在")
                    failed_centerlines.append((centerline_name, "输出目录不存在"))
                    
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(f"失败: {centerline_name} - 输出目录不存在\n")
            else:
                error_msg = result.stderr.strip() if result.stderr else "未知错误"
                print(f"✗ {centerline_name} 处理失败")
                print(f"错误信息: {error_msg}")
                failed_centerlines.append((centerline_name, error_msg))
                
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"失败: {centerline_name}\n")
                    f.write(f"错误: {error_msg}\n")
                    
        except subprocess.TimeoutExpired:
            print(f"✗ {centerline_name} 处理超时")
            failed_centerlines.append((centerline_name, "处理超时"))
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"超时: {centerline_name}\n")
                
        except Exception as e:
            print(f"✗ 处理 {centerline_name} 时发生异常: {e}")
            failed_centerlines.append((centerline_name, str(e)))
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"异常: {centerline_name} - {str(e)}\n")
    
    # 完成统计
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print("批量处理完成")
    print(f"总计处理: {success_count + len(failed_centerlines)}")
    print(f"成功: {success_count}")
    print(f"失败: {len(failed_centerlines)}")
    print(f"耗时: {duration}")
    
    if failed_centerlines:
        print("\n失败的中心线:")
        for name, error in failed_centerlines:
            print(f"  - {name}: {error}")
    
    # 写入最终统计
    with open(log_file, 'a') as f:
        f.write("=" * 50 + "\n")
        f.write(f"完成时间: {end_time}\n")
        f.write(f"耗时: {duration}\n")
        f.write(f"成功: {success_count}\n")
        f.write(f"失败: {len(failed_centerlines)}\n")
        
        if failed_centerlines:
            f.write("\n失败的中心线:\n")
            for name, error in failed_centerlines:
                f.write(f"  - {name}: {error}\n")
    
    print(f"\n日志已保存到: {log_file}")
    return success_count, failed_centerlines


def generate_centerline_dataset(args):
    """
    生成中心线轨迹数据集
    在中心线的每个点上生成RGB图像和深度图像
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建输出目录
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    rgb_dir = os.path.join(output_dir, "rgb_images")
    depth_dir = os.path.join(output_dir, "depth_images")
    info_dir = os.path.join(output_dir, "camera_info")
    
    for dir_path in [rgb_dir, depth_dir, info_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    # 初始化仿真器
    print(f"Initializing simulator for centerline: {args.centerline_name}")
    simulator = onlineSimulator(args.dataset_dir, args.centerline_name, 
                               renderer=args.renderer, training=False)
    
    # 获取中心线轨迹点
    centerline_points = simulator.centerlineArray
    print(f"Total centerline points: {len(centerline_points)}")
    
    # 创建信息文件
    info_file = os.path.join(info_dir, "dataset_info.txt")
    with open(info_file, 'w') as f:
        f.write("# Dataset Information\n")
        f.write(f"centerline_name: {args.centerline_name}\n")
        f.write(f"renderer: {args.renderer}\n")
        f.write(f"image_size: {args.image_size}\n")
        f.write(f"total_points: {len(centerline_points)}\n")
        f.write(f"step_size: {args.step_size}\n")
        f.write("# Format: index, x, y, z, pitch, yaw, distance_to_start\n")
    
    # 遍历中心线点生成数据
    total_distance = 0.0
    generated_count = 0
    
    for i in range(0, len(centerline_points), args.step_size):
        try:
            point = centerline_points[i]
            print(f"Processing point {i}/{len(centerline_points)}: {point}")
            
            # 计算相机姿态（基于中心线方向）
            if i < len(centerline_points) - 1:
                # 使用下一个点计算方向
                direction_vector = centerline_points[i + 1] - point
            elif i > 0:
                # 最后一个点，使用前一个点计算方向
                direction_vector = point - centerline_points[i - 1]
            else:
                # 只有一个点的情况
                direction_vector = np.array([0, 1, 0])
            
            # 生成图像（使用正确的 dir 方法）
            rgb_img, depth_img = generate_images_at_point_with_direction(
                simulator, point, direction_vector, args.image_size
            )
            
            # 计算pitch和yaw用于记录（保持兼容性）
            direction_norm = np.linalg.norm(direction_vector)
            if direction_norm > 1e-6:
                normalized_direction = direction_vector / direction_norm
                pitch = np.arcsin(np.clip(normalized_direction[2], -1.0, 1.0))
                if abs(normalized_direction[0]) > 1e-6 or abs(normalized_direction[1]) > 1e-6:
                    yaw = np.arctan2(normalized_direction[0], normalized_direction[1])
                else:
                    yaw = 0.0
                pitch_deg = pitch * 180.0 / np.pi
                yaw_deg = yaw * 180.0 / np.pi
            else:
                pitch_deg = yaw_deg = 0.0
            
            # 保存图像
            rgb_filename = f"rgb_{i:06d}.jpg"
            depth_filename = f"depth_{i:06d}.jpg"
            
            cv2.imwrite(os.path.join(rgb_dir, rgb_filename), rgb_img)
            cv2.imwrite(os.path.join(depth_dir, depth_filename), depth_img)
            
            # 计算到起始点的距离
            if i > 0:
                distance_diff = np.linalg.norm(point - centerline_points[i - args.step_size])
                total_distance += distance_diff
            
            # 保存相机信息
            with open(info_file, 'a') as f:
                f.write(f"{i}, {point[0]:.6f}, {point[1]:.6f}, {point[2]:.6f}, "
                       f"{pitch_deg:.3f}, {yaw_deg:.3f}, {total_distance:.6f}\n")
            
            generated_count += 1
            # 如果设置了 max_points，则限制生成数量（用于调试）
            if args.max_points > 0 and generated_count >= args.max_points:
                print(f"Reached max_points={args.max_points}, stopping early")
                break
            
            if generated_count % 10 == 0:
                print(f"Generated {generated_count} image pairs")
                
        except Exception as e:
            print(f"Error processing point {i}: {e}")
            continue
    
    print(f"Dataset generation completed!")
    print(f"Total images generated: {generated_count}")
    print(f"RGB images saved to: {rgb_dir}")
    print(f"Depth images saved to: {depth_dir}")
    print(f"Camera info saved to: {info_file}")
    
    return generated_count


def generate_images_at_point_with_direction(simulator, position, direction_vector, image_size):
    """
    在指定位置使用正确的方向向量生成RGB和深度图像（使用WORKING dir方法）
    """
    # 使用正确的 'dir' 方法构建相机姿态
    # forward = normalized dir_vec, up = [0,0,1] (if nearly parallel choose alternative)
    f = direction_vector / (np.linalg.norm(direction_vector) + 1e-12)
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(f, up)) > 0.999:
        up = np.array([0.0, 1.0, 0.0])
    r = np.cross(up, f)
    r = r / (np.linalg.norm(r) + 1e-12)
    u = np.cross(f, r)
    R = np.vstack([r, u, f]).T
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = position
    
    # 设置光照强度
    light_intensity = 0.3
    
    # 清空场景并添加节点
    simulator.scene.clear()
    simulator.scene.add_node(simulator.fuze_node)
    
    # 添加光源和相机
    from pyrender import SpotLight
    spot_l = SpotLight(color=np.ones(3), intensity=light_intensity,
                      innerConeAngle=0, outerConeAngle=np.pi/2, range=1)
    spot_l_node = simulator.scene.add(spot_l, pose=pose)
    cam_node = simulator.scene.add(simulator.cam, pose=pose)
    
    # 设置姿态
    simulator.scene.set_pose(spot_l_node, pose)
    simulator.scene.set_pose(cam_node, pose)
    
    # 渲染图像
    rgb_img, depth_img = simulator.r.render(simulator.scene)
    rgb_img = rgb_img[:, :, :3]
    
    # 自动调整光照强度以获得合适的亮度
    mean_intensity = np.mean(rgb_img)
    target_intensity = 140
    tolerance = 20
    max_iterations = 50
    iteration = 0
    
    min_light = 0.001
    max_light = 20.0
    
    while abs(mean_intensity - target_intensity) > tolerance and iteration < max_iterations:
        if mean_intensity > target_intensity:
            max_light = light_intensity
        else:
            min_light = light_intensity
        
        light_intensity = (min_light + max_light) / 2
        
        # 重新渲染
        simulator.scene.clear()
        simulator.scene.add_node(simulator.fuze_node)
        
        spot_l = SpotLight(color=np.ones(3), intensity=light_intensity,
                          innerConeAngle=0, outerConeAngle=np.pi/2, range=1)
        spot_l_node = simulator.scene.add(spot_l, pose=pose)
        cam_node = simulator.scene.add(simulator.cam, pose=pose)
        
        simulator.scene.set_pose(spot_l_node, pose)
        simulator.scene.set_pose(cam_node, pose)
        
        rgb_img, depth_img = simulator.r.render(simulator.scene)
        rgb_img = rgb_img[:, :, :3]
        mean_intensity = np.mean(rgb_img)
        
        iteration += 1
    
    # 处理深度图
    depth_img[depth_img == 0] = 0.5
    depth_img[depth_img > 0.5] = 0.5
    depth_img = depth_img / 0.5 * 255
    depth_img = depth_img.astype(np.uint8)
    
    # 调整图像大小
    rgb_img = cv2.resize(rgb_img, (image_size, image_size))
    depth_img = cv2.resize(depth_img, (image_size, image_size))
    
    # 转换RGB到BGR（OpenCV格式）
    rgb_img = rgb_img[:, :, ::-1]
    
    return rgb_img, depth_img


def generate_images_at_point(simulator, position, pitch, yaw, image_size):
    """
    在指定位置和姿态生成RGB和深度图像
    """
    # 转换角度为弧度
    # pitch, yaw are provided in degrees; convert to radians
    pitch_rad = pitch / 180.0 * np.pi
    yaw_rad = yaw / 180.0 * np.pi
    
    # 创建相机姿态矩阵
    import pybullet as p
    # Match onlineSimulation.run convention: add +pi/2 to pitch and use Euler order [pitch, 0, yaw]
    # (onlineSimulation applies pitch = pitch/180*pi + pi/2 before building quaternion)
    pitch_rad = pitch_rad + np.pi / 2.0
    quat = p.getQuaternionFromEuler([pitch_rad, 0.0, yaw_rad])
    R = p.getMatrixFromQuaternion(quat)
    R = np.reshape(R, (3, 3))
    
    pose = np.identity(4)
    pose[:3, 3] = position
    pose[:3, :3] = R
    
    # 设置光照强度
    light_intensity = 0.3
    
    # 清空场景并添加节点
    simulator.scene.clear()
    simulator.scene.add_node(simulator.fuze_node)
    
    # 添加光源和相机
    from pyrender import SpotLight
    spot_l = SpotLight(color=np.ones(3), intensity=light_intensity,
                      innerConeAngle=0, outerConeAngle=np.pi/2, range=1)
    spot_l_node = simulator.scene.add(spot_l, pose=pose)
    cam_node = simulator.scene.add(simulator.cam, pose=pose)
    
    # 设置姿态
    simulator.scene.set_pose(spot_l_node, pose)
    simulator.scene.set_pose(cam_node, pose)
    
    # 渲染图像
    rgb_img, depth_img = simulator.r.render(simulator.scene)
    rgb_img = rgb_img[:, :, :3]
    
    # 自动调整光照强度以获得合适的亮度
    mean_intensity = np.mean(rgb_img)
    target_intensity = 140
    tolerance = 20
    max_iterations = 50
    iteration = 0
    
    min_light = 0.001
    max_light = 20.0
    
    while abs(mean_intensity - target_intensity) > tolerance and iteration < max_iterations:
        if mean_intensity > target_intensity:
            max_light = light_intensity
        else:
            min_light = light_intensity
        
        light_intensity = (min_light + max_light) / 2
        
        # 重新渲染
        simulator.scene.clear()
        simulator.scene.add_node(simulator.fuze_node)
        
        spot_l = SpotLight(color=np.ones(3), intensity=light_intensity,
                          innerConeAngle=0, outerConeAngle=np.pi/2, range=1)
        spot_l_node = simulator.scene.add(spot_l, pose=pose)
        cam_node = simulator.scene.add(simulator.cam, pose=pose)
        
        simulator.scene.set_pose(spot_l_node, pose)
        simulator.scene.set_pose(cam_node, pose)
        
        rgb_img, depth_img = simulator.r.render(simulator.scene)
        rgb_img = rgb_img[:, :, :3]
        mean_intensity = np.mean(rgb_img)
        
        iteration += 1
    
    # 处理深度图像
    depth_img[depth_img == 0] = 0.5
    depth_img[depth_img > 0.5] = 0.5
    depth_img = depth_img / 0.5 * 255
    depth_img = depth_img.astype(np.uint8)
    
    # 调整图像尺寸
    rgb_img = cv2.resize(rgb_img, (image_size, image_size))
    depth_img = cv2.resize(depth_img, (image_size, image_size))
    
    # 转换RGB颜色通道顺序 (RGB -> BGR for OpenCV)
    rgb_img = rgb_img[:, :, ::-1]
    
    return rgb_img, depth_img


def main():
    args = get_args()
    
    if args.batch:
        # 批量处理模式
        print("=== 中心线数据集批量生成器 ===")
        print(f"渲染器: {args.renderer}")
        print(f"图像尺寸: {args.image_size}x{args.image_size}")
        print(f"步长: {args.step_size}")
        print(f"处理范围: {args.start_index} 到 {args.end_index}")
        print("=" * 50)
        
        success_count, failed_centerlines = generate_batch_datasets(args)
        
        if failed_centerlines:
            print(f"\n注意：有 {len(failed_centerlines)} 条中心线处理失败")
            print("您可以使用 --start-index 和 --end-index 参数重新处理失败的部分")
        else:
            print(f"\n🎉 所有 {success_count} 条中心线数据集生成成功！")
    else:
        # 单个中心线模式
        print("=== 中心线数据集生成器 ===")
        print(f"中心线: {args.centerline_name}")
        print(f"渲染器: {args.renderer}")
        print(f"输出目录: {args.output_dir}")
        print(f"图像尺寸: {args.image_size}x{args.image_size}")
        print(f"步长: {args.step_size}")
        print("=" * 40)
        
        # 生成数据集
        generated_count = generate_centerline_dataset(args)
        
        print("=" * 40)
        print(f"数据集生成完成！生成了 {generated_count} 对图像。")


if __name__ == '__main__':
    main()
