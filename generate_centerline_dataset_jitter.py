import os
import sys
import argparse
import numpy as np
import cv2
import time
from PIL import Image
import scipy.spatial.transform

import pybullet as p

from lib.engine.onlineSimulation import onlineSimulationWithNetwork as onlineSimulator

np.random.seed(42)


def get_args():
    parser = argparse.ArgumentParser(description='Generate centerline dataset with camera jitter',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dataset-dir', dest='dataset_dir', type=str, default="all_jittered_set",
                       help='Path of dataset directory')
    parser.add_argument('-o', '--output-dir', dest='output_dir', type=str, default="centerline_dataset_jitter",
                       help='Output directory for generated dataset')
    parser.add_argument('--centerline-name', dest='centerline_name', type=str,
                       default="siliconmodel3 Centerline model", help='Name of centerline to use')
    parser.add_argument('--renderer', dest='renderer', type=str, default='pyrender',
                       choices=['pyrender', 'pybullet'], help='Renderer to use')
    parser.add_argument('--image-size', dest='image_size', type=int, default=200,
                       help='Size of output images')
    parser.add_argument('--step-size', dest='step_size', type=int, default=1,
                       help='Step size for sampling points along centerline')

    # jitter params
    parser.add_argument('--pos-jitter-radius', dest='pos_jitter_radius', type=float, default=0.005,
                       help='Max position jitter radius in meters (e.g. 0.01 for 1cm)')
    parser.add_argument('--angle-jitter-deg', dest='angle_jitter_deg', type=float, default=30.0,
                       help='Max orientation jitter in degrees (<= 180)')

    # batch mode (optional)
    parser.add_argument('--batch', action='store_true', help='Batch mode across all centerlines')
    parser.add_argument('--start-index', dest='start_index', type=int, default=0,
                       help='Start index for batch')
    parser.add_argument('--end-index', dest='end_index', type=int, default=59,
                       help='End index for batch')
    parser.add_argument('--output-base-dir', dest='output_base_dir', type=str, default="all_centerline_datasets_jitter",
                       help='Base output dir when batch mode')
    parser.add_argument('--max-samples', dest='max_samples', type=int, default=0,
                       help='For testing: only process first N samples (0 means all)')

    return parser.parse_args()


# utility: quaternion multiply (x,y,z,w)
def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return [x, y, z, w]


def random_point_in_sphere(radius):
    # uniform distribution within sphere
    u = np.random.rand()
    cos_theta = 1 - 2 * np.random.rand()
    sin_theta = np.sqrt(max(0.0, 1 - cos_theta ** 2))
    phi = 2 * np.pi * np.random.rand()
    r = radius * (u ** (1.0 / 3.0))
    x = r * sin_theta * np.cos(phi)
    y = r * sin_theta * np.sin(phi)
    z = r * cos_theta
    return np.array([x, y, z])


def apply_orientation_jitter(quat_base, max_angle_deg):
    # random axis
    axis = np.random.randn(3)
    axis = axis / np.linalg.norm(axis)
    angle_rad = np.deg2rad(np.random.uniform(-max_angle_deg, max_angle_deg))
    s = np.sin(angle_rad / 2.0)
    q_perturb = [axis[0] * s, axis[1] * s, axis[2] * s, np.cos(angle_rad / 2.0)]
    # compose: q_new = q_perturb * quat_base
    q_new = quat_mul(q_perturb, quat_base)
    # normalize
    q_new = np.array(q_new)
    q_new = q_new / np.linalg.norm(q_new)
    return q_new.tolist()


def quat_conjugate(q):
    # q: [x,y,z,w]
    return [-q[0], -q[1], -q[2], q[3]]


def generate_images_at_point_with_jitter_and_direction(simulator, position, direction_vector, image_size,
                                                      pos_jitter_radius=0.01, angle_jitter_deg=30.0,
                                                      max_attempts=20):
    """
    在指定中心线位置生成带抖动的RGB和深度图，使用正确的方向向量方法。
    保证抖动后位置仍在气道内部（使用 simulator 的 vtk 数据和 pointLocator 检查）。
    返回：rgb_img_bgr (H,W,3 uint8), depth_img_gray (H,W uint8), 
           jitter_info (dict with dx, dy, dz, dist_m, rot_deg, final_quat)
    """
    
    # build base camera pose using WORKING 'dir' method
    f = direction_vector / (np.linalg.norm(direction_vector) + 1e-12)
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(f, up)) > 0.999:
        up = np.array([0.0, 1.0, 0.0])
    r = np.cross(up, f)
    r = r / (np.linalg.norm(r) + 1e-12)
    u = np.cross(f, r)
    R_base = np.vstack([r, u, f]).T
    
    # convert to quaternion for jitter calculations
    import scipy.spatial.transform
    base_quat = scipy.spatial.transform.Rotation.from_matrix(R_base).as_quat()  # [x,y,z,w]
    # keep base_quat in [x,y,z,w] format for quat_mul compatibility

    # try sampling jittered positions that remain inside airway
    chosen_pos = position.copy()
    chosen_quat = base_quat

    for attempt in range(max_attempts):
        # sample jitter
        jitter = random_point_in_sphere(pos_jitter_radius)
        candidate_pos = position + jitter

        # transform to vtk coordinates used in onlineSimulation
        transformed_point = np.dot(np.linalg.inv(simulator.R_model), candidate_pos - simulator.t_model) * 100
        transformed_point_vtk_cor = np.array([transformed_point[0], transformed_point[1], transformed_point[2]])

        # check closest point distance
        pointId_target = simulator.pointLocator.FindClosestPoint(transformed_point_vtk_cor)
        cloest_point_vtk_cor = np.array(simulator.vtkdata.GetPoint(pointId_target))
        distance_to_surface = np.linalg.norm(transformed_point_vtk_cor - cloest_point_vtk_cor)

        # check inside
        points = __import__('vtk').vtkPoints()
        points.InsertNextPoint(transformed_point_vtk_cor)
        pdata_points = __import__('vtk').vtkPolyData()
        pdata_points.SetPoints(points)
        enclosed_points_filter = __import__('vtk').vtkSelectEnclosedPoints()
        enclosed_points_filter.SetInputData(pdata_points)
        enclosed_points_filter.SetSurfaceData(simulator.vtkdata)
        enclosed_points_filter.SetTolerance(0.000001)
        enclosed_points_filter.Update()
        inside_flag = int(enclosed_points_filter.GetOutput().GetPointData().GetArray('SelectedPoints').GetTuple(0)[0])

        # require inside and distance to surface > a small threshold (e.g., 0.1 in vtk coords -> 1mm)
        if inside_flag == 1 and distance_to_surface > 0.5:  # 0.5 in vtk units ~= 0.5 mm (original code used 0.1 check earlier)
            chosen_pos = candidate_pos
            break
    else:
        # fallback: use original position
        chosen_pos = position

    # orientation jitter
    if angle_jitter_deg > 0:
        chosen_quat = apply_orientation_jitter(base_quat, angle_jitter_deg)  # returns [x,y,z,w]
        # use jittered quaternion for pose
        R = p.getMatrixFromQuaternion(chosen_quat)  # pybullet expects [x,y,z,w]
        R = np.reshape(R, (3, 3))
        pose = np.identity(4)
        pose[:3, 3] = chosen_pos
        pose[:3, :3] = R
    else:
        # no angle jitter - use the correct dir-method pose directly
        chosen_quat = base_quat  # [x,y,z,w] format
        pose = np.eye(4)
        pose[:3, :3] = R_base
        pose[:3, 3] = chosen_pos

    light_intensity = 0.3

    # build scene
    simulator.scene.clear()
    simulator.scene.add_node(simulator.fuze_node)
    from pyrender import SpotLight
    spot_l = SpotLight(color=np.ones(3), intensity=light_intensity,
                      innerConeAngle=0, outerConeAngle=np.pi/2, range=1)
    spot_l_node = simulator.scene.add(spot_l, pose=pose)
    cam_node = simulator.scene.add(simulator.cam, pose=pose)
    simulator.scene.set_pose(spot_l_node, pose)
    simulator.scene.set_pose(cam_node, pose)

    rgb_img, depth_img = simulator.r.render(simulator.scene)
    rgb_img = rgb_img[:, :, :3]

    # auto adjust light roughly (limited iterations)
    mean_intensity = np.mean(rgb_img)
    target_intensity = 140
    tolerance = 20
    iters = 0
    min_light = 0.001
    max_light = 20.0
    while abs(mean_intensity - target_intensity) > tolerance and iters < 50:
        if mean_intensity > target_intensity:
            max_light = light_intensity
        else:
            min_light = light_intensity
        light_intensity = (min_light + max_light) / 2

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
        iters += 1

    # process depth same as original
    depth_img[depth_img == 0] = 0.5
    depth_img[depth_img > 0.5] = 0.5
    depth_img = depth_img / 0.5 * 255
    depth_img = depth_img.astype(np.uint8)
    
    # resize
    rgb_img = cv2.resize(rgb_img, (image_size, image_size))
    depth_img = cv2.resize(depth_img, (image_size, image_size))
    rgb_img = rgb_img[:, :, ::-1]  # BGR

    # compute jitter info for logging
    jitter_vec = chosen_pos - position
    dist_m = np.linalg.norm(jitter_vec)
    
    # compute rotation difference (approximation)
    # compare chosen_quat and base_quat (both in [x,y,z,w] format)
    q_diff = p.getDifferenceQuaternion(chosen_quat, base_quat)  # pybullet expects [x,y,z,w]
    axis_angle = p.getAxisAngleFromQuaternion(q_diff)
    rot_deg = abs(axis_angle[1]) * 180.0 / np.pi
    
    jitter_info = {
        'dx': jitter_vec[0],
        'dy': jitter_vec[1], 
        'dz': jitter_vec[2],
        'dist_m': dist_m,
        'rot_deg': rot_deg,
        'final_quat': chosen_quat
    }

    return rgb_img, depth_img, jitter_info


def generate_images_at_point_with_jitter(simulator, position, pitch_deg, yaw_deg, image_size,
                                         pos_jitter_radius=0.01, angle_jitter_deg=30.0,
                                         max_attempts=20):
    """
    在指定中心线位置生成带抖动的RGB和深度图。
    保证抖动后位置仍在气道内部（使用 simulator 的 vtk 数据和 pointLocator 检查）。
    返回：rgb_img_bgr (H,W,3 uint8), depth_img_gray (H,W uint8)
    """
    # base orientation
    pitch_rad = pitch_deg / 180.0 * np.pi + np.pi / 2
    yaw_rad = yaw_deg / 180.0 * np.pi

    # try sampling jittered positions that remain inside airway
    chosen_pos = position.copy()
    chosen_quat = p.getQuaternionFromEuler([pitch_rad, 0, yaw_rad])

    for attempt in range(max_attempts):
        # sample jitter
        jitter = random_point_in_sphere(pos_jitter_radius)
        candidate_pos = position + jitter

        # transform to vtk coordinates used in onlineSimulation
        transformed_point = np.dot(np.linalg.inv(simulator.R_model), candidate_pos - simulator.t_model) * 100
        transformed_point_vtk_cor = np.array([transformed_point[0], transformed_point[1], transformed_point[2]])

        # check closest point distance
        pointId_target = simulator.pointLocator.FindClosestPoint(transformed_point_vtk_cor)
        cloest_point_vtk_cor = np.array(simulator.vtkdata.GetPoint(pointId_target))
        distance_to_surface = np.linalg.norm(transformed_point_vtk_cor - cloest_point_vtk_cor)

        # check inside
        points = __import__('vtk').vtkPoints()
        points.InsertNextPoint(transformed_point_vtk_cor)
        pdata_points = __import__('vtk').vtkPolyData()
        pdata_points.SetPoints(points)
        enclosed_points_filter = __import__('vtk').vtkSelectEnclosedPoints()
        enclosed_points_filter.SetInputData(pdata_points)
        enclosed_points_filter.SetSurfaceData(simulator.vtkdata)
        enclosed_points_filter.SetTolerance(0.000001)
        enclosed_points_filter.Update()
        inside_flag = int(enclosed_points_filter.GetOutput().GetPointData().GetArray('SelectedPoints').GetTuple(0)[0])

        # require inside and distance to surface > a small threshold (e.g., 0.1 in vtk coords -> 1mm)
        if inside_flag == 1 and distance_to_surface > 0.5:  # 0.5 in vtk units ~= 0.5 mm (original code used 0.1 check earlier)
            chosen_pos = candidate_pos
            break
    else:
        # fallback: use original position
        chosen_pos = position

    # orientation jitter
    quat_base = p.getQuaternionFromEuler([pitch_rad, 0, yaw_rad])
    quat_jittered = apply_orientation_jitter(quat_base, angle_jitter_deg)

    # prepare scene and render
    R = p.getMatrixFromQuaternion(quat_jittered)
    R = np.reshape(R, (3, 3))
    pose = np.identity(4)
    pose[:3, 3] = chosen_pos
    pose[:3, :3] = R

    light_intensity = 0.3

    # build scene
    simulator.scene.clear()
    simulator.scene.add_node(simulator.fuze_node)
    from pyrender import SpotLight
    spot_l = SpotLight(color=np.ones(3), intensity=light_intensity,
                      innerConeAngle=0, outerConeAngle=np.pi/2, range=1)
    spot_l_node = simulator.scene.add(spot_l, pose=pose)
    cam_node = simulator.scene.add(simulator.cam, pose=pose)
    simulator.scene.set_pose(spot_l_node, pose)
    simulator.scene.set_pose(cam_node, pose)

    rgb_img, depth_img = simulator.r.render(simulator.scene)
    rgb_img = rgb_img[:, :, :3]

    # auto adjust light roughly (limited iterations)
    mean_intensity = np.mean(rgb_img)
    target_intensity = 140
    tolerance = 20
    iters = 0
    min_light = 0.001
    max_light = 20.0
    while abs(mean_intensity - target_intensity) > tolerance and iters < 50:
        if mean_intensity > target_intensity:
            max_light = light_intensity
        else:
            min_light = light_intensity
        light_intensity = (min_light + max_light) / 2

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
        iters += 1

    # process depth same as original
    depth_img[depth_img == 0] = 0.5
    depth_img[depth_img > 0.5] = 0.5
    depth_img = depth_img / 0.5 * 255
    depth_img = depth_img.astype(np.uint8)

    rgb_img = cv2.resize(rgb_img, (image_size, image_size))
    depth_img = cv2.resize(depth_img, (image_size, image_size))
    rgb_img = rgb_img[:, :, ::-1]  # RGB->BGR for OpenCV

    # compute translation (ground-truth) and rotation difference (ground-truth)
    translation_vec = chosen_pos - position
    translation_norm = float(np.linalg.norm(translation_vec))

    # relative quaternion q_rel = quat_jittered * conj(quat_base)
    q_rel = quat_mul(quat_jittered, quat_conjugate(quat_base))
    q_rel = np.array(q_rel)
    q_rel = q_rel / np.linalg.norm(q_rel)
    # rotation angle (rad) = 2*acos(w)
    w = np.clip(q_rel[3], -1.0, 1.0)
    angle_rad = 2.0 * np.arccos(w)
    angle_deg = float(np.degrees(angle_rad))

    return rgb_img, depth_img, chosen_pos, quat_jittered, translation_vec, translation_norm, angle_deg, q_rel.tolist()


def generate_centerline_dataset_jitter(args):
    # prepare output dirs
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    rgb_dir = os.path.join(output_dir, 'rgb_images')
    depth_dir = os.path.join(output_dir, 'depth_images')
    info_dir = os.path.join(output_dir, 'camera_info')
    for d in (rgb_dir, depth_dir, info_dir):
        if not os.path.exists(d):
            os.makedirs(d)

    # init simulator (single process) - this will open pybullet GUI by default (same as original)
    simulator = onlineSimulator(args.dataset_dir, args.centerline_name, renderer=args.renderer, training=False)

    centerline_points = simulator.centerlineArray
    total = len(centerline_points)

    info_file = os.path.join(info_dir, 'dataset_info_jitter.txt')
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write('# Dataset with jitter\n')
        f.write(f'centerline_name: {args.centerline_name}\n')
        f.write(f'pos_jitter_radius: {args.pos_jitter_radius}\n')
        f.write(f'angle_jitter_deg: {args.angle_jitter_deg}\n')
        f.write(f'image_size: {args.image_size}\n')
        f.write(f'total_points: {total}\n')
        f.write('# Format: index, orig_x, orig_y, orig_z, pos_x, pos_y, pos_z, qx, qy, qz, qw, dx, dy, dz, dist_m, rot_deg\n')

    generated = 0
    # process every point on the centerline (one-to-one mapping: original -> jittered)
    for i in range(total):
        point = centerline_points[i]
        # compute direction using neighbor (for the WORKING dir method)
        if i < total - 1:
            dir_vec = centerline_points[i+1] - point
        elif i > 0:
            dir_vec = point - centerline_points[i-1]
        else:
            dir_vec = np.array([0, 1, 0])

        try:
            rgb_img, depth_img, jitter_info = generate_images_at_point_with_jitter_and_direction(
                simulator, point, dir_vec, args.image_size,
                pos_jitter_radius=args.pos_jitter_radius,
                angle_jitter_deg=args.angle_jitter_deg
            )
            
            # extract jitter info for compatibility
            chosen_pos = point + np.array([jitter_info['dx'], jitter_info['dy'], jitter_info['dz']])
            chosen_quat = jitter_info['final_quat']
            translation_vec = np.array([jitter_info['dx'], jitter_info['dy'], jitter_info['dz']])
            translation_norm = jitter_info['dist_m']
            rotation_deg = jitter_info['rot_deg']

            rgb_name = f'rgb_{i:06d}.jpg'
            depth_name = f'depth_{i:06d}.jpg'
            cv2.imwrite(os.path.join(rgb_dir, rgb_name), rgb_img)
            cv2.imwrite(os.path.join(depth_dir, depth_name), depth_img)

            with open(info_file, 'a', encoding='utf-8') as f:
                f.write(f"{i}, {point[0]:.6f}, {point[1]:.6f}, {point[2]:.6f}, {chosen_pos[0]:.6f}, {chosen_pos[1]:.6f}, {chosen_pos[2]:.6f}, {chosen_quat[0]:.6f}, {chosen_quat[1]:.6f}, {chosen_quat[2]:.6f}, {chosen_quat[3]:.6f}, {translation_vec[0]:.6f}, {translation_vec[1]:.6f}, {translation_vec[2]:.6f}, {translation_norm:.6f}, {rotation_deg:.6f}\n")

            generated += 1
            if generated % 50 == 0:
                print(f'Generated {generated} images...')

        except Exception as e:
            print(f'Error at point {i}: {e}')
            continue

    # clean up
    try:
        p.disconnect()
    except:
        pass
    try:
        simulator.r.delete()
    except:
        pass

    print(f'Dataset generated: {generated} image pairs saved to {output_dir}')
    return generated


def generate_batch(args):
    # use subprocess to avoid multiple GUI in same process
    import subprocess
    if not os.path.exists(args.output_base_dir):
        os.makedirs(args.output_base_dir)

    centerline_names = ["siliconmodel3 Centerline model"] + [f"siliconmodel3 Centerline model_{i}" for i in range(1,60)]
    log_file = os.path.join(args.output_base_dir, 'generation_log_jitter.txt')
    with open(log_file, 'w', encoding='utf-8') as lf:
        lf.write('Batch jitter generation log\n')

    success = 0
    failed = []
    for idx in range(args.start_index, min(args.end_index+1, len(centerline_names))):
        name = centerline_names[idx]
        outdir = os.path.join(args.output_base_dir, f'centerline_{idx:02d}_' + name.replace(' ', '_'))
        cmd = [sys.executable, __file__, '--dataset-dir', args.dataset_dir, '--output-dir', outdir,
               '--centerline-name', name,
               '--image-size', str(args.image_size), '--step-size', str(args.step_size),
               '--pos-jitter-radius', str(args.pos_jitter_radius), '--angle-jitter-deg', str(args.angle_jitter_deg)]
        print('Running:', ' '.join(cmd))
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if res.returncode == 0:
                success += 1
                with open(log_file, 'a', encoding='utf-8') as lf:
                    lf.write(f'SUCCESS: {name} -> {outdir}\n')
            else:
                failed.append((name, res.stderr[:200]))
                with open(log_file, 'a', encoding='utf-8') as lf:
                    lf.write(f'FAILED: {name} -> {res.stderr}\n')
        except Exception as e:
            failed.append((name, str(e)))
            with open(log_file, 'a', encoding='utf-8') as lf:
                lf.write(f'EXCEPTION: {name} -> {e}\n')

    print(f'Batch finished. Success: {success}, Failed: {len(failed)}. Log: {log_file}')
    return success, failed


if __name__ == '__main__':
    args = get_args()
    if args.batch:
        generate_batch(args)
    else:
        generate_centerline_dataset_jitter(args)
