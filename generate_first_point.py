import os
import sys
import argparse
import numpy as np
import cv2
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-dir', required=False, default='AI-Co-Pilot-Bronchoscope-Robot')
    p.add_argument('--centerline-name', required=False, default='siliconmodel3 Centerline model')
    p.add_argument('--output-dir', required=False, default='first_point_out')
    p.add_argument('--image-size', type=int, default=200)
    p.add_argument('--num-points', type=int, default=1, help='Number of consecutive points to render for quick check')
    p.add_argument('--start-index', type=int, default=-1, help='If >=0, use this centerline index as starting point (deterministic)')
    p.add_argument('--seed', type=int, default=None, help='Optional random seed for reproducibility')
    return p.parse_args()


def safe_import_simulator():
    try:
        from lib.engine.onlineSimulation import onlineSimulationWithNetwork as onlineSimulator
        return onlineSimulator
    except ModuleNotFoundError as e:
        print('Failed to import simulator module. Make sure the environment has required packages (torch, vtk, pybullet, etc).')
        raise


def render_with_direction_pose(simulator, pose, image_size):
    """Render using the WORKING 'dir' method - direct pose from tangent direction."""
    from pyrender import SpotLight
    # pose is 4x4 numpy array
    simulator.scene.clear()
    simulator.scene.add_node(simulator.fuze_node)

    light_intensity = 0.3
    spot_l = SpotLight(color=np.ones(3), intensity=light_intensity,
                      innerConeAngle=0, outerConeAngle=np.pi/2, range=1)
    spot_l_node = simulator.scene.add(spot_l, pose=pose)
    cam_node = simulator.scene.add(simulator.cam, pose=pose)

    simulator.scene.set_pose(spot_l_node, pose)
    simulator.scene.set_pose(cam_node, pose)

    rgb_img, depth_img = simulator.r.render(simulator.scene)
    rgb_img = rgb_img[:, :, :3]

    # automatic light adjust (same as render_at_quat)
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

    depth_img[depth_img == 0] = 0.5
    depth_img[depth_img > 0.5] = 0.5
    depth_img = depth_img / 0.5 * 255
    depth_img = depth_img.astype(np.uint8)

    rgb_img = cv2.resize(rgb_img, (image_size, image_size))
    depth_img = cv2.resize(depth_img, (image_size, image_size))
    rgb_img = rgb_img[:, :, ::-1]
    return rgb_img, depth_img


def main():
    args = get_args()
    onlineSimulator = safe_import_simulator()

    print('Initializing simulator...')
    sim = onlineSimulator(args.dataset_dir, args.centerline_name, renderer='pyrender', training=False)

    # Read centerline OBJ via VTK and apply same resampling + smoothing as onlineSimulation
    file_path = sim.centerline_model_dir
    reader = vtk.vtkOBJReader()
    reader.SetFileName(file_path)
    reader.Update()
    mesh = reader.GetOutput()
    points = mesh.GetPoints()
    data = points.GetData()
    centerlineArray = vtk_to_numpy(data)
    print(f"Raw OBJ points: {len(centerlineArray)} points")
    print(f"Raw point range: min={np.min(centerlineArray, axis=0)}, max={np.max(centerlineArray, axis=0)}")
    
    # Apply coordinate transform (same as onlineSimulation)
    centerlineArray = np.dot(sim.R_model, centerlineArray.T).T * 0.01 + sim.t_model
    print(f"After coordinate transform: {len(centerlineArray)} points")
    print(f"Transformed point range: min={np.min(centerlineArray, axis=0)}, max={np.max(centerlineArray, axis=0)}")
    
    # Apply resampling (same as onlineSimulation)
    length = 0.007  # same as in onlineSimulation
    distances = np.linalg.norm(np.diff(centerlineArray, axis=0), axis=1)
    cumulative_distance = np.cumsum(np.concatenate(([0], distances)))
    total_distance = cumulative_distance[-1]
    num_resampled_points = int(total_distance / length) + 1
    
    # Create evenly spaced distance points for resampling
    resampled_distances = np.linspace(0, total_distance, num_resampled_points)
    
    # Interpolate x, y, z coordinates
    resampled_points = []
    for i in range(3):  # x, y, z
        resampled_coord = np.interp(resampled_distances, cumulative_distance, centerlineArray[:, i])
        resampled_points.append(resampled_coord)
    
    centerlineArray = np.column_stack(resampled_points)
    print(f"After resampling: {len(centerlineArray)} points")
    
    # Apply smoothing (same as onlineSimulation smooth_centerline method)
    window_width = 10
    smoothed_points = []
    for i in range(len(centerlineArray)):
        start_idx = max(0, i - window_width // 2)
        end_idx = min(len(centerlineArray), i + window_width // 2 + 1)
        window_points = centerlineArray[start_idx:end_idx]
        smooth_point = np.mean(window_points, axis=0)
        smoothed_points.append(smooth_point)
    
    centerline = np.array(smoothed_points)
    print(f"After smoothing: {len(centerline)} points")

    if len(centerline) == 0:
        print('No centerline points found')
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # deterministic seed if provided
    if args.seed is not None:
        np.random.seed(int(args.seed))

    # determine starting index - simplified to always start from 0 unless explicitly set
    if args.start_index is not None and args.start_index >= 0:
        nearest_idx = int(min(max(0, args.start_index), len(centerline) - 1))
        print(f"Using user-specified start index: {nearest_idx}")
    else:
        nearest_idx = 0  # Always start from first point for simple testing
        print(f"Using default start index: {nearest_idx}")

    # determine list of consecutive indices to render starting at nearest_idx
    num_points = max(1, int(args.num_points))
    end_idx = min(nearest_idx + num_points, len(centerline))
    indices_to_render = list(range(nearest_idx, end_idx))

    rgb_dir = os.path.join(args.output_dir, 'rgb_images')
    os.makedirs(rgb_dir, exist_ok=True)

    for idx_i, ci in enumerate(indices_to_render):
        p0 = centerline[ci]
        # compute direction using neighbor
        if ci < len(centerline) - 1:
            p1 = centerline[ci + 1]
        elif ci > 0:
            p1 = centerline[ci - 1]
        else:
            p1 = p0 + np.array([0, 1, 0])
        dir_vec = p1 - p0

        print(f"\n--- Rendering point {ci} (check {idx_i+1}/{num_points}) ---")
        print('point:', p0)
        print('direction:', dir_vec)

        # Use the WORKING 'dir' method: build rotation matrix that aligns camera forward with dir_vec
        # forward = normalized dir_vec, up = [0,0,1] (if nearly parallel choose alternative)
        f = dir_vec / (np.linalg.norm(dir_vec) + 1e-12)
        up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(f, up)) > 0.999:
            up = np.array([0.0, 1.0, 0.0])
        r = np.cross(up, f)
        r = r / (np.linalg.norm(r) + 1e-12)
        u = np.cross(f, r)
        R = np.vstack([r, u, f]).T
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = p0
        rgb, depth = render_with_direction_pose(sim, pose, args.image_size)
        rgb_path = os.path.join(rgb_dir, f'rgb_{idx_i:06d}.jpg')
        cv2.imwrite(rgb_path, rgb)
        print('Saved pyrender RGB to', rgb_path)

    # Removed pybullet lookat comparison per user's request

    print('\nDone. Compare the saved images in', rgb_dir)


if __name__ == '__main__':
    main()
