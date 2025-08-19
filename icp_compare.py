"""
icp_compare.py
- 用于对比两组深度图（原始 vs 扰动）使用 ICP 得到的平移/旋转与 ground-truth 的差异。
- 依赖: open3d, numpy, opencv

基本流程：
1. 读取两个数据集文件夹（each contains depth_images/ and camera_info/dataset_info_jitter.txt）
2. 解析 camera_info，按行一一对应取深度图和相机内参（脚本使用固定 intrinsics，与 generate 脚本保持一致）
3. 将深度图转换为点云（单位：米）并下采样/裁剪
4. 对每对点云使用 Open3D 的 ICP（先粗配准再精配）
5. 输出每帧的平移（m）和旋转（deg），并与 info 文件中记录的 ground-truth 做差值统计

对黑图/空深度图的处理策略建议：
- 如果深度图有效像素过少（例如小于阈值 500 点），则跳过该对并标记为 invalid。
- 或者尝试使用图像金字塔/膨胀等预处理来扩展有效像素，但风险是会引入错误匹配。
- 另一个选择是结合 RGB 做基于特征的配准（SIFT/ORB->RANSAC->ICP）以增强鲁棒性。

此脚本实现了第一种策略（跳过空/近空深度图），并在日志中输出跳过原因。
"""

import os
import sys
import argparse
import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt

def imread_gray_unicode(path):
    """Unicode 兼容灰度读取: 解决 Windows 下含中文/特殊字符路径 cv2.imread 返回 None 问题"""
    try:
        # 使用 np.fromfile + cv2.imdecode 规避编码问题
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        return img
    except Exception:
        return None

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ref-dir', required=True, help='Reference dataset dir (no jitter)')
    p.add_argument('--src-dir', required=True, help='Source dataset dir (jittered)')
    p.add_argument('--min-points', type=int, default=500, help='Minimum valid depth points to accept a frame')
    p.add_argument('--voxel-size', type=float, default=0.002, help='Voxel size for downsampling (m)')
    p.add_argument('--out-csv', default='icp_results.csv', help='CSV to write per-frame results')
    return p.parse_args()


def load_info(info_path):
    import re
    with open(info_path, 'r', encoding='utf-8') as f:
        raw = [ln.rstrip('\n') for ln in f]

    # find last line that starts with '#'
    last_header_idx = -1
    for i, ln in enumerate(raw):
        s = ln.strip()
        if s.startswith('#'):
            last_header_idx = i

    data_lines = raw[last_header_idx+1:]
    lines = []
    for ln in data_lines:
        ln = ln.strip()
        if not ln:
            continue
        parts = [x.strip() for x in ln.split(',')]
        # try direct float conversion first
        try:
            vals = list(map(float, parts))
            lines.append(vals)
            continue
        except Exception:
            # fall back: extract numeric substrings from each token
            nums = []
            for tok in parts:
                found = re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', tok)
                for fnum in found:
                    try:
                        nums.append(float(fnum))
                    except Exception:
                        pass
            if len(nums) == 0:
                continue
            lines.append(nums)

    return lines


def depth_to_pointcloud(depth_img, intrinsics, depth_scale=255.0/0.5):
    # depth_img: uint8 where 255 maps to 0.5 m (as generator did)
    # convert back to meters
    depth = depth_img.astype(np.float32) / 255.0 * 0.5
    h, w = depth.shape
    fx, fy, cx, cy = intrinsics
    mask = depth > 0.001
    ys, xs = np.where(mask)
    zs = depth[ys, xs]
    xs_cam = (xs - cx) * zs / fx
    ys_cam = (ys - cy) * zs / fy
    pts = np.stack([xs_cam, ys_cam, zs], axis=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def run_icp(source_pcd, target_pcd, voxel_size=0.002):
    # preprocess
    src_down = source_pcd.voxel_down_sample(voxel_size)
    tgt_down = target_pcd.voxel_down_sample(voxel_size)
    src_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    tgt_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))

    # initial alignment with identity
    trans_init = np.eye(4)
    # run point-to-plane ICP
    reg = o3d.pipelines.registration.registration_icp(
        src_down, tgt_down, voxel_size*1.5, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return reg.transformation, reg.inlier_rmse, len(reg.correspondence_set)


def transform_to_translation_rotation_deg(T):
    t = T[:3, 3]
    R = T[:3, :3]
    # rotation angle from rotation matrix
    trace = np.clip(np.trace(R), -1.0, 3.0)
    angle_rad = np.arccos((trace - 1.0) / 2.0)
    angle_deg = np.degrees(angle_rad)
    return t, angle_deg


def main():
    args = get_args()
    import glob

    def find_info_file(dataset_dir):
        pattern = os.path.join(dataset_dir, 'camera_info', 'dataset_info*.txt')
        matches = glob.glob(pattern)
        if len(matches) > 0:
            # prefer jitter if present
            for m in matches:
                if 'jitter' in os.path.basename(m):
                    return m
            return matches[0]
        return None

    ref_info = find_info_file(args.ref_dir)
    src_info = find_info_file(args.src_dir)
    if ref_info is None:
        raise FileNotFoundError(f"Reference dataset info not found in {args.ref_dir}/camera_info. Expected dataset_info_jitter.txt or dataset_info.txt")
    if src_info is None:
        raise FileNotFoundError(f"Source dataset info not found in {args.src_dir}/camera_info. Expected dataset_info_jitter.txt or dataset_info.txt")

    ref_lines = load_info(ref_info)
    src_lines = load_info(src_info)

    if len(ref_lines) != len(src_lines):
        print(f'Warning: ref info lines = {len(ref_lines)}, src info lines = {len(src_lines)}; will process up to min length')

    # intrinsics: assume square image and same intrinsics as generator (image_size default 200)
    image_size = 200
    fx = fy = 200.0 * 0.5  # approximate focal length used by generator (if different, adjust)
    cx = cy = image_size / 2.0
    intrinsics = (fx, fy, cx, cy)

    out_lines = []
    # process all available lines (one-to-one mapping of info files)
    n = len(ref_lines)

    for i in range(n):
        ref_vals = ref_lines[i]
        src_vals = src_lines[i]
        # parse indices in file: we expect CSV with numeric fields; original format started with index
        # We rely on depth file naming convention: depth_{index:06d}.jpg
        idx = int(ref_vals[0])
        depth_ref_path = os.path.join(args.ref_dir, 'depth_images', f'depth_{idx:06d}.jpg')
        depth_src_path = os.path.join(args.src_dir, 'depth_images', f'depth_{idx:06d}.jpg')

        if not os.path.exists(depth_ref_path) or not os.path.exists(depth_src_path):
            print(f"Missing depth for index {idx}, skip")
            continue
        depth_ref = cv2.imread(depth_ref_path, cv2.IMREAD_GRAYSCALE)
        depth_src = cv2.imread(depth_src_path, cv2.IMREAD_GRAYSCALE)
        # 若普通读取失败，使用 unicode 安全方式再试
        if depth_ref is None:
            depth_ref = imread_gray_unicode(depth_ref_path)
        if depth_src is None:
            depth_src = imread_gray_unicode(depth_src_path)
        if depth_ref is None or depth_src is None:
            print(f"Index {idx}: 读取失败 (可能是路径含中文导致) ref={os.path.exists(depth_ref_path)} src={os.path.exists(depth_src_path)} 跳过")
            out_lines.append((idx, False, 0, 0, None, None))
            continue

        # quick check for near-empty depth
        valid_ref = np.count_nonzero(depth_ref > 1)
        valid_src = np.count_nonzero(depth_src > 1)
        if valid_ref < args.min_points or valid_src < args.min_points:
            print(f"Index {idx}: too few depth points (ref {valid_ref}, src {valid_src}), skip")
            out_lines.append((idx, False, valid_ref, valid_src, None, None))
            continue

        pcd_ref = depth_to_pointcloud(depth_ref, intrinsics)
        pcd_src = depth_to_pointcloud(depth_src, intrinsics)

        try:
            T, rmse, corr = run_icp(pcd_src, pcd_ref, voxel_size=args.voxel_size)
            t_icp, angle_icp = transform_to_translation_rotation_deg(T)
            gt_dist = None
            gt_rot = None
            gt_dx = None
            gt_dy = None
            gt_dz = None
            try:
                if len(src_vals) >= 16:
                    # older format: dist at index 14, rot at 15
                    gt_dist = float(src_vals[14])
                    gt_rot = float(src_vals[15])
                    # per-axis dx,dy,dz stored earlier in jitter file
                    try:
                        gt_dx = float(src_vals[11])
                        gt_dy = float(src_vals[12])
                        gt_dz = float(src_vals[13])
                    except Exception:
                        gt_dx = gt_dy = gt_dz = None
                elif len(src_vals) >= 2:
                    # fallback: assume last two columns are dist and rot
                    gt_dist = float(src_vals[-2])
                    gt_rot = float(src_vals[-1])
                    # try to extract per-axis from last 4 if available
                    if len(src_vals) >= 6:
                        try:
                            gt_dx = float(src_vals[-6])
                            gt_dy = float(src_vals[-5])
                            gt_dz = float(src_vals[-4])
                        except Exception:
                            gt_dx = gt_dy = gt_dz = None
            except Exception:
                gt_dist = None
                gt_rot = None
                gt_dx = gt_dy = gt_dz = None
            out_lines.append((idx, True, valid_ref, valid_src, t_icp.tolist(), angle_icp, gt_dist, gt_rot, gt_dx, gt_dy, gt_dz, rmse, corr))
            print(f"Index {idx}: ICP t={t_icp}, angle_deg={angle_icp:.3f}, gt_dist={gt_dist}, gt_rot={gt_rot}, gt_dx={gt_dx},gt_dy={gt_dy},gt_dz={gt_dz}, rmse={rmse:.4f}, corr={corr}")
        except Exception as e:
            print(f"Index {idx}: ICP failed: {e}")
            out_lines.append((idx, False, valid_ref, valid_src, None, None))

    # write CSV
    import csv
    with open(args.out_csv, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.writer(cf)
        header = ['index', 'ok', 'valid_ref', 'valid_src', 't_x', 't_y', 't_z', 'angle_deg', 'gt_dist_m', 'gt_rot_deg', 'rmse', 'corr']
        writer.writerow(header)
        for row in out_lines:
            if row[1] and row[4] is not None:
                # row structure: idx, ok, vr, vs, t(list), ang, gt_dist, gt_rot, gt_dx, gt_dy, gt_dz, rmse, corr
                idx, ok, vr, vs, t, ang = row[0], row[1], row[2], row[3], row[4], row[5]
                # attempt to unpack gt and stats
                try:
                    gt_d = row[6]
                    gt_r = row[7]
                    rmse = row[11]
                    corr = row[12]
                except Exception:
                    gt_d = gt_r = ''
                    rmse = corr = ''
                writer.writerow([idx, ok, vr, vs, t[0], t[1], t[2], ang, gt_d, gt_r, rmse, corr])
            else:
                # skipped or failed
                if len(row) >= 4:
                    idx, ok, vr, vs = row[0], row[1], row[2], row[3]
                    writer.writerow([idx, ok, vr, vs, '', '', '', '', '', '', '', ''])

    print(f"Done. Results written to {args.out_csv}")

    # --- analysis: compute per-axis and rotation absolute errors and plot + print means ---
    # collect errors
    errs_x = []
    errs_y = []
    errs_z = []
    errs_deg = []
    idxs = []
    for row in out_lines:
        if not (row[1] and row[4] is not None):
            continue
        # unpack
        # expected successful row len >= 13
        try:
            idx = row[0]
            t = row[4]
            ang = row[5]
            gt_dx = row[8]
            gt_dy = row[9]
            gt_dz = row[10]
            gt_rot = row[7]
        except Exception:
            continue
        if gt_dx is None or gt_dy is None or gt_dz is None or gt_rot is None:
            continue
        tx, ty, tz = t[0], t[1], t[2]
        # convert meter errors to millimeters for presentation
        errs_x.append(abs(tx - float(gt_dx)) * 1000.0)
        errs_y.append(abs(ty - float(gt_dy)) * 1000.0)
        errs_z.append(abs(tz - float(gt_dz)) * 1000.0)
        errs_deg.append(abs(ang - float(gt_rot)))
        idxs.append(idx)

    if len(errs_x) > 0:
        mean_x = float(np.mean(errs_x))
        mean_y = float(np.mean(errs_y))
        mean_z = float(np.mean(errs_z))
        mean_deg = float(np.mean(errs_deg))
        print(f"Mean absolute errors -- x: {mean_x:.3f} mm, y: {mean_y:.3f} mm, z: {mean_z:.3f} mm, deg: {mean_deg:.6f} deg")

        # plot
        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        axes[0].scatter(range(len(errs_x)), errs_x, label='x')
        axes[0].scatter(range(len(errs_y)), errs_y, label='y')
        axes[0].scatter(range(len(errs_z)), errs_z, label='z')
        axes[0].set_ylabel('abs translation error (mm)')
        axes[0].legend()
        # 自定义坐标轴范围与刻度：0~35 mm，间隔 5
        import numpy as _np
        axes[0].set_ylim(0, 35)
        axes[0].set_yticks(_np.arange(0, 36, 5))
        axes[0].grid(True, linestyle='--', alpha=0.4)

        axes[1].scatter(range(len(errs_deg)), errs_deg, color='k')
        axes[1].set_ylabel('abs rotation error (deg)')
        axes[1].set_xlabel('frame index (in processed set)')
        # 角度范围 0~60 度，间隔 10
        axes[1].set_ylim(0, 60)
        axes[1].set_yticks(_np.arange(0, 61, 10))
        axes[1].grid(True, linestyle='--', alpha=0.4)

        out_plot = os.path.splitext(args.out_csv)[0] + '_analysis.png'
        fig.tight_layout()
        fig.savefig(out_plot)
        print(f'Analysis plot saved to {out_plot}')
    else:
        print('No per-axis ground-truth found in info files; analysis skipped')


if __name__ == '__main__':
    main()
