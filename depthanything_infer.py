"""\
DepthAnything 推理脚本 (支持两种后端)
功能: 调用 DepthAnything 预训练模型，对输入 RGB 图像生成深度图，保存格式与自训练 UNet 脚本一致：
    - 若输入为中心线目录 (含 rgb_images)，输出到 <output>/depth_images/depth_XXXX.jpg
    - 深度以 0~0.5 米范围归一化后再按 uint8 (0~255) 保存 (depth/0.5*255)

后端选择 (--backend):
    da   : 使用本地 depth-anything / depth_anything 包 (原始仓库实现)
    hf   : 使用 Hugging Face transformers (LiheYoung/depth-anything-*-hf)
    auto : 优先 da, 不可用则回退 hf

注意:
1. 原模型输出是相对深度(无绝对尺度)，本脚本对每张图做自适应归一：min-max 或分位数裁剪后映射到 [0, 0.5]。
2. 若你需要批量高吞吐，hf 后端支持 batch；da 后端目前按原实现逐 batch tensor。
3. da 安装困难时可直接用 hf： pip install "transformers>=4.40" accelerate safetensors pillow
4. 评估使用统一 0~0.5 映射即可与自训练 UNet 比较。

安装示例 (PowerShell):
    # 后端 da:
    pip install torch torchvision timm depth-anything  (或 depth_anything)
    # 后端 hf:
    pip install torch torchvision transformers accelerate safetensors pillow

若无网络，请手动下载 HF 模型权重并放入本地缓存，再指定 --hf-model 自定义目录。
"""

import os
import argparse
import sys
import cv2
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
from typing import List, Tuple
from PIL import Image

try:  # 本地包后端
    try:
        from depth_anything.dpt import DepthAnythingV2 as DepthAnythingModel  # type: ignore
    except Exception:
        from depth_anything_v2.dpt import DepthAnythingV2 as DepthAnythingModel  # type: ignore
    HAS_DA = True
except Exception:
    HAS_DA = False

try:  # HF 后端
    from transformers import AutoProcessor, DepthAnythingForDepthEstimation
    HAS_HF = True
except Exception:
    HAS_HF = False


def load_da_model(model_type: str, device: torch.device, weights: str = None):
    if not HAS_DA:
        raise ImportError("未找到本地 depth-anything 包，可切换 --backend hf")
    size_map = {'small': 'v2-small','base': 'v2-base','large': 'v2-large'}
    model_key = size_map.get(model_type, model_type)
    model = DepthAnythingModel(model_key, pretrained=True if weights is None else False)
    if weights:
        ckpt = torch.load(weights, map_location='cpu')
        sd = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model


def load_hf_model(hf_model: str, device: torch.device):
    if not HAS_HF:
        raise ImportError("未安装 transformers，请先 pip install transformers accelerate safetensors pillow")
    processor = AutoProcessor.from_pretrained(hf_model)
    model = DepthAnythingForDepthEstimation.from_pretrained(hf_model).to(device).eval()
    return processor, model


def preprocess_image_cv(path: str, target_size: int):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f'无法读取图像: {path}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]
    img_resized = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    img_tensor = torch.from_numpy(img_resized.astype(np.float32)/255.).permute(2, 0, 1)
    return img_tensor, (orig_h, orig_w), img


def load_images_pil(paths: List[str]) -> List[Image.Image]:
    ims = []
    for p in paths:
        im = Image.open(p).convert('RGB')
        ims.append(im)
    return ims


def scale_depth_relative(depth: np.ndarray, method: str, max_depth: float, clip_min: float, clip_max: float):
    d = depth.copy()
    if method == 'minmax':
        d_min = d.min()
        d_max = d.max()
        if d_max - d_min < 1e-9:
            d[:] = 0.0
        else:
            d = (d - d_min) / (d_max - d_min)
    elif method == 'percentile':
        p1 = np.percentile(d, clip_min)
        p2 = np.percentile(d, clip_max)
        if p2 - p1 < 1e-9:
            d[:] = 0.0
        else:
            d = (d - p1) / (p2 - p1)
        d = np.clip(d, 0, 1)
    else:
        raise ValueError('未知 scale-method')
    d = d * max_depth  # 映射到 [0, max_depth]
    return d


def save_depth(depth: np.ndarray, path: str, max_depth: float):
    depth = np.clip(depth, 0, max_depth)
    uint8 = (depth / max_depth * 255).astype(np.uint8)
    cv2.imwrite(path, uint8)


def run_infer(args):
    device = torch.device(args.device)
    backend = args.backend
    if backend == 'auto':
        backend = 'da' if HAS_DA else 'hf'
    print(f"选择后端: {backend}")

    da_model = None; hf_processor = None; hf_model = None
    if backend == 'da':
        da_model = load_da_model(args.model_type, device, args.weights)
    elif backend == 'hf':
        hf_processor, hf_model = load_hf_model(args.hf_model, device)
    else:
        raise ValueError('backend 必须是 da / hf / auto')

    if args.input_format == 'centerline':
        rgb_dir = os.path.join(args.input_dir, 'rgb_images')
        if not os.path.isdir(rgb_dir):
            raise ValueError(f'rgb_images 不存在: {rgb_dir}')
        image_paths = [os.path.join(rgb_dir, f) for f in sorted(os.listdir(rgb_dir)) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        depth_out_dir = os.path.join(args.output_dir, 'depth_images')
    else:
        image_paths = [os.path.join(args.input_dir, f) for f in sorted(os.listdir(args.input_dir)) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        depth_out_dir = args.output_dir

    os.makedirs(depth_out_dir, exist_ok=True)
    print(f'图像数量: {len(image_paths)} 输出目录: {depth_out_dir}')

    for i in tqdm(range(0, len(image_paths), args.batch_size), desc='DepthAnything Infer'):
        batch_paths = image_paths[i:i + args.batch_size]
        batch_tensors = []
        metas = []
        if backend == 'da':
            for p in batch_paths:
                t, (h, w), _ = preprocess_image_cv(p, args.image_size)
                batch_tensors.append(t); metas.append((p, h, w))
            batch_tensor = torch.stack(batch_tensors).to(device)
            with torch.no_grad():
                try:
                    pred = da_model(batch_tensor)
                except Exception:
                    pred = da_model.forward(batch_tensor)
            if pred.dim() == 3:
                pred = pred.unsqueeze(1)
            pred_np = pred.squeeze(1).cpu().numpy()
        else:  # hf
            # 读取 PIL images 原尺寸
            ims = load_images_pil(batch_paths)
            orig_sizes = [im.size[::-1] for im in ims]  # (H,W)
            with torch.no_grad():
                inputs = hf_processor(images=ims, return_tensors='pt').to(device)
                outputs = hf_model(**inputs)
                pred = outputs.predicted_depth  # [B,1,H,W] 或 [B,H,W]
            if pred.dim() == 3:
                pred = pred.unsqueeze(1)
            pred_np = pred.squeeze(1).cpu().numpy()
            metas = [(p, h, w) for p, (h, w) in zip(batch_paths, orig_sizes)]

        for (p_img, orig_h, orig_w), d_rel in zip(metas, pred_np):
            # 归一化到 [0, args.max_depth]
            d_scaled = scale_depth_relative(d_rel, args.scale_method, args.max_depth, args.clip_min, args.clip_max)
            # 可选平滑
            if args.median_ksize > 1:
                k = args.median_ksize if args.median_ksize % 2 == 1 else args.median_ksize + 1
                d_scaled = cv2.medianBlur(d_scaled.astype(np.float32), k)
            # 恢复回原尺寸 or 指定统一尺寸：与训练保持 (args.image_size)
            if args.keep_original_size:
                d_resized = cv2.resize(d_scaled, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            else:
                d_resized = cv2.resize(d_scaled, (args.image_size, args.image_size), interpolation=cv2.INTER_LINEAR)

            base = os.path.splitext(os.path.basename(p_img))[0]
            if args.input_format == 'centerline' and base.startswith('rgb_'):
                out_name = base.replace('rgb_', 'depth_') + '.jpg'
            else:
                out_name = base + '_depth.jpg'
            save_depth(d_resized, os.path.join(depth_out_dir, out_name), args.max_depth)

    print('推理完成')


def build_parser():
    p = argparse.ArgumentParser(description='DepthAnything 推理脚本 (da/hf 后端, 输出与 UNet 格式对齐)')
    p.add_argument('--input-dir', required=True, help='输入目录 (centerline 模式: 包含 rgb_images)')
    p.add_argument('--output-dir', default='depthanything_predictions', help='输出根目录 (会生成 depth_images)')
    p.add_argument('--input-format', choices=['centerline', 'single'], default='centerline')
    p.add_argument('--image-size', type=int, default=200, help='网络推理尺寸 (方形)')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--model-type', default='v2-small', help='模型类型: v2-small / v2-base / v2-large (或 small/base/large)')
    p.add_argument('--weights', default=None, help='自定义权重路径 (可选)')
    p.add_argument('--backend', choices=['auto','da','hf'], default='auto', help='选择后端: da 本地包 / hf transformers / auto 自动')
    p.add_argument('--hf-model', default='LiheYoung/depth-anything-small-hf', help='HF 模型名称或本地路径 (hf 后端)')
    # 深度缩放相关
    p.add_argument('--scale-method', choices=['minmax', 'percentile'], default='percentile')
    p.add_argument('--clip-min', type=float, default=2.0, help='percentile 下界 (0-100)')
    p.add_argument('--clip-max', type=float, default=98.0, help='percentile 上界 (0-100)')
    p.add_argument('--max-depth', type=float, default=0.5, help='映射目标最大深度值 (与训练保持一致)')
    p.add_argument('--median-ksize', type=int, default=0, help='中值滤波核(>1 启用, 自动转奇数)')
    p.add_argument('--keep-original-size', action='store_true', help='保存为原始尺寸 (否则统一为 image-size)')
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_infer(args)


if __name__ == '__main__':
    main()
