"""
统一UNet单目深度估计脚本
通过 --mode 选择功能：
  train  : 训练模型
  infer  : 推理生成深度图
  eval   : 评估预测结果
  all    : 训练 -> 推理 -> 评估 一条龙（需指定必要参数）

示例：
  训练:
    python unet_depth_pipeline.py --mode train --data-dir all_centerline_datasets --centerlines centerline_00 centerline_01
  推理(对单条中心线数据集):
    python unet_depth_pipeline.py --mode infer --model-path unet_depth_checkpoints/best_model.pth \
        --input-dir all_centerline_datasets/centerline_00_siliconmodel3_Centerline_model --output-dir unet_predictions/centerline_00
  评估(多个中心线):
    python unet_depth_pipeline.py --mode eval --gt-dir all_centerline_datasets --pred-dir unet_predictions --output-dir evaluation_results
  一条龙:
    python unet_depth_pipeline.py --mode all --data-dir all_centerline_datasets --centerlines centerline_00 centerline_01 \
        --infer-centerline centerline_00_siliconmodel3_Centerline_model --pred-dir unet_pipeline_predictions

注意：为保持原脚本兼容，未删除旧脚本，可逐步迁移。
"""

import os
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

# ===================== 模型与数据集 ===================== #
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class UNetDepthEstimation(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=(64,128,256,512)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f))
            ch = f
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f*2, f, 2, 2))
            self.ups.append(DoubleConv(f*2, f))
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)
        self.out_act = nn.Sigmoid()
    def forward(self, x):
        skips = []
        for d in self.downs:
            x = d(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            sc = skips[i//2]
            if x.shape[2:] != sc.shape[2:]:
                x = F.interpolate(x, size=sc.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([sc, x], dim=1)
            x = self.ups[i+1](x)
        x = self.final_conv(x)
        x = self.out_act(x) * 0.5  # 深度范围[0,0.5]
        return x

class CenterlineDepthDataset(Dataset):
    def __init__(self, dataset_dirs, image_size=200, max_samples_per_centerline=None):
        self.rgb_paths = []
        self.depth_paths = []
        self.image_size = image_size
        for ds in dataset_dirs:
            rgb_dir = os.path.join(ds, 'rgb_images')
            depth_dir = os.path.join(ds, 'depth_images')
            if not (os.path.isdir(rgb_dir) and os.path.isdir(depth_dir)):
                continue
            files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.jpg')])
            if max_samples_per_centerline:
                files = files[:max_samples_per_centerline]
            for f in files:
                dfile = f.replace('rgb_', 'depth_')
                dpath = os.path.join(depth_dir, dfile)
                if os.path.exists(dpath):
                    self.rgb_paths.append(os.path.join(rgb_dir, f))
                    self.depth_paths.append(dpath)
        if not self.rgb_paths:
            raise ValueError('未找到匹配的RGB/Depth样本')
        print(f'加载样本数: {len(self.rgb_paths)}')
    def __len__(self):
        return len(self.rgb_paths)
    def __getitem__(self, idx):
        rgb = cv2.imread(self.rgb_paths[idx])
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(self.depth_paths[idx], cv2.IMREAD_GRAYSCALE)
        rgb = cv2.resize(rgb, (self.image_size, self.image_size))
        depth = cv2.resize(depth, (self.image_size, self.image_size))
        depth = depth.astype(np.float32)/255.0*0.5
        rgb = rgb.astype(np.float32)/255.0
        rgb_t = torch.from_numpy(rgb).permute(2,0,1)
        depth_t = torch.from_numpy(depth).unsqueeze(0)
        return rgb_t, depth_t

# ===================== 公共工具函数 ===================== #

def list_centerline_dirs(root, prefixes=None):
    """根据前缀列表返回匹配的中心线目录; prefixes 为 None 时返回全部 centerline_* 目录"""
    p_root = Path(root)
    if not p_root.exists():
        return []
    if prefixes is None:
        return [str(p) for p in sorted(p_root.glob('centerline_*')) if p.is_dir()]
    # 允许显式传入 ALL 代表全部
    if any(p.upper() == 'ALL' for p in prefixes):
        return [str(p) for p in sorted(p_root.glob('centerline_*')) if p.is_dir()]
    dirs = []
    for prefix in prefixes:
        matches = list(p_root.glob(prefix + '*'))
        for m in matches:
            if m.is_dir():
                dirs.append(str(m))
    # 去重并排序
    dirs = sorted(list(dict.fromkeys(dirs)))
    return dirs

def save_prediction_samples(model, dataset, device, save_dir, num=5):
    os.makedirs(save_dir, exist_ok=True)
    idxs = np.random.choice(len(dataset), min(num, len(dataset)), replace=False)
    model.eval()
    with torch.no_grad():
        for k,i in enumerate(idxs):
            rgb, depth_gt = dataset[i]
            pred = model(rgb.unsqueeze(0).to(device)).squeeze().cpu().numpy()
            rgb_img = rgb.permute(1,2,0).numpy()
            gt = depth_gt.squeeze().numpy()
            fig, ax = plt.subplots(1,3, figsize=(12,4))
            ax[0].imshow(rgb_img); ax[0].set_title('RGB'); ax[0].axis('off')
            im1=ax[1].imshow(gt, cmap='plasma', vmin=0, vmax=0.5); ax[1].set_title('GT'); ax[1].axis('off'); plt.colorbar(im1, ax=ax[1], fraction=0.046,pad=0.04)
            im2=ax[2].imshow(pred, cmap='plasma', vmin=0, vmax=0.5); ax[2].set_title('Pred'); ax[2].axis('off'); plt.colorbar(im2, ax=ax[2], fraction=0.046,pad=0.04)
            plt.tight_layout()
            fig.savefig(os.path.join(save_dir, f'sample_{k:02d}.png'), dpi=120)
            plt.close(fig)

# ===================== 训练 ===================== #

def run_train(args):
    device = torch.device(args.device)
    # 自动发现中心线目录（默认全部）
    ds_dirs = list_centerline_dirs(args.data_dir, args.centerlines)
    print(f'使用中心线目录数量: {len(ds_dirs)}')
    if not ds_dirs:
        raise ValueError('未找到任何中心线数据目录')
    dataset = CenterlineDepthDataset(ds_dirs, image_size=args.image_size, max_samples_per_centerline=args.max_samples)
    test_size = int(len(dataset)*args.test_split)
    train_size = len(dataset) - test_size
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    model = UNetDepthEstimation().to(device)
    print(f'模型参数: {sum(p.numel() for p in model.parameters()):,}')
    criterion = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', patience=10, factor=0.5)
    os.makedirs(args.save_dir, exist_ok=True)
    # TensorBoard (简化：仅记录loss / lr)
    writer = None
    if args.tensorboard:
        if SummaryWriter is None:
            print('警告: 未安装 tensorboard，无法启用可视化')
        else:
            log_dir = args.log_dir or os.path.join('runs', 'unet_depth', datetime.now().strftime('%Y%m%d-%H%M%S'))
            writer = SummaryWriter(log_dir=log_dir)
            print(f'TensorBoard 日志目录: {log_dir}')
    best = float('inf')
    train_losses, val_losses = [], []
    global_step = 0
    for epoch in range(args.epochs):
        model.train(); tl=0; nb=0
        for rgb, depth in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} Train'):
            rgb, depth = rgb.to(device), depth.to(device)
            optim.zero_grad(); out = model(rgb); loss = criterion(out, depth); loss.backward(); optim.step()
            tl += loss.item(); nb += 1; global_step += 1
            if writer and args.tb_log_steps:
                writer.add_scalar('StepLoss/train', loss.item(), global_step)
        train_loss = tl/nb; train_losses.append(train_loss)
        model.eval(); vl=0; vb=0
        with torch.no_grad():
            for rgb, depth in tqdm(val_loader, desc='Val'):
                rgb, depth = rgb.to(device), depth.to(device)
                out = model(rgb); loss = criterion(out, depth)
                vl += loss.item(); vb += 1
        val_loss = vl/vb; val_losses.append(val_loss); sched.step(val_loss)
        lr_now = optim.param_groups[0]['lr']
        print(f'Epoch {epoch+1}: train {train_loss:.6f}  val {val_loss:.6f}  lr {lr_now:.2e}')
        if writer:
            writer.add_scalar('EpochLoss/train', train_loss, epoch+1)
            writer.add_scalar('EpochLoss/val', val_loss, epoch+1)
            writer.add_scalar('LR', lr_now, epoch+1)
            writer.flush()
        if val_loss < best:
            best = val_loss
            torch.save({'epoch': epoch,'model_state_dict': model.state_dict(),'val_loss': val_loss}, os.path.join(args.save_dir,'best_model.pth'))
            print(f'  * 新最佳模型 (val={val_loss:.6f}) 已保存')
        if (epoch+1) % 20 == 0:
            torch.save({'epoch': epoch,'model_state_dict': model.state_dict(),'val_loss': val_loss}, os.path.join(args.save_dir,f'checkpoint_epoch_{epoch+1}.pth'))
    if writer:
        writer.close()
    torch.save({'model_state_dict': model.state_dict(),'train_losses': train_losses,'val_losses': val_losses,'args': vars(args)}, os.path.join(args.save_dir,'final_model.pth'))
    plt.figure(figsize=(8,5)); plt.plot(train_losses,label='train'); plt.plot(val_losses,label='val'); plt.legend(); plt.grid(True); plt.xlabel('epoch'); plt.ylabel('MSE'); plt.tight_layout(); plt.savefig(os.path.join(args.save_dir,'training_curves.png'), dpi=130); plt.close()
    save_prediction_samples(model, test_ds, device, os.path.join(args.save_dir,'prediction_samples'))
    print(f'训练完成 最佳val={best:.6f}')
    return os.path.join(args.save_dir,'best_model.pth')

# ===================== 推理 ===================== #

def load_model(model_path, device):
    model = UNetDepthEstimation().to(device)
    ckpt = torch.load(model_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)
    model.eval()
    print(f'加载模型: {model_path} (val={ckpt.get("val_loss", "?")})')
    return model

def preprocess_image(path, image_size):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f'无法读取图像: {path}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size))
    img = img.astype(np.float32)/255.0
    return torch.from_numpy(img).permute(2,0,1)

def postprocess_depth(t):
    d = t.squeeze().cpu().numpy(); d = np.clip(d,0,0.5); return d

def save_depth(depth, path):
    uint8 = (depth/0.5*255).astype(np.uint8); cv2.imwrite(path, uint8)

def create_viz(rgb, pred, gt=None):
    if gt is not None:
        fig,ax=plt.subplots(1,3,figsize=(12,4))
        ax[0].imshow(rgb); ax[0].set_title('RGB'); ax[0].axis('off')
        im1=ax[1].imshow(gt,cmap='plasma',vmin=0,vmax=0.5); ax[1].set_title('GT'); ax[1].axis('off'); plt.colorbar(im1, ax=ax[1],fraction=0.046,pad=0.04)
        im2=ax[2].imshow(pred,cmap='plasma',vmin=0,vmax=0.5); ax[2].set_title('Pred'); ax[2].axis('off'); plt.colorbar(im2, ax=ax[2],fraction=0.046,pad=0.04)
    else:
        fig,ax=plt.subplots(1,2,figsize=(8,4))
        ax[0].imshow(rgb); ax[0].set_title('RGB'); ax[0].axis('off')
        im=ax[1].imshow(pred,cmap='plasma',vmin=0,vmax=0.5); ax[1].set_title('Pred'); ax[1].axis('off'); plt.colorbar(im, ax=ax[1],fraction=0.046,pad=0.04)
    plt.tight_layout(); return fig

def run_infer(args):
    device = torch.device(args.device)
    model = load_model(args.model_path, device)
    if args.input_format == 'centerline':
        rgb_dir = os.path.join(args.input_dir, 'rgb_images')
        if not os.path.isdir(rgb_dir):
            raise ValueError(f'rgb_images 不存在: {rgb_dir}')
        image_paths = [os.path.join(rgb_dir,f) for f in sorted(os.listdir(rgb_dir)) if f.lower().endswith(('.jpg','.png','.jpeg'))]
        depth_out_dir = os.path.join(args.output_dir, 'depth_images')
    else:
        image_paths = [os.path.join(args.input_dir,f) for f in sorted(os.listdir(args.input_dir)) if f.lower().endswith(('.jpg','.png','.jpeg'))]
        depth_out_dir = args.output_dir
    os.makedirs(depth_out_dir, exist_ok=True)
    viz_dir = None
    if args.save_visualization:
        viz_dir = os.path.join(args.output_dir,'visualizations'); os.makedirs(viz_dir, exist_ok=True)
    for i in tqdm(range(0,len(image_paths), args.batch_size), desc='Infer'):
        batch_paths = image_paths[i:i+args.batch_size]
        batch = [preprocess_image(p, args.image_size) for p in batch_paths]
        batch_tensor = torch.stack(batch).to(device)
        with torch.no_grad():
            preds = model(batch_tensor)
        for p_img, pred_t in zip(batch_paths, preds):
            depth = postprocess_depth(pred_t)
            base = os.path.splitext(os.path.basename(p_img))[0]
            if args.input_format=='centerline' and base.startswith('rgb_'):
                out_name = base.replace('rgb_','depth_') + '.jpg'
            else:
                out_name = base + '_depth.png'
            save_depth(depth, os.path.join(depth_out_dir,out_name))
            if viz_dir:
                rgb_orig = cv2.imread(p_img); rgb_orig = cv2.cvtColor(rgb_orig, cv2.COLOR_BGR2RGB); rgb_orig = cv2.resize(rgb_orig,(args.image_size,args.image_size))
                gt=None
                if args.input_format=='centerline':
                    gt_dir = os.path.join(os.path.dirname(args.input_dir), 'depth_images')
                    gt_path = os.path.join(gt_dir, out_name)
                    if os.path.exists(gt_path):
                        g=cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                        if g is not None:
                            g=cv2.resize(g,(args.image_size,args.image_size)); gt=g.astype(np.float32)/255.0*0.5
                fig = create_viz(rgb_orig, depth, gt)
                fig.savefig(os.path.join(viz_dir, base+'_viz.png'), dpi=130, bbox_inches='tight'); plt.close(fig)
    print('推理完成')

# ===================== 评估 ===================== #

def load_depth_img(path):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None: return None
    return im.astype(np.float32)/255.0*0.5

def calc_metrics(gt, pred, max_depth=0.5):
    mask = (gt>0)&(pred>0)&(gt<=max_depth)&(pred<=max_depth)
    if mask.sum()==0: return None
    g=gt[mask]; p=pred[mask]
    ae = np.abs(g-p); se=(g-p)**2
    metrics = {
        'mae': ae.mean(),
        'rmse': np.sqrt(se.mean()),
        'mse': se.mean(),
        'abs_rel': (ae/g).mean(),
        'sq_rel': (se/(g**2)).mean()
    }
    ratio = np.maximum(g/p, p/g)
    for th in [1.25, 1.25**2, 1.25**3]:
        metrics[f'delta_{str(th).replace(".", "_")[:5]}'] = (ratio < th).mean()
    return metrics

def eval_centerline(gt_root, pred_root, name):
    gt_dir = os.path.join(gt_root, name, 'depth_images')
    pr_dir = os.path.join(pred_root, name, 'depth_images')
    if not (os.path.isdir(gt_dir) and os.path.isdir(pr_dir)):
        return None
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.jpg')])
    pr_files = sorted([f for f in os.listdir(pr_dir) if f.endswith('.jpg')])
    common = sorted(set(gt_files)&set(pr_files))
    if not common: return None
    coll=[]
    for fn in common:
        gt = load_depth_img(os.path.join(gt_dir, fn))
        pr = load_depth_img(os.path.join(pr_dir, fn))
        if gt is None or pr is None: continue
        if gt.shape!=pr.shape:
            pr = cv2.resize(pr, (gt.shape[1], gt.shape[0]))
        m=calc_metrics(gt, pr)
        if m: coll.append(m)
    if not coll: return None
    avg={}
    keys = coll[0].keys()
    for k in keys:
        vals=[c[k] for c in coll if k in c]
        avg[k]=float(np.mean(vals))
        avg[k+"_std"]=float(np.std(vals))
    avg['num_images']=len(coll); avg['centerline']=name
    return avg, coll

def run_eval(args):
    if not os.path.isdir(args.gt_dir): raise FileNotFoundError(args.gt_dir)
    if not os.path.isdir(args.pred_dir): raise FileNotFoundError(args.pred_dir)
    gt_cls=[d for d in os.listdir(args.gt_dir) if d.startswith('centerline_') and os.path.isdir(os.path.join(args.gt_dir,d))]
    pr_cls=[d for d in os.listdir(args.pred_dir) if d.startswith('centerline_') and os.path.isdir(os.path.join(args.pred_dir,d))]
    common=sorted(set(gt_cls)&set(pr_cls))
    if not common: raise ValueError('无公共中心线用于评估')
    results={}
    for name in common:
        print('评估', name)
        r=eval_centerline(args.gt_dir, args.pred_dir, name)
        if r: results[name]=r
    if not results:
        print('无评估结果'); return
    os.makedirs(args.output_dir, exist_ok=True)
    report=os.path.join(args.output_dir,'evaluation_report.txt')
    with open(report,'w',encoding='utf-8') as f:
        f.write('UNet深度估计评估报告\n'+'='*60+'\n')
        overall={'mae':[],'rmse':[],'abs_rel':[]}
        for name,(avg,_all) in results.items():
            f.write(f'中心线: {name}\n')
            f.write(f'图像: {avg["num_images"]}\n')
            f.write(f"MAE: {avg['mae']:.6f}  RMSE: {avg['rmse']:.6f}  AbsRel: {avg['abs_rel']:.6f}\n")
            for k in avg.keys():
                if k.startswith('delta_') and not k.endswith('_std'):
                    f.write(f'{k}: {avg[k]:.4f}\n')
            f.write('-'*40+'\n')
            for k in overall.keys(): overall[k].append(avg[k])
        f.write('\n总体:\n')
        for k,vals in overall.items():
            f.write(f'{k.upper()} mean={np.mean(vals):.6f} std={np.std(vals):.6f}\n')
    # 简单可视化
    names=list(results.keys()); mae=[results[n][0]['mae'] for n in names]; rmse=[results[n][0]['rmse'] for n in names]
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1); plt.bar(range(len(names)), mae); plt.title('MAE'); plt.xticks(range(len(names)),[f'CL{i}' for i in range(len(names))], rotation=45)
    plt.subplot(1,2,2); plt.bar(range(len(names)), rmse, color='salmon'); plt.title('RMSE'); plt.xticks(range(len(names)),[f'CL{i}' for i in range(len(names))], rotation=45)
    plt.tight_layout(); plt.savefig(os.path.join(args.output_dir,'evaluation_metrics.png'), dpi=140); plt.close()
    print('评估完成, 报告:', report)

# ===================== 参数解析 ===================== #

def build_parser():
    p = argparse.ArgumentParser(description='统一UNet深度估计脚本')
    p.add_argument('--mode', choices=['train','infer','eval','all'], required=True, help='运行模式')
    # 通用
    p.add_argument('--image-size', type=int, default=200)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    # 训练相关
    p.add_argument('--data-dir', default='all_centerline_datasets')
    p.add_argument('--centerlines', nargs='+', default=None, help='中心线前缀列表; 省略=全部; 传 ALL 亦表示全部')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--test-split', type=float, default=0.2)
    p.add_argument('--max-samples', type=int, default=None)
    p.add_argument('--save-dir', default='unet_depth_checkpoints')
    p.add_argument('--tensorboard', action='store_true', help='启用TensorBoard日志')
    p.add_argument('--log-dir', default=None, help='TensorBoard日志目录 (默认 runs/unet_depth/时间戳)')
    p.add_argument('--tb-log-steps', action='store_true', help='记录每个训练 step 的 loss')
    # 推理相关
    p.add_argument('--model-path')
    p.add_argument('--input-dir')
    p.add_argument('--output-dir', default='unet_predictions')
    p.add_argument('--input-format', choices=['centerline','single'], default='centerline')
    p.add_argument('--save-visualization', action='store-true' if False else 'store_true')
    # 评估相关
    p.add_argument('--gt-dir')
    p.add_argument('--pred-dir')
    # all 模式附加
    p.add_argument('--infer-centerline', help='all模式指定推理的中心线目录(完整名称)')
    return p

# ===================== 主入口 ===================== #

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == 'train':
        run_train(args)
    elif args.mode == 'infer':
        # 校验必要参数
        if not args.model_path:
            parser.error('--model-path 必须提供 (infer)')
        if not args.input_dir:
            parser.error('--input-dir 必须提供 (infer)')
        run_infer(args)
    elif args.mode == 'eval':
        if not args.gt_dir or not args.pred_dir:
            parser.error('--gt-dir 与 --pred-dir 必须提供 (eval)')
        run_eval(args)
    elif args.mode == 'all':
        # 1. 训练
        best_model = run_train(args)
        # 2. 推理 (使用指定中心线或第一条)
        infer_dir = args.infer_centerline
        if not infer_dir:
            # 自动取第一条匹配中心线
            ds_dirs = list_centerline_dirs(args.data_dir, args.centerlines)
            if not ds_dirs:
                raise ValueError('all模式未找到中心线目录用于推理')
            infer_dir = ds_dirs[0]
        infer_args = argparse.Namespace(**{**vars(args), **{
            'model_path': best_model,
            'input_dir': infer_dir,
            'output_dir': args.output_dir,
            'mode': 'infer'
        }})
        run_infer(infer_args)
        # 3. 评估 (需要预测结果结构符合centerline_*/depth_images)
        if not args.gt_dir:
            print('未提供 --gt-dir, 跳过评估')
        else:
            eval_args = argparse.Namespace(**{**vars(args), **{
                'gt_dir': args.gt_dir,
                'pred_dir': args.pred_dir or args.output_dir,
                'output_dir': 'evaluation_results',
                'mode': 'eval'
            }})
            run_eval(eval_args)
    else:
        parser.error('未知模式')

if __name__ == '__main__':
    main()
