# 中心线数据集生成器使用说明

## 功能描述
这个脚本可以沿着中心线轨迹的每个点生成RGB图像和深度图像，支持单个中心线和批量处理模式。

## 使用方法

### 单个中心线模式（默认）
```bash
# 生成默认中心线数据集
python generate_centerline_dataset.py

# 生成特定中心线数据集
python generate_centerline_dataset.py \
    --centerline-name "siliconmodel3 Centerline model_1" \
    --output-dir "my_dataset"
```

### 批量处理模式
```bash
# 生成所有60条中心线的数据集
python generate_centerline_dataset.py --batch

# 处理前20条中心线 (0-19)
python generate_centerline_dataset.py --batch --start-index 0 --end-index 19

# 处理中间20条中心线 (20-39)  
python generate_centerline_dataset.py --batch --start-index 20 --end-index 39

# 处理最后20条中心线 (40-59)
python generate_centerline_dataset.py --batch --start-index 40 --end-index 59

# 从特定位置继续（如果中断了）
python generate_centerline_dataset.py --batch --start-index 30
```

## 参数说明

### 基础参数
| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--dataset-dir` | "train_set" | 数据集根目录路径 |
| `--output-dir` | "centerline_dataset" | 输出数据集目录（单个模式） |
| `--centerline-name` | "siliconmodel3 Centerline model" | 要使用的中心线名称（单个模式） |
| `--renderer` | "pyrender" | 渲染器类型 (pyrender/pybullet) |
| `--image-size` | 200 | 输出图像尺寸 (正方形) |
| `--step-size` | 1 | 中心线采样步长 |

### 批量处理参数
| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--batch` | False | 启用批量处理模式 |
| `--start-index` | 0 | 开始处理的中心线索引 |
| `--end-index` | 59 | 结束处理的中心线索引 |
| `--output-base-dir` | "all_centerline_datasets" | 批量模式的基础输出目录 |

## 输出结构

### 单个中心线模式
```
centerline_dataset/
├── rgb_images/           # RGB图像
│   ├── rgb_000000.jpg
│   ├── rgb_000001.jpg
│   └── ...
├── depth_images/         # 深度图像
│   ├── depth_000000.jpg
│   ├── depth_000001.jpg
│   └── ...
└── camera_info/          # 相机信息
    └── dataset_info.txt  # 包含位置、姿态等信息
```

### 批量处理模式
```
all_centerline_datasets/
├── centerline_00_siliconmodel3_Centerline_model/
│   ├── rgb_images/
│   ├── depth_images/
│   └── camera_info/
├── centerline_01_siliconmodel3_Centerline_model_1/
│   ├── rgb_images/
│   ├── depth_images/
│   └── camera_info/
├── centerline_02_siliconmodel3_Centerline_model_2/
│   └── ...
├── ...
├── centerline_59_siliconmodel3_Centerline_model_59/
└── generation_log.txt  # 处理日志
```

## 数据说明

### RGB图像
- 格式：JPG
- 尺寸：可配置 (默认200x200)
- 颜色空间：BGR (OpenCV格式)
- 自动光照调节，目标亮度140

### 深度图像
- 格式：JPG (灰度图)
- 尺寸：与RGB图像相同
- 深度范围：0-0.5米，归一化到0-255
- 超过0.5米的深度被截断

### 相机信息文件
包含以下信息：
- 索引
- 3D位置坐标 (x, y, z)
- 相机姿态 (pitch, yaw角度)
- 到起始点的累计距离

## 示例用法

### 1. 生成完整数据集
```bash
python generate_centerline_dataset.py
```

### 2. 生成特定中心线的数据集
```bash
python generate_centerline_dataset.py \
    --centerline-name "siliconmodel3 Centerline model_10"
```

### 3. 降低采样密度
```bash
python generate_centerline_dataset.py \
    --step-size 5 \
    --output-dir "sparse_dataset"
```

### 4. 生成高分辨率图像
```bash
python generate_centerline_dataset.py \
    --image-size 400 \
    --output-dir "high_res_dataset"
```

## 注意事项

1. **内存使用**：生成高分辨率图像或处理长中心线时可能需要大量内存
2. **渲染速度**：pyrender渲染质量更高但速度较慢，pybullet速度更快
3. **磁盘空间**：确保有足够的磁盘空间存储生成的图像
4. **GPU加速**：建议使用GPU加速渲染过程

## 错误处理

如果遇到以下问题：
- 内存不足：减少图像尺寸或增加步长
- 渲染错误：检查3D模型文件是否存在
- 路径错误：确保输入和输出目录路径正确
