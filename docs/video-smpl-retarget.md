# Video → WHAM / SMPL → G1 retarget

输入可以是行走、挥手、下蹲或其他动作视频，代码没有动作类别或数据目录的硬编码。流程为：WHAM 视频推理 → 世界坐标 SMPL 骨架 → G1 运动学重定向 → 同步查看与仿真参考。

固定 WHAM 提交：`2b54f7797391c94876848b905ed875b154c4a295`。配置依据该提交的 [官方安装说明](https://github.com/yohanshin/WHAM/blob/2b54f7797391c94876848b905ed875b154c4a295/docs/INSTALL.md)，下载来源核对自 [官方资源脚本](https://github.com/yohanshin/WHAM/blob/2b54f7797391c94876848b905ed875b154c4a295/fetch_demo_data.sh)。

## 1. WHAM 独立环境

WHAM 使用 Python 3.9 / PyTorch 1.11 / CUDA 11.3；Isaac Sim 底座使用 Python 3.11 / PyTorch 2.7。两个环境分别安装，通过数值 NPZ / CSV 交换结果。

在项目根目录执行：

```bash
git submodule update --init --recursive third_party/WHAM
conda env create --prefix "$PWD/.wham-env" --file configs/wham/environment.yml
conda activate "$PWD/.wham-env"
export WHAM_PYTHON="$CONDA_PREFIX/bin/python"
export ME139_ROOT="$PWD"

python -m pip install -c "$ME139_ROOT/configs/wham/constraints.txt" \
  -r "$ME139_ROOT/third_party/WHAM/requirements.txt"
python -m pip install -c "$ME139_ROOT/configs/wham/constraints.txt" \
  -v -e "$ME139_ROOT/third_party/WHAM/third-party/ViTPose"
```

世界坐标推理还需要 DPVO 的 CUDA 扩展。在上述 WHAM 环境中按上游方式安装：

```bash
conda install -c rusty1s pytorch-scatter=2.0.9
conda install -c conda-forge cudatoolkit-dev=11.3.1
# 如果系统 GCC > 10，使用与该 CUDA 工具链匹配的编译器：
conda install -c conda-forge gxx=9.5

cd "$ME139_ROOT/third_party/WHAM/third-party/DPVO"
curl -fL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip -o eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d thirdparty
python -m pip install -c "$ME139_ROOT/configs/wham/constraints.txt" .
cd "$ME139_ROOT"
```

只有 WHAM 自身的 `--visualize` 渲染需要 PyTorch3D；本项目的网页骨架预览不依赖它：

```bash
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
python -m pip install pytorch3d \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1110/download.html
```

本项目不自动创建 WHAM 环境或下载检查点。`WHAM_PYTHON` 指向此独立环境的绝对 Python 路径。G1 重定向使用主项目 `.venv`，CPU 依赖由 `./project.sh setup` 安装，已有环境可补装：

```bash
./project.sh python -m pip install \
  -c configs/constraints.txt -r configs/requirements-motion.txt
```

## 2. 下载文件与放置路径

下表路径均相对于 **`third_party/WHAM/`**。下载来源为上游资源脚本；模型不纳入 Git。

| 文件 | 下载地址 | 放置路径 | 用途 |
| --- | --- | --- | --- |
| WHAM BEDLAM + 3DPW | [下载](https://drive.google.com/file/d/19qkI-a6xuwob9_RFNSPWf1yWErwVVlks/view) | `checkpoints/wham_vit_bedlam_w_3dpw.pth.tar` | 默认推理模型 |
| HMR2 | [下载](https://drive.google.com/file/d/1J6l8teyZrL0zFzHhzkC7efRhU0ZJ5G9Y/view) | `checkpoints/hmr2a.ckpt` | 图像特征 |
| ViTPose-H | [下载](https://drive.google.com/file/d/1xyF7F3I7lWtdq82xmEPVQ5zl4HaasBso/view) | `checkpoints/vitpose-h-multi-coco.pth` | 2D 关键点 |
| YOLOv8x | [下载](https://drive.google.com/file/d/1zJ0KP23tXD42D47cw1Gs7zE2BA_V_ERo/view) | `checkpoints/yolov8x.pt` | 人体检测 |
| DPVO | [下载](https://drive.google.com/file/d/1kXTV4EYb-BI3H7J-bkR3Bc4gT9zfnHGT/view) | `checkpoints/dpvo.pth` | 相机运动 / 世界坐标 |
| WHAM 3DPW | [下载](https://drive.google.com/file/d/1i7kt9RlCCCNEW2aYaDWVr-G778JkLNcB/view) | `checkpoints/wham_vit_w_3dpw.pth.tar` | 可选替代，默认不使用 |
| SMPL 辅助资源包 | [下载](https://drive.google.com/file/d/1pbmzRbWGgae6noDIyQOnohzaVnX_csUZ/view) | 将 `body_models.tar.gz` 解压到 `dataset/` | 回归器、均值参数等 |

SMPL 模型需要登录模型站点获取：

| 模型 | 来源与原文件 | 最终路径 |
| --- | --- | --- |
| Neutral（必需） | [SMPLify](https://smplify.is.tue.mpg.de/) 的 `mpips_smplify_public_v2.zip` 中 `smplify_public/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` | `dataset/body_models/smpl/SMPL_NEUTRAL.pkl` |
| Female（可选） | [SMPL](https://smpl.is.tue.mpg.de/) 的 `SMPL_python_v.1.0.0.zip` 中 `smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl` | `dataset/body_models/smpl/SMPL_FEMALE.pkl` |
| Male（可选） | 同上压缩包中的 `smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl` | `dataset/body_models/smpl/SMPL_MALE.pkl` |

辅助包应解压出 `dataset/body_models/J_regressor_wham.npy`、`J_regressor_h36m.npy`、`J_regressor_feet.npy`、`smpl_mean_params.npz` 等文件。`configs/wham/assets.json` 保存了可机读的文件路径和下载来源。

无需先安装 WHAM 即可查看缺哪些文件：

```bash
./project.sh wham check
```

检查文件是否存在且非空；缺文件时返回非零退出码并列出下载链接，不触发下载。模型站点的许可条款适用于对应资源。

## 3. 视频 → WHAM

```bash
export WHAM_PYTHON="/absolute/path/to/ME139_Project/.wham-env/bin/python"
./project.sh wham run \
  --video data/videos/motion.mp4 \
  --output outputs/wham/run_01
```

输出为 `outputs/wham/run_01/motion/wham_output.pkl` 和 `run.json`，后者记录输入视频哈希、运行参数、版本和坐标模式。多人视频保留各自 track ID。

默认必须能加载 DPVO，缺失时停止，不静默改成相机局部坐标。可用 `--calib data/calibration.txt` 提供内容为 `fx fy cx cy` 的相机内参。`--local-only` 仅用于相机坐标查看，不进入世界坐标重定向流程。

输出目录不复用旧追踪缓存。更换视频、内参或模式时指定新的 `--output`。

## 4. WHAM → 标准 SMPL NPZ

```bash
./project.sh wham export \
  --input outputs/wham/run_01/motion/wham_output.pkl \
  --video data/videos/motion.mp4 \
  --person-id 0 \
  --output outputs/smpl/motion.npz \
  --vertices
```

单人结果可省略 `--person-id`；多人必须选择。`--vertices` 输出可选网格，重定向只需要关节。

导出器读取 `pose_world`、`trans_world`、`betas`，用 Neutral SMPL 重建关节，并按 WHAM 关节回归器校正骨盆原点，再由 WHAM 的 Y-up 转成右手 Z-up（`x'=x, y'=-z, z'=y`）。不直接使用上游相机坐标 `verts`。

| 字段 | 格式 |
| --- | --- |
| `joints_world` | `(F,24,3)`，SMPL 24 关节，米，右手 Z-up |
| `frame_ids` | `(F,)`，原视频中从 0 开始的严格递增整数帧号 |
| `fps` | 原视频帧率 |
| `coordinate_system` | 字符串 `world_z_up` |
| `vertices_world` | 可选 `(F,V,3)` |
| `person_id`、`source_video` | 来源信息 |

其他估计器适配为上述数值 NPZ，也能使用后续模块。摄像机局部结果不能仅修改标签后当作世界坐标。

## 5. SMPL → G1

```bash
./project.sh retarget \
  --input outputs/smpl/motion.npz \
  --output outputs/retarget/motion \
  --config configs/retarget_g1.json \
  --output-fps 30
```

输出包括：

- `motion.csv`：均匀采样的根部 XYZ + 四元数 XYZW + 29 个关节角，与 `convert` 一致。
- `preview.npz`：与 SMPL 原始帧号对齐的机器人骨架。
- `report.json`：尺度、目标误差、收敛帧数、限位检查和关节顺序。

默认由腿长估计统一尺度，可用 `--scale` 覆盖。配置文件定义人体关节与 G1 link 的映射、权重、姿态及相邻帧正则项。长于 0.25 秒的追踪间断会被拒绝，应优先切分片段，或明确设置 `--max-gap`。重采样使用原始帧号时间轴，根部姿态使用四元数插值。

当前为位置 IK 基线：关节角受 URDF 限位约束，但没有动态平衡、碰撞、脚底接触和速度约束，也不保证手足方向。检查误差和预览后，再作为仿真或训练参考。优化器收敛不等于动作质量合格。

## 6. 同步查看与接入仿真

```bash
./project.sh view-smpl \
  --input outputs/smpl/motion.npz \
  --video data/videos/motion.mp4 \
  --robot outputs/retarget/motion/preview.npz \
  --output outputs/viewers/motion

./project.sh convert --headless --device cuda:0 \
  --input_file outputs/retarget/motion/motion.csv --input_fps 30 \
  --output_file data/motions/motion.npz --output_fps 50

./project.sh replay --headless --device cuda:0 \
  --motion data/motions/motion.npz
```

打开 `outputs/viewers/motion/index.html` 查看原视频、人体和机器人。网页不依赖外部 CDN，按视频时间与原始帧号同步，可暂停、逐帧拖动和旋转视角。视频通过相对路径引用，移动预览目录时应保持相对位置。

测试覆盖数据格式、坐标转换、人物选择、时间对齐、IK 和命令包装。未提供权重时，不能完成真实 WHAM 视频推理验证。
