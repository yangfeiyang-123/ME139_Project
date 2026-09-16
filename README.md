# ME139_Project

面向 Unitree G1 的人体动作处理、动作跟踪训练与羽毛球 SMPL 可视化项目，基于 Isaac Sim、Isaac Lab 和 BeyondMimic (`whole_body_tracking`)。

## 目录

- `scripts/`：环境安装与检查、动作转换、G1 验证、训练、回放及 SMPL 可视化脚本。
- `configs/`：Python 依赖约束与额外依赖版本锁定。
- `manifests/`：已有实验的环境信息、依赖清单与验证记录。
- `data/README.md`：本地验证数据的来源与格式说明。
- `third_party/`：固定版本的第三方 Git 子模块。

## 获取代码

```bash
git clone --recurse-submodules https://github.com/yangfeiyang-123/ME139_Project.git
cd ME139_Project
```

已克隆仓库可执行 `git submodule update --init --recursive` 获取依赖源码。

| 依赖 | 固定提交 |
| --- | --- |
| [Isaac Lab](https://github.com/isaac-sim/IsaacLab) | `37ddf626871758333d6ed89cf64ad702aef127d0` |
| [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking) | `cd65172032893724b445448818c34165846d847d` |

## 环境与脚本

已有验证记录使用 Linux、Python 3.11、Isaac Sim 5.1.0.0、Isaac Lab 2.3.2、PyTorch 2.7.0+cu128 和 RSL-RL 3.1.2。具体版本见 `configs/` 和 `manifests/environment.json`。

`scripts/setup.sh` 在已有 Isaac Sim 安装上建立本地 `.venv`，通过 `ISAAC_BASE_PYTHON` 指定基础 Python 路径；脚本中的默认路径来自原开发机器。机器人资源由安装脚本下载并校验，不包含在仓库中。

当前源码快照未包含部分脚本引用的根目录 `badminton.sh` 包装入口；`scripts/setup.sh` 的最后一步也引用了该入口。环境配置完成后，可以直接调用已有的 Python 脚本，例如：

```bash
bash scripts/python.sh scripts/doctor.py
bash scripts/python.sh scripts/smoke_g1.py --headless
bash scripts/python.sh scripts/train.py \
  --motion data/motions/g1_walk_10s.npz --headless
bash scripts/python.sh scripts/visualize_smpl.py \
  --input data/ForehandClear_video_SMPL_pair \
  --output data/ForehandClear_visualization
```

上述命令需要对应的本地环境、机器人资源及动作数据。`manifests/` 为历史实验记录，保留了原机器路径；短时训练记录仅验证流程，不代表已收敛的行走或羽毛球技能。

## 数据与生成文件

公开仓库仅包含代码、配置、说明与小型实验记录。原始视频、SMPL 数据、生成的可视化、`data/motions/`、训练日志、模型检查点和 `outputs/` 均由 `.gitignore` 排除，保留在本地。

验证数据的来源、格式和许可说明见 [data/README.md](data/README.md)。第三方代码遵循各自仓库中的许可证。
