# ME139_Project

通用的 Unitree G1 / Isaac Lab 仿真与动作跟踪项目底座。`main` 提供环境安装、依赖管理、物理验证、动作格式转换、策略训练和回放入口，具体任务可在此基础上扩展。

## 目录

```text
project.sh                    统一命令入口
configs/                      Python 依赖约束与版本锁定
scripts/                      安装、检查、仿真、训练和回放
tests/                        无需 GPU 的入口回归测试
data/                         本地动作数据（仅说明文件入库）
manifests/                    本地验证记录（仅说明文件入库）
third_party/IsaacLab/          Isaac Lab 固定版本子模块
third_party/whole_body_tracking/  G1 动作跟踪固定版本子模块
```

## 环境要求

- Linux、支持 CUDA 的 NVIDIA GPU，以及可运行 Isaac Sim 的驱动。
- 已安装 Isaac Sim `5.1.0.0` 的 Python `3.11` 环境。
- 基础环境中的 PyTorch 为 `2.7.0+cu128`；项目使用 NumPy `1.26.0`、RSL-RL `3.1.2`。
- 安装脚本在项目内创建 `.venv`，复用基础环境的模拟器和 CUDA 软件包，额外依赖安装到本地 `.venv`。

| 依赖 | 固定提交 |
| --- | --- |
| [Isaac Lab 2.3.2](https://github.com/isaac-sim/IsaacLab) | `37ddf626871758333d6ed89cf64ad702aef127d0` |
| [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking) | `cd65172032893724b445448818c34165846d847d` |

## 安装

```bash
git clone --recurse-submodules https://github.com/yangfeiyang-123/ME139_Project.git
cd ME139_Project

ISAAC_BASE_PYTHON=/absolute/path/to/isaac/python ./project.sh setup
./project.sh doctor
```

将 `ISAAC_BASE_PYTHON` 替换为已有 Isaac Sim 环境的 Python 路径。未指定时使用当前 `PATH` 中的 `python3`，版本不匹配时会停止并给出提示。普通克隆遗漏的子模块会在安装时初始化；已有子模块版本不匹配时停止，不覆盖本地修改。机器人资源由安装脚本下载并校验。

`./project.sh help` 无需安装模拟器即可查看命令。所有入口命令均以仓库根目录为工作目录，文件参数中的相对路径也以该目录为准。

## 验证基础流程

以下流程使用模拟器生成的默认站姿参考，无需外部数据集：

```bash
# 验证 G1 默认站姿物理稳定性并生成静态参考
./project.sh smoke --headless --device cuda:0

# 使用静态参考验证训练流程
./project.sh train --headless --device cuda:0 \
  --motion data/motions/g1_stand_smoke.npz \
  --num_envs 16 --max_iterations 2 --run_name smoke

# 回放参考动作
./project.sh replay --headless --device cuda:0 \
  --motion data/motions/g1_stand_smoke.npz --steps 100
```

仿真验证结果写入 `manifests/g1-smoke.json`；训练配置、日志、检查点和 `result.json` 写入 `logs/rsl_rl/`。静态参考和短时训练用于验证流程，不代表已学习有效技能。

回放训练出的策略时，替换为实际生成的检查点路径：

```bash
./project.sh play --headless --device cuda:0 \
  --motion data/motions/g1_stand_smoke.npz \
  --checkpoint logs/rsl_rl/g1_flat/<run>/model_final.pt --steps 100
```

## 动作转换与远程回放

将符合格式的本地 CSV 转换为参考 NPZ，格式见 [data/README.md](data/README.md)：

```bash
./project.sh convert --headless --device cuda:0 \
  --input_file data/motions/example.csv --input_fps 30 \
  --output_file data/motions/example.npz --output_fps 50
```

通过 Isaac Sim WebRTC 客户端回放时，显式设置可访问的服务地址，并确保客户端能访问相应端口：

```bash
ISAAC_STREAM_HOST=<server-ip> ./project.sh replay-remote \
  --motion data/motions/example.npz
```

可用 `./project.sh python <script-or-options>` 在项目环境中运行其他 Python 命令。

## 开发检查

以下检查不依赖 GPU 或 Isaac Sim，GitHub Actions 也会执行这些检查：

```bash
bash -n project.sh
for script in scripts/*.sh; do bash -n "$script" || exit; done
python3 -m compileall -q scripts tests
python3 -m unittest discover -s tests -v
```

GPU 仿真、训练与远程串流需在安装了相应运行环境的机器上单独验证。基础脚本保留了当前 Kit 环境的进程退出处理，见 `scripts/runtime.py`。

## 本地文件与许可证

代码、配置、文档和测试进入版本控制；数据、`.venv`、缓存、训练结果、视频和机器验证记录由 `.gitignore` 排除。仓库不预置历史运行成功记录。

第三方代码遵循各子模块中的许可证；外部数据按来源自行获取并遵守其许可。
