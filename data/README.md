# 验证动作数据

`motions/` 为本地数据，不加入版本控制。

- `walk1_subject1.csv`：BeyondMimic 官方 README 指向的 [LAFAN1 Retargeting Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)，路径 `g1/walk1_subject1.csv`。
- `g1_walk_10s.npz`：原始 CSV 的第 1–301 帧，经本项目转换器处理为 500 帧、50 Hz，使用本机 G1 的实际关节/刚体顺序。
- `g1_stand_smoke.npz`：由模型默认站姿生成的静态测试输入，仅用于安装验证，不是人体运动数据或已训练技能。

CSV：36 列，根部 XYZ + 四元数 XYZW + 29 个关节角（弧度），30 Hz。
NPZ：`fps`、`joint_pos`、`joint_vel`、`body_pos_w`、`body_quat_w`、`body_lin_vel_w`、`body_ang_vel_w`。NPZ 四元数为 WXYZ，关节顺序为 Isaac Sim 导入后的顺序；不能直接把 CSV 的列当成 NPZ 的关节顺序。

数据集页面注明原始 LAFAN1 数据许可为 CC BY-NC-ND 4.0，代码许可与数据许可不同。此处仅下载到本地作环境验证，没有重新发布数据。原始/转换文件的 SHA256 记录在 `manifests/environment.json`。
