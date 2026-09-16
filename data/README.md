# 本地动作数据

此目录只提交说明文件。动作文件保存在 `data/motions/`，由使用者自行生成或获取。

## 无外部数据的验证输入

在环境安装完成后运行：

```bash
./project.sh smoke --headless --device cuda:0
```

该命令生成 `data/motions/g1_stand_smoke.npz`，包含模型默认站姿的静态参考，仅用于物理和训练流程验证。

## CSV 输入

`./project.sh convert` 接受无表头的数值 CSV，每帧 36 列：

- 根部位置 XYZ：3 列。
- 根部四元数 XYZW：4 列。
- G1 关节角：29 列，单位为弧度，顺序须匹配转换脚本的输入映射。

通过 `--input_fps` 指定原始采样率，通过 `--output_fps` 指定输出采样率。可使用 `--frame_range START END` 选择帧段，帧号从 1 开始且包含两端。

## NPZ 参考

NPZ 包含 `fps`、`joint_pos`、`joint_vel`、`body_pos_w`、`body_quat_w`、`body_lin_vel_w`、`body_ang_vel_w`。

NPZ 四元数使用 WXYZ；关节与刚体顺序须匹配 Isaac Sim 导入后的模型顺序，不能直接照搬 CSV 列顺序。默认训练环境需要 50 Hz、29 个关节的 G1 参考，转换脚本会执行顺序转换。

外部动作数据不随仓库分发，请按其来源和许可获取。
