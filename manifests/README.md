# 本地验证记录

此目录用于保存当前机器生成的验证记录，输出文件不加入版本控制。

- `./project.sh smoke --headless` 生成 `g1-smoke.json`，记录 G1 关节、刚体和默认站姿物理检查结果。
- 训练配置、检查点和 `result.json` 保存在 `logs/rsl_rl/` 下。
- 需要保存当前依赖快照时，可执行 `./project.sh python -m pip freeze > manifests/pip-freeze.txt`。

运行后应以本机生成的结果判断环境是否通过验证；仓库不预置成功记录。短时仿真与训练仅验证基础流程，不代表策略已收敛。
