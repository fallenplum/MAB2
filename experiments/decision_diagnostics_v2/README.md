# 已有 G2 输出的决策诊断

此目录提供 CPU 探索诊断。原 G2 120 题已知结果，本目录输出不承担独立确认职责。

纯函数分别实现单步动作、符号、排序、不确定性代理诊断。适配器复用原评分与封存检查，模型参数保持既有版本。数据、标签及私有执行证据由本地项目根目录提供。

运行软件回归可使用以下命令。

```powershell
python -m unittest discover -s experiments/decision_diagnostics_v2 -v
```

在具有原 G2 完整本地归档的项目上，固定新配方后运行。

```powershell
python experiments/decision_diagnostics_v2/run_existing_g2_decisions.py --root "E:\图结构演化" --mode freeze
python experiments/decision_diagnostics_v2/run_existing_g2_decisions.py --root "E:\图结构演化" --mode run
python experiments/decision_diagnostics_v2/replay_existing_g2_decisions.py --root "E:\图结构演化"
```

结果采用单次发布保护，已有输出或中途失败应先检查现场与访问记录。适配器要求 NumPy、SciPy 以及原 G2 的冻结来源与辅助发布器。

v2 在读取特征时绑定实际 NPZ 字节哈希，并使用已校验的同一缓冲区解码。v1 结果、程序和访问记录保存在本地历史目录，两版诊断数值已对齐。

质量诊断使用 λ=0、零阈值、一次切换或 stay，均匀随机动作参照覆盖全部 13 个动作。主动复检和在线图编辑需要新的独立机制协议。参见 `protocol/revisions/DECISION_UTILITY_RESEARCH_REVISION_20261105.md`。
