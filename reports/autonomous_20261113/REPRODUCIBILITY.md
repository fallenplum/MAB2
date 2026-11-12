# 复现与数据边界

本包包含两个不同层次：公开聚合数据可重绘图表；完整数值复算需要项目所有者保留的已验收 G2 派生数组。公开仓库本身不包含原题、答案、逐题预测、模型权重或服务器凭据，不声称仅凭公开包即可重新生成全部推理数据。

## 环境与代码

Python 3.11，NumPy、SciPy、threadpoolctl 用于诊断；matplotlib 用于报告图。实际版本记录在每次运行的 STARTED.json 和图表 validation.json；复现优先匹配该记录。正式计算限制为 4 个 BLAS 线程。历史源码保持原始字节；新增 Python/协议使用 LF。

在仓库根目录可以执行纯合成测试（无需历史数据或 GPU）：

```powershell
python -m unittest discover -s experiments/d2_diagnostics -p 'test_*.py'
python -m unittest discover -s experiments/d3_diagnostics -p 'test_*.py'
```

已有运行的协议、实际来源 commit、输入哈希及完成清单构成复算依据：

| 运行 | 私有结果相对项目根目录 | 状态 |
|---|---|---|
| D2 | results/autonomous_20261113/d2_20261113/28fb250d1c46 | 完成；独立复核；提名失败 |
| D3 | results/autonomous_20261113/d3_20261113/4ac9c44a72db | 完成；独立复核；STOP 当前候选 |

两者输入数组绑定为 `7639dad2581ea52536b7c274d33fd019ad726a9bce9e8e4ec6792c68b77f4963`。D2 结果 SHA256 为 `242024766bd77215f6e3175c6a3d22f6d01d1a5b6cddec16be4da9b348e80406`。D3 后续来源与结果哈希见聚合报告和私有 COMPLETE.json。不要编辑接受的运行目录。

需要重跑时，先准备单独的项目副本及匹配哈希的历史输入，再调用下面接口；该说明不是停止后自动新增候选或重复看开发数据的指令。

```powershell
python experiments/d2_diagnostics/run_d2.py --root <private-project-copy> --repo <checkout>
python experiments/d3_diagnostics/run_d3.py --root <private-project-copy> --repo <checkout>
```

运行器检查协议与 Git HEAD 的字节一致性，保留失败尝试，先冻结参数/选边再评分，创建唯一结果目录。D3 明确绑定已接受的 D2 原始工件；重新生成的压缩 NPZ 即使数值相同也可能字节不同，不能静默替换哈希。复现比较应报告数值误差并保留原绑定；修改输入协议需要单独的复现修订记录。

公开重绘只需 `d2/d2_evidence.json` 与同目录的 `render_d2_report.py`。其 CLI 以 `--help` 为准，写入新的输出目录，避免覆盖已封存报告。D3 的聚合结果和重绘接口见其目录。图表重绘属于派生展示，不是新的模型实验。

## 统计和成本约定

旧 train200/tune80/test120 已全部成为开发材料；这里的 exposed120 不是最终独立测试集。每个题目的两个图、12 条边和5个生成种子保持成块，bootstrap 重采样单位为题目。选定 α 只用旧 train/tune，未根据 exposed 结果替补候选。

跨种子信用风险和平方漂移估计允许负值。95% 区间用于探索性描述或投入筛查，不代表方法成功概率，也不构成等效性证明。LOSO 五个折共享训练信息，不当作五次独立实验。

成本账本区分新增 GPU 占用、历史 CUDA 事件时间、实际调用/token、CPU主进程和未知开销。未知值保留 null；本轮 GPU=0 不表示使用历史数据没有成本。没有运行完整预算下的在线机制比较，因此不提供虚构的系统净收益表。

## 审计范围

内部第二实现复算是对算术与协议对应的独立检查，不是新数据复现，也不是正式期刊审稿。技能来源版本、源码哈希、公开文件清单和隐私扫描记录随包保存。私有完整账本保留在项目 artifacts/private/autonomous_20261113 与分支 .research 下。
