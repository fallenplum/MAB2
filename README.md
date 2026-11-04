# MAB2：图结构演化下的多小模型协同推理

检验并维护图演化后的条件边信用，在完整服务、反事实重执行和评分成本下评估预算化复检。

## 状态与证据边界

2026-11-04后续执行更新。G2已在8张A6000通过真实资格后启动正式采集，目标51,970个去重输出，固定200训练、80调参、120留出题和5个种子。当前阶段按冻结计划持续运行，结束后进行独立评分、预测和统计验收。[后续实验规划](protocol/NEXT_EXPERIMENT_PLAN_20261104.md)与[G2代码及测试](experiments/g2/README.md)列出完整合同和阶段门。

**2026-11-04：G1动机实验60000观测已完成，预注册统计门PASS；完整镜像及独立评分、统计、费用复核通过，历史首次标签访问缺持久时间戳，保留这一限制。主项目和ACL投稿准备尚未完成。**

主估计U=0.00133333333333，95%问题级bootstrap区间[0.000133333333333, 0.00263416666667]。这估计期望条件2×2交互的平方，不是准确率提升。信号稀疏且集中，所有背景、边对和负估计均保留。参见[阶段报告](results/G1_STAGE_REPORT_20261104.md)、[汇总JSON](results/G1_STAGE_SUMMARY_20261104.json)及[全部15组合](results/G1_DESCRIPTIVE_APPENDIX_20261104.md)。

当前执行v3。历史v1执行一致性通过但消息协议失败，初始G1停止并排除；v2消息门未通过而未启动正式G1；v3只强化汇总字段提示，真实S0执行/消息门及16题无标签资格通过后才启动本次G1。早期失败、费用和未知在途成本保留。修复历史见[记录](protocol/MESSAGE_PROTOCOL_REPAIR_V2.md)。

首阶段使用同一冻结Qwen2.5-3B-Instruct的4个固定角色与同骨干汇总器、2轮、12条可恢复有向候选边、128-token消息。3轮只作执行依赖资格及后续敏感性。完整覆盖200道开发用途题、5种子、5背景、3组边对的全部2×2单元；按整题重采样、跨独立种子估计平方交互，负值不截零。尚无异构模型、在线采集或等完整预算质量优势结论。

后续仍需G2图条件预测、G3同完整预算采集机制对照、G4至少两任务与5流种子、近邻基线、恢复/敏感性及封存测试。当前代码不把这些未执行阶段标记为完成。

## 主要入口

| 文件 | 作用 |
|---|---|
| `protocol/EXECUTION_CONTRACT.md` | 原方案合同、阶段门、信息权限与完整成本 |
| `protocol/ACL_TARGET.md` | 官方投稿时间、优势风险和交付边界 |
| `configs/qualification_v3.json` | 真实S0资格参数，明确非科研结果 |
| `configs/motivation_v3.json` | G1开发动机预声明参数 |
| `src/graph_credit/executor.py` | 多轮DAG、可恢复边、联合闭包及严格缓存 |
| `src/graph_credit/backend.py` | 冻结BF16真实推理、独立随机数和物理账本 |
| `src/graph_credit/scoring.py` | GSM8K显式最终答案Decimal评分，拒绝歧义 |
| `scripts/prepare_project.py` | 官方数据只读副本、题级去重、固定划分与方案来源 |
| `scripts/download_model.py` | 固定模型版本、断点续传与官方文件哈希校验 |
| `scripts/run_pilot.py` | 先冻结计划/生成全部输出，再读取诊断标签 |
| `scripts/run_message_sanity.py` | 16道保留资格题、无标签144次调用的消息合规检查 |
| `scripts/analyze_pilot.py` | 完整性检查、问题级统计及G1门 |
| `scripts/server_supervisor.py` | 共享GPU显存准入、S0通过后G1分片；失败不重试推理 |
| `scripts/local_coordinator.py` | 初始部署、传输与核验；当前只读镜像使用execution_tools/elastic/mirror_only.py |
| `tests/` | 执行语义、评分与统计反例软件检查 |

## 运行和查看

在项目虚拟环境安装所需依赖后，可用 `python -m unittest discover -s tests -v` 验证软件（需src在PYTHONPATH）。服务器使用项目隔离环境继承已安装的Torch，仅覆盖兼容HF依赖；实际版本见requirements-server.txt及每次run_manifest，不修改共享环境。

`run_pilot.py --help`和`analyze_pilot.py --help`给出参数。模型、无标签题目、标签文件、配置与设备均需显式提供。`--qualification-only --limit 1`执行真实2/3轮同输入同seed重跑、完整/闭包切换和空图资格。

本地 `COORDINATOR_STATUS.json` 显示下载/传输/同步，镜像 `RUN_STATUS.json` 显示服务器阶段。`results/pilot/*/QUALIFICATION.json`是S0执行证据；G1各分片有冻结计划、原始输出、真实调用、评分及状态。服务器最终仅运行到G1分析，后续方法需先核对该门的真实证据。

密码仅从运行时环境 `GRAPH_CREDIT_SSH_PASSWORD`读取，不写配置。SSH必须匹配已有known_hosts，未知主机或变化密钥拒绝连接。主控只管理本项目进程，不停止其他任务。

## 数据、费用与同步

GSM8K train去重后分配：资格16、动机200、校准400、在线流6857；官方test1319封存。主推理只上传开发题和当阶段诊断标签，官方test不上传到当前运行。

分别报告服务、复检、开发审计、评分、物理输入输出token、逻辑消息、CUDA事件时间和显存。共享GPU时间仅作运行观测；正式效率比较须固定设备、调度与竞争条件。稀疏边数不能代替真实算力或显存节省。

GitHub采用明确白名单，仅代码、配置、协议与汇总。数据、权重、逐题输出、日志、私有资源清单、服务器配置和凭据不上传；`.gitignore`是补充防线。服务器模型与结果由本地核验后传输/镜像。该仓库不包含数据和权重许可证的再授权。

v3固定输入10次真实探针全部合规，随后真实完整资格通过；这些工程检查不计入正式60000动机观测。


2026-11-04执行更新：真实v3 S0、16题无标签资格及G1已完成。按用户授权保持原3个逻辑分片和冻结协议，物理推理由3路交接为7路，第8卡通过资源准入、同种子真实资格及恢复资格后加入；最终8路均正常退出。调度源码见[execution_tools/elastic](execution_tools/elastic/README.md)，完整成本与边界见[执行补充合同](protocol/ELASTIC_EXECUTION_ADDENDUM.md)。本地镜像恢复未重启推理或重新采样。
