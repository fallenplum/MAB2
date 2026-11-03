# MAB2：图结构演化下的多小模型协同推理

检验并维护图演化后的条件边信用，在完整服务、反事实重执行和评分成本下评估预算化复检。

## 状态与证据边界

当前执行v3。v1真实执行一致性通过但消息协议检查失败，初始G1已停止并排除。v2修复数值JSON解析、短消息及失败消息表示，并新增消息合规门和16题无标签资格检查；参见[修复记录](protocol/MESSAGE_PROTOCOL_REPAIR_V2.md)。软件检查现为64项通过，真实v3资格仍以产物为准。

2026-11-03：已完整审阅研究方案，建立独立数据切分、执行合同和S0→G1代码。软件检查通过不代表真实动机成立；真实阶段以运行产物和独立审计为准。尚无经验证的质量提升、成本优势或ACL投稿就绪结论。

首阶段使用Qwen2.5-3B-Instruct固定revision、4个冻结角色及同骨干汇总、2轮、12条可恢复有向边、128-token消息。3轮用于执行依赖资格和后续敏感性。先检验真实完整重跑与闭包反事实一致，再跑200开发题、5seed、5背景、3组边对的2×2审计；共60,000图观测。同题全部观测按问题簇处理，使用跨独立seed乘积估计平方交互，保留负估计。

完整主实验还需图条件预测器、同预算采集机制对照、BBH第二任务、直接近邻与独立测试。当前代码不把这些未执行阶段标记为完成。

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
| `scripts/local_coordinator.py` | 本地数据拥有者：传输、服务器核验和增量结果同步 |
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

v2消息门自动失败而未启动G1；v3只强化汇总字段提示，固定输入10次真实探针全部合规，完整资格仍待实际验证。
