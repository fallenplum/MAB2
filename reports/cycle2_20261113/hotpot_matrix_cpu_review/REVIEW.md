# 单题 Hotpot 适用性矩阵独立工程审查

**结论：所审 staging 版本的静态检查及 20 项 CPU 合成测试通过，未发现阻断性的矩阵、费用或信息隔离错误。该结论不授予真实模型工程资格、科学准入或 GPU 预算。**

审核对象 `matrix_executor.py` SHA256 为 `016b175f75fca838c83f3c26073a656b901e3d2fda12133c917073304c67bb8c`，工作清单 SHA256 为 `bfdd98e7778aa7d0d46dc44169f5ff1d31814de2a04c3f0b92b61ce051f62960`。清单内 15 个文件在测试前后逐字节校验，未改动被审实现。这里审查的是未封存工作快照。

## 复核结果

- 四条预定 probe 边均为 source-major 0-based `{1,5,7,9}`；B 是四节点有向环。独立可达闭包与冻结 Executor 一致。
- 同题首 seed 的 A、B、NO_EDGE、A−e、B−e、independent12、singleM 依次需 13、9、9、6、4、13、1 次成功物理生成；第二 seed 需 13、9、6、4 次。总计 87 次生成，完整逻辑调用 131 次。每个多节点对照均为 Qwen 6 次、Mistral 7 次；没有将 no-edge 的 self-state 修订当作 independent12。
- 缓存只来自同题、同 seed 的完整原 A 或原 B；原冻结算子继续核验题目、协议、seed、节点输入 fingerprint 和输出完整性。partial base 不复用，缺原 base 时 fresh 执行同一条件，因此失败路径可能超过 87 请求。
- sender 的归一化与 receiver 的 token 计数沿用已 pin 的路由；观察器保持模型 identity、提示、seed、generate 返回值。44 个条件的提示、节点、wire、费用与直接冻结入口对照通过。
- physical ledger 按请求追踪。相同 call_id 的多个状态不重复计算；缺失或冲突费用保持未知；跨条件复用 call_id 会使总费用未解决并中止。成功图另保留完整逻辑费用；对异常矩阵不补全虚构逻辑费用。resident GPU hours 明确为 null，必须由外层监管记录。
- 仅固定 backend 的两种生成前输入容量异常可继续；已付费后输出违反 token cap、OOM、身份改变、日志失败等均为 INCOMPLETE。已返回但尚未形成 NodeRecord 的 Generation 和已付费用证据可保留。单 M 输入拒绝不会取消第二 seed，其失败也不被偷偷评分。
- 所有生成都得到同一完整 question/context。independent12 的 12 个样本没有跨控制候选或 self-state，聚合器可看到 12 个样本；singleM 是单独 pin 的直接问答提示、完整 context、256 输出 token，没有候选。额外 metadata 不进入 prompt；明确 gold 字段被冻结接口拒绝。

## 测试证据与权限边界

`audit_001/RECEIPT.json` 和 `CPU_TESTS.log` 记录作者的 15 项 CPU 用例实际重跑，以及本审查另写的 5 项边界测试：全 context/metadata 隔离、单 M 拒绝后继续第二 seed、首个付费请求后模型身份变化、生成后输出超限、各对照模型组成。共 20 项，零失败、零错误、零跳过。没有导入 torch/transformers、读取模型权重、读取 benchmark/R1 数据、调用 GPU、改动 E 工作树或提交 Git。

优势是保留固定控制流、原始生成、合法缓存血缘及完整费用；实质局限如下。

1. CPU mock 不证明长 Hotpot context 下的真实输入容量、格式质量、重放一致性或吞吐。87 是完整成功的调用数，不是 GPU 小时预测，也不是异常路径上限。
2. 模块返回结果对象，不是持续落盘执行服务。SIGKILL 不能返回 Python partial result；正式入口必须另验原始事件及时落盘、监督硬上限、模型装载/失败驻留费用与恢复血缘。
3. 所有后端对象要求调用者独占；本地锁不防另一个外部调用者绕过入口并发修改后端。真实 collector 应维持该契约。
4. singleM 是新提示算子。它是便宜参照，不能因 CPU 语义正确就称最强单模型，也不能忽略其与多调用架构在输出/费用上的区别。independent12 与通信图调用组成相同，不保证 token 或 GPU 成本相等。
5. 本文件不决定 384 题合同门、统计结果、题目选择或预算。工程通过不能代替研究适用性证据。

置信度：对所审字节版本的 CPU 语义与记账结论为高；对真实双模型长上下文运行尚无本审查提供的实测证据。
