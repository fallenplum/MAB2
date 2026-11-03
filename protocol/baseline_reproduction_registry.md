# 直接近邻复现登记：AgentPrune 与 AgentDropout

核对日期：2026-11-03（Asia/Shanghai）。状态：**官方资源与适配边界已核对；尚未克隆、安装、运行或复现。** 本登记不改变已冻结的项目 `src/`、`scripts/` 或 `configs/`。只保存研究文档和元数据，不保存第三方代码、数据或任何凭据。

## 结论与能力边界

AgentPrune 和 AgentDropout 均有可读取的作者官方仓库，已取得固定提交与关键源文件 blob SHA。它们的原执行图都允许同轮空间通信，而本项目固定为第1轮独立、第2轮读取上轮消息；**不能把两个原算法直接等同于当前12条边掩码的策略。** AgentDropout 的跨轮删节点是核心能力，禁止它删节点后只能报告为适配/消融版本。最小公平设计应同时保留原算法系统轨与统一合同的机制轨。

原生系统结果、统一骨干的算法复现、固定12边受控适配和简化启发式必须分别命名。当前登记未产生任何性能结果或“已复现”状态。

## 固定资源与许可证核对

| 方法 | 一手论文/作者入口 | 已锁定代码提交 | 当前资源状态 |
|---|---|---|---|
| AgentPrune | [ICLR 2025官方论文](https://openreview.net/forum?id=LkzuPorQ5L)、[作者arXiv](https://arxiv.org/abs/2410.02506)、[官方仓库](https://github.com/yanweiyue/AgentPrune) | [`c544dd6a1858c02c6d5d371d23c6e6ff55e0be21`](https://github.com/yanweiyue/AgentPrune/commit/c544dd6a1858c02c6d5d371d23c6e6ff55e0be21)；2025-03-23 | 远程只读：完整目录树、GSM入口、图执行与剪枝、最终汇总源码 |
| AgentDropout V1 | [ACL 2025官方论文](https://aclanthology.org/2025.acl-long.1170/)、[作者arXiv](https://arxiv.org/abs/2503.18891)、[官方仓库](https://github.com/wangzx1219/AgentDropout) | [`855befab08df60ac734c9608b3b2f317ce687e6e`](https://github.com/wangzx1219/AgentDropout/commit/855befab08df60ac734c9608b3b2f317ce687e6e)；2026-06-12 | 远程只读：完整目录树、GSM入口、轮次图/节点剪枝、最终汇总源码；该提交README增加V2资讯，本文锁的是V1 |

三个公开仓库的GitHub REST仓库元数据均返回 `license: null`，递归目录树未发现名称包含 LICENSE/COPYING 的文件：上述两仓库及 `csgen/SafeSieve`。这是**未检出许可证声明**，不等于任何特定许可证，不作授权推断。本阶段只上传本项目撰写的登记文档。可复核来源：[AgentPrune仓库元数据](https://api.github.com/repos/yanweiyue/AgentPrune)、[AgentDropout仓库元数据](https://api.github.com/repos/wangzx1219/AgentDropout)、[SafeSieve仓库元数据](https://api.github.com/repos/csgen/SafeSieve)。

## AgentPrune：算法、执行及反馈边界

论文将通信分为空间边与跨轮时间边，对图连接参数进行任务反馈驱动的策略梯度优化，结合低秩目标并实施剪枝；最终输出可由汇总模型或投票产生。原算法不是当前项目的图条件信用再校准器，不提供同一问题“当前图/单边切换图”的预算化复检选择。来源：[作者方法正文第2–3节](https://arxiv.org/html/2410.02506)。

已审查官方GSM入口采用最终数值答案与标签的比较作为效用，默认使用 `FinalRefer`。这需要**训练问题标签和完整多节点执行成本**；不能把“仅训练图参数、冻结语言模型”写成无训练或无反馈成本。原图执行对空间边做DAG采样，并另有跨轮连接和最终汇总。来源：[GSM入口](https://github.com/yanweiyue/AgentPrune/blob/c544dd6a1858c02c6d5d371d23c6e6ff55e0be21/experiments/run_gsm8k.py)、[图执行](https://github.com/yanweiyue/AgentPrune/blob/c544dd6a1858c02c6d5d371d23c6e6ff55e0be21/AgentPrune/graph/graph.py)、[汇总节点](https://github.com/yanweiyue/AgentPrune/blob/c544dd6a1858c02c6d5d371d23c6e6ff55e0be21/AgentPrune/agents/final_decision.py)。

| 与当前合同的差异 | 对复现的影响 | 最低公平处理 |
|---|---|---|
| 空间DAG边与时间边均可学习；当前合同只有r1→r2的12条可选边，自状态/汇总固定 | 原空间信息流不能无损映射到12位逻辑掩码；空间DAG约束也不是当前展开图的逻辑图约束 | 系统轨保留其空间/时间计算语义；机制轨明确只优化12条跨轮边且保护自状态/汇总，标 `AgentPrune-adapted-12edge` |
| 原方法可用其原prompt与汇总；当前为统一三字段128-token消息、固定3B汇总器 | 更改协议和汇总会改变目标及信息量 | 原生设置单列；受控轨所有方法共享本项目协议，完整计入汇总调用 |
| 原剪枝不维护暂停边的主动恢复通道 | 本项目的共享小额探索不能静默加到原版后仍称完全相同 | 原版保持原行为；额外共享探索/允许恢复标为独立适配版本 |
| 原校准/优化与本项目在线复检不是同一反馈流程 | 不能直接把原离线训练当在线服务免费预热 | 全部校准执行与标签计入总成本；另报冻结策略纯服务成本 |

### 官方GSM入口中需要继续核验的两处静态发现

**只对已审查固定提交的这一入口及已读路径成立，未运行，不推广为整篇论文或整个仓库不存在相应实现。**

1. `experiments/run_gsm8k.py` L122–151 的可见优化损失为 `-log_prob * utility` 的批均值；**该入口已审查路径中未找到论文所述核范数/低秩正则项**。因此尚不能把直接运行此入口标为“逐项忠实论文目标复现”。需进一步逐路径核验、记录差异；若另写论文目标版本，单列 `AgentPrune-paper-objective-reimplementation`，不能静默补项。
2. 同文件 L72–73 读取单一 `dataset_json`，L92–101 按同一dataset分批，L122–145 累计并保存解答率，L147–163 完成更新并在指定迭代后切换参数。**该入口本身未显示独立的train/test文件接口与分开的校准/测试指标。** 这不是“必然测试泄漏”的断言，实际含义取决于传入文件及外层流程；适配时必须显式提供本项目校准/留出清单，不能把该累计计数直接当官方留出测试成绩。永久文件：[固定提交GSM入口](https://github.com/yanweiyue/AgentPrune/blob/c544dd6a1858c02c6d5d371d23c6e6ff55e0be21/experiments/run_gsm8k.py)。

图文件 `update_masks` L317–339 根据当前logit排列将掩码置0，已关闭边参与排除；本次已读函数未包含恢复动作。观察仅约束该函数，不证明其他入口或后续方法绝无恢复设计。永久来源：[固定提交图代码](https://github.com/yanweiyue/AgentPrune/blob/c544dd6a1858c02c6d5d371d23c6e6ff55e0be21/AgentPrune/graph/graph.py)。

## AgentDropout V1：两阶段学习与不可省略的节点能力

论文先学习每轮通信邻接参数并按加权度选择节点，再在节点裁剪后的图上重新学习边参数并剪边，最终按各轮图执行并汇总。节点/边两阶段不是任意删边启发式；已有“删节点后需重新训练边”的结构依赖，因此本研究不能笼统说前人完全忽略图改变后的重新评价。来源：[作者方法第3节](https://arxiv.org/html/2503.18891)。

官方GSM入口分别读取待评估文件与固定 `datasets/gsm8k/train.jsonl`；`--dec`进入节点阶段，`--diff`使用每轮不同参数，再进入边训练。最终效用依据可验证答案，骨干冻结但图参数需要训练标签和多次执行。默认汇总为 `FinalRefer`；投票等其他汇总是不同配置。来源：[GSM入口](https://github.com/wangzx1219/AgentDropout/blob/855befab08df60ac734c9608b3b2f317ce687e6e/experiments/run_gsm8k.py)、[图和节点剪枝](https://github.com/wangzx1219/AgentDropout/blob/855befab08df60ac734c9608b3b2f317ce687e6e/AgentDropout/graph/graph.py)、[汇总节点](https://github.com/wangzx1219/AgentDropout/blob/855befab08df60ac734c9608b3b2f317ce687e6e/AgentDropout/agents/final_decision.py)。

| 与当前合同的差异 | 不可混淆的边界 | 最低公平处理 |
|---|---|---|
| 跨轮删节点是核心能力；本项目4节点每轮都执行 | 将节点固定后只保留边模块，不能直接叫原版AgentDropout | 系统轨保留Node+Edge两个阶段并完整计费；固定合同轨明确 `AgentDropout-adapted-edge-only` |
| 各轮空间/时间邻接与同轮DAG；本项目首轮独立、后轮同步读旧消息 | 动作、执行深度、调用节省机制不同 | 系统轨原计算语义单列；不得把同轮输出误作上一轮输入而不披露 |
| 作者README示例5agents；当前试验4agents | 不是改一个参数便自动兼容 | 先保留5节点系统轨；若推广维度，独立适配提交、测试和说明 |
| 两阶段重新优化，需校准数据 | 不得把其训练预算设0，而给本文额外审计 | 同训练题清单/标签权限和可比总资源上限，单列实际消耗 |
| 原指标与本项目在线时序不同 | 原离线训练不应被迫遵循另一个在线更新算法后仍称原版 | 报全成本校准+冻结测试；在线拓展另行命名 |

**明确兼容障碍：**在已锁提交的 `AgentDropout/graph/graph.py` L570–599，`update_masks_dec` 使用 `reshape((5,5))`、`range(5)`及按5计算索引；README GSM示例也明确5agents。因此当前 `--dec` 路径不能未经检查直接改为4agents。静态发现还不是运行失败报告；若改为任意N并调整选择数量，属于需说明的实现修补/适配。永久来源：[节点剪枝函数所在文件](https://github.com/wangzx1219/AgentDropout/blob/855befab08df60ac734c9608b3b2f317ce687e6e/AgentDropout/graph/graph.py)。

另一个容易误算的地方：已读入口节点训练有2批×20题、边训练有4批×10题的显式循环，唯一题数与实际执行次数不同，不能只报“40条数据”并把第二阶段执行当免费。具体成本必须从真实运行账本读，不用源码循环代替测量。该路径的代码与论文其他配置仍需逐项对齐。

当前README又介绍AgentDropoutV2，链接跳转到 [MASRubric](https://github.com/TonySY2/MASRubric)。这是另一项运行期审计/修复方法，**不等于已锁定ACL 2025 AgentDropout V1**，不得用其结果填V1基线。本登记不启动V2复现。

## 两项基线的最低公平执行锁

在真实G1门通过前只锁文档，不启动下述正式计算。后续至少保留两种结果轨，且不混在一个“同动作空间”平均值里。

| 轨道 | AgentPrune | AgentDropout | 可支持的结论 |
|---|---|---|---|
| 原算法系统轨 | 固定官方commit，保留空间/时间优化、DAG及原剪枝流程；骨干或prompt变化明确列出 | 保留每轮节点选择+边优化及原计算图；优先5节点官方路径 | 比较真实系统的质量与完整成本；统一骨干后不能声称重现论文表格原数字 |
| 固定合同机制轨 | 4node/2round/12edge/128token/统一汇总，固定自状态，保留可适配的策略梯度剪枝 | 同合同下明确edge-only；原节点能力被移除，不能拿此版本代表原版全部能力 | 检查图条件复检相对受控边优化的新增作用；必须搭配强机制对照 |

共同锁：相同问题与已给证据、问题级去重和独立留出；校准数据不得来自官方测试；角色/汇总/温度/输出限长在读结果前固定；校准、探索、图训练、失败重试、在线服务和评分全部记账；速度比较同GPU与调度条件；至少5流/算法seed并问题级配对统计。既有论文报告的token指标不可直接等同本项目完整账本。

建议使用 `official_commit / protocol_patch_commit / adapter_commit / source_hash / split_manifest_hash / reward_definition / action_space / training_queries_unique / executions / labels / aggregator / total_cost` 作为每次运行的必要记录。原算法轨先验证目标与动作一致性，再做小规模真实运行；任何修补均要给差异文件，不因“代码能跑”就认定复现合格。

## 其他直接近邻：资源登记和未解决事项

| 方法 | 已核对资源 | 原版能力与固定合同差异 | 当前状态 |
|---|---|---|---|
| SafeSieve | [作者论文v3](https://arxiv.org/html/2508.11733v3)、[官方仓库](https://github.com/csgen/SafeSieve)、[提交0af1b31474ae0107a4f6876b3dd6beb5dda6a9ac](https://github.com/csgen/SafeSieve/commit/0af1b31474ae0107a4f6876b3dd6beb5dda6a9ac) | 语义初始化、历史反馈、结构化剪枝；初始化LLM/embedding与在线更新费用均需计入；去掉节点/连通性行为会成为适配 | 远程核对目录与README；核心源码尚未逐路径审计；license:null |
| E2-Explainer | [作者论文2608.12921v2](https://arxiv.org/html/2608.12921v2) | 成对屏蔽/重复执行构造贡献并蒸馏预算子图；可含节点剪枝，固定全图信用与部署期再校准是需要落实的区别 | 本次有限官方入口与搜索未确认作者官方代码链接；**未确认不等于证明未发布**；不伪称有可运行仓库 |
| AgentRevive | [ACL 2026官方论文](https://aclanthology.org/2026.acl-long.373/)、[作者arXiv](https://arxiv.org/html/2605.17348) | Active/Standby/Terminated状态、压缩记忆和状态驱动边优化；状态策略训练/风险估计有额外成本，固定全节点会移除核心能力 | 本次有限官方入口与搜索未确认作者官方代码链接；只可登记待取得代码/独立重实现，不称原生复现 |

E2与AgentRevive是否发布代码仍有不确定性，置信度为中；AgentPrune/AgentDropout固定提交、关键函数及无许可证元数据的观察置信度高。所有方法的运行有效性和性能均**未验证**。

## 优势与劣势（风险）

这种双轨比较的优势是既保留竞争方法核心能力，又能用统一协议隔离本文新增机制，降低“削弱基线后取胜”的解释风险。劣势是系统轨与机制轨都需要预算，原计算图的适配和训练反馈对齐工作明显多于调用几个baseline名字；代码/论文目标差异、固定节点数实现以及许可声明缺失均需在复现记录中保留。

## 拓展思考

AgentDropout本来就包含节点裁剪后的再次学习，因此更有力的区别应是**持续部署中在当前图上选择有限的新证据，并证明何时复检比静态复用划算**。只比较一次剪枝或减少边数，仍不能建立该区别。

## 你可能想知道

把同一官方仓库代码换成Qwen2.5-3B，不会自动复现论文原表；关掉AgentDropout的节点阶段也不会留下完整原算法。文档锁与远程源码审查只意味着后续实现有可追溯起点，不能增加“完成实验”的计数。
