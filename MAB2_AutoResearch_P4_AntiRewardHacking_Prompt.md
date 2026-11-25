# MAB2 P4 AutoResearch：可证伪的图条件序贯复检（反奖励黑客执行规约）

**运行期间不需要普通阶段性聊天汇报。** 在专用研究分支逐轮记录完整结果、失败、预算和实验状态，并继续研究；只在通过全部最终门禁，或出现硬停止/耗尽已确认预算/预注册研究空间被证伪/会话无法继续时，交付一次可审计的最终报告。

### 0.1 必读源码和材料

根目录 `R=experiments/autoresearch_strict_graph_voi_20261124`：

- `$R/cycles/h2_fast.py`、`strict_graph_voi_cpu.py`（后者在 `$R/cycles/`）、`cycle_005_topology_h2.py`、`cycle_006_h2_confirmatory.py`；
- `$R/results/cycle_002_aggregate.json` 至 `cycle_006_aggregate.json`、`P3_PHASE2_REAL_SERVICE_SUMMARY.json`、`P3_CPU_SEED_AGGREGATE.json`；
- `$R/contracts/RESEARCH_CONTRACT.md`、`PROTOCOL_LOCK.json`、`$R/protocols/`、`$R/independent_tests/`、`$R/theory/SAFE_TRANSPORT.md`、`$R/reports/REAL_TASK_MODEL_AUDIT.md`、`G3_CLEAN_REPLAY.md`（在 reports 下）、`$R/ledgers/`；
- `experiments/p2_multihop_20261124/p2_service_runner.py`、`p2_logic.py`、`prepare_p2_multihop.py`；
- `paper/p3_strict_graph_voi/main.tex`；此前的 M60、P1/P2 报告与 `research/cycle3-credit-transport-20261116` 中 HotpotQA、2Wiki、Qasper、APPS 的负结果。
- 上一份研究者 Prompt：`MAB2_AutoResearch_P3_StrictGraphVOI_Prompt.md`，若在工作区有原件则核对，未找到不得假造其内容。

### 0.2 必须保留的事实与问题（本轮待复核，不许伪装成已修复）

| 已知结果/问题 | 本轮科学状态 |
|---|---|
| Cycle 005，6,400 个开发实例，Masked Pairwise 30.3226、H1 29.8561、H2 27.1277、Exact DP 26.9214，action exact 均为 1 | H2 对 H1 约 9.14% 的探索性信号；H1/H2 的 leaf cost 等实现不同，不是同核 horizon 消融 |
| Cycle 006，160,000 个实例，Masked Pairwise 30.0467、H2 26.8312，下降 10.70% | `EXPLORATORY_GAIN_WITH_CONFOUNDS`，不是独立 H2 成功；chain 23.73%、fanout 10.54%、diamond 7.00%、mixed 6.41% |
| `chain` 中每个 terminal 直接依赖 Source 0，Source 1 不参与决策 | 不是多层链；无关节点/已知未编辑节点不能作为强基线强制付费项 |
| `run_static` 先查询全部 Source；H2 可 Source/Terminal 交替、早停 | 基线不公平，必须同动作空间、同信息、同停止规则 |
| H2 枚举二进制全部状态，直接使用 `%3` 合成标签生成函数与 salt | 已知模型理想实验，不是黑盒预测；真实任务不能免费获得此生成函数；均匀先验未利用真实编辑约束 |
| H2/DP 等异常路径可能 `return action=target, certificate=False` | 目标答案回填属于禁止路径；确认聚合必须统计失败/未决 |
| P3 service 是 `certificate_voi`（静态 mass 排序+早停），不是 `topology_voi_h2_fast` | CPU 与真实服务不得拼成同一个方法的成功证据 |
| P3 service terminal `(i,j)` 同时读取 `relay[i]` 与 `relay[j]`；dirty 却仅检查 `cell[0] in changed` | 漏检第二父节点；严格证书的可信集合 K 不成立 |
| `exact_cache` 只重算 Relay，仍复用可能失效的 Terminal；graph-mask 重算全部 16 个 Terminal | 都不是最强依赖缓存；以真实完整输入指纹判断 |
| 真实 runner 声称的 R0→R1/R2/R3 并未实际进入子节点模型输入；RAG 仍预填旧证据并硬设 retrieval_calls | 不是真多层协作与真实 RAG；SciQ 4 选项还需检查 8 标签读取是否产生非法动作 |
| QASC/SciQ、Qwen-0.5B/14B 的旧有限接口 | 不能满足独立复杂任务门槛；Qwen-14B QASC 的留出准确率仍只有 0.125，不能凭模型规模认定能力 |
| `strict_graph_contract.py` 等测试依赖未在已核查 HEAD 的文件树出现；T01 等只测字符串玩具；GPU/CPU ledger 为零但报告真实执行 | 公开复现与真实成本审计不完整；必须补齐实际依赖、真实测试、账本，不许只改报告 |

不要试图证明上表每一条负面意见都错误。若新提交真实解决某项，提供对应源码、逐次测试和同口径重跑证据。历史原始结果永远保留，修订用新 artifact ID。

### 0.3 科学目标（不绑定 H2）

在真实多层 NLP 协作 DAG 的必要版本更新之后，利用合法历史 transcript 与少量新查询，在**严格有限动作证书或明确有效的风险合同**下，降低恢复新图决策所需的完整成本。证明图依赖带来的跨节点状态变化及查询策略拥有**不可由精确缓存、普通加权证书、同动作空间的自适应贪心、通用主动测试或已知模型 Oracle 解释的独立价值**。

本研究允许最终候选为 H2、H1、非概率鲁棒规划、其他结构化方法，也允许结论为 `NO_ALGORITHMIC_HEADROOM`。不允许只因名字叫 Graph-VOI 或代码多就把它定为主方法。

---

## 1. 反奖励黑客制度：执行者没有修改评价规则的权限

### 1.1 评价规则高于候选代码

按字典序判断：

1. **诚信与信息权限**（一项失败即 INVALID）；
2. **证明、真实依赖与可复现性**（一项失败即 INVALID 或 BLOCKED）；
3. **公平强对照上的独立算法收益**；
4. **真实任务质量、净服务成本与泛化**；
5. 最后才是论文叙事和效率数字。

任何收益指标都不能抵消 1–2 的失败。**保存可靠的负结果是合格执行，不是失败的代理表现。** 禁止为交差而优化 PASS 数量、摘要或图表。没有证据时状态必须是 `UNVERIFIED`、`INCONCLUSIVE`、`REJECT`、`INVALID`、`BLOCKED` 或 `STOP_NO_CLAIM`，而不是 PASS。

### 1.2 角色隔离与锁定

- **研究者所有**：本文件、原始 independent audit kit、验收阈值、独立测试生成种子/测试集、最终外部签收权。执行者不得编辑它们或生成替代版本自称同一测试。
- **执行者可写**：候选策略、开发集生成器、合法预测器、真实任务调用 adapter、诊断、追加式结果与实现文档。任何候选只能提交给冻结的评测接口，不能修改评测接口、Scorer、oracle、强基线和数据 split 来改善成绩。
- **可信执行/评分侧**：环境 broker 独占真实未查询输出、封存 gold、新图 reference、私有测试 manifest；候选只获得合法的 PublicState 和实际 paid query 返回。优先使用不同进程、只读代码、单独账号/容器或受控 RPC 隔离，禁止候选直接 import 私有模拟器生成函数、全量结果或评分数据。
- 本地若所有文件对同一代理可见，只能称为“逻辑隔离/本地自检”，不能宣称真正独立盲测。不能让执行者自生成“独立 reviewer”然后把它当研究者签收。真实的独立签收需研究者或其授权的隔离评测器运行原始核验工具并核对 SHA。无此条件，最多 `CANDIDATE_READY_FOR_EXTERNAL_AUDIT`，**绝不能 `RESEARCH_SUCCESS`**。
- 公开测试可以用来开发；真正私有反例、种子、blind IDs 由独立评测侧保管。若 agent 已读取封存数据，即暴露，必须记录，并使用全新独立划分，不能重新命名为 test。
- 首先从研究者原件建立 `LOCK_RECEIPT.json`：记录文件 hash、原始 Prompt hash、审计工具 hash、协议版本、baseline implementation hash、数据 manifest hash、时间戳和 commit。每轮运行前后校验；若变动则 FAIL 并停止该轮。无独立写保护时不能把 hash 自评当防篡改证明。
- 锁定测试只有研究者能授权修改；发现规约自身的 bug 应标记 `CONTRACT_DEFECT`，保存最小反例和旧结论，不允许降低门槛。必须经过单独版本审核并重新执行所有受影响比较。不得将候选失败改写为“协议不适用”。

### 1.3 明确禁止的替代实现（命中一项直接 INVALID）

1. 新算法未部署，却以另一个静态调度器的服务结果替代；不同候选算法 ID 或同名异实现不得共用主结果。
2. 使用 target/new_labels/gold/未购买的输出决定下一步、停止或回填失败答案；在 `not options`、`mask==0`、异常或超预算时返回 target；将 None、未完成、缺失样本计成正确。
3. 将真实多跳图写在 README/manifest 而不进入子节点实际 model input；把双层星型图标为多层 chain；保留死节点用于拉高基线成本。
4. 强基线被迫执行无关节点、未编辑节点、已经有效的缓存；基线与候选具有不同查询权、不同后代集合、停止机会或 budget；用最差对照代替最强对照。
5. 用合成真实生成器 `%3`、隐藏 salt、后验真实状态或 test 统计当免费预测器；把理想已知模型实验称为可部署算法。
6. H1/H2 使用不同代码、不同叶价值、不同 tie policy、不同缓存权限，然后把差异全归因 horizon；将不同 seed 或规模不同的表格拼为匹配比较。
7. 未知/失效的 terminal 仍计入可信分数；漏掉第二父节点；用漂移阈值、语义相似度、LLM judge 或经验分位数宣称逐例严格安全；未知权重偷偷取旧权重。
8. 伪造检索次数、耗时、GPU 小时、token、样本数、复现收据；只运行静态 `assertIn`、模板字符串示例、happy-path 测试便宣称真实 runner 通过；仅提交日志截图/手写 JSON，无可追踪原始账本。
9. 将开发集、重复问题、重复模型/编辑密度复制样本当独立测试；根据 test 结果选择模型、拓扑、样本、漂移强度、horizon 或阈值；删除失败单元、选择 seed、只报告正向分层。
10. 更改原定 CI、独立任务数、最强基线或收益阈值以宣布 PASS；仅增加合成样本数量，不解决机制假设；另写评审文件宣称 Main-ready。

---

## 2. 独立数学规约与可审计执行接口

### 2.1 统一图语义与合法状态转移

给定真实 DAG `G=(V,E)`、目标终端集合 T、old snapshot 与 new snapshot。每个节点 v 的执行函数仅消费：真实上游输出、节点本地数据/文档版本、prompt/template、模型/tokenizer/version、decode/seed contract、工具/外部状态、结构化 schema。`key_v = SHA256(canonical(actual_effective_input))`。若 LLM 使用 tokenized input，则保留实际 `input_ids` 或可复核 digest 和各项版本信息；不能仅 hash 节点名或手写图清单。

每一条 `u→v` 必须对应 v 实际消费 u 的有效输出。对完整有权重的终端计算祖先集合 `A(T)`；无关节点从所有主方法的合法付费动作集合删除（负控制可保留，但零费用且不会进入评价）。未直接改变且其所有父节点输入保持有效的节点按精确契约免费复用。已查询或经完整输入键验证的节点才是 resolved。缺失指纹、版本、父输入或状态 -> fail closed，不能标记有效。

`legal_actions(H)` = 当前未 resolved、位于 A(T)、全部有效父节点已经 resolved、且预算可支付的节点；如果选择预取/联合调用等不同 API，须在所有比较中开放同样权限并完整计费。每次付费观测后必须：

1. 记录 broker 返回值、实际调用 ID、成本、输入指纹；
2. 将结果写入已知状态，按拓扑传播；
3. 递归执行精确缓存闭包：仅当完整新输入键=已缓存键且执行合同适用，才复用旧输出；
4. 更新 dirty、known、合法动作、未知权重范围、当前证书；
5. 再决定下一步（如证书已经成立则立即停止）。

**链式**至少存在 Source→Intermediate1→Intermediate2→Terminal 的真正消费路径；**扇出**真实共享父节点；**汇聚/菱形**至少存在同一子节点消费两条不同父路径；**混合**同时包含多层与多父。需要输出实际运行边集、最大深度、每个节点实际祖先/后代；若 manifest 与实际 input 依赖不一致，以实际依赖为准，并标记该配置无效。对每个节点做有效输入扰动（不是无意义字符串测试）。

确定性执行合同：相同完整输入、模型版本、工具状态、解码与随机性契约得到同一个规范化输出。不能保证时只采用统计/分布层面目标，不能声称和另一次自由采样逐字/逐动作一致。worker/service 的实际 reference 采用相同执行语义；发生不一致则先修复或记为 `ENVIRONMENT_NONDETERMINISM`。

### 2.2 严格已知权重证书（所有比较共用一个冻结实现）

对于合法离散动作集 A，已知新图非负权重 `w_j`，终端标签 `y_j∈A`，定义

`S_a = Σ_j w_j * 1[y_j=a]`。

`K` 是已查询新标签或由完整输入契约合法复用的标签集合；`U=T\K`。`L_a=Σ_{j∈K} w_j*1[y_j=a]`，`W_U=Σ_{j∈U}w_j`。只有当

`L_a > max_{b≠a} L_b + W_U`

时返回 `CERTIFIED(a)`。若全部终端 resolved，采用提前冻结的 deterministic tie rule 返回 `FULL_RESOLVED`；不能以平局冒充严格早停。数值临界使用整数/有理数验证或有向舍入保守区间，不能用提高 `eps` 让错误证书通过。

证明：任意未知标签补全对每个竞争者最多贡献 `W_U`，对 a 至少贡献 0，故严格式保证 a 为唯一胜者。此结论不依赖黑盒 LLM 连续性、预测先验或语义相似性；**仅保证完整新图聚合动作，不保证 gold 正确或每条边的信用精确。**

反例：旧全分数 A=.55/B=.45，旧投 A 的未知终端权重 .08 可以转投 B，导致 A=.47/B=.53；因此使用含旧未知投票的分差时，简单扣一次未知权重不安全。严禁遗漏 U。若权重随尚未观测上游变化，要么先支付必要权重观测，要么使用对全部可行权重、标签的保守鲁棒证书：`inf_{(w,y)∈F(H)}(S_a-S_b)>0` 对所有 b。未能证明可行域覆盖真实状态时不允许早停。未知标签超出有限集合 -> INVALID。模型非法标签必须明确处理并计入失败率，不得悄悄映射到有效答案。

P2 `(i,j)` 实际消费两个 Relay，权重 `s_i*q_i(j)`。当两个 Relay 的有效输出改变且其他输入不变时，潜在失效终端为 `16-(4-2)^2=12`，并非 8，也非无条件 16；如果输出序列化实际相同，可更少。终态有效性必须由真正新旧 prompt/token fingerprint 确定。所有查询策略共享此逻辑。

### 2.3 预测与安全必须彻底解耦

- `CERTIFIER(H)` 只能读取实际已核验数据、合法缓存和可证明不确定域，**不接受预测器输出作为事实**。即使预测器被随机噪声或恶意错估替换，确定性证书仍应无错（成本允许变差）。
- `PLANNER(H,P)` 只能影响查询选择。预测器 P 输出有限结构事件，例如“上游有效状态保持不变/发生变化”“各可能结构状态概率”“下游可恢复缓存数/已知质量变化/剩余成本”。训练只使用开发集；概率支持不足的情况设置保守 OTHER 桶、无先验回退或上界，而非假造具体文本。
- 对有限已知模型的动态规划，明确其为 **IDEAL_MODEL** 实验。若直接公开确定性生成规律使所有终端可由父状态无调用计算，则增加 `analytic_known_model` 对照，或显式说明实验强制查询合同及其非现实意义；不得将模拟器隐藏转移核作为真实服务先验。
- `P_deploy` 不得读取生成器真实函数、隐藏参数、测试标签或实际未来输出。用冻结开发样本拟合轻量预测器（计入训练/推理/规划成本），或选不依赖概率的鲁棒策略。
- 不保证统计预测在编辑分布外校准。错误率、Brier/校准曲线、漂移压力与成本分解分别记录。若使用 conformal/CRC，先独立证明或核验交换性、目标损失、策略选择与自适应停止的有效性；若不能成立，只给经验风险，不使用“distribution-free”或“100% safe”的无条件表达。

### 2.4 Broker/Policy/Scorer 接口（核心代码必须改造）

在线候选仅获得 `PublicState`：公开 DAG、旧 transcript、版本化新输入、已知输出/权重、合法动作、预算、固定费用信息和冻结预测器。**不包含 `new_labels`, `new_outputs`, `target`, `gold`, `case` 全量对象、私有 fixture 路径**。可见的真实输入不能包含答案标签或未来执行结果，除非任务本身允许且所有方法相同。

```
Policy.next_action(public_state) -> Query(node_id) | Stop(proof) | Abstain(reason)
Broker.query(node_id) -> Observation(output, actual_input_digest, paid_cost, call_id)
Broker.update_and_close_cache(observation) -> PublicState
IndependentScorer.score(full_reference, sealed_gold, immutable_trace) -> metrics
```

Broker 独立校验合法动作和预算，按实际调用获取输出并写入不可由候选修改的 event log。候选无合法动作/无后验支持/超预算时返回 `UNRESOLVED`、`BUDGET_EXHAUSTED` 或继续付费保守求解，不得回填参考目标。所有失败仍进入分母；拒答需单独计 coverage、cost、risk，不能只比较已解决子集。输出中区分 `CERTIFIED`、`FULL_RESOLVED`、`UNRESOLVED`、`INVALID`，以及概率证书的单独状态。每轮逐项保存这些计数。

实际模型在线查询不能直接使用提前缓存的完整新图结果当免费调用；允许公平离线 replay，但必须明确是 replay，以独立 broker 按真实记录收费，且只在查询时揭示；实际 GPU 前向和服务延迟需独立真实运行。

---

## 3. 独立验收工具与真实集成测试（G0，不是自评 PASS）

研究者提供的 `MAB2_P4_Research_Gate.zip` 包含 `verify_trace.py`、`test_verify_trace.py`、公开示例、`SHA256SUMS.txt`。它是**确定性、固定非负整数终端权重、有限标签 DAG 的最小独立轨迹验收器**，不依赖候选算法的实现，不负责验证真实 GPU 的调用真实性、浮点未知权重或风险校准。必须另外做真实 runner 集成检查。不要更换 verifier 以使候选通过；候选适配输出格式，不改变原 verifier。

工具运行示例（由研究者/隔离评分进程执行）：

```
python -m unittest -v test_verify_trace
python verify_trace.py --fixtures example_private_fixture.json --traces example_trace.json
# 正式盲测时把示例换成评测侧保管的独立私有 fixture 文件，候选不可读取。
```

`verify_trace.py` 从固定拓扑顺序、old/new 完整上下文和 oracle 端输出独立重建输入指纹及缓存闭包，核对付费调用、预算、证书、最终答案。退出码 0：全部 solved 且轨迹合法；2：轨迹合法但存在未解；1：轨迹错误。**0 只是该有限 fixture 集合的门禁，不是 G1/G2/G3 通过。** 公开示例不可作为论文 test。研究者应使用不交给 Codex 的新随机 fixture 与独立 runner 重放。不能把 JSON 中自报的 call_id 当真实 GPU 调用证据。

### 3.1 强制真实候选集成测试（新增 T01–T16）

| 编号 | 实际被测对象 | 必须得到的断言/负控制 |
|---|---|---|
| T01 | 生产/真实 runner 的 prompt renderer + model call hook | 对每个有效父边替换父输出，真实模型 input_ids/digest 必须对应改变；修改非消费字段时不得伪造依赖 |
| T02 | 同一 broker 的精确缓存 | 第二父节点变化、模型/工具版本变化、缺失 digest、上游变化但结构化输出恢复；所有有效命中与无效复用都被独立逐项核验 |
| T03 | 所有候选实际停止函数 | 枚举≤4未知标签的完整反例、权重临界、并列、零权重、旧投票双向翻转；证书成立则全补全同一唯一动作 |
| T04 | `next_action` 实际状态转移 | 同一公共状态注入不同合法观测，验证变化会传播到合法动作、证书及候选评分；构造至少一组后续查询选择变化的实例，不能只测试 helper |
| T05 | 公共状态与进程边界 | 扰动未支付的未来输出和 sealed gold，在相同已付费观测历史下，在线动作/停止不受影响；字段探针、读权限和实际调用追踪 |
| T06 | 真 RAG/强单模型 | 真实检索器产生版本化证据 ID、原始文本、索引、排序和日志；不同方法面对同一更新后合法信息源；禁止硬编码次数 |
| T07 | 真实 call ledger 与独立聚合器 | 每个模型/工具/预测调用从实际 hook 产生，按 call_id 对账、重复 ID/漏记/负费用/假 token 都拒绝；不得依赖候选填写总数 |
| T08 | mask/digest/dirty 的故障注入 | 删除第二父边、错误 manifest、接口变更未声明、缺失版本、未知输入等，必须保守失效或终止，不得假安全 |
| T09 | 执行语义 | worker/service 一致性、固定 checkpoint/tokenizer/decode/seed 的重复运行；不一致时禁止严格逐动作声明 |
| T10 | 新 checkout | 所有 import 依赖、原始 verifier、测试、命令可复现；缺 `strict_graph_contract.py` 等立即失败，不允许自称 PASS |
| T11 | 图拓扑与死亡节点 | chain 最大真实深度、diamond 真实双父，目标祖先集合；加入一个无关昂贵节点不改变任何主方法的合法付费调用和成本 |
| T12 | H1/H2 单因素 | 同一函数、同一哈希、相同 leaf、prior、动作集合、证书和 tie 规则，仅 horizon 改变；逐题比较并证明成本差来源 |
| T13 | 先验/生成器权限 | mock 隐藏生成器禁止读取，替换 prior 为噪声/错误 prior，严格证书仍零错；IDEAL_MODEL 与 DEPLOYABLE 明确不同实验层 |
| T14 | 未完成/目标回填 | 无候选、无后验、模型异常、超预算、非法标签，要求非成功状态且 action 不取 reference；注入该类失败至少一次，评分分母保留 |
| T15 | 算法身份一致 | CPU 与服务共用的 planner/证书核心 API、source hash、配置、真实 action trace；服务中 H2 若未实际使用标记 `NOT_DEPLOYED`，不可写为 H2 result |
| T16 | 完整覆盖与预算 | 每个锁定 case × method × seed 必须恰好一条结果，逐记录哈希与独立聚合，漏样/重复/预算超额/后验重选均 FAIL |

测试必须运行真实 adapter、候选、broker 与 scorer，不能用另写小函数代替生产路径。**每项还要设计 mutation test**：故意删除第二父边、给 `return target`、把 `>` 改成 `>=`、注入假 ledger、换掉 H2、放开 oracle 字段等，确认测试确实能拒绝相应错误；若恶意变体仍通过，则该测试无效，G0 不通过。记录真实 exit code、输入/输出 hash、反例、运行命令、测试版本。不要只测试源码中是否存在某个字符串或 `assertLess` 在固定种子上成立。

---

## 4. P4-0：冻结基线与历史缺陷复核（不新增昂贵模型调用）

1. 验证真实 HEAD、干净工作树、已授权服务器环境、CPU/GPU/存储、现有历史数据/模型和当前实际累计成本。不能把旧 ledger 中 `gpu_hours_used:0` 当真实余额；先从作业/租赁记录与完整调用账本重建预算。权限不明，只做 CPU/已合法缓存。
2. 阅读原始 P4 指令与旧 Prompt，生成 `P4_SCOPE_AND_LOCK.json`，锁定下列门禁、统计、种子/划分、方法列表、预算、候选权限、算法身份、数据私有边界。研究者提供的 verifier hash 不可被替换。建立 `ISSUE_REGISTRY.jsonl`，以上已知问题逐项记录复现和修复证据。
3. 对旧 P3 的 `h2_fast`、`run_static`、`run_h1` 和实际服务 runner 做最小反例测试；验证死节点、初始不确定性、H1/H2 leaf 不同、`target` 回填路径、终端第二父节点漏检、伪 RAG、非法标签、GPU ledger、缺失测试依赖。不需要为复现这些 bug 再跑 16 万实例。
4. 用既有可访问真实 trace 检查旧 128 题及 P3 rows2/rows4 所有分层：旧/新 action flip、gold 纠错/破坏、真正的依赖缓存成本、有效/无效终端权重与证书 headroom。若原始私有 trace 不可访问，明确 `MISSING_PRIVATE_TRACE`，禁止从汇总推断逐题结果。所有这些已曝光队列标为 development/history，不可再次独立确认。
5. 冻结 `BASELINE_REPLAY.md`（真实命令、源码/输出 sha、逐记录路径、缺陷及是否重跑）、`HISTORICAL_RESULTS_INDEX.json`、`DATA_AND_BUDGET_LEDGER.json`。旧 M60 的结果若不进入新主比较，不必反复全量 GPU 重跑；如改其评分或主张则必须完整重新评分。不得覆写旧结果。

**出口**：生产接口、真实依赖与 G0 相关测试通过方可进入正式 CPU 归因；缺少外部签收可继续 development，但不能宣称独立确认通过。

---

## 5. P4-1：决定性 CPU 归因，不允许“10.70% 大样本”绕过强基线

### 5.1 统一的基线梯级（预注册，之后不可删改）

所有**证书型**方法使用同一个 public state、合法 query broker、cache closure、真实依赖、同一停止证书、价格、预算和 tie 语义。全量精确重算/无风险保障 Stale 的目标不同，单独标注。允许静态算法选择固定次序，但不能强迫它购买已知无关节点；基线的实现以最强合法合理版本为准，不取弱化版本。

| ID | 方法 | 要求 |
|---|---|---|
| S0 | Stale | 零增量调用；单独报告动作风险、gold 与成本；不得忽略 |
| B0 | Exact Dependency Cache | 真正完整输入复用；按拓扑只运行需要且位于目标祖先的节点，全部相关终端 resolved，非“只买 Relay” |
| B1 | B0 + Strict Certificate | 可提前停止；任何阶段一旦证书成立即停 |
| B2 | Adaptive Masked Pairwise | 与候选同合法源/中间/终端动作；动态权重/最坏影响/成本排序，允许交替查询和任意阶段早停；不仅先买全部 Source |
| B3 | Myopic VOI (H1) | 同一个 rollout 核、先验、叶价值和 tie policy；只看一层可能输出，不能用旧 `run_h1` 不同 leaf 冒充 |
| B4 | H2 | B3 唯一变化为 `horizon=2` |
| B5 | H3 / bounded beam | 在可负担条件下同核 `horizon=3`，记录复杂度与额外规划耗时 |
| B6 | Strong Generic DRD/VOI | 与相关真实文献的决策区域/主动测试思想合理对应；同合法动作、同先验、同缓存及停止；说明近似/适配的区别，不以随机排序冒充 DRD/HEC |
| B7 | Random / Periodic / Max-Uncertainty / Mass-per-cost / Topology greedy | 同权限、同证书，逐步选择；保留公平低成本简单方法；随机种子固定 |
| B8 | Full new graph / Descendant recompute | 真实完整目标 reference 和保守依赖基线；只作成本锚点，不作唯一 strong baseline |
| B9 | Tiny-DAG optimal DP | 同一公开预测模型与约束下的可执行最优策略；区分已知模型 DP 和 hindsight 诊断；不要求近似 H2 在相同假设下超越最优 DP 的期望查询成本；若 DP 可部署则比较含规划耗时的总成本 |
| B10 | Analytic known-model | 仅当模拟器生成函数被公开时必须实现直接推断，报告免费推断能达到的结果，不用理想环境虚抬模型购买需求 |

明确两种 oracle：已知预测分布的 online exact DP 可以是强合法参考；知道实际未来输出的 hindsight 只可用于 scorer 侧的上限/下限诊断，不得当在线预测或回填。无法找到合法计划则报告不可行，不报告伪零成本。若 B9 已知模型场景与部署候选信息不同，分表。

### 5.2 单核 horizon 消融与机制正交归因

先做小规模公开诊断：原 `run_h2_fast(case,horizon=1/2)` 对照，但这仍含原有图和 prior 限制，仅帮助复现；**正式实现先修复，不能直接将原接口进入主实验**。

正式 `plan(state, prior, horizon, leaf_value, legal_actions, transition, certificate, tie_policy)` 除 horizon 外完全共享代码、状态转移和边界。固定 H=1,2,3；H=0 对照无前瞻贪心。叶价值统一使用同一个合法 completion policy（先完成必要父节点且对缓存闭包正确计价、再按可执行强策略停止），不能假设免费知道实际未购买结果，不能计入已支付调用。若采用预测的 rollout，则所模拟状态与观测必须来自冻结预测分布，逐分支记录预测和真实返回区别。

对每一 case 做成对成本分解：

- A0：原弱基线（历史诊断）→ 正确目标祖先/初始合法缓存（删除死节点和未编辑可复用节点）；
- A1：B0→B1 的证书早停收益；
- A2：B1→B2 的动态交替/合法动作收益；
- A3：B2→B3 的一步预测收益；
- A4：B3→B4 的**纯两步前瞻收益**；
- A5：H2→H3/DP 的剩余头部空间、规划成本和 regret。

A0 与 A1… 不得混在同一个“algorithm saving”。报告逐 case 差值、每个拓扑和编辑/成本条件分层、最坏情况、实际 action exact/证书覆盖/未决/失败率、planner CPU 时间与内存。不能只报告平均比率。以 B2、B3、B6 和其他满足同风险的最强可部署对照为主要比较对象。若 H1 更好则允许 H1 或其他候选作为最终主方法，不为 H2 强凑贡献。

### 5.3 真多层 DAG 与模型不确定性实验矩阵

生成器必须先做图性质验收，再运行。所有正权重终端祖先合法；无关节点仅为负控制。每类拓扑实际存在所声称的边和深度，生成器的真实输出由父输出经过逐层转移，非扁平函数伪装。

| 因素 | 预注册开发矩阵 |
|---|---|
| 拓扑 | 真实 chain、fanout、diamond、mixed DAG；额外 flat/star 作为零跨层价值负控制 |
| 规模/深度 | 小图 4–12 节点供穷举/DP；中图 16/32；较大 64 节点及 2–6 跳分层，预算允许再扩张；不把 2–4 个二进制源伪称大图 |
| 编辑 | 单/多源编辑、内部节点接口编辑、共享父变化、局部证据版本更新；固定预注册稀疏/密集条件 |
| 漂移 | no-edit、真实输出无变化、小幅、中等、翻转；不可把测试标签或最终赢家用于选编辑强度 |
| 决策 | 宽/窄 margin、平局、冲突与冗余；不人为挑容易证书化的题 |
| 成本 | 均匀/异质/重尾；贵父便宜子、便宜父贵子；记录真实规划成本 |
| 接口 | 完整、缺失、不一致、双父/高扇出；未知权重单列，不得混用旧权重 |
| 预测分布 | 理想已知模型、无先验、均匀与历史频率、廉价学习预测、漂移/错误先验；先验支持不足、概率置换、结构 placebo |

先做开发小矩阵：每拓扑/条件 20–100 例，至少 5 个开发种子；只保留具有机制证据的候选。**开发集用于选候选、horizon、先验、阈值和 leaf；锁定后由研究者/隔离评测侧生成全新 seed 与 case manifest。** 禁止使用旧 `20261124..28` 五个种子作为唯一独立确认，它们已公开。独立 CPU 至少 5 个未曝光种子、每主拓扑至少 500 个独立图实例（根据预注册精度计划可更多）；不要求无意义地跑 160,000 个或更多实例。只对已完成公平归因的候选扩大样本量。

### 5.4 预测能力与部署成本可行性（第二道硬门槛）

不要直接把合成 H2 接到自然语言 LLM。并行比较：

- `IDEAL`: 真模型/模拟器生成规律只用于机会诊断，与部署结果隔离；
- `FREQ`: 开发历史中按节点类型、版本编辑类型统计结构化不变/变化概率；
- `LIGHT`: 使用已合法获得的节点成本、扇出、旧结构状态、输入编辑大小等特征，训练低成本逻辑回归/树/小模型估计“缓存恢复事件与未来证书质量/成本”；所有预测训练/校准只在开发数据；不需要预测完整自然语言文本；
- `ROBUST`: 无准确概率时，基于显式有限可能结构状态/最坏未来成本做鲁棒调度，或以图结构与证书有效上界作可执行保守贪心；不得将未经验证的 heuristic 叫严格 VOI；
- `SHIFT`: 预测器训练与测试编辑类型/节点状态分布不同；故意错误先验、概率扰动、支持外事件、空 posterior 反例；证书正确性不能依赖预测准确。

将已观测状态与合法预测分布区别开；任何预测错误只影响选点成本，不能进入 K 或缩小安全集合。若 `P(y|H)` 无法低成本估计，允许降级至无先验鲁棒策略。证明/测量真正跨节点价值：上游查询改变后代 cache validity、合法动作/权重界或剩余成本，而不仅减少某个独立标签的 uncertainty。用去掉图相关特征、保持入出度的合法重接线 placebo、同预测模型的无图策略、只保留强缓存的正交消融检验。图 placebo 不得改变真实参考或为算法提供无效伪依赖。

总成本（不同单位不要直接相加）至少按一套预注册的可比服务资源账本：

`C_total = C_model + C_tool/retrieval + C_predict + C_plan + C_validate + C_IO`。

报告真实 forward/input/generated tokens、GPU 秒、CPU 秒、wall clock、工具费用、缓存命中、内存。若有可解释的统一计价，用冻结价格换成同一单位（货币或资源成本）；否则分别报告并检查端到端壁钟/成本 Pareto，不得把 token 与秒简单相加。预测训练、索引构建、旧 transcript 初始成本另表，并报告 K=1、10、100 次真实更新/任务的摊销与 break-even；无法证实长期复用就不能只展示极大 K 的收益。CPU 策略模拟时间与真实模型调用分账。实际预测与规划成本必须测量，不以 0 代替。

### 5.5 CPU Gate G1：不可降低的成功条件

下列全部满足才可标为 `G1_CANDIDATE_PASS_PENDING_EXTERNAL_AUDIT` 并启动非冒烟真实模型验证：

1. 修复所有 G0 适用项，独立反例核验无错，所有正式 strict 预测的错误、未决和非法路径计数明示；不得以降低覆盖换取低成本。充足预算主比较：候选与严格基线均完整 solved，动作一致率 1.0；预算受限曲线另表，未决计入分母。
2. 至少一个具有实质区别的候选在真实多层合成 DAG、**DEPLOYABLE/ROBUST 信息权限**下，相对最强合法且同安全/覆盖的非候选实用基线，主指标独立留出配对成本点估计下降 **≥10%**（延续原项目内部实用目标），按独立图聚类的 95% CI 下界 >0；至少 3 个预先注册的真实多层拓扑族具有正向配对收益，完整报告所有族。若实际可执行 exact DP 成本更优且规划开销合理，不能把近似规划当胜者。该 10% 不是 ACL 官方门槛。
3. 若主张 H2 的创新，必须有同核 H1↔H2 单因素消融：在独立验证上 H2 仍有额外净成本收益，95% paired CI 下界 >0，规划开销计入；至少两个预注册拓扑族不为负。若 H2 没有独立收益，停止 H2 主张，转向其他有证据的候选，不得给 H2 改名。
4. 预测先验、真实因果机制与安全证书分离；非理想预测和合理域移下保留可用收益，或者有明确可验证的保守回退边界；若收益只存在于知道模拟器真实生成函数的 IDEAL 条件，`REJECT_ORACLE_DEPENDENT`。
5. 有实际跨节点机制及消融支持，能够区分缓存、提前停止、动态交替、前瞻预测和图特有收益；文献对照后不能被已有方法简单等价解释。理论只宣称已证明的充分条件/边界，不把普通加权早停、VOI 概念当原创数学。

CI 使用锁定的 paired cluster bootstrap（至少 2,000 次，固定 seed；独立图为抽样簇；分层权重在锁定前决定），同时报告绝对差和相对 pooled 成本差的 CI。不能把多 method、同图多编辑/seed 复制当独立 N。若存在多候选选择，只在开发选择主候选，独立确认不反复挑最佳；确认后改变候选需新独立队列或按预注册多重比较处理。若独立数据不足，`INCONCLUSIVE`，不能改为“No CI under project contract”。

**不要以 G1 失败为理由悄悄改成 G2 单独通过；不要用扩大样本数解决基线不公平。**

---

## 6. P4-2：真实复杂场景机会验证（先证实任务价值，避免重复 Cycle3）

可与 P4-1 做 CPU/已有缓存的并行资产审计，但正式 GPU 扩展必须通过 G1。旧 QASC/SciQ/四 Relay fixture 只用于冒烟与有限接口 sanity，不可计入 G2 的复杂任务数量。

### 6.1 数据与任务筛选（所有选择发生于开发集）

优先查找已有真实文本和模型资源，重点考察 HotpotQA、2WikiMultiHopQA、MuSiQue、Qasper 或其他具有合法文档更新/多证据推理的数据。阅读历史 Cycle3 的负结果及任务谱系，先解释过去为何单模型/廉价检索更好。不要为了得到复杂 DAG 人为新增大量无意义模型调用。若原始数据不存在，先利用授权资源合法获取；不能访问时记录 BLOCKED，不得伪造输入或把旧 MCQA 改名。

允许探索以下场景，最终冻结至少 2 个独立自然语言任务设置、其中至少 1 个真实多跳检索/长文跨证据推理，且具备真正多层/多父依赖：

- 文档/证据版本更新：同一问题的旧证据快照→新证据快照；更新有自然意义、来源可追踪；标注新版正确答案或可靠可审核的评估标准。
- 局部检索/工具结果变化：分解查询、检索、结构化证据提取、跨证据验证、有限标签或答案候选聚合；真实地将父输出喂给子模型。
- 合法的模型/验证接口更新：同一信息源和真实任务需要更换某个专家或规则；声明更新必要性，不人为制造测试翻转。

禁止用题目 gold 选编辑或构造让新图答案恰好正确的案例。冻结一个有代表性的总体与编辑生成规则，保留所有自然翻转/不翻转问题；翻转率低是负结果或适用边界，不是剔除样本的理由。若开发发现零调用 Stale 已足够且更新没有实际服务必要性，则停止该任务，或重新预注册一个业务意义独立的新任务设置。不得将 5%、15–40% 等翻转率当 ACL 硬规定。可使用开发集上的翻转率、强缓存剩余成本、gold 纠错/破坏和真实更新频次作为 headroom 诊断。

### 6.2 模型与架构资格

至少 2 个合法本地 checkpoint，优先 7B/8B/14B 等已有且通过基础能力测试的模型；Qwen-0.5B 仅作为 smoke 对照，不计入最终两模型目标。使用开发集预注册的指令遵循、有效输出比例、基本问答性能、资源限制和模型/Tokenizer 版本来选模型。规模本身不保证能力，禁止因 test 正结果而选模型。真实模型执行需使用可验证 token/标签解析，SciQ 等任务仅允许合法选项；模型不具备能力则换预注册备选或 BLOCKED。

明确一个可证书化的有限**动作**（候选答案选择、证据验证状态、是否采用某个答案等），并说明该受限接口如何对应真实任务质量。若开放式答案由另一个生成器输出，不能偷换为仅保证某个中间标签；必须测真实最终任务质量、生成器成本、候选召回率，并说明最终动作保证的范围。不能将结构化投影不变当成自由文本语义不变；下游必须只消费规定的结构状态或其完整输入被验证。多模型协作与同模型多次调用区分，固定真实协作分工，不把单模型 21 次重复调用直接称为多智能体优势。

### 6.3 真实任务价值与强非图基线

对比：同证据的强单模型（必要时长上下文与同预算 self-consistency/多样本/批量方案）、真实 BM25 或合理稠密检索 RAG、完整新图、旧图、严格依赖缓存、共享证书的强序贯基线。检索索引及文档版本、top-k、排序、语料和运行成本锁定；所有方法在相同合法更新信息源上执行，不得让单模型只读取旧证据。工具与检索的实际调用及耗时记账。Graph 的更高 gold accuracy 并非所有效率论文的必要条件，但**在真实更新服务需求下必须具有可解释且可比较的价值，不能在完整服务质量与总成本上被简单替代方案稳定支配**。若强 RAG 直接更便宜且同等或更高质量、没有更新维护必要性，`NO_REAL_TASK_HEADROOM`。

先开发小样本 32–100 个独立组、只为资格与预测器开发；不以小面板作为独立成功。报告预定任务的旧/新翻转率、gold 转移（correct→wrong/wrong→correct）、完整图与单模型质量—成本曲线、严格缓存剩余待复检质量/成本、planner 开销；确定实际更新次数/可复用条件。能否推广不得从合成 160k 或 QASC/SciQ fixture 推断。

---

## 7. P4-3：真实服务候选同核部署与锁定确认

### 7.1 算法身份证明（不可替代）

必须将 G1 保留候选的同一个 planner/transition/certifier 核心接入真实 broker，记录 `algorithm_id, core_source_sha256, planner_config_sha256, predictor_sha256, adapter_sha256`。如真实有限接口与 CPU 不同，只允许明确的 observation adapter、开发集训练的部署 predictor 或权重合同变动；不能替换核心策略，若方法发生实质变化则注册为新候选并重新做 G1 公平归因。

逐步日志至少包含 `case_id, method_id, step_id, public_state_digest, legal_actions, queried_node, action_scores_or_rollout_values, horizon, predictor_version, cache_hits_and_reason, dirty_before/after, known_scores, unknown_mass_or_bound, actual_input_digest, call_id, actual_cost, proof_status`。日志只揭示已合法观察值，不能公开私有 gold。若 H2 没有真实调用前瞻核，而只执行 mass 排序，则 `H2_NOT_DEPLOYED`，不得把其结果纳入 H2 主表。至少给出有真实父节点观测导致后续动作变化的成对轨迹，以及前瞻相对 H1 确实改变选点的真实案例；不能只展示示意图。

### 7.2 独立确认与指标

- 至少 2 个满足 6.1 的任务设置 × 2 个 checkpoint，主确认每个任务—模型组合目标不少于 **200 个独立问题/文档/更新组**，在样本不足或成本不允许时保留开发结果、标记 `INCONCLUSIVE`，不降低 G2。预注册按问题组、文档族、更新实体去重分割，跨模型复用相同问题是配对设计，不是倍增独立 N。以前所有已观察队列归为开发。
- 主要盲测 manifest 与 candidate hash 在真实执行前由评分侧封存；模型/先验/horizon/编辑规则/阈值/风险合同不得基于盲测改动。若失败，保留整个批次，新候选仅在开发修改，再申请全新未曝光确认队列。
- 真实 runner 模型调用、检索、缓存、规划、验证账本采用同一 hook，实际计费，固定 batch/decode/tokenizer/环境；控制执行顺序、warmup 和模型加载成本；任何服务/worker mismatch 明示。报告单次调用成本与端到端壁钟，不能声称端到端节约而只报告逻辑 token。
- 正式比较 **与 G1 一致的最强 B0/B1/B2/B3/B6 及实际候选**，同时保留 S0、full graph、强单模型/RAG；如某强方法不适配明确证明或实现共享等效版本，不能直接删掉最强者。
- 风险匹配曲线：锁定实际预算点（开发集定相对完整合法重算成本的 0/10/25/50/75/100% 等或合理绝对资源），每个预算重新运行合法策略；剩余预算不足必须停止/回退计费，不能把一个大 nominal cap 的单点误称 matched-budget Pareto。风险、覆盖、失败、净总成本同时报告。
- gold accuracy/F1/EM 或任务原生指标，与 full-new-graph action agreement 分开；评估必要任务更新和原始任务质量，不将动作一致写成答题正确。统计旧→新、强基线→候选的逐题 paired transition；按独立问题/文档更新组聚类 bootstrap ≥2,000，95% CI，固定统计代码，预先报告全体与分层。统计不确定时不宣布普适优势。
- 确定性合同下：正式充足预算主比较必须 100% 新图动作一致且完整 solved，含 tie/非法标签/失败；有任一不一致先 fail-close 定位，不得调测试阈值。若随机合同/统计风险方法为新主路线，必须在开发前锁定风险 `alpha` 与验证方法并经独立审查，另建同风险比较；不得把现有 0 错证书放宽为统计路线以过门。

**G2 成功**：两任务、两模型、真实图、同核候选、强真实 RAG/单模型、原任务质量与有意义的真实更新；在事前定义的相同安全/质量及覆盖合同下，至少两个独立任务设置和两 checkpoint 的合并配对净服务成本相对最强可部署同风险强基线有正向实用收益（延续项目目标：主 pooled 相对下降≥10%，95% paired CI 下界>0，分层完整且不允许把失败任务藏掉；若某分层无优势，单列其限制，不得普适宣称）。证书型与统计风险策略分开；在实际相同质量/风险条件下保留零成本 Stale 的经验结果和风险差别。质量等效声明需有预先锁定的配对容忍界；确定性动作完全一致自然给相同动作级准确率，但不自动保证开放式最终生成质量。净服务收益必须覆盖预测、规划、验证等开销。不能把 H2 模拟收益与另一算法的真实收益相加。

---

## 8. P4-4：G3 独立复现、理论和文献（非论文自评）

### 8.1 复现材料和独立签收

在新目录 `experiments/p4_falsification_graph_voi_20261124/` 保持完整结构：

```
contracts/                 # 研究者规约副本、LOCK_RECEIPT、验收门禁（只读）
independent_audit/         # 原始 verifier hash、运行收据、mutation 失败证据
broker/                    # 统一真实调用、合法动作、缓存、计费
candidates/                # 每个候选唯一 ID，统一 API
baselines/                 # 冻结强基线，同权同目标
protocols/                 # 版本与预注册
cycles/cycle_XXX/          # 全量证据，包括负结果
results/                   # 从逐题 trace 自动聚合，禁止手填
ledgers/                   # 实际资源、训练/预测/调用和预算
repro/                     # clean checkout 安装、测试、重放命令
literature/                # 公开原始文献与 novelty 矩阵
reports/                   # 最终门禁决策与证据链接
```

完整 CPU 公开 replay 从真实远程新 checkout 或独立无缓存环境执行，安装依赖并核对 source hash；复制部分当前 workspace 不算独立 checkout。无法网络 clone 可使用确切 commit 的可信归档核对所有 blob hash，标记环境边界。不要漏掉测试 import 或 `.gitignore` 掉必要源文件。原始结果可私有，但每个实验必须保留可验证访问路径、文件数、SHA-256、样本/调用数、脱敏逐题 trace、commit、命令、实际 stdout/stderr、退出码。不可仅存公开平均数。若私有数据/模型不应公开，提供研究者可在授权环境重放的收据，不将 sealed gold/凭据/私有原始文本上传公开仓库。真实 GPU 工时/设备类型/运行记录与费用汇总必须一致；未授权预算不可启动；不能将 `0` 当未知。报告源、聚合器、原始 ledger 需一一对应。

独立验收需研究者或其授权隔离评分者实际运行：

1. 检查代码与研究者原始 Prompt/工具 SHA 一致；
2. 运行真实 T01–T16 与 mutation test，核对 case 完整性与反例；
3. 在独立私有 DAG fixtures 和真实锁定服务队列运行 broker/评分；
4. 独立聚合风险/成本/CI 与主表核对；
5. 审查 algorithm identity、模型/检索公平性、原始失败列表、预算。

没有外部收据时写 `G3_EXTERNAL_SIGNATURE_MISSING`，`CANDIDATE_READY_FOR_EXTERNAL_AUDIT` 不是 `RESEARCH_SUCCESS`。同一个执行者生成的 `INDEPENDENT_REVIEW.md` 不能视为独立同行评审或最终签收。不得伪造第三方审批、签名或“ACL Main ready”。

### 8.2 Novelty 与论文（仅在有效证据形成后）

检索可核验原始出版物：decision region determination/HEC、DiRECt/非近视 VOI、adaptive submodularity、成本敏感主动测试、增量计算/依赖缓存、语义缓存、LLM 多智能体通信优化、图条件证书与风险控制。逐条核验论文作者/标题/DOI/出版页面，保留原始来源与具体对比。已有方法的通用决策早停、精确缓存、有限加权投票边界、普通两步 rollout 或贝叶斯 DP **不是自动创新**。要求可明确复述的一条新机制/新约束/新算法及同条件证据：为什么在 graph edit + 历史 transcript 下，跨节点价值不能由现有强合法方法解释？若最强现有方法基本等价，接受机制/边界贡献，不声明原创算法。

数学证明的假设必须与真实代码一致。对 `P→Q` 的漂移传播不能在黑盒 LLM 上假设无依据的 Lipschitz、收缩核或联合相关性；需要已知结构或显式校准合同。证明严格停止不等于证明调度最优，有限模拟先验不等于真实后验。若主张复杂度、近似界、自适应次模性等，给出完整条件和反例/证明；不要求为了理论外观强凑定理。

生成论文草稿仅在有效结果达到对应边界后；所有表从冻结汇总自动生成，引用从可核验来源取得。若未达到 G1/G2，写 `MECHANISM_AND_FAILURES` 技术报告即可，不用润色替代科研。若准备投稿，另验证 ACL 官方模板和完整编译、页面视觉检查、引用与结果一致性。投稿接收不存在指标保证。

---

## 9. AutoResearch 自主搜索计划：允许探索更多候选，但不给无限预算

### 9.1 状态机与每轮交付

`REGISTER_HYPOTHESIS -> G0_INTEGRATION/MUTATION_TEST -> DEV_PILOT -> MATCHED_BASELINE -> (ONLY_IF_ELIGIBLE) LOCKED_CONFIRMATION -> SCORE -> KEEP/REJECT/INVALID/INCONCLUSIVE -> NEXT_HYPOTHESIS`。

每轮开始前冻结 `IDEA.md`（可证伪机制假设与失败判据）、`CONFIG_LOCK.json`（候选 hash、共享基线/数据/seed/价格/预算、指标、开发/确认属性）、`IMPLEMENTATION_ID.json`。结束追加 `RAW_TRACE_INDEX.json`、`RESULTS.json`（全方法、全 case，失败行也在）、`COST_LEDGER.json`、`FAILURES.md`、`DECISION.json`、`GATE_RECEIPTS.json`。`PROGRESS_LOG.jsonl` 只增不删，每次记录父 commit、当前 hash、artifact SHA 和实际 CPU/GPU 用量。研究者保留 Git 历史；不可 force push/删负结果。每轮提交先注册、后结果，不能先跑 test 再补注册。报告不是唯一证据。

### 9.2 优先研究假设（建议 8–12 个有区别的开发轮，不是必须全部跑到 GPU）

| Cycle | 机制假设 | 要证伪的具体条件 |
|---|---|---|
| 001 | 旧 10.70% 的收益源于死节点/强制查询 | 修复 graph/祖先/合法缓存后重做 B0–B4 与 H1/H2 同核；若 H2 不增益，停止旧主张 |
| 002 | 上游实测可让真实后代缓存恢复 | 多层链/多父/扇出构造，比较强精确缓存与严格证书；含脏锥塌缩 counterexample |
| 003 | 自适应交替本身具有收益 | B1↔B2，使用同证书、同权限；观察结果不同的轨迹 |
| 004 | 纯 H2 前瞻额外有价值 | 同一 kernel horizon 0/1/2/3；small DP，估计 leaf 与规划开销；旧 prior 与真正可用 prior 区分 |
| 005 | 低成本事件预测足以指导查询 | FREQ/LIGHT，开发校准、训练/推理成本和 OOD 先验；不免费使用真实生成器 |
| 006 | 无先验鲁棒规划有独立价值 | ROBUST vs 共享风险/证书的强 greedy/DRD，风险集合与最坏成本；若太保守保留失败 |
| 007 | 结构与成本耦合带来非近视收益 | 去图特征、拓扑安慰剂、真共享父与多汇聚、已知聚合器更紧可行域；与已有通用 VOI 比较 |
| 008 | 规划开销和规模限制可控 | 16/32/64 节点、异质成本/长路径，bounded beam/DP 边界、真实 CPU 和净成本 |
| 009 | 真实复杂任务存在可利用 headroom | 开发任务/模型资格、强 RAG、真实更新与严格缓存，保留旧 Cycle3 否证 |
| 010 | 同核算法可在真实服务上净省 | 小规模真实 smoke，预测/规划、真实依赖 T01/T15；不使用已曝光测试作确认 |
| 011 | 冻结候选独立确认 | 新问题组、两任务两 checkpoint、同风险/预算强对照，完整 ledger 与 CI |
| 012 | 外部复现与独立验收 | 原始工具、私有反例、新 checkout、论文主张边界；只有外部收据才最终成功 |

Cycle 编号仅示意执行逻辑，允许根据发现改变开发候选顺序，但不允许取消门禁或把开发负结果隐瞒。若先验可行性明显失败，允许立即转向鲁棒/简单策略；每个新候选必须提出不同可证伪假设，不能只换 seed、学习率、门槛、样本数量反复找正结果。CPU 优先，GPU 资格通过再扩大。对同一机制连续三个正式失败，关闭该机制族；只有不同、明确的新假设才重启。若三类核心候选（非近视、鲁棒、图结构贪心）均在强基线与合法信息下无实用 headroom，保存全部失败，`STOP_NO_CLAIM`；不要为了“必须成功”无限发明后验条件。

### 9.3 预算与自主执行的硬边界

- 先核实真实剩余 `MAX_NEW_GPU_HOURS`、`MAX_CPU_HOURS/RUNS`、`MAX_PAID_API_COST`、可用服务器时长和项目权限。旧文档的 160h 是历史记录，不等于新授权余额；报告 GPU=0 与实际作业矛盾时先重建账本。
- 无可核实新增付费预算：仅做 CPU/已有合法缓存及可合法使用的现有资源；不得自行租新机器、购买 API、增加配额、上传秘密、修改外部权限。真实调用前估算和锁定 per-cycle 预算，至少预留确认/复现所需且不低于已确认新预算 20%。若无法完成确认，不启动超出预算的探索。
- 每轮保存累计成本与剩余，达到授权上限立即停止。若硬件/私有数据/外部审核/会话寿命无法支持继续，记录可恢复 checkpoint，不能谎称后台运行或将 BLOCKED 写成 PASS。
- 不发送普通阶段性聊天报告；内部审计文件必须及时提交。用户若主动询问则如实回答。硬阻塞、预算耗尽、严重信息权限问题、无法独立验收时可一次性报告，不必为“未达到目标不汇报”隐瞒失败。

---

## 10. 最终成功与终止逻辑（硬门禁，不允许自改）

由独立 verifier/scorer 的可复核证据生成 `FINAL_GATE_STATUS.json`，字段至少为：

```
G0_integrity: PASS | FAIL | UNVERIFIED
G1_algorithm: PASS | FAIL | INCONCLUSIVE
G2_real_complex_nlp: PASS | FAIL | INCONCLUSIVE
G3_reproduction_external_review: PASS | FAIL | BLOCKED
algorithm_id / core_source_sha / chosen_horizon / predictor_sha
primary_baseline_id / matched_budget / risk_contract / n_independent_groups
paired_effect / CI / per_topology / per_task_checkpoint / failure_counts
actual_budget_used / remaining_budget / raw_artifact_receipts
final_status
```

- **`RESEARCH_SUCCESS`**：G0、G1、G2、G3 全部 PASS，且真实候选同核、原始研究者规约未变、独立验收由研究者/隔离评分侧签收。输出一次完整报告：主要新机制和数学定义、必要假设与限制、强公平 CPU/真实主表、H1/H2 因果归因、部署预测/规划开销、真实任务质量与成本、CI、负结果、完整复现命令/hash、文献区别、已用预算和剩余。只说明证据是否支持 ACL Main 级研究准备，不保证录用。
- **`CANDIDATE_READY_FOR_EXTERNAL_AUDIT`**：内部实际证据达到候选门槛但缺外部签收或隐私隔离盲测，交付给研究者执行的具体命令与待核验材料，不等于最终成功。
- **`STOP_NO_CLAIM` / `RESEARCH_INCONCLUSIVE` / `BLOCKED` / `INVALID`**：说明触发的硬规则、已经证伪的路线、实际剩余预算、有效与无效结果目录、缺口、恢复条件。全部负结果也属于科学交付，不许更换任务定义或包装为成功。

开始执行：先检查实际 HEAD 和预算，冻结研究者规约、外部工具 SHA 与统一强基线，运行 P4-0 和真实 T01–T16；完成最小反例，再做 B0–B6 公平归因与非理想先验可行性。不要先写论文，不要先启动大规模 GPU，不要仅凭 10.70% 历史数字宣布 H2 胜利。**科学目标是发现和验证真正的新增机制；诚信与证伪优先于任何成功指标。**
