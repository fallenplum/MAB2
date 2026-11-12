# Hotpot图条件信用适用性：可冻结合同候选 v1

状态：CANDIDATE_NOT_FROZEN_NOT_ADMITTED。本文件未分配GPU额度、未选题、未读取标签或R1结果。主代理须在首个新benchmark调用前冻结代码、数据描述符、工程资格及资源表；必填pin缺失则不可执行。旧R1/R2/R3及旧G2/D2/D3的合同均不改变。

## 问题、范围与不可替换项

评价固定Qwen3B/Mistral7B混合系统在Hotpot distractor train上的通信作用，以及删除其他通路后同一边的信用是否变化。本轮不识别异构的因果效应、不证明主动复检收益、不声称新的CS/HT/CV理论。

**B是唯一预定通信候选。** A仅提供稠密背景，不可在B不利时用A、F1、某边、某seed或观测最佳图救援。所有worker和single M看到同一完整官方问题与上下文；不分片证据、不依据supporting_facts取段、不削弱aggregator信息、不删除self-state。模型revision、tokenizer、prompt、采样、解析、评分和缓存算子在执行前固定。

H1的具体预测是B优于两种13调用无通信基线，且保留边在稀疏B中的平均存在信用高于稠密A，即平均δ>0。若δ方向相反，只能支持另一方向的条件依赖，必须明确原“冗余释放”解释受反证。

## 固定全集、划分与随机数

总额恰为384个新问题，无效果驱动续样或删题。官方test、旧开发题及全部已记录G1/G2/资格/新周期R1/工程探针/明确文献暴露题均排除。重复组至少按规范化question文本构造，不能仅检查question+context整体payload：使用NFKC、Unicode空白折叠、casefold得到question key，同key即同组，不依据答案或supporting_facts合并。精确规范化payload重复作为附加检查；这些关系的连通组只保留UTF-8字节序最小ID代表。组内任一成员有已知暴露则整组排除，即使该成员不是代表。若与既有normalizer不一致，先在标签前版本化并核对全部已知排除映射，不静默覆盖旧hash。未知外部暴露记录UNKNOWN，不制造预训练无污染保证。排除元数据、normalizer源码和来源文件SHA须先冻结。采样入口只接收ID、question与完整context，不接收answer/supporting_facts/gold。

1. 合格ID先按UTF-8字节升序排列；重复ID或payload映射冲突直接拒绝。
2. `Generator(PCG64(261113101)).permutation(M)[:384]`确定选入顺序。M不足384则不准入；不补其他数据源。
3. 四边标签构造为`repeat([1,5,7,9],96)`，由独立`PCG64(261113102)`置换后顺序配给384题。因此每边96题；两seed相同边。
4. 生成seed1=26111301，seed2=26111302。沿已资格核验的节点seed派生规则；两seed命名空间独立。同一题/seed/node在图间使用共同随机数，图间配对；不得把它当独立图样本。independent12沿其独立槽位seed域，须在冻结工程资格中验证12槽位没有复制采样。
5. bootstrap使用PCG64(261113103)。盲计时前缀排序使用PCG64(261113104)。这些数字不能依据已有R1结果修改；准入前查既往种子注册表，发现冲突则在首题前发布新版本，不能静默换seed。

冻结的selection manifest只包含ID/hash、原始选入序号、分配边、种子与不含标签的长度元数据。label vault由评分器单独保管。不会将题目token长度/格式成功与正确率混为筛题依据；全集选定后不能因输入长或输出失败剔除题。

## 执行矩阵

四worker路由固定[Q,M,Q,M]，aggregator=M；Q=Qwen2.5-3B-Instruct，M=Mistral-7B-Instruct-v0.3。三轮，worker输出上限128，aggregator/single输出上限256，接收方输入上限8192，采样temperature0.7/top_p0.9。全部实际operator pin优先于名称，差异必须在首题前修订合同。

source-major边索引；A mask=4095，B mask=674（{1,5,7,9}），no-edge mask=0。A\e/B\e清除预分配边bit。详细行见MATRIX.csv。

- seed1全体：A、B、no-edge、A\e、B\e、independent12+aggregate、single M。
- seed2全体：A、B、A\e、B\e。
- 共4224个逻辑condition终态记录；其中3456个graph条件、384个independent12、384个single。
- 各graph及independent12独立服务均13次逻辑调用，single为1。全成功保守采集估计物理33408、独立逻辑50304调用。异常或不能复用时可以增加实际调用，故该数不是预算硬上界。

independent12为4角色各3个互不见历史/peer输出的样本，aggregator接收全部12槽位，不能只给末4条。no-edge保留三轮self-state。剪peer边不减少worker调用，主要改变输入token；反事实后缀缓存节约不代表部署整图服务省call。

同题/seed执行次序固定A→B→no-edge→A\e→B\e→independent12→single；seed2为A→B→A\e→B\e。B/no-edge/A\e均以合法A记录为base，B\e以B为base；跨架构不假定缓存。A失败时其他条件仍按原协议独立执行，不能伪造base，增加的费用如实计入。不能通过重排/追加图挑选更好表现。

## 完整覆盖、失败与溢出

采集期间不读取EM/F1、信用、答案内容或中途组间效果。合法monitor仅接收完整性、状态码、调用/token/驻留费用和硬件健康。每个逻辑条件必须有唯一可审计终态；seed、mask、输入身份、节点依赖与账本绑定一致。重复终态、缺记录、错误cache、无解释重试或源版本混用均阻断分析。

预定系统失败与基础设施失败分开：

| 情况 | 保存与分析规则 |
|---|---|
| 输入超过8192 | 不截断、不重分上下文、不换题。写FAIL_INPUT_OVERFLOW，停止该condition后续节点；该系统条件EM/F1记0。其他条件保留固定计划，合法复用只用已完成节点。 |
| 完成生成但格式不合法 | 使用已冻结formatter/fallback和严格parsed.answer入口；不临时从raw提取答案。评分按该固定输出执行，空/不可用answer记0。格式计数单独报告。 |
| 达max_new_tokens | 是固定输出预算内的完成记录；照原parser评分，不追加生成或扩长度。 |
| OOM、进程异常、硬预算截止、传输损坏、缺模型文件 | 基础设施/执行不完整，不擅自记为答错。保留失败及费用，停止当前尝试；无静默retry。恢复须有显式attempt血缘及统一累计账，不能仅评分完成子集。 |

“4224终态覆盖”不等于“4224成功生成”。只有基础设施完成且账本闭合，才能释放评分；合法overflow计0后仍保留全部384题的意向评价，不以共同成功子集代替主分析。为避免容量差被误称为通信机制，**任一预定主图/主基线发生输入overflow时，不签语义L1/L2升级门**，输出COMPLETE_WITH_CAPACITY_FAILURES及全样本系统成绩；可支持容量限制描述，不能自动进入机制实验。single overflow也阻断完整强基线比较。该规则保守，但不是事后剔除样本或把可预期溢出伪装为程序bug。

## 评分与统计对象

主指标为官方Hotpot answer EM；answer F1为次指标。无supporting-fact/joint指标。官方评分器固定提交`3635853403a8735609ee997664e1528f4480762a`及本地adapter SHA须核对。所有条件同一输入接口：固定formatter产生的parsed.answer，经官方normalization评分；禁止某基线使用raw答案回退。标签只用于封闭评分，模型和选择器不可见。

每题q构造四个主EM统计量：

    D1_q = EM(B,seed1) − EM(no-edge,seed1)
    D2_q = EM(B,seed1) − EM(independent12,seed1)
    γ_b,q,s = EM(G_b,q,s) − EM(G_b\e,q,s)
    δ_q,s = γ_B,q,s − γ_A,q,s
    D3_q = (δ_q,1 + δ_q,2)/2
    D4_q = δ_q,1 · δ_q,2

主参数为各D的384题等权均值，等价于四边各自题均再等权。D4对应θ=E_{q,e}[(E_seed δ_q,seed)^2]，依赖两seed条件独立与同一固定生成分布；不是两个总体平均δ的乘积。原始D4均值及区间允许负，不能裁零或称真实平方为负。seed共用流/重复输出身份错误会阻断该解释。

每个bootstrap重复在四边各自96题内有放回抽96个题簇，保留该题全部seed/图/基线。边顺序[1,5,7,9]，各层原始行号升序；一次调用`rng.integers(0,96,size=(10000,4,96))`，按C顺序消耗随机流。所有主/次统计量共用同一抽样矩阵，不各自重设seed。

四个主EM统计量构成一个比较族：Bonferroni将双侧错误预算0.05分为4份，每项98.75% percentile区间，分位0.00625和0.99375，线性插值。它是题簇bootstrap的近似同时区间，不是严格有限样本或部署风险保证。报告原始点估计、所有4区间、非零题数、正负支持、分边描述。F1、B−single、每seed信用/各边均值及成本为预定次分析，给未校正95%描述区间并明确不替补主门。

稀疏保护：任一统计量的非零题簇少于20，或样本方差为0，或bootstrap估计量取值全同/其区间退化，其bootstrap区间仅描述，不签该项方向/排除门。总体样本方差和bootstrap估计量方差分别报告；四层各自常数但层间不同，仍可能产生退化分层区间，不能因pooled方差>0而签门。20是保守资源规则，不是保证覆盖的数学阈值。特别是θ全零时[0,0]不能排除未观察到的稀有变化。**本合同不根据θ的bootstrap上界小于0.0025自动排除RMS≥5pp；也不把上界为负解释成负的真实平方。** 所有不满足支持条件的L2情况均保留INCONCLUSIVE。

## 阈值体系与结果分类

门依据没有读取本轮或R1结果。2pp、5pp仅为预数据效应尺度参照；θ尺度参照0.0025对应RMS5pp，不是自动PASS门。标准误/区间宽度与支持数优先于点估计排名。

L1：两项主质量差均满足支持条件且同时区间下界>0，记SUPPORTED_BOTH。**即使点估计不到5pp，也可以成为进一步独立机制检查的依据。** 点估计达到5pp只记POINT_ESTIMATE_AT_LEAST_5PP；只有区间下界达到5pp才可说该近似区间支持至少5pp。若任一必要差有支持且上界<0，记B_DISFAVORED，停止B优越性主张；不能用A/F1替补。其余为INCONCLUSIVE。若上下界均位于[-0.02,+0.02]且支持条件满足，额外标记APPROX_SMALL_EFFECT_RESOLUTION；只说明在所用近似区间口径下效应被限制在2pp内，不称严格等价。

L2：D3有支持且区间下界>0，记REDUNDANCY_DIRECTION_SUPPORTED；上界<0则REDUNDANCY_DIRECTION_REVERSED。D4有支持且下界>0，记NONZERO_CONDITIONAL_DRIFT_SUPPORTED。这三者任一成立即具有图条件变化的开发证据，但方向反转须保留H1反证。D4点估计为正而下界≤0，不算支持。另报告区间宽度/0.0025以显示漂移幅度精度；不因宽而把显著非零改称不存在，也不因非零而假定幅度已足以编辑。全部未获支持则L2_INCONCLUSIVE，不能把非显著当无变化。

| 系统完成和证据状态 | 允许的结论/行动 |
|---|---|
| 身份/账本错误、缺覆盖或基础设施失败 | INVALID或INCOMPLETE；不得评分完成子集签科学结果。 |
| 容量失败 | 全样本系统成绩可报告；语义机制升级阻断，不变更输入取正结果。 |
| L1支持且L2支持 | 只允许考虑下一份独立机制验证协议，不是方法成功；完整费用与single等强静态参照仍须审核，不自动批准GPU。小但有方向证据的效应不因低于5pp而一律禁止研究。 |
| L1支持、L2不确定 | 固定通信用途有支持；信用维护依据不足。结束本轮，不加seed/加题。 |
| L1不确定或B不利、L2支持 | 条件反应存在，但可能只是有害通信变化；不能以剪坏边救成净系统改进。新的风险控制问题可以另提，但不得宣布原B通过或自动投入。 |
| L1不确定、L2不确定 | EVIDENCE_INSUFFICIENT；停止当前候选的自动算力升级，保留小效应/稀疏性的不确定性，不判整个主题无效。 |
| B不利或完整费用被强静态方案支配 | STOP_CURRENT_B_DIRECTION；不从A或同题oracle菜单换赢家。 |

“仅允许进一步独立机制检查”也不是净效益已证实：剪peer边仍13逻辑调用，主要省输入token。single M及独立12/no-edge的完整费用要列明；B−single与费用本轮是描述性支持，不能伪装为新的已控制误差的非劣检验。若绝对成本前沿审核不清楚，则该后续状态为PENDING_COST_FRONTIER_REVIEW，不直接准入。

## 精度与资源停止

四指标多重比较比旧两质量比较建议更保守。若质量配对差方差0.1，N384的正态规划半宽约4.03pp；5pp效应的粗略功效约73%，不是80%保证。信用交互方差更大且未知，不能把384×2×4当样本量3072。小样本/稀疏统计的保守门可能产生INCONCLUSIVE，这是设计的已知限制，不授权后续扩样。

48 GPU小时仅为未承诺的本轮上限建议。正式cap、每卡份额、累计已用预算及准入签名必须在任何benchmark调用前填写；未填写不可执行。本轮专属的新增qualification/profile、模型加载、驻留、失败、盲计时、恢复费用进入本轮及总160小时账。此前独立reservation下的异构/independent12工程资格仍计总160小时，在共享工程栏引用其原始费用，**不得再次累加进本轮48小时**。每笔实际支出以唯一ledger transaction/attempt ID去重；共享工程、候选增量和端到端完整系统归属视图分别呈现，合计只计一次，reservation与实际消耗也不混加。允许先用人工合成输入测工程，但短合成吞吐不能直接外推Hotpot。

若采用benchmark盲前缀，则384题全集先冻结；用无标签、双方tokenizer的question+完整context最大token数排序，等分8个48题长度层，PCG64(261113104)在每层抽1题作为前8个完整unit。其余376题保持原选入顺序。前8题也跑全部11条件且计入正式轮次；不读效果，不能换题或重选样本量。它不是免费qualification。

根进程在前8题闭合后根据真实各节点/模型/长度层耗时、一次性加载、I/O和共享负载作完整剩余成本预测，至少留25%时间余量。若不容纳预先签定cap，停止INCOMPLETE_RESOURCE并保存所有前缀，不改N、不依据质量续样、不扩cap。长度分层8题仍不足以保证尾部耗时；独立子进程与全局/每卡hardcap仍必须有效。

最终只有完整4224终态、384题覆盖、全部账本及源身份闭合后一次性释放评分。完整资源停止条件优先于科学结果；既定384题完成即停止本轮，无中途显著性停止、无结果驱动的新seed、换图或追加任务。

## 待填硬依赖

PROTO.json列出必须补齐的source/model/tokenizer/prompt/scorer/operator/qualification/data-exclusion/selection/runtime/budget pins。候选合同的数字、矩阵和统计代码可以先审查，但这些依赖未齐时不具备执行权限。冻结前可因明确理论/资源理由修订并留版本；不能读R1选择阈值，也不能在本轮结果后修改。
