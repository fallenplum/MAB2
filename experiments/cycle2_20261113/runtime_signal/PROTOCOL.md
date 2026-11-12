# 执行条件复检信号：历史数据必要条件诊断 v1

研究假设：G2 的静态图/题目特征遗漏已经可观察的求解状态。首轮四名求解者
是否给出不同答案、谁处于少数、消息格式及长度，可能帮助识别“翻边后最终
正确性会改变”的情形。预测器仍是工具，不作为算法创新。

这里只检验这个必要条件，不声称单seed非零信用就是可靠总体信用、历史信用
失效或未来图编辑收益。已有历史格式关联已经被阅读，故属于新冻结的探索性
重分析，绝非盲法确认或 D2/D3 结果修复。

## 数据和信息预算

只使用原 train200/tune80；全部旧heldout120排除分析，R1 I/C/S和封存test不读。
旧D2输入SHA256为7639dad2581ea52536b7c274d33fd019ad726a9bce9e8e4ec6792c68b77f4963。
从已冻结历史index按首次出现顺序为每个(qid,seed)选择一个执行文件，选择不看
quality。最多1400个文件。只提取r1:w0..w3，绝不将r2或aggregate用于特征。
首轮跨图不变性已有逐图历史审计支持；本次绑定其索引、审计和文件哈希。

每题2个anchor×12个edge×5个seed。edge-off与edge-on仅相差所测边。
目标为1[Q(on)!=Q(off)]；原评分器和旧已揭示分数不变。整个question是分区和
bootstrap单位，不能把24个图背景或5个seed当作独立样本。
首轮4调用对全部方法共用且必须计费；每个反事实配对在首轮缓存基础上最多
7新调用，6个配对保守上界46调用/题/seed与12次评分。这里不声称实际token相等。
真实缓存共享、所有训练/初始化费用和未来服务成本必须在新实验另行核对。

## 固定特征与模型

图特征：edge-off的12bit，目标边12维one-hot，目标接收者入度和发送者出度。
运行特征：每名首轮worker的格式正常、数字可解析、log1p输出token、
log1p消息token、非空uncertainty；6个worker对的可解析数字相等指示；
目标源/接收者上述特征；二者合法且答案不同；各自与其他worker的答案一致比例；
接收者与edge-off现有入邻居的合法答案分歧比例。
数字规范化复用原scorer.numeric，不调用gold，不保存明文答案或原题到特征。
缺失数字不算相等，也不冒充意见分歧；缺失/格式标记独立保留。

固定 HistGradientBoostingClassifier：learning_rate=.05,max_iter=100,
max_leaf_nodes=15,min_samples_leaf=100,l2_regularization=1,early_stopping=False,
random_state=94173；按train正负类频率给反频率权重。无超参搜索、无tune早停。
预测值仅用于排序，不声称概率校准。

三次固定拟合：graph_only（图特征）、full（图+运行特征，唯一候选）、
runtime_only（运行特征消融）。runtime_only不能在full失败后成为替代winner。
固定启发式baseline：sender_receiver_disagreement、receiver_incoming_disagreement、
source_uncertainty（非空1，空0）、inverse_receiver_degree=1/(1+入度)、uniform。

## 固定评价与投入门

每个(question,seed)在24个背景/边配对中选6个。并列边界按均匀随机选择的
精确包含概率求期望，不用边号破同分制造差异。主指标为所选配对中非零信用比例，
先平均5seed，再平均80题。AUC/AP只是辅助，全局难度信号不能替代组内采集收益。

所有策略输出先写入带hash的冻结预测文件，然后评估tune目标，不回传训练。
比较full与graph_only及全部5个固定启发式baseline。唯一候选的投入门全部满足：

1. 80题完整，非零信用覆盖至少20题，train两类都有支持；
2. full−uniform >=.02；full−最强固定baseline >=.01；
3. 10000次按question整体bootstrap（seed94174），每次重新求最强固定baseline
   的均值；full减此均值的90%区间下界严格>0；
4. full相对最强固定baseline的逐题差值，严格正值至少20题。

上述只是资源筛选门；不作确认性p值、多重假设阳性、未来决策或ACL达标主张。
门失败保留全部模型、预测、分母和负结果，只结束这个实例，项目继续。
门通过最多进入独立“运行信号驱动的信用复检”协议设计，不自动启动GPU。

## 原理边界、优势与风险

优势：不增加LLM特征提取调用；直接测试目前静态预测器可能遗漏的执行信息。
风险：分歧可能仅是难度代理；标签稀疏；非零效果不等于可预测效果方向；
不同anchor背景不能拼成同一张当前图，所以本诊断不能报告“最佳图编辑准确率”。
只有新的同图动作比较、完整成本、独立未来题，才能检验核心科研方法。
ACTS (2610.25851)、CATS (2610.29292)、Graph-GRPO
(https://aclanthology.org/2026.findings-acl.1010/)已有执行状态、归因和拓扑优化，
因此执行分歧与普通分类器本身不构成独立新颖性。

本次新增GPU小时0、正式候选轮次0。CPU和原始读文件成本记录。运行前提交本协议。
