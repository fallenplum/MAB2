# G2条件信用预测实验

本目录保存G2的准备、无标签采集、资源监督和CPU预测评估入口。原始系统与信息权限沿用项目执行合同，冻结执行器和评分器位于仓库的 `src/graph_credit`。[执行合同](../../protocol/EXECUTION_CONTRACT.md)

## 设计与状态

G2使用原400道校准题，按题划分为训练200、调参80和heldout120。每题预抽p=.25与p=.75两个独立锚图，覆盖12条边的全部单边翻转，使用5个节点种子。当前设计共2,000个题目与种子任务、51,970个去重图输出，正式生成调用上界为124,000次。8卡真实资格各至多89次调用，工程费用单列。

冻结MiniLM表示经过训练题拟合的PCA8后增加常数维。二阶模型有711维，一阶模型有117维。F训练按唯一题目与图聚合5种子质量，每个图标签加入一次，每题总权重为1。无条件边均值、一阶模型和二阶模型共享预声明的信息来源。

主指标是heldout题上的无条件边均值风险减二阶模型风险，使用10,000次整题paired bootstrap。完整性门通过且95%区间下界大于0时，科学门为 `G2_PASS`。一阶与二阶的比较另行报告，负风险估计与负方法差均保留。

当前公开材料对应无标签准备与软件验证阶段。真实资格、采集、预测冻结和评估各自形成独立状态与证据。

## 两阶段执行与标签门

第一阶段完成无标签生成。`prepare.py` 的 `design` 模式生成私有设计草案，`features` 模式建立训练题拟合的表示与特征收据。受控冻结步骤核对全部来源和特征字节后，生成正式 `frozen_plan.json` 与旁置的 `PLAN_FROZEN.json`。

`supervisor.py` 依次运行候选GPU资格、完整采集和全局输出核验。子进程使用明确的物理GPU掩码与 `cuda:0`，显存不足时等待准入。全局输出冻结后，监督器状态为 `WAITING_TRAIN_TUNE_LABELS`。

```sh
python experiments/g2/supervisor.py --root PROJECT_ROOT --plan PLAN_JSON --gpus 0,1,2,3,4,5,6,7 --python PROJECT_PYTHON
```

第二阶段在完整原始产物所在的根目录执行CPU工作流。受控发布器先验证全局冻结，再发布train与tune标签。预测阶段拟合模型和选择alpha，随后冻结参数、2,880行heldout预测及来源哈希。

```sh
python experiments/g2/workflow.py --root PROJECT_ROOT --plan PLAN_JSON --mode predict
```

缺少train与tune发布凭据时，工作流返回 `WAITING_FOR_TRAIN_TUNE_LABEL_RELEASE`。预测冻结后的状态为 `PREDICTIONS_FROZEN_AWAITING_HELDOUT_LABELS`。发布器核对该冻结凭据后，才发布heldout标签。

```sh
python experiments/g2/workflow.py --root PROJECT_ROOT --plan PLAN_JSON --mode evaluate
```

完成状态为 `G2_EVALUATION_COMPLETE_REVIEW_REQUIRED`，科学结果为 `G2_PASS` 或 `INSUFFICIENT_PREDICTIVE_GAIN`，随后进入独立审查。每次阶段调用同时保留CPU尝试和标签访问事件。进程退出码需要与这些产物共同核对。

实际运行依赖本地已验收的G1产物、合法取得的模型与编码器、校准数据和受控标签发布。全局输出、模型预测与标签发布凭据必须属于同一正式计划。完整工作流会验证原始unit、执行引用与费用账本，轻量元数据镜像用于查看状态。

## 运行软件测试

在仓库根目录、项目Python环境中运行以下命令。CPU测试使用NumPy与psutil，执行器和评分器从仓库 `src` 导入，模型输出由合成后端提供。

```sh
python -m unittest discover -s experiments/g2 -p 'test_*.py' -v
```

四份测试保留采集、预测器、监督器和工作流的回归断言。存在原暂存参考目录时，测试继续使用该处执行器或评分器。其他布局使用仓库 `src/graph_credit/executor.py` 和 `src/graph_credit/scoring.py`。

`test_real_draft_question_to_partition_map_and_five_seed_coverage` 需要在本地生成的私有 `DESIGN_DRAFT.json`，并与测试文件位于同一目录。缺少该文件时，仅此案例明确skip。其余合成实体、数学断言和阶段门回归继续执行。

项目本地完整G2组件验证已运行60项CPU回归，包含本目录之外的镜像组件回归，全部通过且0项skip，其中上述案例使用真实无标签计划。本目录四份测试共有45项，公开布局预期为44项通过、1项skip。真实GPU资格与科研结果分别由各阶段产物确认。

## 来源字节与可复现性

`pipeline_source_manifest` 与原始源码清单绑定目标运行目录中的实际文件字节，正式计划保存相应SHA。GitHub文本可按LF换行规范化，因而仓库文件哈希可能与已有运行目录的字节哈希不同。

新的正式运行应在目标环境重新生成计划、核对来源与模型文件并完成真实资格。已有运行的分析与恢复继续使用其原冻结文件和凭据。模型版本、Python依赖、表示文件和完整执行来源均通过正式计划核对。

测试、代码、协议与聚合汇总可公开。题目、私有计划、标签、逐题输出、权重、访问凭据和运行主机信息由本地实验目录保管。
