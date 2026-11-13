# 冻结评价后的聚合报告生成器：人工数据准备包

本目录只做 CPU 展示准备，未读取当前 Hotpot 题目、原始生成、真实标签或成绩，没有运行评分器、bootstrap、模型或 GPU。实际接入前需 root 独立审阅本包。本包不授权标签、评分、发布或新科学实验。

## 接入方式

从本目录运行，保留同目录 `FROZEN_REPORTING_PLAN.md`。正式入口只读取已完成冻结评价的 JSON 以及两份元数据回执，不接收原题、原始矩阵或标签路径。

```text
python -B report_cli.py
python -B report_cli.py --synthetic --output synthetic_preview_001
python -B check_preparation.py --output cpu_run_new
```

默认入口为 `PREFLIGHT_NO_EVALUATION_OR_LABEL_IO`。人工模式只使用内存 fixture，拒绝同时传入真实输入参数；输出路径须含 `synthetic`。所有演示报告、CSV、JSON 和图均标明 SYNTHETIC，演示不是正式科学结果。fixture 中主区间和支持数是手工构造的 schema 测试值，故意不由人工数组重算，不能把它们解释为该数组的一次 bootstrap。

以下命令仅为将来实际接入方式，本次未执行：

```text
python -B report_cli.py --run-released --evaluation PRIVATE_EVALUATION.json --evaluation-receipt EVALUATION_RECEIPT.json --receipt-sha256 ROOT_TRUSTED_RECEIPT_SHA256 --score-review SCORE_REVIEW.json --score-review-sha256 ROOT_TRUSTED_SCORE_REVIEW_SHA256 --output report_run_new
```

两个 SHA 必须由 root 从独立已解封链提供，不能让本工具自行选择或自签。CLI 先仅作文件身份检查，拒绝 evaluation 与两份元数据路径指向同一文件，包括操作系统可识别的别名；此时不读任何文件内容。入口随后验证评价回执 SHA、冻结 adapter/postclosure 身份、完成状态、score review SHA、`APPROVED_FOR_SCORING`、384/4224 全量闭合和冻结 helper pins，再调用唯一 evaluation reader。读取后继续核评价 SHA、评估/review bindings 和物理账本元数据 pins。测试包含无授权或坏 hash 时 reader 调用次数为零。本工具实现 hash 绑定，并不能认证传入 SHA 的人是否真是 root；外部可信签发及输入文件在执行期间不被并发替换仍是运行前提。

输出目录必须不存在。失败可能留下部分报告，应保留并使用全新目录重跑，不能把无 `private/REPORT_RECEIPT.json` 的目录当完整交付。主命令只输出状态和文件数；异常只输出异常类，避免将输入字符串写到终端。

## 字段与解释边界

| 内容 | 处理方式 |
|---|---|
| 11 条件 EM/F1、输入拒绝数 | 对已绑定 384 题数组描述性求均值，完整固定顺序，拒绝保留冻结置零 |
| D1-D4、原区间、支持数、L1/L2/action | 直接从 `statistics.primary_EM/decision` 投影，不重算主估计、区间、门或前沿 |
| 全局格式失败 | 原样列全局计数；逐条件格式失败与 finish reason 为 NA |
| 有符号信用 | 按冻结 gamma_A、gamma_B、delta 公式作描述；EM/F1 各 seed 负/零/正计数，EM 固定离散直方图及跨 seed 乘积保留负值 |
| 四方法的 S1 完整服务逻辑费用 | 原 `full_logical_costs_by_model` 按模型投影，包含复用的逻辑节点；未提供/未知为 NA |
| 其它七条件逻辑费用 | NA，不外推 |
| 物理调用、缓存节约、驻留、初始化、等待和失败费用 | 数值均未由冻结评价提供，单独列 NA；相关 hashes 留私有回执 |
| 长度分组 | 输出无该字段，不造分层 |

有符号分布仅是预定描述，不替代 D3/D4。F1 的零按原浮点值精确比较，不调容差。各题等权、四边原始分配各 96，不选子集。CSV 中 NA、JSON 中 null 均指不可用，不指零或未发生；真实给定的零仍为零。

前沿只引用冻结的 observed-point status，不作总体支配声明。B 仍是唯一通信候选，A/F1/单 seed/单 edge/长度层不能救援。当前 B 路线失败不等于项目停止；即使原门通过，也不自动授权新机制实验，不证明主动反馈胜随机或异构因果优势。

**限制：本包没有覆盖全部成本。** CUDA 事件秒不是物理驻留，不估算 GPUh；以后 root 可在既定报告计划范围内另做已解封 raw/外部审计描述性补充。本工具不为补图改变 adapter。

## 输出与隐私

每次完成输出分为 `public/` 与 `private/`。`public/` 恰含 16 个聚合文件：

- `REPORT.md`、`SUMMARY.json`；
- `conditions.csv`、`primary_contrasts.csv`、`signed_support.csv`、`em_credit_histograms.csv`、`logical_service_costs.csv`、`physical_research_costs.csv`；
- 四个 `figure_01` 至 `figure_04` 的 PNG 和 PDF。

公开产物只含固定标签、聚合数和固定说明，不复制 qid、来源 hash、prompt、answer、逐题向量或私有自由文本。图像只接收这个小聚合字典；正文有额外 hash/标识扫描，测试还检查 PDF 文本与人工隐私哨兵。`write_reports()` 是 CLI 完成白名单投影后的内部输出函数，不是接受任意外部 summary 的独立净化边界；真实使用应走上述 CLI。该约束是字段白名单，不是对任意匿名数据的形式化隐私保证。

`private/REPORT_RECEIPT.json` 保存输入与源码 hash、物理账本 pins、公开白名单 hash；`private/mplconfig/` 是字体缓存。它们不发布。测试日志可含本机路径，仅私有保留。预览子目录也只是人工展示；本轮未发布任何文件。

## 源码与证据

冻结报告计划副本 SHA256：`fc1ec977b51a08ba19b39236409ed1c3361f3f58a348cc6e33df4e050f6a4d31`。

冻结评价 adapter SHA256：`f8ba38be9ce1ef6bbfee66c17606685b117d25c52b947e66937ea5574a6105b8`。

冻结 postclosure SHA256：`8f701f52ac7fa15ea39f55f2beb877ce400774cc88484e7280c45da322ccb544`。

本报告代码、依赖 helper 的精确 SHA 见 `SOURCE_MANIFEST.json` 与私有运行回执。仅报告计划作为原字节副本随包提供；运行不导入冻结 scorer/statistics/adapter，只检查其已绑定身份。最小部署源为 `reporting.py`、`plotting.py`、`report_cli.py`、`FROZEN_REPORTING_PLAN.md`；人工测试另需 `synthetic_fixture.py`、`test_reporting.py`、`check_preparation.py`。

Python 使用已存在 numpy/matplotlib；PNG 300 dpi，PDF 保留矢量与 TrueType 字体。测试另用 Pillow/pypdf，PDF QA 使用已有 PyMuPDF。未安装或升级依赖。`FIGURE_CONTRACT.md` 记录图表单位、范围与缺项；`QA_REPORT.md` 记录真实人工测试和视检。

优势是主判据保真、完整条件展示和显式缺项；风险是当前 adapter 没有全部格式/结束原因/物理成本字段，报告不能定位因果来源或补齐实验账本。对已覆盖代码路径的结论置信度高，依据是人工权限/聚合测试和渲染核验；正式数据适配尚待 root 在解封后验证。
