# Hotpot 评分和固定数组绑定 adapter

状态：CPU 实现，未 freeze，未进行真实评分，未读取真实题、标签、结果或运行模型。三个评分/统计/费用 helper 为 root 指定原字节副本，不修改四个主统计门。最终科学协议 SHA 必须由外部传入；候选 PROTO 的 SHA 不作为准入。

## 接口和权限

`bind_archive(selection_manifest_bytes, files, review_bytes, expected_review_sha256, expected_bindings)` 是纯字节输入的验证入口，不访问题库或标签文件。`files` 必须恰含三份最终 metadata 与384题的三件套。接口没有从路径自动寻找“最近结果”、选择成功子集或修复记录的行为。

- 最终 metadata：`RAW_OUTPUT_MANIFEST.json`、`PARENT_RESIDENCE.json`、`ATTEMPT_RESULT.json`。
- 每题：`worker_{w}/unit_{index:03d}/INPUT.json`、`MATRIX.json`、`UNIT_RECEIPT.json`，匹配当前 runner 接口。
- `expected_bindings`：最终科学协议、selection manifest、plan、source manifest、matrix/graph/independent12/single protocol hashes，以及 scorer/statistics/frontier/selector/evaluation adapter 源码 SHA。
- `review_bytes` 必须匹配调用者独立提供的 expected SHA，明确 APPROVED_FOR_SCORING，并绑定每题三文件、三份 metadata、label vault 原字节 SHA、全部 attempt/物理账闭合，以及所列 native/durable/cache/源身份/无标签采集审查范围。

`REVIEW_RECEIPT.template.json` 默认 UNAPPROVED、哈希 null、闭账 false。它不能开放评分。本模块不证明签署者身份，预期 SHA 必须来自外部可信归档审查，不能由候选结果文件自行提供。模块重新核对单位文件和结构，但不替代外部对全部 native/durable 日志、模型身份和跨 attempt 物理账的独立审计。

`evaluate_after_release(label_loader=..., **archive_arguments)` 先完整调用 bind_archive，成功后才调用 label_loader。loader 接收审核 SHA、vault SHA 和固定384个原始ID，返回独立私有 vault 的 exact bytes。vault 是恰好384个 `raw_id: answer_string` 的 JSON 对象；实际字节哈希必须匹配外部 release。这里仅定义回调接口，没有真实标签文件读取实现。

`score_bound_archive(bound, label_vault_bytes, run_statistics=True)` 为纯评分函数。BoundArchive 只能由 bind_archive 得到，并在评分前核对语义 seal，防止绑定后修改输入、费用或记录。源代码 SHA 也会复核。这是程序完整性检查，不是 Python 进程内抵御恶意调用者的安全边界。

## 数组、失败和费用

行始终按 selection manifest 原始顺序；不按 worker、完成时间或长度前缀顺序重排。验证384个唯一问题和重复组、4边各96、固定两个 seed，以及全部4224个具名条件。

`graphs[384,2,4]` 的最后一轴严格为 `[A,B,A_MINUS_E,B_MINUS_E]`。S1 的 `no_edge`、`independent12`、`single` 各为384元素。条件通过 `S1/...`、`S2/...` 具名key、实际mask、seed、probe edge和协议身份绑定。独立12/single的aggregate通过明确 node别名与slot index定位，不取“最后一项”猜测答案。

只有 COMPLETE 使用原 `hotpot_support.score_aggregate_node`，严格评分规范化 `parsed.answer`；`format_ok=false` 为0，不从 Generation.text、wire或执行级answer字段救答案。INPUT_REJECTED 仅接受冻结 preflight 分类和失败 trace，EM/F1记0并保留该题；INCOMPLETE、NOT_RUN、FAILED、缺终态或源错误拒绝整个评分批次。没有成功子集分析、追加生成或重试入口。

完整逻辑费用从每个13节点架构（single为1节点）逐节点重算，缓存 reused 节点也计入。节点顺序固定为 graph 的 round-major 或 independent12 的 role-major；Q/M归属同时核对固定node路由和实际请求 trace，缓存输出绑定同题、同seed原A/B记录。每个S1的B/NO_EDGE/INDEPENDENT12总Q2304、M2688；single为Q0、M384。成本字段为 calls、input_tokens、output_tokens、CUDA_event_seconds；CUDA event不是GPU驻留时间。

完整费用缺失/冲突不填0；溢出导致相应完整服务成本为null（single从不调用Q，Q维度仍为可证明的0）。frontier使用原冻结 `review_frontier`，容量失败时 NOT_ASSESSED。物理研究账仅保留独立归档pins，不拿缓存节省冒充完整服务费用。

默认调用原 `applicability_statistics.analyze`：四主EM门、bootstrap种子、边分层及F1次分析均不改变。输出含逐题数组，属于 PRIVATE；不在公开报告中附题目、标签、原始回答或逐题数组。

## 优势与限制

优势：显式数组轴、逐级文件和节点绑定、完整逻辑费用与物理账分开，错误不会静默转换为答错。CPU测试使用384个人工条目与完整4224终态，包含A/B不同F1哨兵和真实helper调用。

限制：尚无真实归档/标签接口验证；runner仍在独立冻结流程，最终接口变更需同步复核。本模块依赖外部可信审核闭合native/durable和多attempt血缘；它不根据已有输出自动授予评分或GPU权限，也不改变科学合同。
