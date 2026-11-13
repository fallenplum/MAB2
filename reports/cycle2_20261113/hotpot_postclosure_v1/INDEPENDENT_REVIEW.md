# Postclosure 标签权限流程独立审查

日期：2026-11-13。范围：只读作者 README、postclosure.py、CLI、人工测试与冻结 evaluation_adapter 相关代码，并执行独立人工 CPU 权限边界测试。**未读真实题目、collector 输出、答案、supporting_facts 或当前结果，未运行模型/GPU。**

## 结论

**未发现阻断当前工程准备的漏洞。** 当前 postclosure.py 的 SHA256 为 `8f701f52ac7fa15ea39f55f2beb877ce400774cc88484e7280c45da322ccb544`，与 root 委托版本相同。独立人工测试 13 项通过，0 失败、0 错误；覆盖 25 个提取拒绝情形和 14 个评分拒绝情形，相关 reader 均零调用。作者 13 项测试没有被当成本次独立测试再次运行。

本结论仅批准工程准备的审查结论，**不授予标签提取、评分、论文结论、发布或新增 GPU 权限**。真实运行仍需完整外部归档审计与 root 的两份独立批准 hash。

## 可确认的顺序与绑定

1. 提取入口先验证独立传入的批准文件 SHA、精确提取权限、wrapper 源码 SHA、五份 evidence 原字节，以及真实 witness 声明、完整覆盖、尝试/驻留闭账和固定来源身份。该验证返回前不调用 reader。
2. 第一阶段只提取已冻结的 384 个 ID 的字符串答案，禁止额外列和重复/缺失答案，输出独立 vault SHA 与提取回执；没有评分授权。实际默认 reader 按完整分片 bytes 校验来源，并请求 id/answer 投影。
3. 第二阶段要求新的 APPROVED_FOR_SCORING 原字节 SHA。它必须由原 template 仅改变 decision、vault pin，并追加规定的提取回执/批准/audit/template/wrapper 血缘构成。修改 unit、科学字段或 vault pin 会拒绝。
4. wrapper 调用未经修改的冻结 `evaluation_adapter.evaluate_after_release`。后者先完成 `bind_archive` 的 384 题/4224 条件及文件绑定，再调用 `label_loader`；wrapper 的 guarded loader 再核 request 的批准/vault/ID 身份，读取后检查 vault 原字节 SHA。
5. CLI 默认 preflight 不打开 IO 描述符。显式流程拒绝覆盖已存在输出目录；失败后可能留下私有空目录/部分文件，这不是成功，不允许自动重试或复用。

## 独立人工测试

测试代码：`reviewer_tests.py`。回执与日志：`run_001/RECEIPT.json`、`run_001/CPU_TESTS.log`。

| 检查族 | 证据 |
|---|---|
| 首次 pin 缺失/错误、批准或 evidence 字节改变 | reader 零调用 |
| 重新签署的人工 witness、0/1/字符串布尔别名 | 仍在 reader 前拒绝 |
| 重新完整绑定的缺题/缺终态/缺 unit/未闭账/运行失败 | reader 零调用；不是仅依赖旧 hash 不匹配 |
| 重新完整绑定的错误 split、revision、原描述符投影 | reader 零调用 |
| 提取权限扩张、错误 wrapper pin、重复 JSON 权限键 | reader 零调用 |
| 合法人工提取 | 一次人工 reader 调用，384 个原字符串保留；评分/发布授权仍 false |
| 缺失/错误第二次 pin、用第一次 pin 替代、错误权限阶段 | vault reader 零调用 |
| 重新签署评分文件但改变科学字段、unit、提取回执、vault 或未知费用确认 | vault reader 零调用 |
| 真实冻结 adapter，空 archive | vault reader 零调用 |
| 真实冻结 adapter，人工三份 metadata 合法但缺 unit_000 INPUT | 确实运行至 unit 文件绑定才拒绝，vault reader 零调用 |
| 真实 adapter 源字节被追加换行（仅修改独立临时副本） | 导入/调用标签前拒绝 |
| 默认 CLI 带不存在的 IO 路径 | monkeypatch 禁止 Path.read_bytes 后仍可完成 preflight |

测试复用了作者的**人工 metadata fixture 构造器**，独立编写重新绑定、篡改和零调用断言；没有用作者测试结果替代断言。没有重建完整 384 题模型执行 mock，也没有宣称独立完成完整成功评分回放。临时 adapter 副本只在本审查目录创建与删除；被审源码执行前后 SHA 一致。人工成功路径中的答案全部是测试字符串。

## 必须保留的信任边界

- 两个外部 SHA 的可信性来自 root 独立决定。程序无法从自报 JSON 或一个自行算出的 hash 认证操作者身份；人工 fixture 声称 witness=false 并不证明存在真实 witness。
- 提取入口信任 root 已复核的外部 AUDIT，它不重做 native/durable/账本或 tokenizer 真实检查。当前工程测试不能为仍运行的正式采集签闭账。
- wrapper 核对 adapter 原字节；adapter 的 helper 实际源文件在后续评分加载时才校验，bind 阶段只核 helper 固定 pin 的一致性。因此 README 中“root 对整个部署包 manifest 外部冻结”的条件必须执行，不能只冻结 postclosure.py 后忽略 helper/CLI。此处是已明确的部署前提，不将不满足该前提的部署称为本次通过。
- Parquet hash 会读取整分片不透明 bytes，Arrow 可能暂时解码其他行的 id/answer；只保证返回 Python 的集合恰为选中 ID。不能写成行级字节隔离。
- Windows 私有 ACL 由 root 提供。POSIX mode、哈希和不可覆盖文件不构成对恶意并发替换或进程内任意代码的安全沙箱。
- 本审查不改变容量失败的冻结评分规则、统计参数、科学门或资源合同。

## 优势、风险与置信度

优势：权限阶段清晰；原字节及语义绑定兼顾；使用已有冻结评价器而未另写评分。人工篡改中可区分“hash 不一致”与“重新绑定后仍因语义不合法拒绝”。

风险：真实闭账与来源可信性尚不在本次证据范围内；完整成功归档评分依赖冻结 adapter 的既有审计，不能由本次缺档测试推出全部统计正确。

置信度：对当前受审源码的拒绝顺序及人工输入下零 reader 调用为**高**；对未来真实数据/部署的正确性不作概率声明。下一步是待真实归档闭合后由 root 核 audit 与 manifest，再分别决定两阶段权限；本文件不自动触发该步骤。

