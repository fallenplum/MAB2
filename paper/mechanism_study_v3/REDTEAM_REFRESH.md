# Pairwise Gap Certificate v3 终审刷新

## 范围

本审查覆盖当前 `main.tex`、Pairwise 27-panel replay、独立 ARC-Challenge 队列、Qwen-0.5B service、匹配行序基线、图条件半径消融、M60 负消融和当前已生成的 PDF。此前 `REDTEAM_FINAL.md` 中关于“缺少通用行序基线、未进入主稿的半径消融、未生成 PDF”的部分已经被当前版本修复。

## 当前判定

**仍为 `CONTINUE_RESEARCH / NOT MAIN-READY`，但剩余阻塞已从方法缺失缩小为冻结协议与复现交付问题。**

当前主线已经有足够清晰的机制候选。27 个 panel 的 pooled $q=.9$ replay、独立 ARC 队列、匹配行序对照、图条件半径对照和历史 CTDR 负消融共同支持“图条件信用证书传输”这一范围明确的研究方向。当前版本仍不能直接宣称 ACL Main-ready，原因如下。

## 已修复的问题

1. 主稿已切换到 Pairwise Gap Certificate，并给出 TV pairwise gap 上界和条件命题。
2. 主稿已加入 random、periodic、maximum-uncertainty 的匹配行序控制。结果显示 Pairwise 与 maximum-uncertainty 在许多 cell 上同成本，这个负结果应保留，并支持把 gap-per-token 排序降级为实现细节。
3. 主稿已加入独立 graph-conditioned radius ablation，并明确这是 answer-region controller 的机制层对照，不把它误写成 Pairwise scheduler 的直接因果消融。
4. 历史 Full CTDR 的负模块消融已进入正文，避免把旧路线的精确缓存收益归因给 drift radius 或 generic ranking。
5. Server 上已生成并通过视觉检查的 5 页 ACL 风格 PDF，表格重叠已修复。
6. Qwen-0.5B service 的指标已限定为 cached source/base/terminal 之后的 relay 增量成本，Qwen-14B service mismatch 也已从主 claim 中排除。

## 当前 P0 阻塞

### 1. ARC worker 的 q 协议仍未统一

正文目前报告的 ARC worker 25.2%、16.2%、11.5% 使用队列内 calibration/selection 选出的 $q=.5$，而 ARC service 使用预冻结 pooled $q=.9$。正文虽已披露差异，但摘要和 transfer narrative 仍容易被读成一个统一部署协议。

该问题已有低成本修复路径。使用现有 ARC 原始 worker JSONL 离线按 pooled $q=.9$ 重放得到：

| edit density | action exact | Pairwise tokens | fair mask tokens | reduction |
|---:|---:|---:|---:|---:|
| rows1 | 1.000 | 146.33 | 151.38 | 3.3% |
| rows2 | 1.000 | 297.71 | 302.75 | 1.7% |
| rows4 | 1.000 | 574.25 | 599.50 | 4.2% |

这些结果来自现有 73/31/24 ARC split，不使用新增模型推理。应将 `ARC_Q09_WORKER_REPLAY.json` 固定为正式结果，把 ARC worker 主结果改为 pooled $q=.9$。若保留 $q=.5$，必须把它明确降级为队列内开发校准诊断，不能与 service 主协议合并。

### 2. ARC 数据版本尚未固定到可复现协议

协议只记录 `allenai/ai2_arc` validation split 和“前 128 个四选项记录”，没有 Hugging Face revision、Arrow 文件哈希或公开 question-key manifest。数据排序变化会改变 128 题和所有结果。提交前必须在协议和复现说明中固定 revision 或本地 Arrow SHA-256，并说明 unique questions=128、每个 density 使用同一 128 题的配对评测。

### 3. 论文中 service cohort 的命名仍应更严格

正文已经区分原始 Qwen-0.5B service 和 ARC service，但摘要只写“actual Qwen-0.5B service trace”。应在摘要或结果表中标记 `original Qwen-0.5B cohort`，并把 ARC service 的 .992/.961/.859 明确称为 worker-agreement diagnostic，而不是 service exactness 或质量保证。

## P1 风险

- graph-conditioned ablation 的 `decision_exact=0.9968` 与 `answer_action_exact=1.0` 应保持两个指标分开，不能用“完整决策正确”概括。
- 27 panel 和 ARC density 是重复因素，不应解释成 27 或 384 个独立数据集。ARC 需明确 128 个 unique questions。
- hidden-interface 证据覆盖 worker replay，不能扩展为跨模型 service latency。Qwen-14B service mismatch 仍应留在限制中。
- pooled empirical radius 和 split-conformal 结果都是条件性证据。不得写成无条件安全保证或一般性有限样本保证。
- 当前 Pairwise 与 max-uncertainty 许多 cell 同成本，不能把 gap-per-token order 作为已证明的独立算法创新。主贡献应聚焦图条件有效性、fingerprint gate、fail-closed transport 和 conditional answer-action certificate。
- service ledger 仍只覆盖 relay 增量成本，不覆盖模型加载、队列、source/base/terminal 完整端到端成本。

## 通过的证据门

- Pairwise gap 上界和实现一致，条件命题假设已写明。
- 27-panel pooled $q=.9$ held-out action exactness 为 1.0，且 test residue 未用于选择。
- 现有 ARC raw worker trace 可在统一 pooled $q=.9$ 下离线重放并保持三种 density action exactness 为 1.0。
- 独立 graph-conditioned radius control 与 pooled-radius control 具有相同 answer-action exactness，且 relay token 差异可复现。
- matched row-order controls、hidden-interface replay、Qwen-0.5B relay service ledger 和 M60 negative ablation 均有对应 JSON/Markdown 工件。
- 论文已在服务器编译为 5 页 PDF，且本地渲染检查通过。

## 终审建议

先把 ARC worker 统一到 pooled $q=.9$，固定 ARC 数据 revision/hash 和 unique-question 口径，再重新编译并检查摘要、表格和正文中的百分比。完成后应重新审查 novelty boundary 和 service-cost wording。若这些修复全部落地，证据可支持范围明确的机制型 ACL 主会投稿主线，但仍不能声称一般多模型任务质量提升或无条件安全保证。


