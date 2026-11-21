# 任务语义变化与图变化的可辨识性

2026-11-21 的定向检索覆盖成对代码需求、API 更新和需求切换。检索词包含 EvoEval、CodeEditorBench、CodeUpdateArena、requirements change、paired version。通过 arXiv 论文、作者仓库、会议论文记录交叉核对。Google 搜索结果页未能直接打开，补用了网页检索与原始页面。覆盖面限于已核对的三项工作。

| 研究 | 已有能力 | 对 MAB2 的约束 |
| --- | --- | --- |
| Xia、Deng、Zhang，EvoEval，COLM 2024 | 从 HumanEval 派生需求变化和语义等价改写，作者文档定义 parent 字段 | 可以据此检验版本谱系。公开历史、衍生任务和父题高度相关，不能把不同版本计作独立样本。[论文](https://arxiv.org/abs/2403.19114) · [作者定义](https://github.com/evo-eval/evoeval) |
| Guo 等，CodeEditorBench | 将需求切换列为代码编辑任务，另含修复、翻译与优化 | 应比较拿到相同旧程序和新需求的普通编辑器。单纯支持需求变化不足以构成创新。[论文](https://arxiv.org/abs/2404.03543) · [作者仓库](https://github.com/CodeEditorBench/CodeEditorBench) |
| CodeUpdateArena | 将合成 API 更新与依赖该更新的程序合成任务配对 | 将任务世界的变化和模型更新区别于通信图边的变化。API 更新上的成功不能直接归因于边信用维护。[论文](https://arxiv.org/abs/2407.06249) · [作者仓库](https://github.com/leo-liuzy/CodeUpdateArena) |

在可观察条件完全匹配时，只有跨图的复检排序优于通用不确定性、随机复检和同成本直接修复，才支持 MAB2 的机制主张。D27 只核对分组来源，后续推理仍需要独立协议、历史暴露核查与可负担的完整成本。
