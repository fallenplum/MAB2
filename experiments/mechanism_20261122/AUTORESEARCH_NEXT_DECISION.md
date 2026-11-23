# AutoResearch 当前候选路线决策

## 当前证据

- Qwen-0.5B 的真实单条 forward 服务测量显示，TCR guard 在 rows1/2 使用 1 次 relay 调用，answer-region q=1 使用 1.31 次。rows4 分别为 3.00 和 3.02 次。两者在单条调用口径下均达到 action exact 1.0，answer-region 没有降低实际 relay forward 成本。此前微批量结果造成的 action 差异已排除。
- Qwen-14B 的 calibrated route 在 rows2 独立面板上只触发 1 次 relay forward，128 题平均 0.0078 次调用，decision/action exact 均为 1.0。该结果只覆盖 rows2，不能外推到更高编辑密度。
- 语义接口漂移面板覆盖三类模型和三个任务族的九个单元。TCR guard 在所有已完成单元达到 exact-set 1.0。stale 与 warm reweighted 在多个单元失效，说明接口变化需要进入复检依据。
- 隐藏接口面板显示，事实编辑掩码会漏掉未声明的接口变化。加入接口指纹后，三种模型的 answer-action exact 均恢复到 1.0，但 Qwen-14B 仍有一个 edge-decision 边界错误。
- 独立正交消融的主要节约来自 all-row 校准中未观测行的零漂移假设。可信 changed-row mask 下，row-specific radius 和 source-aware priority 的额外收益较小。
- 本地历史三模型 route selector 的低调用版本在 Qwen-14B 服务面板上出现 action exact 0.78--0.88，说明仅按 relay 行数选择路线会漏掉需要补充条件核的情况，当前不能作为主算法。

## 研究判断

当前不能把 answer-region 作为跨模型普适算法。更稳健的候选是按开发集选择复检路线，低密度或小模型优先使用图后代掩码，高密度或接口不确定时启用 answer-region 和接口指纹的保守路径。该路线必须在同一模型、同一编辑密度的开发划分上选择，再在独立问题上评估。当前真实服务证据只证明 Qwen-14B rows2 存在低调用路线，尚未证明跨密度收益。

## 下一轮低成本实验

1. 用单条 forward 口径完成 Qwen-14B rows1/rows4 的 service 对照，至少保留 stale、TCR、answer-region q=1 和 calibrated route 四个结果。
2. 在 Qwen-0.5B、Qwen-14B 和 Llama-1B 上运行同一套 route selector，输出开发选择与独立测试结果，记录 forward calls、输入 token、wall time、decision exact、answer-action exact 与 answer accuracy。
3. 对 semantic、formatting 和 metadata 三类接口漂移使用相同 route selector。接口指纹缺失时强制 all-row fallback，并单独统计调用增加量。
4. 如果 calibrated route 在两个模型和两个编辑密度上同时达到 action exact 1.0 且低于 TCR 的 forward/token 成本，再进入论文主方法候选。否则冻结该候选，转为机制边界和失败条件论文路线。

## 停止条件

在单条 forward 口径下，若 calibrated route 不能在至少两个模型、两个编辑密度上同时保持 action exact 1.0 并降低强 baseline 成本，则不再扩展 answer-region 结构，只保留接口指纹导致信用失效的机制结果与负结果。

