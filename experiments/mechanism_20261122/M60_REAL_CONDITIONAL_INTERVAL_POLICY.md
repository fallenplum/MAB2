# 真实条件漂移区间迁移诊断

在问题切分的训练部分估计每个编辑行的最大历史 TV 漂移，再应用条件漂移区间。未被区间确定的样本查询全部图编辑行。

| 任务 | 方法 | decision exact | answer accuracy | token |
| --- | --- | ---: | ---: | ---: |
| QASC | warm / max-uncertainty / conditional interval | 98.41% / 98.41% / 98.41% | 26.98% / 26.98% / 25.40% | 0 / 99.48 / 230.33 |
| SciQ | warm / max-uncertainty / conditional interval | 95.65% / 95.65% / 100% | 65.22% / 62.32% / 52.17% | 0 / 89.48 / 206.99 |
| CommonsenseQA | warm / max-uncertainty / conditional interval | 95.83% / 100% / 100% | 43.75% / 45.83% / 52.08% | 0 / 90.75 / 209.75 |

该迁移结果说明，条件漂移区间能在两个任务上恢复 finite decision exactness，但查询成本约为最大不确定性基线的两倍，且答案质量方向与 decision exactness 不一致。QASC 的训练最大 TV 半径在留出题上仍出现少量漏判，因此经验半径不能替代理论上界。该候选不晋升为主方法。

