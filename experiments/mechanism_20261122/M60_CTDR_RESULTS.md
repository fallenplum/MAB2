# M60 CTDR 独立复现实验结果

- 正式面板：QASC-A、QASC-B、SciQ-A、SciQ-B，每个 192 题，3 个模型团队，共 2304 个题-团队单元。
- ARC 无证据面板保留为 null control，不纳入正向机制聚合。
- 评价：图编辑后的 action set 与新图完整证书的一致性，反馈成本使用增量逻辑 token。

| 方法 | action error | exact set | 增量 token |
| --- | ---: | ---: | ---: |
| Stale | 0.031 | 0.931 | 447.6 |
| CTDR | 0.000 | 1.000 | 98.2 |
| Full restart | 0.000 | 1.000 | 493.2 |
| Max uncertainty | 0.069 | 0.826 | 104.3 |
| Periodic | 0.069 | 0.818 | 113.3 |
| Random | 0.069 | 0.821 | 105.3 |

## 预注册门核对

1. CTDR 在四个正式面板的每个团队均达到零 action error，且相对 stale 与 max-uncertainty 满足至少两个团队条件。
2. 相对同增量预算的 max-uncertainty，CTDR 的 aggregate error 为 0.000，对照为 0.069。
3. CTDR 与 full restart 的完整 decision set 一致，按面板平均的增量 token 节省为 0.796，按汇总 token 比值为 0.801。
4. 去除 action-margin transport 后退化为 stale，四个正式面板均保留非零 stale error，消融门通过。

修正后的 worker 将无证据状态表示为空字符串。修正前的运行保留在私有目录中，不计入正式结果。
所有数字均由私有 worker 输出和封存标签离线重放产生，未把封存标签用于算法选择。
