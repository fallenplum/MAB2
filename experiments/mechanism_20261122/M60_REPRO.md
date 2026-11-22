# M60 CTDR 复现说明

正式机制面板使用 QASC-A、QASC-B、SciQ-A、SciQ-B。每个面板 192 题，使用三个固定模型团队和相同提示模板。ARC 面板作为无证据 null control 保留在私有目录，不计入正式正向聚合。

## 运行入口

- `m60_mcqa_worker.py` 是服务器执行的无答案键 worker。
- `ctdr.py` 实现图编辑后的证书传输和阈值边界判定。
- `M60_CTDR_RESULTS_SUMMARY.json` 是脱敏正式汇总。
- `M60_ABLATION.json` 保存无传输和无选择排序的消融。
- `paper/main.tex` 和 `paper/main.pdf` 是论文源文件与编译结果。

## 评价口径

`full restart` 在编辑后从空 transcript 运行到完整证书。`stale` 直接复用旧 action set。`max uncertainty`、`periodic` 和 `random` 每题使用与 CTDR 相同的增量反馈预算。增量 token 不重复计算编辑前已支付的旧 transcript。

答案键、逐题模型输出、原始 token 数组和模型权重存于私有工件。公开汇总只保留面板和团队级统计。

## 验证

- 四个正式面板共 2,304 个题-团队单元。
- CTDR 与 fresh edited-graph decision set 的 exact equality 为 1.000。
- 正式聚合 action error 为 0.000，stale 为 0.031，budget-matched max-uncertainty 为 0.069。
- 面板平均增量 token 节省为 0.796，汇总 token 比值节省为 0.801。
- 200 个随机合成有限接口通过 transport soundness replay。
- 论文使用 MiKTeX `pdflatex` 双遍编译并渲染检查。
