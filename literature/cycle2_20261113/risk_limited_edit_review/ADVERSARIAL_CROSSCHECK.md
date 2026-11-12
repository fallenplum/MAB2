# 独立反向核查摘录

来源：委派代理 risk_novelty_adversary 的只读文献核查。此文件是人工整理的要点，未冒充原论文全文；没有实验数据访问或 GPU 使用。主报告中的 aLTT 是委派复核完成后由本代理补查的更直接先例，不将其标为该子代理已审。

委派结论：旧信用排序、新鲜配对验证与证据充分后替换，尚不足以成为独立方法创新。共享费用也有先例；图特定闭包结构可能剩余空间，但目前没有证据建立这一差异。

| 论文 | 实际深度 | 独立核查重点 |
|---|---|---|
| [Learn then Test](https://arxiv.org/pdf/2110.01052) | §1.1、§2、Appendix D，出版元数据另核 | 独立排序、固定顺序和图式多重检验；当前校准样本不能反过来决定未经修正的检验顺序。 |
| [Kim 2003](https://informs-sim.org/wsc03papers/065.pdf) | §1–3 | 替换 standard 的顺序验证早已有直接框架；CRN 版本仍需相应假设。 |
| [Zheng et al. 2020](https://sli.csail.mit.edu/pdfs/publications/zheng2020neurips.pdf) | §3.1–3.2、§4开头 | 共享后验采样、候选累计计算费用和 AE/LUCB 比较；原 BAI 保证不可不加验证地搬用。 |
| [Nelson & Goldsman 2001](https://pubsonline.informs.org/doi/10.1287/mnsc.47.3.449.9778) | 官方摘要/元数据 | CRN 不是无条件有利，未读全文。 |
| [Healey et al. 2015](https://www.tandfonline.com/doi/abs/10.1080/0740817X.2015.1009198) | 官方索引摘要/元数据 | 采样与切换成本；不等样本量下 CRN 的正确选择概率问题。未读全文，不扩大为任意 DAG 结论。 |
| [McNemar 1947](https://link.springer.com/article/10.1007/BF02295996) | 官方摘要/元数据 | 配对二元比较为经典问题；没有核到可直接照搬的 sequential McNemar 原文。 |

本代理对 Kim §2–3、Zheng §3.1–3.2 再次阅读原文交叉核实。所见边界均已纳入主报告：总风险预算、条件目标、图变后证据失效、共享 seed 非独立任务、零差样本对绝对净收益的重要性。

独立视角没有提出第三个候选；最终仅保留主报告 H1（排序信息）与 H2（图闭包分配），均为待证伪问题。

