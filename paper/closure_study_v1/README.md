# MAB2 完整负结果工作稿

主文件 `main.tex`，包括摘要、引言、相关工作、方法、协议、结果、讨论、局限、声明、历史与复现附录。固定第五轮六门四败；五轮正式预算用尽，没有有效新方法被确认。本稿不是投稿就绪稿；作者/机构/基金/利益冲突及人类逐条核验仍待提供，没有伪造。

`generate_results.py --repo <checkout>` 从冻结公开汇总重建本轮表格并核对 1360 行；`generate_history.py --repo <checkout>` 重建历史点估计。均不调用模型、不计算区间或 bootstrap。现有 source/fragments 的准备快照保留；本 README 和最终 provenance 取代旧“结果待开放”的状态含义。

使用已安装 LaTeX：pdflatex → bibtex → pdflatex → pdflatex。包含官方 ACL style；microtype 可选。建议在独立构建目录编译，禁用 shell escape 与自动安装。最终 PDF、逐页视觉检查与哈希由交付核验记录绑定。

所有具体比较受同一数据、固定权重和有限图族约束；没有新独立任务确认、没有外部 SOTA 全复现、没有外部人类审稿。完整成本和失败路线见 `reports/cycle2_20261113/final_delivery_v1`。
