# 复现入口

1. 从同一 MAB2 提交检出代码。项目历史和 D2/D3 的读取索引在 `reports/cycle2_20261113/research_audit_index_v1`。本次续跑合同在 `protocol/cycle3_20261116`；每份新诊断合同的提交早于对应结果。数据、模型、逐题矩阵与凭据始终留在受控本地归档。
2. APPS 原矩阵和恢复脚本在 `experiments/cycle3_20261116/function_graph_v1`、`function_resident_v2` 和 `function_resume_20261121`。核对两个实际收集段的清单，重放已批准的隔离执行环境、已冻结的输入输出与资源上限。`reports/cycle3_20261116/formal10_function` 留有费用、资格、矩阵和独立隐藏评分回执。只用公开表无法从零重建私有程序输出。
3. 时间问答用 `d31_temporal_sources/sources.py` 下载四份固定 Git 对象的 TRAIN 文件。`retrieval_credit.run` 接收 `read_all` 读入的 TimeQA human train easy 记录。D33 `replay.evaluate` 接收 WinoGrande `valid` 六模型目录。D34 `attribution.analyze` 接收第五轮公开 `EPISODE_SEED_AGGREGATES.csv`。D35 `ceiling.evaluate` 接收每任务的 `em,tokens,shock` 三个 D16 数组。
4. D36 `replay.score_task` 除上述数组，还接收 D16 `ENTRIES.json` 中相应任务的请求条目。`independent.audit(private_cycle3_root, public_report_dict)` 通过另一套实现核对 288 个策略折行。三次采集种子为 17、31、47，正式数值与资源口径详见合同。不需要 GPU 即可重放这些开发估计。运行使用 Python 3、NumPy 和 scikit-learn；公开稿中计费为实际观测的输入加生成 token，两个未返回请求的未知量不作插补。
5. 复制旧稿的 `acl.sty` 或使用随新稿提供的相同 ACL 样式，在 `paper/revalidation_study_cycle3` 下连续两次执行 `pdflatex -halt-on-error -jobname=MAB2_revalidation_study main.tex`。所有文内文献在源文件内，生成 PDF 另有独立页数、引用、哈希核验。
