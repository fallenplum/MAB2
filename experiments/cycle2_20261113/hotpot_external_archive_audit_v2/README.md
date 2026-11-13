# 正式归档的独立 CPU 审计器

该工具重建消息输入、图影响闭包、缓存来源、调用顺序、token 成本与驻留账本。它只产生 REVIEW_REQUIRED，不能自动释放标签或授予 GPU 权限。

原始科学设计与资源执行合同分别固定 SHA。选择清单绑定原始设计；执行合同除资源约束外，九项科学字段须具有完全相同的 canonical JSON 表示。公开 runner freeze 绑定原 snapshot002 与七个实际执行源码。

在此目录运行 `python -B run_cpu_tests.py --output NEW_ARTIFICIAL_DIRECTORY`，可构建完整384题/4224条件的人工归档。全部依赖已按字节打包，不需要模型、真实数据或 GPU。`python -B -m unittest test_scientific_lineage -v` 单独检查合同谱系。

正式命令为 `python -B audit_cli.py --io PRIVATE_IO.json --contract-sha256 TRUSTED_SHA`。默认只核验合同；只有完整采集和驻留闭合后才使用 `--audit --output NEW_DIRECTORY`。IO 显式映射原路径到本地副本，禁止改写原计划后声称相同哈希。正式 CPU tokenizer 必须与资格环境版本、文件、类和 chat template 一致；无需权重。原始输出、题目和 IO 都保持私有。

优势：独立重算、多来源交叉核验、失败关闭、明确区分物理研究成本与完整服务逻辑成本。风险：外部历史账本完整性和可信准入仍依赖 root；CPU 审计不重新生成 GPU 输出，不独立测量 CUDA 时间，不证明训练污染不存在。人工测试通过仅证明准备情况，尚非真实结果或标签释放。
