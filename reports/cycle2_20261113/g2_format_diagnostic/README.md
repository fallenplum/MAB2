# G2 格式机制诊断：公开审计包

这是旧 G2 train200/tune80 的探索性机制诊断，不是新候选实验或独立确认。
先读 FINDINGS.md；完整分层、指标口径与限制见 REPORT.md。原始研究记录仅扫描一次，后续均读取私有数值cache。

公开内容仅含去标识聚合统计、代数与旧D2指标复算摘要、分析协议、CLI参数化分析代码及合成测试。没有题目ID、题文、答案、逐题数组、数值cache、选择索引或模型权重。原始脚本、旧分析版本和完整receipt保留在项目私有归档。

## 可复现范围

合成测试不需要私有数据：

```sh
python -B -m unittest -v test_analyze_cache
```

实际统计需要数据持有人提供三个**私有**文件，文件hash固定在 INPUT_PINS.json：

```sh
python analyze_cache.py --cache PRIVATE_NUMERIC_CACHE.npz --d2-inputs PRIVATE_D2_INPUTS.npz --d2-predictions PRIVATE_D2_PREDICTIONS.npz --pins INPUT_PINS.json --output NEW_PRIVATE_OUTPUT_DIRECTORY
```

输出目录必须不存在。代码仅读取这些明确指定的派生文件，不扫描raw、不加载模型、不访问网络。私有输入中的 train/tune 掩码固定，代码不会将旧exposed120或新的R1加入分析。缺少私有输入时不能宣称已复现真实数据结果；可核对公开聚合表及运行合成测试。

REPORT.md 中描述的是完整私有审计流程；本公开包只提供上述参数化cache复算入口，不包含原始目录专用的inventory/scan/render脚本。CLI副本与原脚本SHA对应见 SOURCE_LINEAGE.json，文件清单与脱敏扫描结果见 PUBLIC_MANIFEST.json。

## 指标边界

单seed非零率、五seed均值非零率、五seed经验MSE和原跨seed误差乘积风险分别报告。原风险采用每题24context再题均；分层贡献保持相同分母且加回原风险。相同完整cell与权重下，模型间MSE差等于跨seed风险差，因此绝对尺度差不能解释G2模型比较失败。

所有关联和bootstrap区间均为开发集探索结果。没有消息修复反事实，不做因果归因。exact-cache子诊断仅有固定fallback语义字段，未保留一般逐对wire或aggregate hash，不能声称已建立普遍零效应证书。
