# M60 条件 relay 真实推理运行协议

## 输入

新建 QASC 和 SciQ 各 128 题，题面与 M60 四个面板去重。每题固定 old source、new source、base 三行。relay 条件行 `[none,A,B,C]` 中只对 A 建立图编辑后的新条件。新条件删除第二条证据，表示 relay edge removal。正式推理输出 old relay 四行和 new relay 四行，终端 4×4 labels 保持同一题面。

## 输出接口

`conditional_mcqa_worker.py` 输出 27 个字段。0–2 是 source rows，3–6 是 old relay rows，7–10 是 new relay rows，11–26 是 terminal labels。`conditional_schema.py` 在评分前校验形状、归一化和字段完整性。

## 运行资源

三个模型团队与 M60 相同。每个模型只在新输入上运行。post-edit scorer 使用同一 old transcript 生成规则和真实生成 token 长度。服务器连接恢复后才提交 GPU 任务，服务器不可达期间不伪造结果、不把 synthetic replay 当作独立确认。

## 评分

source-only CTDR 继续使用旧 relay rows。TCR 根据输入中声明的受影响 A relay 行重查 new relay row，复用合法 terminal labels。另设 warm/reweighted、token ranking 和 cold restart。主要指标是 exact decision-set、false drop、false keep 和增量 token。
