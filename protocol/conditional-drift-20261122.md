# 条件 relay 通道漂移诊断合同

## 假设

source-only CTDR 只在 relay 条件核不变时安全。若图编辑改变一个或多个 relay 行，复用旧 relay 观测可能造成错误 keep/delete。一个条件感知守卫在检测到变化的行后重新查询这些行，并复用仍合法的 terminal 观测，应恢复准确决策。局部变化时它应少于冷重启成本，所有行都变化时不应承诺节省。

## 数据和限制

使用四个 M60 冻结面板的既有 worker 输出，2304 个题目×团队单元。没有新增模型推理。对 relay 概率列执行预先固定的循环变换，生成强度 0.25、0.5、1.0，以及只改变单行和双行的稀疏场景。该阶段是机制诊断，不能作为真实模型条件漂移的独立确认。

## 对照

- stale，零新增查询，直接复用旧 relay 观测。
- source-only CTDR，使用现有 source-TV 半径并假设 relay 条件不变。
- TCR guard，重查发生变化的 relay 行，复用合法 terminal 观测。
- TCR token，在 TCR guard 上加入记录 token 价格的决策区域排序。
- cold restart，从空审计器重新查询新条件核。

## 验收门

- 任何条件变化场景中，TCR 的 exact decision set mismatch 必须为零。
- 单行或双行变化场景中，TCR post-edit token 应低于 cold restart。
- 所有四行变化场景中，若 TCR 高于冷重启，记录为效率失败，不通过正算法门。
- source-only CTDR 的错误率用于验证其适用范围，不能作为 TCR 调参信号。
