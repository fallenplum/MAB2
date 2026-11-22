# 学习型图条件复检策略试验

在三张 A6000 上训练了一个轻量行级复检策略。输入使用旧/新 source 分布、base 分布、旧 relay 分布、旧 relay 熵与间隔、复检 token 价格及图变化掩码。目标是预测单条 relay 行是否足以恢复新图的有限决策集。

跨任务短协议 top-1 测试中，策略 exact set 平均 0.9560，显式 max-uncertainty 为 0.9514，平均 token 分别为 83.76 和 83.73。四行密度测试中策略为 0.9370，max-uncertainty 为 0.9167，但策略 token 为 110.53，max-uncertainty 为 106.89。语义接口测试中策略为 0.9247，max-uncertainty 为 0.9259，未取得优势。

该试验只使用既有 conditional 输出，标签来自同一有限决策审计，属于候选方法诊断。它没有独立任务流、端到端回答质量或真实部署延迟，因此不能宣称 ACL Main 结果。pairwise ranking 版本仍在运行。

