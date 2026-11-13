# 长合成上下文资格入口：初步静态检查

状态：IN_PROGRESS，等待作者可审 manifest。不是 PASS，也不准入 GPU。

已读作者当前可用的 synthetic 构造器、资格算术检查器及原 Linux guard；未读取或运行真实模型/benchmark。没有修改作者源。

已见支持：synthetic 输入完全程序构造；低层 tokenizer 仅用于预构造长度，报告明确实际 HF 计数尚待服务器验证；两轮 fresh matrix 对逐节点语义和合法缓存来源精确比较；174 native/route/request 唯一调用、各节点自身 digest 与 physical token 汇总均有核验；原 guard 绑定监管 PID、PDEATHSIG 和绝对 monotonic deadline。

待关闭的审核边界已发作者：

1. 最终入口必须显式绑定实际 matrix protocol hash、源码与双模型/tokenizer身份；两个相同错误输出的重放一致性不能单独替代算子身份核对。
2. 当前 verify_matrix 对 logical cost 只检查 calls，而未从全部逻辑 NodeRecord 独立重算每条件及总 input/output/received/self-state/timing。建议补齐，防两轮一致地记录错误逻辑费用。
3. 新 durable helper 应在返回 Generation 给外层之前完成原文及调用身份落盘，失败须停止并保留已付费证据。最后要按174个call_id核对durable raw与matrix Generation，而非只检查事件数量。helper尚未交付，暂未审。
4. 当前 guard 固定1760.5秒最大窗口；只有同一0.5h资源合同才可直接复用。新增编排器长窗口需另建guard并保留原版本。

优势是工程资格与科学结果分离、固定合成输入及逐节点可复算；风险是目前入口和 durable helper 尚在实现，CPU静态检查不覆盖实模型长上下文容量、重放确定性、驻留时间或SIGKILL后的文件闭合。
