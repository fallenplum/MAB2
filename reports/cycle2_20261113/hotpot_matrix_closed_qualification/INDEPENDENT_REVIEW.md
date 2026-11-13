# 已闭账长输入完整矩阵资格复核

**结论：PASS_ENGINEERING_CLOSED_REPLAY_ROOT_ACCEPTANCE_REQUIRED。** 独立重算5571项检查通过，等待root最终验收。没有修改原运行、预算或准入记录，也没有新增模型推理。该审计只涉及本次自编人工两跳输入，不涉及真实Hotpot题、标签或R1结果。

## 主要证据

| 项目 | 独立重算 |
|---|---:|
| 完整条件 | 11 × 2 = 22 |
| 物理调用 | 174，Qwen 80 / Mistral 94 |
| 逻辑节点 | 262，包含全部缓存复用节点 |
| 物理输入 / 输出 tokens | 670168 / 8682 |
| CUDA event 秒数 | 600.770263671875 |
| 实际驻留秒数 | 642.057382708881 |
| 实际驻留 GPUh | 0.1783492729746892 |
| overflow / 格式失败 / 基础设施失败 | 0 / 0 / 0 |
| 结束原因 | 全部174次 EOS |

完整用户文本的服务器计数为Qwen3295、Mistral3744；实际含chat template输入范围分别为3446–3718和3899–4858，全部低于8192。所有控制保留同一完整人工上下文。

## 独立核对方法

新审计源码不导入、不调用生产 `matrix_qualification_audit.py`，也不以自带QUALIFICATION的PASS为判据。它自行解析全部durable START/RETURNED行，核唯一request/call ID、每请求唯一native SUCCESS、Generation原文和数值、seed、route、原始消息以及native/runtime身份；再与174个实际trace及262个逻辑节点逐项绑定。

审计独立重建三轮图消息、self-state、peer输入、最终四worker聚合消息，以及independent12的role×sample顺序和全部12槽位聚合消息；single使用其独立协议。独立计算union图影响闭包，核同seed原A/B缓存来源、未受影响节点的完整记录一致性及每条件物理调用几何。逐节点重算完整逻辑费用与实际物理费用，分别对账，未混用CUDA event和驻留时间。

两次fresh按输入、seed、fingerprint、parsed、wire、原始生成text/token/finish、缓存来源条件及调用费用语义逐项比较。call ID、时间和峰值不作为跨fresh相等条件；每份记录的自身digest仍单独验证。全部语义一致。

13份下载文件、13个实际部署源SHA、部署回执、源码freeze回执、私有plan及启动审核/预算回执均有绑定。模型权重沿既有完整hash链核对，没有再次读取大权重文件。身份仅stub重建，不构造HF模型；实际模块字节核对通过。记录运行版本为PyTorch2.9.1+cu128、Transformers4.54.0、Python3.11.9、CUDA12.8、BF16/eager。

小tokenizer.json先核模型identity中的原字节SHA，再用本地低层tokenizers0.22.2独立计数。262个节点的sender wire、receiver peer及self-state token数全部一致。此项不等于重新执行服务器Transformers chat template，也不从重新编码生成text推断原始generated token IDs。

父进程CLOSED/returncode0、选卡时无其他compute进程、所有native调用时间落在驻留窗口内、实际GPUh除法及无超预留标志一致。原输出和部署源审计前后SHA保持不变。该核对不冒称事后再次访问服务器实时进程表。

## 限制与用途

优势是原始生成、消息权限、源码身份、缓存与两种成本都有分开的可追溯证据。限制是只有一个人工长输入和固定edge1的真实回放；其他边来自既有CPU几何核验，不能据此宣称真实Hotpot覆盖、答案质量或科学门通过。

两矩阵累计墙钟608.2901900541037秒。按相同输入长度、硬件和全成功缓存路径机械外推384题约32.4421434695522 GPUh，仅作条件吞吐证据；它不包含真实任务长度尾部与并发负载变化，也不构成预算批准。

审计者曾实现资格入口。本次采用独立代码和原始数据重算，但不把它称为作者身份独立的最终验收；root后续验收仍必须保留。任何科学准入或预算变更均不由此报告授权。

## 证据哈希

- `audit_001/AUDIT.json`：`3d4783417622e786896b243ffb321aae6927257be439f689c11754015186182e`
- `audit_001/RECEIPT.json`：`870a6701b6f6a752a558e9ba9fabeb3e7492d39328bcc314aedb3feb55ce1604`
- `review_closed.py`：`f9f19d4e05ec02624c4bd37ac8ce8b65a171d094ef5b927a5e67f98ab567573f`

逐文件SHA与各类检查计数在AUDIT.json中。本公开摘要不包含主机、原始提示/回答或私有绝对路径。
