# 本地tokenizer版本，样本容量预检：HotpotQA 未标注输入

`server_runtime_equivalence=false`。3轮是沿用的工程配置，不是正式科学合同；4096-token额外预留仅为情景假设。完整上下文保持不变，没有读取gold或supporting_facts值。

**本地固定样本预检支持继续考虑 8192 输入预算，但不构成正式任务准入。** 两个 tokenizer 的 2000 条样本首轮输入均低于8192；加入空消息占位的 aggregator 模板后，样本最大值分别为3426和3820 tokens。原始全部上下文保留。没有任何答案、supporting_facts、validation/test或新R1结果读取，没有模型推理、权重加载或CUDA初始化。

## 数据与模板边界

使用 HotpotQA distractor train 固定 revision `1908d6afbbead072334abe2965f91bd2709910ab` 的两个 parquet，共90447行。先固定 PCG64 seed=2026111302，再从拼接的全局行序号不放回均匀抽取2000条并排序读取；不按长度、题型、难度或结果过滤/重采样。**这是容量样本，不是正式数据 split。** 样本索引、身份哈希、payload哈希与逐行计数只存于 private/。

Parquet只投影 `id, question, context, type, level` 五列；只将预选行转换为Python值。沿用已有collector：全部文档title/sentences保持原顺序，以紧凑JSON拼接；不检索、不删句、不截断。payload计数指实际进入user prompt的题目加 `Supplied context` 和完整context，不含不进入模型的id。

直接使用已有 `Executor._messages`、四个ROLE_PROMPTS、MESSAGE_INSTRUCTION与AGGREGATOR_PROMPT。工程qualification_v2明确是3轮；代码默认2轮只作为元数据记录，未冒充本次配置。当前没有正式Hotpot科学模板，这里只是沿用工程模板做容量探针。

首轮是四个真实模板分别加system后进行chat-template计数；下表取每条样本四worker的最大值。aggregator采用四条空字符串wire占位：**模板渲染与token计数是真实本地操作，空消息内容是人工占位，不是真实后续生成结果。**

## 本地精确计数

| tokenizer／计数对象 | 中位数 | P95 | P99 | 样本最大值 |
|---|---:|---:|---:|---:|
| Qwen payload | 1434 | 2085.25 | 2444.06 | 3137 |
| Qwen 首轮四worker最大 | 1586 | 2237.25 | 2596.06 | 3289 |
| Qwen aggregator空wire占位 | 1723 | 2374.25 | 2733.06 | 3426 |
| Mistral payload | 1622 | 2353.05 | 2726.10 | 3503 |
| Mistral 首轮四worker最大 | 1779 | 2510.05 | 2883.10 | 3660 |
| Mistral aggregator空wire占位 | 1939 | 2670.05 | 3043.10 | 3820 |

每个对象分母均为2000。首轮与空占位base的8192超限数均为0；样本最大base距8192的余量：Qwen4766、Mistral4372 tokens。**样本最大值不覆盖全部train最大值。**

## 后续消息的假设预留

分别给当前base额外加0、512、1024、2048、4096个“接收端tokenizer下的总输入tokens”，每种情景两模型均为0/2000超限。最紧的4096预留情景，最大总量为Qwen7522、Mistral7916，剩余670、276。

这不是实测后续消息，也不是最坏情况保证。sender的128-token wire cap经过异构tokenizer和JSON转义后，不等于接收端128 tokens；四条消息也不能直接宣称最多512。真实后续轮次需继续按接收端实际模板计数，禁止靠截断获得通过。8192是输入上限，输出预留和模型位置容量仍须在正式执行前独立核对。

## tokenizer与CPU成本

环境：Python3.11.14、NumPy1.26.4、PyArrow23.0.0、Transformers5.12.1、tokenizers0.22.2；sentencepiece未安装，未修改或升级环境。只复制8个明确白名单tokenizer小文件（共14,177,862 bytes）到私有临时目录；Mistral缺失的tokenizer.model补件来自固定revision `c170c708c41dac9275d15a8fff4eca08d52bab71`，没有修改原模型目录。源文件与副本的8项哈希最终全部一致，原Mistral目录仍未补写tokenizer.model。

Qwen使用本地AutoTokenizer fast。Mistral显式调用LlamaTokenizerFast、`legacy=False`；在此Transformers版本中返回LlamaTokenizer fast，legacy属性为false。它与本地AutoTokenizer的backend序列化和chat template相同，2000条payload的token IDs逐条一致。**这只证明本地两种加载方式一致，未建立与服务器旧运行时的等价性。** 如需要声称服务器实际容量，须另行CPU模板核对；本次不把本地结果冒充服务器证明。

完整容量扫描（不含文件复制和前期合成预检）wall=117.19秒、CPU=95.02秒。其中未标注样本读取wall13.64秒；Qwen计数wall50.32秒、CPU44.31秒；Mistralwall40.81秒、CPU38.33秒。进程采样峰值RSS约757 MiB，末尾约686 MiB。它衡量CPU预检开销，不推算GPU推理时间或显存。

首次合成预检因本地Transformers返回BatchEncoding而检查器把它当list比较，触发断言；该失败记录保留。修复只处理Mapping返回结构，没有改变模板、样本或tokenizer。正式样本计数在修复后一次完成。

## 验证、优势与风险

独立验证脚本从私有索引和计数重建固定抽样、全部统计和预留超限数，精确一致；不重新读取题文或任何标签。当前文件大小/metadata行数与先前审计一致；parquet全文件SHA沿用先前固定审计，本轮没有重哈希含其他列的整个文件。源码、协议、样本与计数的哈希保存在RESULTS和VERIFICATION。

优势：上下文完整、两种真实本地tokenizer、固定随机样本、system/chat模板实际计入；CPU成本低且不占GPU。风险：仅2000条样本；本地/服务器运行时等价待验；后续消息预留是假设；不能推出任务正确率、graph质量或GPU可运行性。样本内计数结论置信度高；全部训练集与正式运行容量尚未确认。

拓展思考／你可能想知道：下一项必要工程检查是用最终服务器tokenizer对同一私有样本与真实模板核对token IDs，并在正式合同下对实际后续消息做不截断的输入上限检查。这不是本报告触发的正式轮次或GPU启动。
