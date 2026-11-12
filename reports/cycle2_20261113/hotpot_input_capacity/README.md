# 本地tokenizer版本，样本容量预检

**`server_runtime_equivalence=false`。** 本包公开HotpotQA未标注训练输入的本地tokenizer容量汇总与计数实现，不包含私有样本索引、逐题计数、题文、标签、tokenizer副本或模型权重。

历史固定样本为2000/90447条；完整上下文未过滤或截断，没有读取gold或supporting_facts值。3轮仅沿用已核验的工程配置，不是正式Hotpot科学合同；GraphConfig类默认2轮与工程显式3轮有区别。4096-token额外预留是接收端输入总量的情景假设，不是实测后续消息或跨tokenizer的最坏情况证明。样本最大值不能代替全train最大值。

## 历史结果与接口修订

REPORT.md、HISTORICAL_RESULTS.json、HISTORICAL_VERIFICATION.json保留历史聚合结果与核验边界。发布接口在历史计数之后修改：硬路径改为参数，结果写全新目录并拒绝覆盖；源码与冻结容量规范分开读取；禁止写bytecode到源目录。原始run_capacity.py没有修改，旧/新SHA在SOURCE_HASH_MAP.json中。

本次包装只做3条合成输入的接口和模板验证，**没有重复2000条历史计数**。SYNTHETIC_VALIDATION.json只能证明所述接口检查通过，不能称新数据复现、服务器等价或额外科学证据。

## 使用授权输入复现

已验证本地环境为Python3.11.14、NumPy1.26.4、PyArrow23.0.0、Transformers5.12.1、tokenizers0.22.2、psutil7.2.2。依赖声明不要求修改既有环境；可在自行管理的隔离环境中配置。没有安装sentencepiece；已有fast tokenizer路径成功。Mistral显式legacy=False在当前版本返回LlamaTokenizer fast，本地Auto/显式加载一致不等于原服务器实现一致。

以下路径均为占位符，需要使用已授权的原始代码/元数据/数据/tokenizer。公开包不提供或授权取得这些私有输入。

```sh
python run_capacity.py \
  --collector-module /authorized/hotpot_support.py \
  --executor-module /authorized/executor.py \
  --engineering-protocol /authorized/engineering_protocol.json \
  --qwen-tokenizer /authorized/qwen_tokenizer \
  --mistral-tokenizer /authorized/mistral_tokenizer \
  --data-root /authorized/hotpot_train_revision \
  --audit-summary /authorized/train_audit_summary.json \
  --output capacity_replay_new
python verify_capacity.py --output capacity_replay_new
```

collector/executor/工程协议的冻结哈希见HISTORICAL_RESULTS.json，tokenizer文件哈希见TOKENIZER_HASHES.json。historical模式使用原样本seed及2000样本量，仅投影五个白名单字段；样本索引及逐行计数写入输出目录的private/，不得公开。数据SHA沿用既有审计，当前metadata/文件大小核对不是新一次全parquet哈希验证。

若仅做合成接口验证，保留前五个源文件/tokenizer参数、`--output`，加`--synthetic-only`并省略两个数据参数。该路径不读取任何benchmark样本或标签，使用三个自带合成输入验证完整上下文序列化与chat-template计数。

## 公开范围与限制

公开白名单与文件哈希在PUBLIC_MANIFEST.json；不存在private/子目录、逐题数组或tokenizer模型文件。数据及tokenizer未随包发布，所以完整复现需要授权输入，不能称无外部输入的端到端公共复现。

优势是保持上下文和模板、记录本地版本、CPU成本可测；限制是固定样本、后续消息预留假设、服务器等价未验，以及不涉及GPU显存/推理时间、答案质量或正式候选准入。
