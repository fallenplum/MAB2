# 服务器384题 tokenizer-only 长度工具

**仅计数已选无标签输入，不加载权重/backend、不启动GPU、不评分。** 本地8项人工测试通过；尚未在服务器实际 tokenizer 上运行。首次7通过/1测试错误保存在 cpu_validation_001：测试把 probe bindings 直接当 runner bindings，漏了 collector 顶层字段；修正人工适配后 cpu_validation_002 全部通过。未改 runner snapshot002。

生产工具只需 tokenizer_lengths.py。服务器使用既有资格环境（本轮 Python 3.11.9、Transformers 4.54.0），不安装或升级依赖；版本实际值以 qualified identity 为准，严格不等即拒绝。AutoTokenizer 从新私有目录的小文件副本加载；目录没有权重。仅使用标准库及服务器已安装 transformers/tokenizers/SentencePiece 依赖。环境显式禁用Torch/TF/Flax后端、网络与CUDA设备；若仍导入torch/backend，则拒绝成功。

## TOKENIZER_PROBE_BINDINGS.json

```json
{
  "schema": "hotpot_server_tokenizer_probe_bindings_v1",
  "matrix_protocol_hash": "<qualified matrix hash>",
  "engineering_acceptance": {
    "path": "<server acceptance bundle>/ACCEPTANCE.json",
    "sha256": "<acceptance SHA>"
  },
  "sources": {
    "collector": {"path": "<frozen hotpot_support.py>", "sha256": "<SHA>"},
    "backend": {"path": "<frozen src/graph_credit/backend.py>", "sha256": "<SHA>"},
    "executor": {"path": "<frozen src/graph_credit/executor.py>", "sha256": "<SHA>"}
  },
  "models": {
    "qwen": {"path": "<server model directory>", "identity": {}},
    "mistral": {"path": "<server model directory>", "identity": {}}
  }
}
```

两个 identity 必须完整复制资格 IDENTITY.json 的 backend.qwen/backend.mistral 对象，不能用上例空对象。原资格schema已只读确认：顶层 source_manifest、matrix_protocol_hash、backend。工具从 acceptance.evidence.identity 的 bundle-relative path/SHA 读取原资格身份，逐个对照，且 backend/executor source SHA 要属于该 qualification source_manifest。不需要拷贝GPU runtime主机字段，不打印prompt。acceptance必须已由root独立审查签署；不接受原资格 REVIEW_REQUIRED 自动转PASS。

初次 probe bindings 不填 tokenizer_identity_sha256；由真实 tokenizer 构造后生成。完成时将 LENGTHS.json.tokenizer_identity_sha256.NAME 填入最终 runner bindings.models.NAME.tokenizer_identity_sha256，完整 models.NAME.identity 继续用原资格对象。若在再次独立计数时已填此SHA，工具会核对生成身份完全一致。这消除了输入与待生成回执之间的循环依赖。

## 执行

每个SHA均为对应文件原始bytes，输出目录必须不存在：

```text
python -B tokenizer_lengths.py \
  --selection SELECTED_MANIFEST --selection-sha256 MANIFEST_SHA \
  --selection-receipt SELECTION_RECEIPT --selection-receipt-sha256 RECEIPT_SHA \
  --questions QUESTIONS_JSONL --questions-sha256 QUESTIONS_SHA \
  --bindings TOKENIZER_PROBE_BINDINGS --bindings-sha256 BINDINGS_SHA \
  --output NEW_PRIVATE_OUTPUT
```

只读 selector 五字段JSONL，不接受 Parquet 或 label 参数。每题完整 qtext=question + 非空context的 "\n\nSupplied context:\n" + context 原字符串；调用 encode(add_special_tokens=False)。不计chat/system/后续消息，不截断或筛掉长题；因此不是8192完整推理容量保证。完整collector payload图式SHA无末尾换行，manifest input_sha则有末尾换行。

输出 LENGTHS.json 正好含 snapshot002 要求字段及384条id/hash/双长度；RUNTIME.json含小文件manifest、真实tokenizer identity及版本、资格引用、CPU耗时，COMPLETE.json绑定两者。private_tokenizers只包含小文件副本，所有逐题记录保持私有。控制台只输出status/count/error type，不输出原题、生成内容或异常文本。故障没有自动retry、不覆盖原目录，没有成功COMPLETE则不可给runner。

server_runtime_equivalence=true 仅表示实际服务器输入tokenizer计数的文件、Python/Transformers版本、构造class/chat模板已与资格核对；不是重新验证GPU推理。tokenizers/SentencePiece/protobuf当前版本与backend序列化SHA进入新tokenizer身份，正式runner部署应保持同一环境。GPU模型身份通过已签资格SHA引用，本工具不读取任何权重。

## 人工CPU测试

复制 test_tokenizer_lengths.py 和 tokenizer_lengths.py；让 MAB2_RUNNER_DIR 指向未修改的 snapshot002 源目录。标准库+NumPy即可，真实Transformers由测试假模块替代：

```text
MAB2_RUNNER_DIR=/path/to/snapshot002 python -B test_tokenizer_lengths.py --output NEW_CPU_RECEIPT_DIR
```

8项包括384人工输入接入未改 runner 的 validate_lengths/verify_payloads/build_schedule、完整上下文不变、文件/版本/资格漂移拒绝、拒绝覆写、真实loader代码的AutoTokenizer参数与pad fallback、以及拦截任何.safetensors/.bin读取。测试不读取真实selection和gold，不把假tokenizer验证称为实际服务器tokenizer验证。
