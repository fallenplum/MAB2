# 固定384题服务器 tokenizer-only 长度回执

本包独立于已审 runner snapshot002；仅实现 CPU 工具和人工测试，不选题、不加载 backend/权重、不读取 Parquet 或标签、不申请推理预算。正式输入仅是 selector 已产的五字段 questions.jsonl 和对应 manifest/receipt。

计数文本固定为冻结 Executor 的 qtext：question 加非空完整 context 时的双换行、Supplied context 标记与 context 原字符串。使用与 HFBackend 相同的 AutoTokenizer.from_pretrained(local_files_only=True,trust_remote_code=False) 和 encode(add_special_tokens=False)。不加入 system/chat/轮次指令，不截断、不按长度排除；这些计数仅用于已冻结全集内部的长度层，不证明完整 prompt 或未来消息在8192内。

工具输入 TOKENIZER_PROBE_BINDINGS 绑定冻结 collector、backend 与 executor 的源码 path/SHA，root 独立 engineering acceptance path/SHA，以及 qwen/mistral 的服务器模型目录与完整 qualified identity。acceptance 的 matrix hash和两个 model_identity_sha256必须一致。工具不会 import backend/executor 或打开权重，仅 hash 已指定源码及少量 tokenizer 文件，并核对它们与 qualified identity.model_files 的 SHA/大小。

实际 tokenizer 小文件先复制到全新输出目录的 private_tokenizers，严禁权重进入临时目录。AutoTokenizer 只能从这份无权重副本加载，网络关闭，CUDA不可见，Transformers 的 Torch/TF/Flax后端均关闭；不安装或升级依赖。匹配 qualified Python/Transformers版本、实际 tokenizer class 和 chat template SHA 后，记录小文件清单、tokenizers/SentencePiece等当前依赖版本、backend tokenizer序列化SHA与encode选项形成 tokenizer identity。该新 identity 的 SHA供最终 runner bindings.models.NAME.tokenizer_identity_sha256 填入，避免输入与待生成长度回执循环依赖。

server_runtime_equivalence=true 的明确范围仅为该服务器环境中已核文件/版本/构造及输入编码；完整 model_identity 从既有资格回执绑定引用，不表示重新加载或复核GPU数值推理。身份不匹配则无成功LENGTHS回执。

输出 LENGTHS.json 含 runner 的最小字段与384条私有ID/hash/count；另 RUNTIME.json、COMPLETE.json、CPU timing、源码/输入/qualification来源pin，无原始题文/答案副本。任何失败不重试、不改变384、不输出伪成功；所有输出目录拒绝覆盖。测试只用人工384输入与假 tokenizer，真实服务器执行由root单独审查后安排。
