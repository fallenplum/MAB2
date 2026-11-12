# Cycle2 新执行入口与最小 schema

`make_plan_template.make_plan(...)` 返回 DRAFT，调用方审阅后改成 FROZEN、落盘并固定文件 SHA256。生成器只读调用方指定的无标签题目文件和排除表，不挑题、不读答案、不运行模型。`source_root` 是新 checkout 或部署 bundle；`data_root` 是含 data/models/results 的项目数据根目录。

必填输入：

```python
episodes = [
  {"episode": 0, "seed": 3701, "permutation": [0,1,2,3], "complement": False},
  # 共 8 项，episode 0..7，seed 3701+episode，覆盖 4 cyclic permutations × 2 complements。
]
units = [
  {"unit_id": "qual_00", "question_id": "gsm8k_<normalized_hash20>",
   "phase": "qualification", "seed": 3799, "worker_index": 0},
  {"unit_id": "ep0_I_00", "question_id": "gsm8k_<normalized_hash20>",
   "phase": "I", "episode": 0, "seed": 3701, "worker_index": 0},
]
```

每 episode 精确 I12/C12/S24，正式题跨 role/episode 不重叠，另 6 个资格题。I 执行 A+6toggle，C/S 执行 A/B 各 7 项。图由 `episode_menu(permutation,complement)` 生成，unit 可冗余写 menu 但必须一致。菜单每项字段 name/mask/base/background/action，toggle 另有 edge，资格独立全量 replay 另有 replay_of。模板提供 1 次 B_toggle_9 全量 replay；固定规则可在资格执行前补充，其 token 与驻留成本都计账。

questions JSONL 每行 `{id,question,source_split:'train',context?}`，允许 task/split 元数据，禁止 answer/gold/label/target/solution/quality。完整文件绑定 descriptor `{path,sha256,payload_sha256:{qid: Executor.digest(Executor.question_payload(row))}}`，path 相对 data_root。

exclusions JSON 为：

```json
{"schema":"mab2_cycle2_exclusion_v1",
 "excluded_ids":[],
 "excluded_normalized_question_sha256":[],
 "excluded_normalized_question_sha256_prefix20":[],
 "coverage":{"G1":"ALL_KNOWN_RECORDED","G2":"ALL_KNOWN_RECORDED",
             "qualification":"ALL_KNOWN_RECORDED","official_test_ids":"ALL_KNOWN_RECORDED",
             "literature_exposure":"KNOWN_RECORDS_REVIEWED"},
 "unknown_exposure_beyond_recorded":true}
```

排除表覆盖已知记录；无法宣称未知公共文献暴露被完整排除。官方 test 只允许从 split_manifest 元数据获取 ID，执行入口不读 test 题目/答案。exclusions descriptor 为 `{path,sha256}`。

命令：

```sh
python new_batch.py --mode preflight --source-root /bundle --data-root /project --plan /frozen/plan.json --plan-sha256 <sha>
python new_batch.py --mode supervise --source-root /bundle --data-root /project --plan /frozen/plan.json --plan-sha256 <sha> --worker-index 0 --phase qualification
python new_batch.py --mode audit --source-root /bundle --data-root /project --plan /frozen/plan.json --plan-sha256 <sha>
```

每 GPU 一次独立 supervise，内部才启动单 GPU worker。3 worker 各 8h 累积上限包含模型载入、资格、正式和失败；calls 是另一独立上限。失败停止，显式 `--resume` 最多再尝试一次当前 unit，成功图 checkpoint 校验后复用，无静默重试。已完成 phase 拒绝重新载模。输出 `results/cycle2/<run_id>/worker_<index>/`，含原始执行、native physical ledger、wrapper attempt ledger、驻留账本、30 秒心跳。

资格完成后 root 独立验收并写输出根 `QUALIFICATION_ACCEPTED.json`：`{status:'PASS',plan_sha256,executor_protocol_hash,worker_reports:{'0':SHA256(worker_0/QUALIFICATION_COMPLETE.json),'1':...,'2':...}}`。正式运行将 phase 改为 development。这个文件是本研究流程的验收凭证，不代替用户预算授权。若资格耗时表明不能在剩余预算完成覆盖，应不启动正式采集并保存资格结果。

生成入口完全不评分，`COMPLETE_UNSCORED` 仅代表采集完成。root 必须另外将 S labels 的读取绑定冻结策略收据；穷举菜单的数据生成成本与在线策略可见反馈及逻辑成本分开记账。
