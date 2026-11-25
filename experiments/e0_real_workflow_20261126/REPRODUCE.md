# E0 复现说明

当前已完成官方 FActScore 源码和 `data.zip`/`demos.zip` 下载、ZIP 校验、32 个匿名开发组锁定，以及固定阈值证书 CPU 测试。

```powershell
cd D:\develop\CV\多臂老虎机
python experiments/e0_real_workflow_20261126/scripts/download_factscore_official.py
python experiments/e0_real_workflow_20261126/scripts/download_factscore_source.py
python experiments/e0_real_workflow_20261126/scripts/audit_factscore_assets.py
python experiments/e0_real_workflow_20261126/scripts/lock_factscore_topics.py
python experiments/e0_real_workflow_20261126/tests/test_e0_certificate.py
```

原始下载文件保存在 `experiments/e0_real_workflow_20261126/external/factscore_official`，不会上传 GitHub。完整 F/D 流程还需要官方 `enwiki-20230401.db` 知识源，当前未运行模型调用。不得将 Qasper、LongBench、HotpotQA 或历史 QASC/SciQ/2Wiki 直接替换为主任务，也不得读取封存标签补齐知识源。

