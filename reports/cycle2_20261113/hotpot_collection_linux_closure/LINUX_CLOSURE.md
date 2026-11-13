# Linux CPU 实测闭环附录

本附录保留原 Windows 独立报告中的 2 项 Linux-only skip，不改写历史测试结果。父代理随后在 Linux 运行了相同 snapshot002 的作者全套 CPU 测试与独立实际进程 guard 探针；本代理只读核验返回回执、下载哈希、源码哈希及命令返回证据，未自行 SSH 或再次运行远程测试。

**核验通过：作者套件 26/26 PASS、0 skip，52.699328 秒；独立真实 guard 探针 7/7 PASS，0.918607 秒。未使用 GPU、模型、benchmark 或真实标签。**

绑定 snapshot SHA 为 `8eca7837d990257f2234cac86b2590b69ce802cb0d7441ebb5b7f03e9da34ba8`；实际 guard SHA 为 `666184a17aa880a7303fc8e767f4ced69cbb8b8a93f61890a77ac1b7c349d988`；独立探针 SHA 为 `9bebce64a88b3a322cf9330b85bc463f15f01e39d3e5ca1646648886f051e757`。

7 项覆盖父进程 SIGKILL 后独立 session 子进程终止、绝对期限中止忙循环、只停止自有进程组、正常退出不影响另一自有进程、错误父 PID 拒绝、已过期截止拒绝、继承屏蔽 SIGALRM 时解除屏蔽。父 SIGKILL 测试观测子进程最终状态为 absent。

核验脚本 `verify_linux_closure.py` 只读取指定 CPU 回执目录，以 SHA 验证三项下载原字节、匹配 snapshot002 全部运行源码/依赖、核对测试数与禁用 GPU/数据声明，并将探针命令完整 stdout 解析结果与独立探针回执比较。脱敏结果为 `LINUX_CLOSURE_RECEIPT.json`，不包含主机、凭据或私有服务器路径。

父代理报告首次 SSH 读取在 45 秒超时，但没有重新启动测试，随后只读连接取回完成结果。此项传输历史按父代理说明记录；本附录实证核验的是最终完成回执与跟进命令证据，没有把未读取的初次错误回执声称为独立验证。

该闭环补足 Linux 特有进程控制的工程证据；不扩大测试覆盖，不认证真实 GPU 运行中的所有系统异常，也不授予真实题选择、模型采集或评分许可。
