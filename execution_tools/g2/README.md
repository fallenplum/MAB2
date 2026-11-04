# G2 执行控制与独立验收工具

## 文件与职责

| 文件 | 职责 |
| --- | --- |
| `coordinator.py` | 本地数据所有者控制，已完成 unit 的镜像、全局输出核验、分阶段标签释放、CPU 阶段启动与终态镜像证明 |
| `mirror.py` | 校验并传输完成 unit，核对 claim、done、marker、tar 成员与逐文件哈希 |
| `independent_audit.py` | 本地终态门、独立评分、独立岭回归重建、预测和统计重放、成本范围审计 |
| `../mirror_only.py` | 复用的固定字节路径安全与原子发布 helper |
| `test_*.py` | 纯 CPU 合成回归，SSH/SFTP 使用替身 |
| `SOURCE_HASHES.json` | 原始部署字节、公开文本字节、公开测试适配及依赖路径清单 |

这四个工具源码保留原始部署字节。发布白名单由三份 G2 工具、一份 helper、四份合成 CPU 测试、本文和来源哈希清单组成。运行计划、问题、标签、模型权重、特征和真实输出由数据所有者在项目根目录保管，连接参数与认证值在运行时提供。

## 与已有公开源码的关系

五个冻结实验核心位于仓库的 `experiments/g2`。该目录的 `SOURCE_HASHES.json` 分别记录 `runtime_sha256` 和 `repository_utf8_lf_sha256`。公开测试先核对其中的部署哈希与独立验收器的固定引用，再核对实际公开文件的文本字节哈希。

原评分器位于 `src/graph_credit/scoring.py`，固定 SHA256 为 `da315ade4be7362ca81fc7ed41252cdccaa9271dc32f9a20531356587bb5f37e`。独立验收器的生产评分重放仍校验该原始评分器哈希。

G2 核心源码、此目录的控制工具及其 helper 分别保持自己的来源身份。已有运行以实际冻结字节和计划中的来源清单为准。另起运行需要重新冻结安装来源并完成资格检查。

## Helper 安装位置

`mirror.py` 固定读取项目根目录下的 `artifacts/private/monitoring/mirror_only.py`，并要求其 SHA256 为 `273008dfafa95fe67f522fd116024c50f2176298072f6f4b95cf6267dc7e5ccb`。

部署到新项目根时，将公开的 `execution_tools/mirror_only.py` 按字节复制到该位置。复制前后都核对上述哈希。相对目录中的 `private` 表示既定运行产物命名空间。实际项目根目录由数据所有者在部署时指定。

以下是从仓库根执行的部署示意。`LOCAL_PROJECT_ROOT` 由数据所有者替换为自己的项目根目录。

```python
from pathlib import Path
import hashlib
import shutil

root = Path("LOCAL_PROJECT_ROOT")
source = Path("execution_tools/mirror_only.py")
expected = "273008dfafa95fe67f522fd116024c50f2176298072f6f4b95cf6267dc7e5ccb"
assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
target = root / "artifacts/private/monitoring/mirror_only.py"
target.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(source, target)
assert hashlib.sha256(target.read_bytes()).hexdigest() == expected
```

G2 使用此 helper 的固定函数。helper 自身的历史独立 CLI 仍保留原有 G1 行为。

## CPU 回归

在包含已有 `experiments/g2` 和 `src/graph_credit/scoring.py` 的仓库根目录运行。数学回归需要 NumPy 和 SciPy。

```text
python -m unittest discover -s execution_tools/g2 -p "test_*.py" -v
```

测试生成合成问题标识、合成标签和临时文件，网络交互使用替身。来源校验定位到公开核心目录与公开评分器，数据夹具在测试中构造。路径安全回归会在临时目录建立链接，链接创建权限决定相应测试的实际执行状态。

公开适配仅包括以下改动。

- `test_independent_audit.py` 使用仓库相对位置定位 `experiments/g2` 和原评分器，并把来源校验展开为部署引用哈希与公开文本字节哈希两项。
- `test_mirror_only.py` 从 `execution_tools/mirror_only.py` 导入 helper。历史 G1 终验 mock 在测试临时目录创建合成模块占位，并将该次模块定位限定到临时目录，消除对实际 G1 终验器的依赖。
- `test_coordinator.py` 和 `test_mirror.py` 保留原始测试字节。

数值解析、样本权重、岭回归、alpha 平手规则、风险差、bootstrap、标签门与重试历史等既有测试断言保持其原有内容。

### 验证记录

| 执行环境 | 结果 | 证据范围 |
| --- | --- | --- |
| root 最终公开仓库隔离布局 | 87 通过、0 跳过、0 错误、0 失败，4.061 秒 | 十文件工具包、公开五核心源码与评分器，合成 CPU 夹具，实际链接与并发写入回归 |
| 前序工作区沙箱隔离布局 | 85 通过、2 跳过、0 错误、0 失败 | 两项 Windows junction 创建被沙箱权限拒绝，其余回归通过 |

`SOURCE_HASHES.json` 分别保存两次验证的范围。root 最终验证记录同时绑定收据字节 SHA256、持久化日志字节 SHA256 与日志 UTF8 LF 文本 SHA256，Windows 落盘换行差异通过独立字段记录。

## 生产入口

`coordinator.py` 绑定既定 run ID `g2_v1_20261104`，从数据所有者提供的项目根读取冻结计划。SSH 端点、用户、远端路径和 known-hosts 文件均由命令行提供。认证值仅从运行时环境变量 `GRAPH_CREDIT_SSH_PASSWORD` 读取。

```text
python execution_tools/g2/coordinator.py --root LOCAL_PROJECT_ROOT --host SSH_HOST --port SSH_PORT --user SSH_USER --remote-root REMOTE_PROJECT_ROOT --remote-python REMOTE_PYTHON --known-hosts KNOWN_HOSTS
```

真实协调入口另需 Paramiko 和 psutil。远端 Python 参数须为数据所有者指定的绝对可执行路径。

独立验收在完整本地镜像形成后运行。

```text
python execution_tools/g2/independent_audit.py --root LOCAL_PROJECT_ROOT --plan results/g2/g2_v1_20261104/frozen_plan.json
```

终态链缺失时返回 `WAITING_NOT_TERMINAL`，退出码为 3。完整性或算术验收失败时返回 `FAIL`，退出码为 2。通过时返回 `ACCEPTED_WITH_DOCUMENTED_LIMITATIONS`，退出码为 0。

协调器的 `READY_FOR_INDEPENDENT_AUDIT` 表示已到独立验收入口。验收通过后的 G2 阶段完成标志与科学门分别报告。`G2_PASS` 和 `INSUFFICIENT_PREDICTIVE_GAIN` 均可对应已完成的 G2 算术验收，后者按当前合同停止 G3。整个项目的完成状态另行报告。

## 成本范围与证据边界

审计区分正式生成、资格、CPU 阶段与工程尝试的范围。预处理编码/PCA 的历史 CPU 秒数未记录，保留未知。启动与特征收据之间的观测 wall interval 单列报告。

此公开包提供执行和验证源码及合成回归。真实 G2 的全量评分、711 维模型重建和终态验收，需要数据所有者完整且哈希匹配的本地镜像。
