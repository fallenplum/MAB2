# MAB2 v3 最终可复现性审计

审计日期：2026-11-23  
审计范围：`paper/mechanism_study_v3` 与 `MAB2_recovered/experiments/mechanism_20261122` 的当前工作区版本。

## 已通过检查

- Python 语法检查：`python -m compileall -q` 通过，机制目录全部 `.py` 文件可编译。
- Pairwise Gap Certificate 单元测试：`test_pairwise_gap_certificate.py`，3/3 通过。
- 结果 JSON 解析：机制目录内 118 个 JSON 中 116 个以 UTF-8 JSON 成功解析。
- 论文源文件结构：`main.tex` 的花括号及 `begin/end` 环境计数平衡，引用与标签未发现静态不闭合。
- PDF 产物：`main.pdf` 可由 pdfTeX 读取，共 5 页，文件大小 136155 bytes。最新源/PDF 对的 SHA-256 分别为 `6af9d534dec4d590e87028d63d952c8ded7dacb5cad70e98bd6a0467006e0a9d` 和 `34219a78a11471a5c8698f1a9cba2084d1ef61bc2761bb6ae42475be98ea364b`。5 页渲染 PNG 已完成视觉检查，新增服务表格页面无重叠。
- 时间顺序：`main.tex` 的修改时间早于 `main.pdf` 的生成时间，当前 PDF 对应当前源文件版本。
- 关键结果文件存在：全局 pooled selection、split-conformal、matched baselines、独立 ARC worker/service、fingerprint replay、M60 负消融与机制层 radius 对照均存在。

## 发现并保留的历史工件问题

以下两个文件不是当前 v3 主结果的输入，且不是合法 UTF-8 JSON。它们分别是历史传输/编码残留，审计中未删除，以避免篡改历史记录：

- `ARCC_DENSITY_PUBLIC_SUMMARY.json`：内容为 Base64 文本，文件扩展名与内容格式不一致。
- `ARCC_INDEPENDENT_DEDUP_REMOTE_SUMMARY.json`：包含二进制/乱码字节，无法作为 UTF-8 JSON 读取。

当前论文主稿没有引用这两个文件。后续复现实验应使用同目录下的独立 ARC 协议、`arc_q05_*` 结果和对应 Markdown 报告。

## 证据边界

- worker 结果与 Qwen-0.5B service 结果均有对应输出和报告。Qwen-14B service 路径仍保留为诊断，不作为端到端主结论。
- PDF 编译检查验证了排版和引用闭合，不等同于重新执行全部 GPU 实验。
- pooled q=.9 由开发 panel 的 selection residues 冻结，ARC worker 与 ARC service 均使用该同一 pooled quantile。早期 ARC per-panel q=.5 replay 仅作为历史诊断，不属于当前主结果。
- 逻辑 relay token 与增量 service token 的账本范围已在主稿中声明，不代表完整模型加载、排队和端到端延迟。

## 审计结论

当前 v3 源码、结果文件和论文 PDF 在静态层面可复现，未发现会阻断复现的语法、单元测试或 PDF 生成错误。上述两个历史编码残留应在复现说明中标记为弃用输入，不应被解析器当作正式结果。

