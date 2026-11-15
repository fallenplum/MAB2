# R1 纯展示修订 v2

将原 change/k2 的九方法成本—质量散点改为共享方法行的两联图：左完整逻辑 token bars，右正确率 points。固定原方法顺序，`no_graph_ablation` 改显示名为 `Priority w/o graph`，明确其为优先级消融。原 R1 源码、图和所有结果保持不变。

rendered/r1_main_cost_quality.csv 与原 CSV 逐字节一致；45 个数值字段逐一与最终 R1_RESULTS 比较相等。图中数字仅显示时四舍五入，不改源数值。没有新增区间、误差条、统计门或推断。当前图只使用 R1 主分析，不使用曾标错 A/B 的 architecture run_001。

已由 Python/matplotlib 输出 SVG、PNG 与 PDF；PNG 逐项视检九行标签、横轴单位、数值注释、共享行位置和脚注，无遮挡或裁剪。完整 caption 说明 I+C+S 逻辑成本、8 episode、192 S 题、均匀精确期望、共同 cap 与非等实际花费。

复现使用 `plot_r1_presentation_v2.py --help` 指定原 CSV、最终结果、完成回执、独立 replay、原 figure manifest 与原 Python plot 文件；输出必须是新目录。代码只读汇总和源文件，不读取原题、答案或逐题数组，不进行推理。
