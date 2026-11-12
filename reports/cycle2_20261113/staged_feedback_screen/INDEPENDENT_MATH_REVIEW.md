# 独立数学审查回执

审查者：协作代理 sampling_priorart。2026-11-13。只读取本目录 frontier_math.py、run_screen.py、初版 RESULTS.json、FRONTIER.csv，未执行数据脚本或读取项目原始数据；后续单独核查 MATHEMATICAL_NOTE.md 的解析退化推导。最终代码重跑并非该审查者的独立数据重建。

通过：有限银行SRS方差、全probe Neyman配置、独立两阶段HT方差、按probe触达的Poisson/SRS基图费用。共同exact-wire aggregate免调用实现公平。初版65个lazy且p1=p2=1点与直接Poisson的最大方差差1.08e-19。

发现并处理：原docstring的Global optimum声明不适用于带concave lazy成本的单次bounded搜索，已删除，增加129点网格与局部谷值核查；非退化场景保留数值解措辞。新增直接Poisson比值，避免把相对SRS微差误称响应筛选收益。

必须保留：评分等价比1存在小幅非平凡oracle收益；评分费不是实测；未实现完整跨左右图/节点物理复用；期望费用不是硬预算保证；行级设计方差不是新题总体不确定性。

解析证书独审：q′和dV/dt式正确；非负组成本保证B′(q)≥B′(1)，两方向margin非负足以证明指定uniform-probe、独立两层Poisson类所有正预算t=1最优；q=1饱和段衔接正确。margin为负不证明改进。公式审查置信度高。
