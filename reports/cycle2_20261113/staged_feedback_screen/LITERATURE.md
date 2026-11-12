# 最近原始文献与贡献边界

实际检索：2026-11-13，子代理 sampling_priorart 使用 web.run 关键词搜索及原文打开；没有 Google 专用工具，不声称 Google 检索。这是定向核对，非穷尽综述。

1. Neyman (1934), On the Two Different Aspects of the Representative Method. [原文扫描](https://www.stat.cmu.edu/~brian/905-2008/papers/neyman-1934-jrss.pdf)，[出版记录](https://rss.onlinelibrary.wiley.com/doi/10.1111/j.2397-2335.1934.tb04184.x)。分层抽样及最优样本配置属于既有理论。固定层大小、方差与成本下，连续配置 n_h ∝ N_h S_h / sqrt(c_h)，并处理 census 上限。付费获取层标签的成本必须另外加入。
2. Horvitz & Thompson (1952), A Generalization of Sampling Without Replacement From a Finite Universe. [原文](https://stat.cmu.edu/~brian/905-2008/papers/Horvitz-Thompson-1952-jasa.pdf)。HT 处理固定有限总体、不等纳入概率抽样；真实纳入概率必须对应实际设计。硬预算条件化后不能沿用未条件化 proposal。
3. Poiani et al. (2024), Optimal Multi-Fidelity Best-Arm Identification. [原文第2节](https://arxiv.org/html/2406.03033v2)。要求已知每级成本与最大均值偏差界 |mu_a,m - mu_a,M| ≤ xi_m；目标是最高保真最优臂识别。本任务字符串协变量没有已知有效偏差界，且目标是固定银行均值，不可搬用其保证。
4. Grant et al. (2017), Combinatorial Multi-Armed Bandits with Filtered Feedback. [作者预印本](https://arxiv.org/abs/1705.09605)。filtered semibandit feedback 已有研究；以真实结果为条件的过滤观察模型不同于付费接收者探测后选择是否购买终端结果。HTML 原文链接返回404后使用作者原始摘要，不能声称核验全部证明。
5. Robins, Rotnitzky & Zhao (1994), Estimation of Regression Coefficients When Some Regressors are not Always Observed. [出版方摘要](https://www.tandfonline.com/doi/abs/10.1080/01621459.1994.10476818)。逆概率及增广估计属于既有理论，需要相应缺失/概率与正性条件。出版方正文访问403；此处仅依据可读摘要，不声称核对全文证明。

固定银行中，若 m_i 在抽样前固定且真实 pi_i>0 已知，N^-1 sum_i[m_i+I_i(y_i-m_i)/pi_i] 是设计无偏的；未知 proxy 偏差并不必然破坏此恒等式，但也不保证残差方差更小。独立 Bernoulli 方差不适用于未经重算的硬预算条件抽样。

可检验的差异只能来自：在图反事实任务里，同样 exact-wire cache、相同信息权限、计入基图/探测/完成/评分的完整预算后，具体分阶段策略是否改善独立新数据的目标。既有抽样公式、字符串关联、历史 oracle 本身不构成原创方法或未来服务收益证据。

补充定向检索（同日，主统计子代理 web.run）：`linear bandit experimental design pairwise contrasts best arm identification Soare Lazaric Munos 2014`；`graph feedback bandits observability graph observations Alon Cesa Bianchi Dekel Koren 2015`。原始出版来源 [Soare等2014 NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2014/hash/f8d84caae48546d0934d637ab54f7086-Abstract.html) 明确线性BAI与G-optimal实验设计联系；[Alon等2015 COLT](https://proceedings.mlr.press/v40/Alon15.pdf) 研究反馈图如何控制在线学习难度。本次只核对研究模型主题，不声称其定理已适用于图反事实状态购买。
