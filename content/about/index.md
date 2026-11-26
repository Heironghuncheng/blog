+++
title = "关于 Xinquan Shao"
date = "2023-11-14"
layout = "about"
name_zh = "邵鑫泉"
role = "硕士研究生"
affiliation = "山东大学 控制科学与工程学院"
location = "济南，中国"
github = "https://github.com/Heironghuncheng"
email = "blessingbrandro@gmail.com"
orcid = "0009-0005-7999-2907"
linkedin = "https://www.linkedin.com/in/xinquan-shao-74b163416/"
focus = ["大模型推理系统", "强化学习与多智能体", "微电网优化调度"]
scholar = "https://scholar.google.com/citations?user=ldx77oQAAAAJ&hl=zh-CN&oi=ao"
portrait = "cv/xinquan_shao_bio.png"
github_user = "Heironghuncheng"
notes_tag = "论文笔记"
notes_count = 2
github_pr_count = 5
github_note = "最近在做大模型推理系统、强化学习训练性能分析和开源工具；参与了 Kinetic 与 CodexPlusPlus 的开发。"

[github_fallback]
repos = 8
followers = 19
since = "2021"
prs = 3

[[education]]
period = "2025.09 – 2028（预计）"
degree = "控制工程 · 硕士"
school = "山东大学 控制科学与工程学院"

[[education]]
period = "2020.09 – 2024.07"
degree = "自动化 · 本科"
school = "山东大学 控制科学与工程学院"

[[award]]
year = "2026"
title = "国家奖学金"

[[award]]
year = "2025"
title = "“华为杯”中国研究生数学建模竞赛二等奖"

[[publication]]
year = "2026"
title = "LLM-Driven Reward Evolution for Integrated Energy System Scheduling under Ladder-Type Carbon Trading"
authors = "Xinquan Shao, Kai Zhao, Qianwen Xu, Xin Yu, Shuai Liu"
venue = "SSRN"
venue_short = "SSRN"
badge = "PREPRINT"
group = "⚡ 能源系统与强化学习"
status = "SSRN 预印本（2026）"
summary = "将综合能源调度的任务约束与奖励接口交给 LLM 生成并迭代，在阶梯式碳交易下训练调度策略。"
highlights = ["将调度约束、状态变量与奖励接口组织成提示词，让 LLM 生成可执行的 Python 奖励函数。", "在阶梯式碳交易场景下迭代奖励候选，用独立指标比较调度策略。"]
doi = "https://doi.org/10.2139/ssrn.7540426"
pdf = "https://papers.ssrn.com/sol3/Delivery.cfm?abstractid=7540426"
cover = "images/papers/ssrn.jpg"

[[publication]]
year = "2026"
title = "Fully Distributed Reinforcement Learning for Efficient Networked Microgrids Resource Management: Source-Load-Storage Coordination"
authors = "Xiaowen Wang, Xinquan Shao, Shuai Liu"
venue = "IEEE/CAA Journal of Automatica Sinica"
venue_short = "JAS"
badge = "PUBLISHED"
summary = "面向网络化微电网源-荷-储协调的全分布式强化学习调度。"
highlights = ["面向网络化微电网的源-荷-储协调，采用全分布式强化学习。", "论文算例中，事件触发通信将仿真通信次数降低 52.7%。"]
doi = "https://doi.org/10.1109/JAS.2025.125636"
cover = "images/papers/jas.jpg"

[[publication]]
year = "2025"
title = "Distributed Multi-Agent Reinforcement Learning for Multi-Objective Optimal Dispatch of Microgrids"
authors = "Xiaowen Wang, Shuai Liu, Qianwen Xu, Xinquan Shao"
venue = "ISA Transactions"
venue_short = "ISA"
badge = "PUBLISHED"
summary = "面向微电网多目标优化调度的分布式多智能体强化学习。"
highlights = ["面向微电网多目标优化调度的分布式多智能体强化学习。", "我负责算法代码与仿真实现。"]
doi = "https://doi.org/10.1016/j.isatra.2025.01.009"
cover = "images/papers/isa.jpg"

[[publication]]
year = "2026"
title = "A Piezoelectric Multi-Robot System for Macro Movement and Micro Manipulation"
authors = "Jinshuo Liu, Xinquan Shao, Ni Sui, Jianning Gao, Zijian Yin, Zihan Zhu, Jiang Wu, Shuai Liu, Rui Song, Yibin Li"
venue = "IEEE Transactions on Industrial Electronics"
venue_short = "TIE"
badge = "ACCEPTED"
group = "🤖 多机器人系统"
status = "已录用（2026）"
summary = "压电多机器人系统的宏观运动与微观操作。"
highlights = ["压电多机器人系统的宏观运动与微观操作。", "我负责多机器人控制算法与上位机视觉闭环；论文已录用。"]
doi = "https://doi.org/10.1109/TIE.2026.3737842"
cover = "images/papers/tie.jpg"
+++

我目前在山东大学控制科学与工程学院读硕士，学习 AI Infra，主要关注大模型推理系统与性能优化。之前做过多智能体强化学习、微电网优化调度和大模型工具开发，关注集中式训练与分散式执行（CTDE），以及零和与一般和博弈。

这个博客记录我的论文笔记、代码复现和实验观察。我习惯把论文、实验环境和代码放在一起，方便复现结果，也方便把论文里的方法用到实际项目中。
