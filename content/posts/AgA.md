---
title: "AgA阅读"
date: 2025-10-15T23:00:00+08:00
draft: false
featuredImg: ""
description : 'NIPS2024论文笔记——AgA'
tags: 
  - 论文笔记
  - MARL
author : BLESS
scrolltotop : true
toc : true
mathjax : true
comments: false
---

# 多智能体协作中个体与集体目标的对齐

看着很高级，还是国人写的，感觉应该会很有收获，作者主要在曼彻斯特大学和上交，可惜没代码，什么时候复现试试。

论文地址： https://nips.cc/virtual/2024/poster/96810

## 前置知识

我吃了不会数学的亏，先研究了一下博弈。

简单入门基础概念可以看 https://copy2049.github.io/blog/html/2020/11/Game-Theory.html ，感觉不错，短小精悍。

### 混合动机博弈

**Mixed-Motive Game**，是指合作和竞争并存、非零和的博弈。合作和竞争是指智能体既要考虑整体目标，又要考虑个体目标，既有可能合作扩大整体利益，也有可能破坏合作扩大自身利益；非零和是指智能体间的总体利益不是一个常数。

### 可微分博弈

**differential game**，一种运用在动态系统中的博弈理论，主要研究在连续时间的过程中，多方参与者如何进行策略选择，以实现自己的目标。这种博弈和 MARL 的随机博弈略有差异，侧重点不一样：可微博弈强调策略可微，重点研究的是优化方法；随机博弈强调环境的动态和不确定性，重点研究的是策略互动。

## 摘要

作者将多智能体学习中的混合动机博弈建模为可微分博弈，以解决个体目标和集体目标的协调问题，避免人工的奖励设计。提出了名为**利他梯度调整（AgA）**的优化方法，通过梯度调整逐步协调个体与集体目标，从理论上证明了 AgA 在兼顾个体利益的同时能将梯度有效吸引至集体目标的稳定不动点，并做了验证。

## 介绍

多智能体合作主要分为两大研究方向：**纯动机合作**和**混合动机合作**。纯动机合作是指智能体只考虑整体目标，竞争不明显；混合动机合作是指智能体既要考虑整体目标，又要考虑个体目标，竞争与合作并存。

在混合动机合作方向下，目前促进合作的方式依赖于人工设计某些机制，譬如：
- **声誉体系**：通过记录智能体过去的行动（例如是否合作、是否遵守规则），为其分配声誉分数。其他智能体会根据声誉选择是否与其交互；
- **规范约束**：智能体集体可以制裁不遵循社会规范的智能体，规范约束减少不合作或有害行为；
- **契约机制**：智能体通过"签约"同意遵循特定规则，执行特定任务，并根据契约条款获得奖励或惩罚。

还有将个体奖励与集体奖励相融合的研究；偏好奖励研究，通过分析其他智能体的奖励或行为，偏好信号为智能体提供"社会线索"，鼓励利他；学习智能体行为对他人产生的潜在影响。

贡献总结这块有个点值得注意一下，作者开发了一个名为 **selfish-MMM2** 的大规模混合动机协作环境，同时在多个场景里验证了所提出的算法。

## 背景介绍

### 可微分博弈

微分博弈可以定义为 \(\{N, w, l\}\)，其中 \(N = \{1, \ldots, n\}\) 表示玩家集合，\(w = [w_i]^n \in \mathbb{R}^d\) 代表所有玩家的参数集合，其中 \(w_i \in \mathbb{R}^d\) 是第 \(i\) 个玩家的参数向量，\(d = \sum_{i=1}^n d_i\) 是每个玩家参数向量的维度的综合。损失函数定义为 \(l = \{l_i : \mathbb{R}^d \to \mathbb{R}\}_{i=1}^n\)，损失函数至少二阶可微，每个玩家 \(\{i \in N\}\) 都有一个策略，这个策略由参数 \(w_i\) 确定，目标是最小化损失函数 \(l_i\)。

定义 \(\xi(w) = (\nabla_{w_i} l_i, \ldots, \nabla_{w_n} l_n) \in \mathbb{R}^d\) 为玩家的梯度集合，命名为**同时梯度**，\(\nabla_{w_i} l_i\) 是损失函数 \(l_i\) 对玩家 \(i\) 参数 \(w_i\) 的梯度。微分博弈中**黑塞矩阵**被定义为同时梯度的雅各比矩阵，即 \(H(w) = (\nabla_{w_i} \nabla_{w_j} l_i, \ldots, \nabla_{w_i} \nabla_{w_n} l_n) \in \mathbb{R}^{d \times d}\)。

黑塞矩阵是一个多元函数所有二阶偏导构成的矩阵，雅各比矩阵则是对一个多元函数在某一个点上求梯度的结果。这样描述不够准确，给出严谨定义：假设某多元函数从 \(f : \mathbb{R}^n \to \mathbb{R}^m\) 即从向量 \(x \in \mathbb{R}^n\) 映射到向量 \(f(x) \in \mathbb{R}^m\)，其雅各比矩阵是一个 \(m \times n\) 的矩阵：

$$
\mathbf{J}
= \left[ \frac{\partial \mathbf{f}}{\partial x_1} \quad \cdots \quad \frac{\partial \mathbf{f}}{\partial x_n} \right]
= \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}
$$

不严谨地说，是指每个方向的分函数对于输入向量的每个元素的偏导，组成一个 \(m \times n\) 的矩阵，或者说对每个方向的分函数求梯度，组成一个 \(m \times n\) 的矩阵。

由于同时梯度是每个玩家的损失对自身参数的梯度，即

$$
\xi(w)=
\begin{bmatrix}
\nabla_{w_1}l_1 \\[4pt]
\nabla_{w_2}l_2 \\[4pt]
\vdots \\[4pt]
\nabla_{w_n}l_n
\end{bmatrix}
\in \mathbb{R}^{d}
$$

由于参数向量 \(w\) 是 \(d\) 维向量，所以同时梯度也是 \(d\) 维向量，再对 \(w\) 计算雅各比矩阵，得到黑塞矩阵：

$$
H(w) = \mathbf{J}^T \mathbf{J}
= \begin{bmatrix}
\nabla_{w_1} \nabla_{w_1} l_1 & \cdots & \nabla_{w_1} \nabla_{w_n} l_n \\
\vdots & \ddots & \vdots \\
\nabla_{w_n} \nabla_{w_1} l_1 & \cdots & \nabla_{w_n} \nabla_{w_n} l_n
\end{bmatrix}
$$

其中每个元素 \(H_{ij}= \frac{\partial^{2} l_i}{\partial w_i \partial w_j}\) 表示玩家 \(i\) 的损失函数 \(l_i\) 关于自身参数 \(w_i\) 与玩家 \(j\) 的参数 \(w_j\) 的二阶混合偏导，刻画了玩家 \(j\) 的策略变化对玩家 \(i\) 的交叉影响，其对角线元素为玩家 \(i\) 损失函数 \(l_i\) 关于自身参数 \(w_i\) 的二阶偏导，即 \(H_{ii} = \frac{\partial^{2} l_i}{\partial w_i^2}\)。这就是微分博弈中黑塞矩阵被定义为同时梯度的雅各比矩阵的含义。

作者提到的微分博弈的**学习动态**是一阶方法例如梯度下降，不显式使用黑塞矩阵，可以表示为：

$$
w_{t+1} \leftarrow w_t - \alpha_t \xi(w_t)
$$

其中 \(\alpha_t\) 是学习率，\(\xi(w_t)\) 是同时梯度。这一更新规则意味着所有玩家在同一时刻沿着自身损失函数的负梯度方向同步调整参数，各自独立地最小化自身损失。

### 梯度调整优化

**梯度调整优化**就我理解是指在微分博弈中如何让玩家的策略向**稳定不动点**方向调整，与 MARL 结合，其核心应该是促进收敛，减少循环和不稳定性。

首先是**不动点**的概念，在微分博弈中，若 \(w^*\) 是一个不动点，则说明该点处**同时梯度** \(\xi(w^*) = 0\)，即所有玩家没有动力更新其策略；在不动点 \(w^*\) 处，若**黑塞矩阵** \(H(w^*)\) 为半正定矩阵且可逆时，\(w^*\) 称为**稳定不动点**，若黑塞矩阵 \(H(w^*)\) 为负定矩阵，则 \(w^*\) 称为**不稳定不动点**。

一种朴素的引导学习动态收敛到不动点的思路是，最小化同时梯度的平方范数，也是构造了一个**李雅普诺夫函数**，即：

$$
\min{\frac{1}{2} \| \xi(w) \|^2}
$$

因为，这个就不推了，我矩阵论水平很次

$$
\nabla{\frac{1}{2} \| \xi(w) \|^2} = H^T\xi
$$

当 \(H(w^*)\) 可逆时，\(H^T\xi = 0\) 等价于 \(\xi = 0\)，即在黑塞矩阵可逆时，最小化同时梯度的平方范数，等价于寻找一个不动点，直观地看这个点可能是不稳定的。

还有一种叫**共识优化（CGA）**的方法，引入梯度调整项，然后使用调整过的同时梯度进行更新：

$$
\tilde{\xi} = \xi + \lambda \cdot \nabla\tfrac{1}{2} \| \xi(w) \|^2
$$

还有**辛梯度下降（SGA）**，也引入了一个修改项，即：

$$
\tilde{\xi} = \xi + \lambda \cdot A^T\xi
$$

其中 \(A^T\) 是黑塞矩阵的**广义亥姆霍兹分解**，简单说就是对称反对称分解，即：

$$
S = \frac{1}{2} (H(w) + H(w)^T)
A = \frac{1}{2} (H(w) - H(w)^T)
$$

\(A\) 是黑塞矩阵的**反对称部分**，在零和博弈中，\(A \approx H\) SGA 退化为 CGA，一般和博弈中，只使用反对称部分更新，为什么效果好，我没看懂，效果好的原因参见 https://arxiv.org/abs/1802.05642。

## 主要方法

### 混合动机微分博弈

**可微分混合动机博弈（Differentiable Mixed-Motive Game, DMG）**，基本定义与微分博弈一致，区别在于可微的损失函数具有**混合动机特性**：个体目标与集体目标之间存在冲突；除去同时梯度外，定义**集体损失梯度** \(\xi_c(w) = (\nabla_{w_1} l_c, ..., \nabla_{w_n} l_c)\)。

微分混合动机博弈中存在**对齐困境**，分别优化个体损失和集体损失，会导致玩家策略向不同方向调整。如果使用梯度下降法对集体损失进行优化，如果存在**奇异点**，即梯度为零但黑塞矩阵奇异的点，那么梯度下降无法收敛到这个点。

作者举了一个例子阐述他的优化方法：

一个两玩家的可微分博弈，玩家 1 的损失函数为 
$$l_1(a_1, a_2) = -\sin(a_1 a_2 + a_2^2)$$
玩家 2 的损失函数为
$$l_2(a_1, a_2) = -[\cos(1+a_1-(1+a_2)^2+a_1a_2^2)]$$
其中 \(a_i\) 表示玩家 \(i \in \{1, 2\}\) 的动作变量，且 \(a_i \in \mathbb{R}\)，每个玩家的奖励为损失函数的相反数，定义 \(l_c = l_1 + l_2\) 为**集体损失函数**。

作者对比了四种方法：没有改进的学习动态、CGA、SGA、使用集体损失的学习动态。结论是，前三种方法只能将解收敛到不稳定点和局部最大值，而使用集体损失可以将解收敛到集体奖励的稳定不动点，但是玩家 1 的奖励没有收敛到一个不动点，这表明更新没有考虑到玩家 1 的个体目标。

### AgA - 利他梯度调整

类似于 CGA 和 SGA 的思路，**AgA** 引入了一个**梯度调整项**，即：

$$
\tilde{\xi} = \xi_c + \lambda\xi_{adj} = \xi_c + \lambda (\xi + H_c^T\xi_c)
$$

其中 \(\lambda \in \mathbb{R}\) 是一个**对齐系数**，\(\lambda\xi_{adj}\) 是梯度调整项，\(\xi_c\) 是**集体损失梯度**，在集体梯度的基础上，调整项给它做修正，\(H_c^T\) 是集体损失黑塞矩阵的转置。注意，黑塞矩阵 \(H_c\) 是一个对称矩阵，该形式只是为了和 CGA 和 SGA 的形式保持一致，定义 \(\nabla \mathcal{H}_c\) 为 \(\nabla_w\frac{1}{2} \| \xi(w) \|^2\)，等价于 \(H_c^T\xi_c\)，前文有说明，在该公式里不需要直接算黑塞矩阵而是计算黑塞矩阵和同时梯度向量的积，速度很快。

在选择合适的 \(\lambda\) 时，AgA 可以确保稳定地收敛到**稳定不动点**即
$$
\theta(\tilde{\xi}, \nabla \mathcal{H}_c) \leq \theta(\xi_c, \nabla \mathcal{H}_c) 
$$

也会远离**不稳定的不动点**即

$$
\theta(\tilde{\xi}, \nabla \mathcal{H}_c) \geq \theta(\xi_c, \nabla \mathcal{H}_c) 
$$

当且仅当 \(\lambda\) 满足

$$
\lambda \cdot \langle\xi_c, \nabla\mathcal{H}_c\rangle\,(\langle\xi, \nabla\mathcal{H}_c\rangle + \|\nabla\mathcal{H}_c\|^2) \geq 0.
$$

其中 \(\theta(a, b)\) 是向量夹角，\(\langle a, b\rangle\) 是向量内积，使用 \(\tilde{\xi} = u + \lambda v\) 表示 AgA 的调整梯度，\(u\) 是 \(\xi_c\)，\(v\) 是 \(\xi + H_c^T\xi_c\)。

定义 \(\theta_\lambda(\tilde\xi, w)\) 是 AgA 梯度和参考更新方向的夹角。

拓展**微分对齐**的定义到 AgA 梯度，使用余弦相似度的平方定义 AgA 梯度与参考更新方向的吻合程度，考虑局部变化所以对 \(\lambda\) 求导，反映 \(\lambda = 0\) 的两侧 AgA 梯度的变化率，即：

$$
\text{align}(\tilde\xi, w) := \frac{d\{\cos^2\theta_\lambda\}_{|\lambda=0}}{d\lambda}
$$

直观上，当 \(u^Tw \geq 0\)，\(u\) 与 \(w\) 同方向时，若 \(\text{align} > 0\)，\(v\) 驱动 \(u\) 向参考方向靠拢，若 \(\text{align} < 0\)，\(v\) 推动 \(u\) 向远离参考方向，若 \(u\) 与 \(w\) 方向不一致，\(v\) 会牵引 \(u\) 向参考方向趋近。

令 \(w = \nabla \mathcal{H}_c\) 给出**对齐方向判定法**，

$$
\text{sign}(\text{align}(\tilde\xi, \nabla \mathcal{H}_c)) = \text{sign}(\langle\xi_c, \nabla\mathcal{H}_c\rangle(\langle\xi, \nabla\mathcal{H}_c\rangle + \|\nabla\mathcal{H}_c\|^2))
$$

NIPS2024 下载的原文这块有个公式脚标写错了，证明结论和给出的公式不一样，这块给出的公式是修改过的了，证明过程就不写了，概括为把余弦相似的平方展开，带入 \(\tilde\xi\) 的定义式，分子对 \(\lambda\) 做一阶等价无穷小近似，然后进一步展开，分母为平方项，微分对齐的定义为 \(\lambda = 0\) 时的一阶导的值，公式的一阶导的符号与分母无关，由泰勒恒等式分子的一阶导为上文提到的公式。

对于对齐方向判定公式，对于 \(\langle\xi_c, \nabla\mathcal{H}_c\rangle\) 显然等于 \(\xi_c^T H_C \xi_c\)，由于 \(H_C\) 是对称矩阵，假设 \(\xi_c \neq 0\)，有若 \(H_c\) 半正定，则 \(\langle\xi_c, \nabla\mathcal{H}_c\rangle \geq 0\)，若 \(H_c\) 负定，则 \(\langle\xi_c, \nabla\mathcal{H}_c\rangle < 0\)。

此处相当于是对前文的直观理解给出了一个**分类讨论**：

- **情形一（稳定不动点附近）**：
  若 \(\langle \xi_c, \nabla \mathcal{H}_c \rangle \ge 0\)，则 \(\xi_c\) 与 \(\nabla \mathcal{H}_c\) 指向相同方向，即 \(\theta(\xi_c, \nabla \mathcal{H}_c) \le \frac{\pi}{2}\)。当 \(\langle \xi_c, \nabla \mathcal{H}_c \rangle \ge 0\) 时，align\((\tilde{\xi}, \nabla \mathcal{H}_c)\) 的符号与 \(\langle \xi, \nabla \mathcal{H}_c \rangle + \|\nabla \mathcal{H}_c\|^2\) 的符号一致。若令
  
  $$
  sign(\lambda) = sign(\langle \xi, \nabla \mathcal{H}_c \rangle + \|\nabla \mathcal{H}_c\|^2) = sign(align)
  $$
  
  有：
  - 当 sign(align) \(\ge 0\) 时，sign\((\lambda) \ge 0\)，向量 \(v\) 将把 \(u\) 拉向 \(w\)；
  - 当 sign(align) \(< 0\) 时，sign\((\lambda) < 0\)，负的 \(\lambda\) 反向作用，使 \(v\) 将 \(u\) 推离 \(w\)
  
  因此，只要令 sign\((\lambda)=\)sign\((\langle \xi, \nabla \mathcal{H}_c \rangle + \|\nabla \mathcal{H}_c\|^2)\)，\(v\) 均能确保 \(u\) 被拉向 \(w\)。
  
  在稳定不动点邻域内，若令
  
  $$
  \lambda \cdot \langle \xi_c, \nabla \mathcal{H}_c \rangle (\langle \xi, \nabla \mathcal{H}_c \rangle + \|\nabla \mathcal{H}_c\|^2) \ge 0
  $$
  
  则 AgA 梯度 \(\tilde{\xi}\) 相比 \(\xi_c\) 更靠近 \(\nabla \mathcal{H}_c\)。

- **情形二（不稳定不动点附近）**：
  证明思路与情形一相似。若令 sign\((\lambda)\) 满足同样条件
  
  $$
  \lambda \cdot \langle \xi_c, \nabla \mathcal{H}_c \rangle (\langle \xi, \nabla \mathcal{H}_c \rangle + \|\nabla \mathcal{H}_c\|^2) \ge 0
  $$

则优化过程具有下述性质：在不动点邻域，
1) 若该点稳定，AgA 梯度被拉向该点，即 \(\theta(\tilde{\xi}, \nabla \mathcal{H}_c) \le \theta(\xi_c, \nabla \mathcal{H}_c)\)；
2) 若该点不稳定，AgA 梯度被推出该点，即 \(\theta(\tilde{\xi}, \nabla \mathcal{H}_c) \ge \theta(\xi_c, \nabla \mathcal{H}_c)\)。

这个证明很大程度上是说为什么 \(\lambda\) 要满足 \(\lambda \cdot \langle \xi_c, \nabla \mathcal{H}_c \rangle (\langle \xi, \nabla \mathcal{H}_c \rangle + \|\nabla \mathcal{H}_c\|^2) \ge 0\)，没有说明一定会收敛到稳定不动点，只是说明在稳定不动点附近，\(\lambda\) 满足条件时，AgA 梯度会向稳定不动点靠拢，可能在别的论文里证明过了等等。

## 其他

其他都是实验了，先写这么多，等到复现成了这份工作再详细介绍原文中只提了一小段的 AgA 怎么和 MAPPO 集成。

