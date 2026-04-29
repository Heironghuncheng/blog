---
title: "RAD-2阅读"
date: 2026-04-29T23:00:00+08:00
draft: false
featuredImg: ""
description: '自驾相关的新颖RL算法阅读'
tags:
  - 论文笔记
  - RL
author: BLESS
scrolltotop: true
toc: true
mathjax: true
comments: false
---

# RAD-2阅读

**RAD (Reinforcement Learning for Autonomous Driving)** 系列工作聚焦于自动驾驶中的运动规划，目标是解决强化学习在闭环驾驶中的两大痛点：因果混淆，模型无法确定哪些动作导致了好的结果；开环差距，训练是开环的，不考虑动作执行后的环境反馈，而实际部署是闭环的，导致分布偏移和累计误差。RAD 选择直接构建大规模的 3DGS 闭环仿真环境，用强化学习训练端到端驾驶策略，当然实际训练时先用模仿学习预训练策略然后再RL主训练。首个用3DGS环境的端对端自动驾驶策略训练算法，吃了3DGS的红利。

而2026年4月，华中科大和地平线机器人合作，推出了RAD-2，而且做了实际场景验证，可以说是又一重大提升，思路值得学习。

原论文： https://arxiv.org/abs/2604.15308
原项目： https://hgao-cv.github.io/RAD-2

## 介绍

**RAD-2** 想把扩散模型和强化学习结合起来，使用扩散模型和判别器替代策略网络，充分发挥扩散