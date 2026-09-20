---
title: "不用梯度的学习"
date: 2026-06-30T21:00:00+08:00
draft: false
featuredImg: ""
description: '一种可能的学习新范式的思考'
tags:
  - 论文笔记
  - RL
author: BLESS
scrolltotop: true
toc: true
mathjax: true
comments: false
---

# 前言

五月初，OpenAI 研究员翁家翌提出了一个新的可能的学习范式 Heuristic Learning ，这哥们是 tianshou 和 envpool 的主要贡献者，也是 RL 领域的重要贡献者。这个范式目前没有一篇正式论文，但是翁研究员写了一篇博客仔细分析了这个范式，有官中，说实话我没太跟上他的思路，所以结合评论区一篇和 HL 可能相关的论文我觉得得仔细了解一下这个范式，主要这范式一提，我想到之前一些工作中的创新也和这个新范式有点关系，值得思考。

Learning Beyond Gradients: https://trinkle23897.github.io/learning-beyond-gradients

Meta Harness: https://arxiv.org/abs/2603.28052

