---
title: "从 Kinetic 认识 AI Infra"
date: 2026-11-25T22:00:00+08:00
draft: false
featuredImg: ""
description: '以 Veloresearch/kinetic 为线索，理解本地 LLM 推理引擎的设计与实现'
tags:
  - AI Infra
  - 推理引擎
  - GGUF
  - Rust
  - Kinetic
author: BLESS
scrolltotop: true
toc: true
mathjax: true
comments: false
---

# 从 Kinetic 认识 AI Infra

这是一份学习笔记，不是教程。

我刚开始补 AI Infra 的时候，最难受的不是概念难，而是概念之间没有线。量化、KV Cache、prefill、batch，每个词单看都能查到一个定义，合在一起就不知道谁先谁后、谁管谁。后来我换了个办法：找一个真实仓库，顺着一次请求走一遍。

Kinetic 是那个仓库。它用 Rust 写，读 GGUF，能在 CPU 和 GPU 上跑，自带 OpenAI 兼容服务。代码不算少，但还没有多到读不完；更重要的是，一个请求从进来到出去，中间该有的东西它都有。

下面是我读的时候记的东西。有查来的资料，有读代码得出的结论，也有我自己的理解。不确定的地方我会尽量标出来。要是有讲错的地方，欢迎指正。

**基线。** 除特别说明，仓库结论基于 `Veloresearch/kinetic` 的 `main` 分支提交 `1b671b9`（2026-11-25）。代码结论来自静态阅读，不是运行结果；性能、兼容性和逐位一致性，仍要在对应设备、模型和配置上实测。

两个 PR 的复盘放在 `kinetic-pr.md`。这篇只讲引擎本身，不展开具体 PR。

## 阅读建议

我没有从头到尾读。

第一遍先看概念，把 prefill、decode、KV Cache、量化这几个词的关系理顺。第二遍才进仓库，看一次请求怎么从 HTTP 走到 kernel。第三遍挑自己感兴趣的部分，比如投机解码和设备后端。

笔记也按这个顺序分了三块。第 1–4 章是概念，第 5–14 章是仓库，第 15–19 章是动手。第一次读可以只看每章开头和表格，等需要时再回来查细节。

难度标记只是我自己的感觉。带一个星号的，是我觉得应该能用自己的话讲一遍的；带两个星号的，是能讲清条件和取舍的；带三个星号的，读源码时再看，不必背。

如果时间不多，我是这么读的：先看第 2、10、11 章，把推理引擎的主干搭起来；再回头补 GGUF 和 API；最后按第 15 章的路线读代码。

# 第一部分 AI Infra 基础

## 1. AI Infra 的定位

AI Infra 这个词听起来很大，其实可以先理解成一件事：让模型在真实机器上跑得起来，而且跑得不至于太贵、太慢。它分训练和推理两块。我一开始是从推理入门的，原因很实际——推理在笔记本上就能复现，改一行代码，吞吐数字马上就变，反馈比训练快得多。

一个请求从按下回车到屏幕上出现第一个字，中间大概会经过这么几层。

| 层 | 负责什么 | 典型代表 |
|---|---|---|
| 应用层 | 聊天界面、Agent、RAG | LM Studio、Open WebUI |
| 接口层 | 请求格式、流式输出 | OpenAI API、gRPC |
| 服务层 | 排队、批处理、限流、多用户 | vLLM、TGI、kinetic-server |
| 推理引擎 | 前向计算、KV Cache、采样 | llama.cpp、kinetic、TensorRT-LLM |
| 运行时与内核 | 矩阵乘、注意力、SIMD | CUDA、Vulkan、AVX2 |
| 硬件 | 算力、显存、内存带宽 | CPU、NVIDIA / AMD / Intel GPU |
| 模型文件 | 权重、词表、模板 | GGUF、safetensors |

我第一次看这张图的时候，最容易混的是服务层和推理引擎。后来才分清楚：服务层管排队、协议、多用户，推理引擎管一次前向怎么算、KV Cache 放哪、下一个 token 怎么选。两边都叫“推理”，但管的事不一样。

### 1.1 训练与推理的分工

训练追求吞吐和显存容量：一次处理大量样本，反向传播，更新权重。推理追求延迟和成本：一次处理一个或一批请求，只做前向计算，权重不变。

两者对硬件的要求不一样。训练需要大显存、高速互联和稳定的长时间运行；推理更在意单次请求的首 token 延迟、每 token 延迟，以及每块 GPU 能同时服务多少用户。

本地推理又是另一个场景。你只有一张或两张消费级 GPU，甚至只有 CPU；模型必须放得下，速度必须能接受。Kinetic 面向的就是这个场景，所以它的很多设计，云端引擎不会做。

### 1.2 本地推理与云端推理

| 维度 | 本地推理 | 云端推理 |
|---|---|---|
| 数据 | 不出本机 | 离开本机 |
| 硬件 | 个人电脑，单卡或 CPU | 集群，多卡，专用加速器 |
| 优化目标 | 单请求延迟、内存带宽 | 吞吐、并发、成本 |
| 模型大小 | 受内存和显存限制 | 可以很大，可切分 |
| 典型引擎 | llama.cpp、Kinetic、Ollama | vLLM、TensorRT-LLM、TGI |

本地引擎的很多设计，云端引擎不会做：量化到 4 bit、把层拆到 CPU 和 GPU、把专家留在内存、按前缀复用 KV Cache。反过来说，云端引擎的 PagedAttention、多租户调度、跨节点并行，本地引擎也未必需要。

这不是谁更先进的问题，而是约束不同。理解约束，才能理解设计。

### 1.3 三个指标

- **TTFT（Time To First Token）**：从发出请求到收到第一个 token 的时间。主要受 prefill 和排队影响。
- **TPOT（Time Per Output Token）**：第一个 token 之后，平均每个 token 的时间。主要受 decode 和内存带宽影响。
- **吞吐（tokens/s）**：单位时间生成的 token 数。单请求时约等于 `1 / TPOT`；多请求时还要看批处理效率。

还有一个常被忽略的指标：**goodput**。它只统计满足延迟目标的请求。一个系统吞吐很高，但每个请求都超时，goodput 仍然很低。只答吞吐，解释不了延迟敏感的场景。

### 1.4 三个“一样”的合同

“输出一样”这句话很模糊。读推理引擎时，至少分清三种。

第一种是**逐位一致**：两个张量的每个浮点数完全相同。批内 logits、回滚后的状态，常用这种断言。

第二种是**逐 token 一致**：贪心解码的 token 序列相同；带种子的采样，token 序列也相同。大多数“加速不改变答案”的承诺指这一层。

第三种是**逐字节一致**：量化工具写出的文件字节相同。比如 `kinetic quantize` 和 `llama-quantize` 在受支持预设下的对比。

同一设备、同一路径上可以做到逐位；换设备、换内核、换 KV 精度，通常只能承诺逐 token 或更弱。后文会反复用到这三层，尤其是讲精确性和 CUDA 诊断的时候。

## 2. 推理引擎的职责

### 2.1 一次请求的生命周期

先跟着一个请求走一遍。

它先从磁盘上读到一个模型文件，解析出超参数、词表和权重。然后用户的消息被套进对话模板，变成模型训练时见过的格式；再被分词器切成 token id。接下来是 prefill：整段提示一次性过一遍模型，算出 KV Cache，拿到第一个 token 的 logits。之后进入 decode：每次只输入一个新 token，读历史 KV Cache，算出下一个 token。采样器从 logits 里选一个，流式输出给客户端，遇到停止符就结束。

这七步看起来简单，但每一步都有工程问题。模板是否和训练一致，分词是否和参考实现一致，KV Cache 放在哪，采样是否可复现，流式输出怎样处理半个 UTF-8 字符。推理引擎的复杂度，主要不在“模型怎么算”，而在这些边界上。

### 2.2 Prefill 与 Decode 是两种负载

推理引擎里最重要的一个区分，是 prefill 和 decode。

| 维度 | Prefill | Decode |
|---|---|---|
| 一次处理 | 整段提示，L 个 token | 1 个 token |
| 矩阵形态 | 矩阵 × 矩阵（GEMM） | 矩阵 × 向量（GEMV） |
| 瓶颈 | 算力（compute-bound） | 内存带宽（memory-bound） |
| 指标 | TTFT，首 token 延迟 | TPOT，每 token 延迟 |

Decode 阶段每生成一个 token，都要把模型权重从头读一遍。算术量很小，搬运量很大，所以速度上限由内存带宽决定：

$$
\text{tokens/s} \lesssim \frac{\text{有效带宽}}{\text{每个 token 读取的字节数}}
$$

举个例子。一个 4B 模型用 Q4_K_M 存成约 2.8 GB，笔记本 DDR5 的有效带宽按 50 GB/s 算，理论上限约 18 token/s；RTX 3060 Laptop 的显存带宽约 360 GB/s，上限约 128 token/s。实际值还要打折，但这个估算解释了为什么量化能直接提速，也解释了为什么小模型在本地更实用。

另一个有用的概念是**算术强度**：每读一个字节，能做多少次浮点运算。Prefill 的矩阵乘可以复用读入的权重，算术强度高；Decode 的 GEMV 每个权重只参与一次乘加，算术强度低。GPU 的算力远高于内存带宽，所以 decode 跑不满算力，等的是数据。

### 2.3 KV Cache

自回归生成时，历史 token 的 Key 和 Value 不必每步重算。把它们存下来就是 KV Cache。大小可以估算为：

$$
2 \times n_{\text{layers}} \times n_{\text{kv-heads}} \times d_{\text{head}} \times L \times \text{bytes}
$$

前面的 2 是 K 和 V，GQA 让 $n_{\text{kv-heads}}$ 小于注意力头数。拿 Llama 3 8B 举例：32 层、8 个 KV 头、每头 128 维，上下文 8192，F16 存储：

$$
2 \times 32 \times 8 \times 128 \times 8192 \times 2 \approx 1\ \text{GiB}
$$

上下文越长、并发越多，KV Cache 越大。推理引擎的大量工程都在管理它：

- **复用**：相同前缀不必重算，直接恢复状态。
- **分页**：把 KV Cache 切成块，按需分配，减少碎片。
- **换出**：显存放不下时，换到内存或磁盘。
- **压缩**：把 KV Cache 量化到 8 bit 或更低。
- **滑窗**：只保留最近 N 个位置，旧的丢掉，用环形缓冲。

Kinetic 的 CPU 状态按层保存 KV、卷积或 recurrent 状态；滑窗 KV 是环形缓冲；checkpoint journal 支持撤销和保留选定行。

### 2.4 量化

量化把权重从 16 位浮点压到 4 到 8 位，用更少的字节表示同一组参数。GGUF 里常见 `Q4_K_M`、`Q5_K_M`、`Q6_K`、`Q8_0`。位数越低，模型越小、读得越快，精度损失越大。

量化的基本单位是**块**。一块权重共享一个或几个缩放因子，块内每个权重存整数码。反量化时：

$$
x_i \approx d \cdot q_i + m
$$

其中 $d$ 是缩放因子，$m$ 是最小值偏移，$q_i$ 是整数码。K-quant 用更细的 6 位子块缩放和最小值，所以同样 4 bit 左右，精度比老的 Q4_0 好。

GGUF 的块大小和字节数（来自 `kinetic-quant` 的 `TensorType`）：

| 类型 | 每块元素 | 每块字节 | 约 bit/权重 | 说明 |
|---|---:|---:|---:|---|
| Q4_0 | 32 | 18 | 4.5 | 老格式，一个 f16 缩放 |
| Q4_1 | 32 | 20 | 5.0 | 缩放加最小值 |
| Q5_0 | 32 | 22 | 5.5 | 5 bit |
| Q5_1 | 32 | 24 | 6.0 | 5 bit 加最小值 |
| Q8_0 | 32 | 34 | 8.5 | int8，一个 f16 缩放 |
| Q2_K | 256 | 84 | 2.625 | K-quant，2 bit 级 |
| Q3_K | 256 | 110 | 3.4375 | K-quant，3 bit 级 |
| Q4_K | 256 | 144 | 4.5 | 6 位子块缩放/最小值，4 bit 权重 |
| Q5_K | 256 | 176 | 5.5 | 5 bit 权重 |
| Q6_K | 256 | 210 | 6.5625 | 6 bit 权重 |
| Q8_K | 256 | 292 | 9.125 | 8 bit，带子块缩放 |
| IQ2_XXS | 256 | 66 | 2.0625 | 重要性矩阵量化 |
| IQ2_XS | 256 | 74 | 2.3125 | IQ 系列 |
| IQ3_XXS | 256 | 98 | 3.0625 | IQ 系列 |
| IQ4_XS | 256 | 136 | 4.25 | IQ 系列，4 bit 级 |

注意 `Q4_K_M` 不是一个 `TensorType`，而是一种**量化配方**：大部分张量用 Q4_K，少数敏感张量用 Q6_K 或 Q5_K。所以文件里会同时出现 Q4_K、Q6_K、Q5_K、F32 等类型。

量化的收益和代价，其实是一笔账。收益是模型变小，decode 每步读的字节变少，速度上限提高；显存占用降低，能放更长的上下文或更大的模型。代价是权重有误差，长上下文和难样本上可能退化；不同量化配方和不同内核的舍入不同，输出可能分叉。

### 2.5 采样

模型输出的是 logits，不是 token。采样器把 logits 变成概率，再选一个 token。

常见步骤：

1. **温度**：`logits / temperature`。温度越低，分布越尖；温度为 0 时退化成 argmax（贪心）。
2. **top-k**：只保留概率最高的 k 个 token。
3. **top-p（nucleus）**：按概率从高到低累加，保留到累计概率达到 p。
4. **min-p**：丢掉概率低于“最高概率 × min_p”的 token。
5. **惩罚**：重复惩罚、存在惩罚、频率惩罚，降低已出现 token 的分数。
6. **softmax**：归一化成概率。
7. **随机采样**：按概率抽一个 token；种子固定时结果可复现。

Kinetic 的采样参数包括 `temperature`、`top_k`、`top_p`、`min_p`、`repeat_penalty`、`presence_penalty`、`frequency_penalty`、`seed`，还支持 DRY、Mirostat、XTC、typical、top-n-sigma 等扩展。默认值是 temperature 0.8、top_k 40、top_p 0.95、min_p 0.05。

采样和“精确性”的关系，值得单独说一句。贪心解码只看 argmax，最容易逐 token 对齐；带种子的采样要求随机数序列和 logits 都一致，才能给出同一答案。投机解码的验证必须使用与目标采样策略相同的接受/拒绝规则，否则分布会变。

### 2.6 批处理

单请求时，GPU 的很多单元是空闲的。批处理把多个请求放进同一次前向，复用权重读取，提高吞吐。

两种批处理。静态批处理一次凑齐一批，跑完再换下一批，实现简单，但短请求会被长请求拖住。连续批处理（continuous batching）每步都检查有没有新请求或已完成的请求，动态加入和退出，vLLM、TensorRT-LLM 都用这种。

本地引擎的批处理更谨慎。一张消费级 GPU 的显存有限，KV Cache 很占地方；多用户同时生成，可能互相拖慢。Kinetic 的 Convoy 只在满足条件时合并请求，并承诺每个请求的输出与单独运行时相同。

批处理最难的不是调度，是**数值一致性**。同一批里，每个 token 的矩阵乘可能被合并成更大的矩阵乘；如果合并改变了浮点求和顺序，输出就会漂移。Kinetic 的 batch invariant 要求：同一算术路径中，改变同批其它行，不改变该行的结果。

### 2.7 投机解码

普通 decode 每生成一个 token 都要读一遍权重。投机解码换个做法。

它先用便宜的方法猜接下来 k 个 token，把原提示加上这 k 个猜测，放进一次前向，得到每个位置的 logits。然后从第一个位置开始验证：如果目标模型也认为这个 token 该被选中，就接受；否则拒绝，从该位置回退。接受的前缀直接变成输出，一次前向可以产出多个 token。

收益取决于两个数：**接受率**和**验证成本**。草稿越准，接受率越高；验证一次前向的成本越接近普通 decode，收益越大。写代码、改文件这种高度可预测的文本，接受率高；开放式创作，接受率低。

投机解码必须保证输出分布不变。贪心解码要求逐 token 相同；采样解码要用拒绝采样，保证接受后的分布仍是目标分布。Kinetic 的 RAD-1 就是在这个约束下做的。

## 3. GGUF：一个文件装下模型

GGUF 全称 GPT-Generated Unified Format，由 llama.cpp 社区在 2023 年推出，用来取代旧的 GGML 格式。它现在是本地模型的事实标准：Hugging Face 上的量化模型、Ollama、LM Studio 都发这个格式。

它的核心思路很简单：把模型需要的东西都放进一个文件，让读取者不需要额外的配置文件，也不需要先转换。一个文件里同时有结构、词表和权重。

### 3.1 文件布局

kinetic 的解析代码在 `crates/kinetic-gguf/src/lib.rs` 的 `Gguf::parse_header`，布局和它一致：

| 部分 | 内容 |
|---|---|
| 魔数 | 4 字节 `GGUF` |
| 版本 | `u32`，kinetic 支持 2 和 3 |
| 张量数量 | `u64` |
| 元数据数量 | `u64` |
| 元数据 | 若干键值对，键是字符串，值有类型 |
| 张量信息 | 名称、维数、各维长度、类型 id、相对偏移 |
| 对齐填充 | 默认按 32 字节对齐，可用 `general.alignment` 改 |
| 张量数据 | 按偏移连续存放的量化权重 |

读取顺序是：先读魔数和版本，再读两个计数，然后逐个读元数据键值对，再读每个张量的名称、维数、类型和偏移，最后按对齐算出数据区起点。

### 3.2 元数据

元数据是自描述的核心。常见键：

- `general.architecture`：模型架构，例如 `qwen35`、`llama`、`gemma3`。
- `<arch>.block_count`、`<arch>.context_length`、`<arch>.embedding_length` 等超参数。
- `tokenizer.ggml.model`、`tokenizer.ggml.pre`：分词器类型和预分词方案。
- `tokenizer.ggml.tokens`、`tokenizer.ggml.scores`、`tokenizer.ggml.merges`：词表和合并规则。
- `tokenizer.ggml.bos_token_id`、`eos_token_id`：序列起止 token。
- `tokenizer.chat_template`：对话模板。

`general.architecture` 决定引擎能不能加载这个文件。`kinetic info` 会打印架构名、层数、宽度、上下文、权重类型分布、分词器和模板大小；不认识的架构会标 `(not supported yet)`。

### 3.3 张量与量化块

每个张量有自己的类型和形状。GGUF 的维度顺序是“输入在前、输出在后”：一个线性层的权重 `[cols, rows]` 表示 `cols` 个输入、`rows` 个输出。`Loader::matrix(name, cols, rows)` 会按这个顺序校验。

量化张量按块存储。每行权重的长度必须是块长的整数倍，否则加载会失败。这个约束让内核可以按块解包，不必处理残缺块。

张量数据的偏移是相对于数据区起点的；读取器会检查每个张量不越过文件、偏移对齐、元素数不溢出。头可以单独解析，所以测试可以用无权重 fixture，不需要下载完整模型。

### 3.4 mmap 与零拷贝

GGUF 的数据区按对齐排布，操作系统可以把文件直接映射进内存。`memmap2` 做这件事。读取权重时，`Matrix` 借用文件里的字节，不复制、不反量化；只有需要的小向量（norm、bias）才解量化到 f32。

好处有三个。加载快，因为不用把几 GB 的权重读进内存；内存省，因为多个进程可以共享同一份页缓存；零拷贝，因为量化矩阵的行视图直接指向文件字节。

`--no-mmap` 会改成全量读取，适合文件系统不支持 mmap 或需要 mlock 的场景。

### 3.5 在 kinetic 里对应的代码

| 位置 | 职责 |
|---|---|
| `crates/kinetic-gguf` | 解析头部、校验偏移、mmap 文件、写出 GGUF |
| `crates/kinetic-quant` | 定义每种量化块的布局，提供反量化参考实现 |
| `crates/kinetic-models` | 按 `general.architecture` 选择架构实现并组装模型 |
| `crates/kinetic-tokenizer` | 读词表元数据，实现 BPE、SentencePiece、WordPiece |

### 3.6 局限

GGUF 不是没有代价的。格式由 llama.cpp 主导，新架构和新量化类型要靠各实现自己跟进；不同工具的元数据键偶尔有差异；一个文件只装一个模型，分片 GGUF 需要额外处理；量化块的对齐要求意味着不是任意行宽都能直接跑，某些形状需要特殊内核或回退。

## 4. OpenAI 兼容接口

### 4.1 它是什么

“OpenAI 兼容”不是指用了 OpenAI 的 SDK，而是指服务端实现了 OpenAI 的 HTTP 约定。最常见的是 `POST /v1/chat/completions`：

```json
{
  "model": "local",
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello"}
  ],
  "stream": true
}
```

响应有两种。非流式是一个完整的 JSON，包含 `choices[0].message.content`、`finish_reason`、`usage`。流式是一串 SSE 事件，每个事件带一个增量 `delta`，最后以 `[DONE]` 结束。

### 4.2 流式输出

开启流式后，服务端用 Server-Sent Events 逐块返回：

```text
data: {"choices":[{"delta":{"role":"assistant"}}]}

data: {"choices":[{"delta":{"content":"Hi"}}]}

data: {"choices":[{"delta":{"content":" there"}}]}

data: {"choices":[{"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

SSE 的格式是 `data: <JSON>\n\n`。客户端要按行解析，忽略空行，遇到 `[DONE]` 结束。每个 chunk 的 `choices[0].delta` 可能有 `role`、`content`、`reasoning_content`、`tool_calls` 等字段。

流式输出有三个常见坑。第一，chunk 边界不等于 token 边界：引擎可能把多个 token 合成一个 chunk，也可能把一个 token 拆开（UTF-8 字符）。第二，停止条件可能吞掉尾部 token：stop 字符串匹配后，最后几个 token 不出现在文本里，但可能仍计入 usage。第三，工具调用是增量拼装的：`tool_calls` 可能分多个 chunk，参数是字符串片段，要按 index 合并。

### 4.3 为什么兼容很重要

生态按这个协议写代码。任何 OpenAI 客户端、IDE 插件、Agent 框架，只要允许改 `base_url`，就能指向本地服务：

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
```

对推理引擎来说，兼容协议等于零成本接入现有生态。这也是 Ollama、vLLM、llama.cpp、kinetic 都实现它的原因。

### 4.4 兼容的边界

核心字段（`messages`、`stream`、`temperature`、`max_tokens`、`tools`、`usage`）各家一致。扩展能力和细节会有差异：工具调用的流式增量格式、多模态输入的结构、`logprobs` 的返回形状、token 计数口径、结构化输出的方言、错误码和错误体。

很多项目说的“OpenAI 兼容”，只兼容到聊天能跑通；工具调用、logprobs、usage 这些细节，各家各写各的。写客户端时要按目标服务端实测。这一点我后面在 PR 复盘里还会讲到，因为一个真实的 bug 就出在流式和非流式的 `logprobs` 口径上。

### 4.5 工具调用与结构化输出

工具调用的一般流程是这样：请求里带 `tools` 数组，每个工具有名字、描述和参数 schema；服务端把工具描述塞进模板，模型生成一个工具调用（名字 + 参数 JSON）；服务端解析出工具调用，返回给客户端；服务端本身不执行外部函数；客户端执行工具，把结果作为 `tool` 消息再发回去。

结构化输出用 grammar 约束下一次 token 的选择：GBNF 或 JSON Schema 编译成状态机，采样前把不合法的 token 掩蔽掉。这样模型只能生成符合语法的 token 序列。

注意，grammar 约束的是“每一步合法”，不保证“在长度上限内一定完整”。如果 `max_tokens` 太小，JSON 可能被截断。

### 4.6 kinetic 的三套协议

`kinetic serve` 默认监听 `127.0.0.1:8080`，同时提供三套接口：

| 协议 | 端点 | 用途 |
|---|---|---|
| OpenAI | `/v1/chat/completions`、`/v1/completions`、`/v1/embeddings`、`/v1/rerank`、`/v1/models`、`/v1/health` | 文本生成、向量、重排 |
| llama-server | `/completion`、`/tokenize`、`/detokenize`、`/props`、`/slots`、`/metrics` | 兼容 llama.cpp 客户端 |
| Ollama | `/api/chat`、`/api/generate`、`/api/tags`、`/api/show`、`/api/embed` | 兼容 Ollama 客户端 |

同一个引擎，三套外壳。路由有优先级：不带 `/v1` 的 `/completions` 和 `/embeddings` 会先被 llama 适配层捕获，所以自定义端点要避开这些路径。

# 第二部分 Kinetic 总览

## 5. Kinetic 的定位

### 5.1 定位与边界

Kinetic 是单模型、本地优先的 Rust 推理运行时：从 GGUF 或 `.ktc` 加载一个模型，在 CPU 或受支持的 GPU 上执行，通过 CLI、HTTP 或进程内绑定交付结果。文本生成与 embedding/rerank 使用独立的服务入口。

它适合本地聊天和写作，适合把 OpenAI 客户端、IDE 插件、Agent 框架指向本地 base URL，也适合需要可复现 token 序列的实验。它不适合当任意 GGUF 的通用加载器，也不适合动态多模型网关；一个进程只服务一个模型。它更不适合要求跨 CPU/GPU 逐位一致的服务。

### 5.2 基本盘

| 项目 | 情况 |
|---|---|
| 许可证 | Apache-2.0 |
| 版本 | 0.1.0，beta |
| 语言 | Rust 1.91，edition 2021 |
| 平台 | Windows / Linux x86-64；macOS 计划 0.2 |
| 设备 | CPU（AVX2）、NVIDIA CUDA（Turing 起）、AMD/Intel Vulkan（部分模型） |
| 模型格式 | GGUF v2/v3；自家的 `.ktc` |
| 架构 | 文档列出 19 个架构名，另有 embedding 和 rerank 模型 |
| 接口 | OpenAI `/v1`、llama-server 接口、Ollama `/api`、C/Python/Rust 绑定 |
| 命令 | run、chat、serve、bench、info、verify、verify-receipt、quantize、perplexity、models、pull、doctor、fit |

“支持 19 个架构”不等于每个架构在所有设备上都可用。比如 phi3 目前只在 CPU 上测过，Vulkan 只覆盖 Llama、Qwen2.5、Qwen3、Qwen3.5、LFM3。看模型支持表时要同时看 CPU 和 CUDA 两列。

### 5.3 crate 地图

仓库是 Cargo workspace，14 个 crate，按职责分层。

| crate | 职责 |
|---|---|
| `kinetic-gguf` | GGUF 读写与 mmap |
| `kinetic-quant` | 量化类型、反量化、量化参考实现 |
| `kinetic-tokenizer` | 三种分词器 |
| `kinetic-models` | 模型架构、权重加载、KV Cache、投机解码、批处理 |
| `kinetic-cpu` | CPU 内核，AVX2，整型矩阵乘 |
| `kinetic-cuda` | CUDA 后端，运行时加载驱动 |
| `kinetic-vulkan` | Vulkan 后端，WGSL 内核 |
| `kinetic-server` | 引擎封装与 HTTP 服务 |
| `kinetic-cli` | `kinetic` 命令行 |
| `kinetic-grammar` | GBNF 与 JSON Schema 约束生成 |
| `kinetic-vision` | 多模态投影器 |
| `kinetic-capi` | C API |
| `kinetic-ktc` | 自家 `.ktc` 格式 |
| `kinetic` | Rust 库门面 |

依赖方向：

| 组 | 依赖方向与职责 |
|---|---|
| 入口 | `kinetic-cli`、`kinetic`、`kinetic-capi`；CLI 和库都依赖 `kinetic-server`，C/Python 绑定依赖 `kinetic` |
| 编排 | `kinetic-server` 管协议、请求、槽位和生成调度，依赖 `kinetic-models`；后者管家族、block、状态和投机解码，依赖设备 crate、GGUF、quant |
| 设备 | `kinetic-cpu`、`kinetic-cuda`、`kinetic-vulkan` 执行算子并消费 `kinetic-quant::Matrix`；设备适配仍在 `kinetic-models` |
| 数据 | `kinetic-ktc -> kinetic-gguf -> kinetic-quant`；KTC 和 GGUF 都向上提供 `TensorSource` |
| 功能 | `kinetic-grammar` 依赖 tokenizer；`kinetic-vision` 依赖 GGUF、quant、CPU 和可选 CUDA |

我自己的理解是：**模型层做设备适配，设备 crate 做内核。** 读代码时先把这条线记住，后面不容易乱。

### 5.4 一次请求的执行链

```text
CLI run/chat -------------------------+
  generate::stream / chat::answer    |
HTTP -> serve_with -> handle -> route +-> Engine::complete* -> on_backend -> Backend::generate
         OpenAI / llama / Ollama           encode, slot, Job        |
                                                                     v
                                       sequence::{generate, extend, decode} -> Decoder / Target
                                                                     |
                             Model / CudaModel / VulkanModel -> layer/mixer -> CPU/CUDA/Vulkan kernels
```

组装和执行是两条链。CLI 先打开权重、构建模型、选设备，再交给输出或 `serve_with`；`run` 把输入包装成 user message 并渲染模板，HTTP `/v1/completions` 才走 raw prompt。生成请求在 `Engine::complete_tokens_inner` 检查上下文、组装 `Job`，从空闲槽里选一个可复用前缀最多的 backend。三个 backend 共用 `sequence`，经 `Target` 分派设备前向。

### 5.5 第一次运行

```sh
kinetic pull LiquidAI/LFM3-1.7B-GGUF:Q4_K_M
kinetic run model.gguf "Explain recursion in two sentences."
kinetic serve model.gguf
```

`kinetic` 不带参数会检查机器，推荐几个能跑的模型。`kinetic models` 会列出 Ollama、LM Studio 和 Hugging Face 缓存里已有的模型。

## 6. 与 llama.cpp 的差异

### 6.1 相同点

- 读同一批 GGUF 文件，支持常见量化。
- 常用命令行参数兼容：`-m`、`-c`、`-ngl`、`-p`、`-n`、`--temp`。
- `kinetic quantize` 对 Q4_K_M、Q5_K_M、Q6_K、Q8_0、Q4_0 写出的字节与 llama.cpp 一致。
- 可以把 `kinetic` 改名成 `llama-server`，让按这个名字启动的客户端直接用它。
- 提供 `llama-server` 和 Ollama 的接口。

### 6.2 不同点

| 维度 | llama.cpp | Kinetic |
|---|---|---|
| 语言与实现 | C/C++，ggml 后端 | 从零用 Rust 重写 |
| 投机解码 | 通常要单独准备 draft model | RAD-1，自带三种草稿来源，不需要 draft model |
| 结果一致性 | 加速路径可能改变采样结果 | 声称同设备上逐 token 与朴素解码一致 |
| Prompt 缓存 | 有，主要针对文本前缀 | Warm Start，跨请求、跨进程，支持 recurrent 模型 |
| 预填充时机 | 按下发送后开始 | Zero Wait，打字时就开始 |
| 多请求 | continuous batching | Convoy，多个请求共享一次权重读取 |
| 工程形态 | 多个独立工具 | 一个二进制覆盖 server、cli、bench、quantize、perplexity |
| 成熟度 | 生态大，架构跟进快 | beta，模型覆盖少，Vulkan 和 Mac 支持仍在补齐 |

### 6.3 关键优化一览

**RAD-1（精确投机解码）**：先用便宜的草稿猜接下来几个 token，再一次前向并行验证。猜对的直接采用，猜错的回退。难点是验证必须与逐个生成给出相同的 logits 和相同的采样结果。

**Warm Start（预热启动）**：聊天客户端每次重发整段对话。它把上一次请求结束时的引擎状态存下来，下一次只跑新增部分。状态放内存（上限 1 GiB）和磁盘（上限 2 GiB）。对 Qwen3.5、LFM3 这类 recurrent 模型也能用，这是比普通前缀缓存更难的部分。

**Zero Wait（零等待）**：用户粘贴长文档后还要打字。引擎在打字过程中就把文档预填充掉，按回车时只剩最后几个词。这改善的是 TTFT，不是吞吐。

**Torque**：一次 decode 要启动几百个小 kernel，CPU 启动开销会拖慢 GPU。它把整个 decode 步骤录成 CUDA graph（Vulkan 上是命令缓冲），之后一次调用重放。

**Convoy**：多个请求共享同一次权重读取，每个请求的输出仍与单独运行时相同。这要求批内每一行的求和顺序与单条时一致。

**Shortlist**：输出层可能有二十多万行，不必每行都精确算分。先用小副本估算上界，只对仍可能胜出的行算精确分数。

**Burst**：长提示的预填充用 int8 分块矩阵乘，每个权重块解包一次，复用于多个 token。

**Flywheel**：MoE 模型放不下的专家留在内存，按层决定在 CPU 算还是拷到 GPU 算。

这些优化共同的前提是“精确批处理”：批内每一行的浮点求和顺序与单条推理相同，否则结果会漂移。这也是 kinetic 反复强调“加速不改变答案”的技术含义。

### 6.4 小结

llama.cpp 定义了 GGUF 和本地推理的生态；kinetic 是围绕“同样的答案、更少的工作”重写的挑战者。它兼容 llama.cpp 的文件和命令行，但不兼容它的实现。

## 7. 性能数字的读法

读 kinetic 的跑分，先问三个问题：测的是哪个阶段，prefill 还是 decode；测的是哪种负载，普通生成还是重复编辑；跑在什么设备上，全 GPU、纯 CPU，还是层被拆到 CPU 和 GPU 两边。同一个引擎，这三个答案不同，数字可以差一个数量级。

### 7.1 两种负载

| 负载 | 特征 | 主要受益的技术 |
|---|---|---|
| 普通生成（generate） | 一段提示，生成一段新文本；前缀只出现一次 | 量化、批处理、输出层剪枝；投机解码的收益取决于文本可预测性 |
| 重复编辑（edit） | Agent 反复改文件、重发整段对话；前缀大量重复，改动很小 | 前缀复用、投机解码（代码和文件内容高度可预测） |

看 LFM3 1.7B Q4_K_M 在 RTX 3060 Laptop 上的三行数字：

| 测试 | llama.cpp | Kinetic | 倍率 |
|---|---:|---:|---:|
| edit（重复编辑） | 229.7 | 1033.7 | 4.5x |
| generate（普通生成） | 194.3 | 198.4 | 1.02x |
| prompt 512（预填充） | 7642.7 | 6820.6 | 0.89x |

同一台机器、同一个模型，重复编辑是 4.5x，普通生成只有 1.02x，预填充甚至略慢。拿 4.5x 当整体倍率，多少有点标题党；它只说明在这个负载上，缓存和草稿吃到了，换个负载就未必。CPU 上的 LFM3 也是同样的形状：edit 4.6x，generate 1.16x。

Qwen3.5 4B 的数字小一些，但趋势一致：GPU 上 edit 2.9x、generate 1.18x；CPU 上 edit 2.0x、generate 1.0x。README 的 status 段落还给了一个更保守的口径：首次编辑最高 2.5x，普通生成最高 1.4x。分层卸载的场景能到 2.6x，但那是“8 层在 GPU、24 层在 CPU”这个特殊瓶颈带来的，不是通用提速。

### 7.2 为什么重复编辑能到 4.5x

重复编辑同时吃到三份收益。

一是前缀复用。上一轮的对话和文件内容几乎不变，Warm Start 直接恢复状态，只跑新增的 token。

二是草稿接受率高。改文件时，后面的 token 大多能从文件内容猜出来，RAD-1 一次验证能接受多个。

三是输出层剪枝。Shortlist 只精确计算可能胜出的行，减少最后一步的读取。

普通生成只吃到一部分：前缀只出现一次，草稿接受率取决于文本可预测性，输出层剪枝仍然有效但占比小。所以不能把 4.5x 当成“引擎整体快 4.5 倍”。

### 7.3 对比口径

llama.cpp 的 edit 跑分也开了它自己的投机解码（Qwen3.5 用 `draft-mtp`，LFM3 用 `ngram-simple`）。4.5x 是“两套投机实现 + 前缀复用 + 其他优化”的合计差距，不是“有投机对没有投机”。

还要注意：模型、量化类型、设备、llama.cpp 版本是否写明；负载是 generate、edit 还是 prompt；是首次运行还是重复运行；双方的投机解码是否都开启；输出是否验证过与朴素解码一致；单请求还是并发，并发数是几。

### 7.4 prefill 与 decode 的数字要分开看

README 里“粘贴 22,631 token 再输入问题，首 token 从 10.95 秒降到 0.13 秒”，改善的是 prefill 和调度：Zero Wait 在你打字时就把文档处理掉。它不是 decode 变快了，也不是整体吞吐提高了。

bench 里的三行要分开读：`pp512` 是 512 token 的预填充吞吐，`tg128` 是 128 token 的 decode 吞吐，`edit256` 是编辑负载的吞吐。

用 `kinetic bench model.gguf --compare` 可以在自己机器上按同样口径测一遍，它会调用 `llama-bench` 并打印对比表。0.1 里还有个已知问题：模型拆在 GPU 和 CPU 两边时，第二次及以后的请求可能给出不同答案，bench 会报 “not identical”。这属于正确性问题，不能算性能优势。

# 第三部分 机制与工程

## 8. 模型加载

> 我的理解：模型加载是把 GGUF 里的张量映射成引擎能算的层。

### 8.1 文件识别

`ModelFile::open_with` 先看文件 magic：

- `KTC1`：Kinetic 自家的 `.ktc` 容器，带哈希和可选签名。
- `GGUF`：llama.cpp 生态的格式。

GGUF 默认由 `GgufFile::open` 建只读 mmap；`--no-mmap` 改成全量读取，`--mlock` 把页锁在内存里。

### 8.2 GGUF 解析

`Gguf::parse_header` 只接受 v2/v3。它读取：

1. 魔数、版本。
2. 张量数量、元数据数量。
3. 元数据键值对：键是字符串，值有类型（u8、i8、u16、i16、u32、i32、f32、bool、string、array、u64、i64、f64）。
4. 每个张量的名称、维数、各维长度、类型 ID、相对偏移。
5. 对齐：默认 32 字节，可由 `general.alignment` 改。
6. 数据区起点：头部结束后按对齐向上取整。

检查项：

- 版本必须是 2 或 3。
- 计数不能超过文件剩余大小。
- 每个张量的元素数不溢出。
- 每行长度是块长的整数倍。
- 偏移对齐，张量不越过文件。

完整 `parse` 还会检查每个张量都在 buffer 内。头可以单独解析，所以测试可以用无权重 fixture。

### 8.3 架构分派

`kinetic_models::load_with` 按 `general.architecture` 查 `FAMILIES` 表，找到对应的 `Family`，再调用它的 `body` 构建函数。

构建函数把文件里的张量映射成 `Stack`、`Layer` 和可选的 `Mtp`。`Stack` 是层的集合，`Layer` 是一个 transformer 层（attention、FFN、norm），`Mtp` 是多 token 预测块。共用构建器再应用 RoPE、补输出头、加归一化。没有 `output.weight` 时，复用 embedding 作为输出头。

### 8.4 权重映射

`Loader::matrix(name, cols, rows)` 做三件事：按名字找张量，检查形状是 `[cols, rows]`，借用原始字节，构造 `Matrix`。

`Matrix` 知道类型、行数、列数、每行字节数。`row_bytes = cols / block_len * block_bytes`。行视图直接切片，不复制、不反量化。

小向量（norm、bias）会解量化到 f32。未消费的权重缩放 tensor 会被拒绝，避免“看起来加载成功、实际算错”。

### 8.5 tokenizer

`Tokenizer::from_gguf` 由 `tokenizer.ggml.model` 和 `tokenizer.ggml.pre` 选择：

- `gpt2`：byte-level BPE。
- `llama`、`gemma`、`gemma4`：SentencePiece 风格。
- `bert`：WordPiece。

未知 pre-tokenizer 会报错。分词结果要和参考实现对齐，尤其是特殊 token、空白和前导空格的处理。

### 8.6 chat template

`Engine::new` 从 `tokenizer.chat_template` 编译 MiniJinja 模板，可用 `--chat-template-file` 覆盖。模板注入 messages、tools、BOS/EOS 和 generation prompt。没有模板时，chat 请求会报错。

模板没有文件 loader，并设了执行和输出上限，避免恶意模板读文件或无限循环。

### 8.7 加载失败的诊断

`kinetic info` 会打印架构、层数、宽度、上下文、权重类型、分词器和模板大小。不认识的架构标 `(not supported yet)`。已知架构但缺少特性（比如某种 RoPE scaling、某个张量）时，`kinetic run` 会在加载时给出具体原因。

常见失败：架构没注册、缺少必需张量、张量形状不对、量化类型不支持、RoPE scaling/sliding window/LongRoPE 等特性没实现、上下文长度不匹配。

## 9. 设备与内核

> 我的理解：设备与内核决定同一个模型在哪跑、跑多快、哪些路径不能跑。

### 9.1 CPU

CPU 后端是 `CpuBackend::new` → `Model::run`。内核按权重类型和 CPU 特性分派：

- AVX2/FMA/F16C：x86-64 的常用路径。
- NEON：Apple Silicon 和 ARM。
- 标量/解量化：兜底路径，也是其他内核的参考实现。

CPU 矩阵乘分两类。单行或少量行的 GEMV 给 decode 用；多行 tile 一次解包权重，复用于多个 token，给 prefill 用。

CPU 能处理 CUDA 普通 GEMV 不收的 `IQ4_XS` 等格式，所以 README 说某些 Unsloth 量化只在 CPU 上跑。

### 9.2 CUDA

CUDA 后端运行时加载 driver（`nvcuda.dll` 或 `libcuda.so.1`），所以没有 CUDA 的机器也能编译。内核由 `nvcc` 在构建时编译成 fatbin。

能力大致是这样：普通 GEMV 约 21 种类型，包括 F32/F16、Q4_K/Q5_K/Q6_K、多种 IQ/TQ；没有原生 BF16 或 `IQ4_XS` 的普通 GEMV；专家路径另有约 9 种类型，包含 `IQ4_XS`、`IQ3_XXS`、`IQ2_XS`、`IQ2_XXS`，但受 router、gate/up 类型和专家结构约束；dense 模型可以切 CPU/GPU，也可以多 GPU，MoE、逐层输入、共享 KV 不走多 CUDA GPU。

### 9.3 Vulkan / Metal

Vulkan 后端用 WGSL 写内核，构建时编译成 SPIR-V。macOS 的 `metal` 分支换用 Metal driver。

上传矩阵支持 F32/F16/Q8_0/Q4_K/Q5_K/Q6_K；文档覆盖 llama、qwen2、qwen3、lfm3、qwen35；按层能力判定，拒绝 Mamba-2、相对位置 mixer、路由专家、逐层输入；不支持 MoE 和层切分；需要 GPU 支持 32 宽 subgroup。

### 9.4 设备选择与回退

`--device auto` 依次试 CUDA、Vulkan、CPU。GPU 初始化失败会记录原因；显式指定 GPU 时，初始化失败直接报错。

请求级回退是另一回事：Vulkan 图像请求走 CPU；CUDA 在 `takes_images()` 成立时可走 GPU，否则走 CPU；embedding/rerank 的 CLI 入口用 CPU。

### 9.5 分层卸载

Dense 模型放不下时，Kinetic 把最后几层放 GPU，前面的层放 CPU。每生成一个 token，只有一个 hidden state 在 CPU 和 GPU 之间传递。

MoE 模型用 Flywheel：注意力层和共享权重放 GPU，专家放内存，按需在 CPU 算或拷到 GPU。适合专家很多、每次只激活少数几个的模型。

## 10. 生成与状态

> 我的理解：生成 = prefill 处理提示 + decode 逐 token 生成 + 状态可回滚。

### 10.1 Sequence 状态

`Sequence<S>` 保存已执行的 token、设备状态 `S` 和位置信息。

CPU 状态按层保存 KV、卷积或 recurrent 状态。滑窗 KV 是环形缓冲；checkpoint journal 支持撤销和保留选定行。

### 10.2 Prefill

`sequence::extend` 做三件事：尝试恢复或截短旧状态，分块喂入未覆盖的 token，末段经 `Decoder::prefill` 选第一个输出 token。

分块是为了控制显存和批大小；恢复是为了复用前缀。

### 10.3 Decode

`sequence::generate` 选择解码策略：greedy、`Sampler`（temperature、top-k、top-p、min-p、惩罚、seed）或 grammar policy（约束合法 token）。

`sequence::decode` 先发已确定的 token，再跑下一步；最后一个已选 token 不必执行。这样可以少做一次前向。

### 10.4 KV Cache 与状态

CPU 上，每层保存 K 和 V；滑窗是环形缓冲；checkpoint journal 记录可回滚的状态。

CUDA 上，KV 默认 F16，可以单独给 K/V 选 Q8_0；长 prefill 和逐 token decode 可能走不同算术路径。

### 10.5 停止条件

停止条件包括 EOS 或已识别的轮次结束 token、stop strings、输出预算（max_tokens）和回调停止（客户端断开、用户取消）。

停止条件处理不好，会出现“文本对了、usage 不对”或“文本少了、logprobs 多了”的问题。PR 复盘里会讲。

### 10.6 流式文本

流式 decoder 要处理四件事：未完成 UTF-8 字符（先缓冲，等后续字节补齐）、可能跨片段的 stop string（先缓冲，确认不匹配再发）、reasoning 和 content 的分割（`ThinkSplitter`）、工具调用的分割（`ToolSplitter`）。

### 10.7 Batch invariant

**Batch invariant 的单位是单个输出值。** CPU 量化点积先得到每块的整数和，再按单行内核的顺序折叠成 f32；tile 复用解包权重，但保持该 token 的浮点运算顺序。CUDA 短 batch 用与单 token 相同次序的 GEMV；Convoy 合并矩阵行，每条序列仍持有自己的位置和状态。

长 prefill 可以走另一套算术，所以短 batch 的断言不能推广到长 prompt 和逐 token decode。**精确性是按路径承诺的，不是全局承诺。**

## 11. 八项加速

> 我的理解：八项加速都在减少重复读取或重复计算，但每一项都有成立条件。

### 11.1 RAD-1

三个草稿来源：当前上下文的 lookup、会话历史索引、模型自带的预测块（MTP）。

`Decoder::step` 用目标模型和实际采样策略逐位置验证：保留命中路径，回滚其余状态。无匹配或近期命中差时，缩短或暂停草稿。

RAD-1 在 CPU/CUDA/Vulkan 可用。树验证有实现，但默认成本设置关闭。D-Spark companion 有库代码和示例，扫描未找到默认 CLI/HTTP 接线。

### 11.2 Warm Start

复用相同 token 前缀的内存或磁盘快照。它按“预计省下的 prefill 时间减读取/恢复成本”选择，未必取最长命中。需要兼容状态和可恢复切点；没有收益则冷跑。

`--no-prompt-cache` 或 LoRA 会关闭 Warm。不能据此推断 backend 不再保留最近的 `Sequence`。Vulkan coopmat 缓存另按设备和驱动区分。

### 11.3 Zero Wait

`chat` 输入时或 `POST /v1/prefill` 先跑稳定前缀。`draft_prompt` 丢掉可能继续合并的最后一词；CLI/服务每批让位于正式请求或更新的草稿；回退到合法批边界。

CPU/CUDA 可用。图片、空 prompt、超上下文、Vulkan 或未提前提交草稿不受益。

### 11.4 Torque

CUDA 捕获并重放单 token 提交步。需要满足可录制条件；host 路由专家、speculative checkpoint 验证步和 eager 模式不走该分支。

Vulkan 重放命令缓冲；`--off torque` 在 Vulkan 不关闭它。多 CUDA GPU 可每卡录制，特定录制失败回到逐 kernel。

### 11.5 Convoy

同时就绪的请求合并计算，各自保留状态。CPU 用 `run_lanes`，特殊 `network` 逐条运行；CUDA 共享 pass 需要单 GPU、无 CPU 前段、无专家 offload、各 mixer 可拆分，且总行数受限；Vulkan 没有这个收益。

一个请求、`--parallel 1` 或不满足共享条件时没有收益。多个槽不等于一定合批。

### 11.6 Shortlist

CPU 用 Q4_K 头部副本和误差上界筛行，再用原始内核重算可能进入 top-k 的行。

原头只收 Q6_K/Q8_0，列数须整除 256；需要全量 logits 的策略、上界无效或留下超过四分之一词表时，回到完整 head；筛掉的 logits 为 `-∞`。

### 11.7 Burst

README 的名字对应两套内核分派。CPU tile 一次解包复用多个 token，最低两个 token，代码还有 NEON 实现；CUDA 大 prompt 视格式、shape 和硬件选 GEMM，旧 GPU 有不同算术路径。

单 token、不支持的 tile 或短验证 batch 走其他内核；没有独立 `--off burst`。

### 11.8 Flywheel

CUDA MoE 给专家分配剩余显存：命中留在 GPU，未命中比较 host 计算和拷贝成本，决定在 CPU 执行、拷入缓存或用临时槽。还可以预取下层预测专家，最终按原路由顺序合并。

仅适用受支持的 CUDA MoE；dense、Vulkan、专家全驻留时没有这条动态 offload 路径。

### 11.9 条件表

| 机制 | 做什么 | 什么时候不适用 |
|---|---|---|
| RAD-1 | 上下文 lookup、会话历史索引、可选 MTP 做草稿；目标模型逐位置验证 | 无匹配或近期命中差时缩短/暂停；树验证默认关闭；D-Spark 未接默认 CLI/HTTP |
| Warm Start | 复用相同 token 前缀的内存/磁盘快照，按成本选 | `--no-prompt-cache` 或 LoRA 关闭；没有收益则冷跑 |
| Zero Wait | 输入时或 `/v1/prefill` 先跑稳定前缀 | 图片、空 prompt、超上下文、Vulkan、未提前提交草稿 |
| Torque | CUDA 捕获并重放单 token 提交步；Vulkan 重放命令缓冲 | host 路由专家、speculative checkpoint 验证、eager 模式；`--off torque` 在 Vulkan 不关闭 |
| Convoy | 同时就绪的请求合并计算 | CPU 特殊 `network`；CUDA 条件多；Vulkan 没有 |
| Shortlist | CPU 用 Q4_K 头部副本和误差上界筛行 | 原头只收 Q6_K/Q8_0，列数须整除 256；需要全量 logits 时回退 |
| Burst | CPU tile 解包一次复用多个 token；CUDA 大 prompt 选 GEMM | 单 token、不支持的 tile、短验证 batch；没有独立 `--off burst` |
| Flywheel | CUDA MoE 专家按显存和成本动态放置 | 仅受支持的 CUDA MoE；dense、Vulkan、专家全驻留 |

我看这张表时，会先问一个加速数字属于哪一行、开了哪些条件、在哪个设备上。

## 12. 服务协议与运行边界

> 我的理解：服务协议是把引擎包成别人能调用的接口，边界比端点更重要。

### 12.1 OpenAI

| 端点 | 方法 | 用途 |
|---|---|---|
| `/v1/health` | GET | 健康检查 |
| `/v1/models` | GET | 模型列表 |
| `/v1/chat/completions` | POST | 对话生成 |
| `/v1/completions` | POST | 文本补全 |
| `/v1/embeddings` | POST | 向量 |
| `/v1/rerank`、`/v1/reranking` | POST | 重排 |
| `/v1/prefill` | POST | 提前送草稿，配合 Zero Wait |

流式格式是 `text/event-stream` 和 `data: <JSON>\n\n`。聊天先发 assistant role，再发 `reasoning_content`/`content`，工具调用整理后另发 delta，最终 `finish_reason`、可选 usage、`data: [DONE]`。

### 12.2 llama-server

| 端点 | 用途 |
|---|---|
| `/completion`、`/completions` | 继续 prompt，SSE |
| `/tokenize`、`/detokenize` | 分词/反分词 |
| `/props`、`/slots` | 服务信息、槽位 |
| `/embedding`、`/embeddings` | 向量 |
| `/metrics` | Prometheus 文本 |

SSE 用 llama 形状的 JSON 和最终 stop 对象，不调用 OpenAI 的 `[DONE]`。

### 12.3 Ollama

| 端点 | 用途 |
|---|---|
| `/api/version`、`/api/tags`、`/api/ps` | 版本、模型列表、运行状态 |
| `/api/show` | 模型详情 |
| `/api/chat`、`/api/generate` | 生成，JSON 或 NDJSON |
| `/api/embed`、`/api/embeddings` | 向量 |

用 `done` 对象收尾；没有模型管理端点。

### 12.4 工具与 grammar

工具由模板传入、解析器提取，服务不执行外部函数。`tool_choice=required` 或指定函数走工具 grammar，不能同时再指定一套输出 grammar。

`response_format` 支持 text/json_object/json_schema，也可用独立 GBNF；采样前掩蔽不合法 token。达到长度上限仍可能提前结束，不能把 grammar 约束写成任意预算下的完整 JSON 保证。

### 12.5 并发、认证与限制

- HTTP/1.1 一连接一处理线程，上限 64；超限 503。
- 生成槽用 `Mutex`/`Condvar` 等待；CPU/CUDA 默认尝试至多 4 槽；Vulkan CLI 只建一个。
- 默认绑定 `127.0.0.1`；配置 `--api-key` 后接受 Bearer 或 `X-Api-Key`，`/health` 和 `/v1/health` 例外。
- 没有内建 TLS。
- 读请求限 30 秒，单次写限 60 秒；头/体限 16 KiB/16 MiB。
- 错误码：400 请求无效、401 key 错、403 部分适配的 Origin 拒绝、404 无路由、405 方法错、415 内容类型错；另有 411/413/431/505 协议限制和 llama 的 501。

## 13. 精度与量化

> 我的理解：精度是分层承诺，逐位、逐 token、逐字节不能混。

### 13.1 三张表

先分清三张表。第一张是能识别的类型 ID，在 `quant/types.rs` 的 `TensorType::from_id`。第二张是能标量解码的类型，在 `quant/dequant.rs` 的反量化表里，包括 F32/F16/BF16、legacy、K、IQ、TQ 和部分 FP4/FP8。第三张是能在设备矩阵或专家上直接运行的类型，见第 9 章。

三者不是一回事。一个类型能被识别、能被参考实现解码，不代表某个 GPU 内核能直接跑它。

### 13.2 三个合同

- **逐位**：批内 logits、回滚后的状态用 `to_bits()` 类断言。
- **逐 token**：RAD-1、Warm Start、Zero Wait、Convoy 用同配置的 token 对照；Shortlist 只保留所需 top logits。
- **逐字节**：文档称无 `--imatrix` 的受支持量化输出可与 `llama-quantize` 文件逐字节一致；代码有混合类型和编码参考测试，但这次扫描没有验证外部工具输出，不能外推到所有模型和版本。

跨 CPU/CUDA/Vulkan、改变 CUDA KV 精度、CUDA 长 prefill 与单步、Vulkan coopmat 开关，都没有通用逐位承诺。PR 复盘里的 CUDA 诊断，就是在这个背景下看的。

### 13.3 quantize 预设

`kinetic quantize` 的输出预设只有 Q4_0、Q8_0、Q4_K_M、Q5_K_M、Q6_K。需要重编码的源 tensor 限 F32/F16/BF16/Q8_0。混合预设可以提高个别 tensor 的精度；对不整块的行选较小块格式；最后重写 GGUF 元数据和权重。

## 14. 测试与 CI

> 我的理解：测试是证明这些承诺的地方，CI 绿不等于设备正确。

### 14.1 单元与小型集成

`cargo test --release --locked --workspace` 覆盖：GGUF 坏头、截断、块宽、round-trip；CPU 标量参考与 tile 单行一致；GPU 内核容差、batch、状态；HTTP 解析、认证。

没有设备或 kernel 时，部分 GPU 测试会跳过。

### 14.2 Header fixture 与 digest

`cargo test -p kinetic-models --test headers` 只读头和借出的占位 tensor，走正式 loader。每个注册架构至少一个成功 fixture；digest 记录结构、已读 tensor 名、形状、类型，不是权重或生成摘要。

用 `cargo run --release -p kinetic-models --example fixture -- model.gguf fixture.gguf` 生成；有意变更才用 `KINETIC_BLESS=1` 更新。

### 14.3 真实权重

```sh
KINETIC_TEST_GGUF=model.gguf \
  cargo test --release -p kinetic-models --test architectures -- --ignored
```

覆盖逐 token、tree、hidden taps 和 greedy draft 对照。GPU 另跑：

```sh
cargo test --release -p kinetic-models --features cuda --test cuda -- --ignored
cargo test --release -p kinetic-models --features vulkan --test vulkan -- --ignored
```

### 14.4 CI

- Ubuntu 两套 fmt。
- Ubuntu 24.04 / Windows 2022 / macOS 15：全 feature Clippy、默认 feature workspace tests、C 示例编译和 usage smoke。
- 三平台另跑 Python 3.12 wheel 构建与 unittest。
- 发布构建矩阵：Linux/Windows x86-64 CPU/CUDA，macOS arm64 Metal，Python wheel。

**没有**实体 CUDA/Vulkan 和真实模型验收的 CI job。CI 绿不等于 GPU 行为通过；发布产物构建也不算 GPU 测试。

### 14.5 新家族验收清单

1. header fixture + `tests/headers/SOURCES.md` + digest。
2. 真实权重跑 `architectures` identity gates。
3. 逐 token、tree、hidden taps、greedy draft。
4. tokenizer 与参考实现对齐。
5. chat template 的单轮/多轮渲染。
6. 文档、README、CHANGELOG。
7. 如果声称加速，给同模型、同设备、同 seed 的开关对照。

# 第四部分 动手与参考

## 15. 源码阅读顺序

按依赖顺序读，比从 `main.rs` 一路追下去有效。

先读 `crates/kinetic-gguf/src/lib.rs` 的 `Gguf::parse_header`，看它怎样校验每个偏移；再读 `crates/kinetic-quant/src`，理解一个量化块怎样解包。然后看 `crates/kinetic-tokenizer/src`，对照一个 GGUF 的 `tokenizer.ggml.*` 元数据；再看 `crates/kinetic-models/src/loader.rs` 和 `arch/`，看架构名怎样映射到实现。内核从 `crates/kinetic-cpu/src/matvec` 开始，先读参考实现，再读 SIMD 快路径；服务看 `crates/kinetic-server/src`，看一次请求怎样变成 token 流；最后用 `kinetic bench` 和 `kinetic verify-receipt` 观察“同设备一致”这条约束怎样被测试。

可以配合的命令：

```sh
cargo test -p kinetic-gguf
cargo run -p kinetic-cli -- info model.gguf
cargo run -p kinetic-cli -- bench model.gguf
```

优先读的文件：

| 入口与模型 | 执行与验收 |
|---|---|
| `cli/main.rs` 看模型和设备如何装成服务 | `srv/engine.rs` 看请求、槽位、停止和回调如何串起来 |
| `srv/server.rs` 看三套协议的路由优先级 | `srv/sequence.rs` 看三个 backend 共享的生成循环 |
| `gguf/lib.rs` 看头解析与完整文件校验的差别 | `mdl/speculative/mod.rs` 看草稿怎样验证、接受和回滚 |
| `quant/types.rs` 看 GGUF 类型与块布局 | `mdl/warm/mod.rs` 看恢复决策为什么不只是最长匹配 |
| `tok/lib.rs` 看词表、切分方案和 special token | `mdl/head/mod.rs` 看 Shortlist 在什么条件下保持 top-k |
| `mdl/lib.rs` 看家族和通用输出头如何拼起来 | `cpu/matmul.rs` 看单行与 tile 如何分派 |
| `mdl/arch/mod.rs` 看家族注册和三种 Body | `mdl/cuda/mod.rs` 看 CUDA 放置与 decode/prefill 分流 |
| `mdl/arch/qwen35.rs` 看实际家族怎样组装 gated attention | `mdl/vulkan/mod.rs` 看 coopmat 与 decode 的精度边界 |
| `mdl/model.rs` 看 CPU 合批怎样分隔每路状态 | `tests/headers/check.rs` 看 digest 实际记录了什么 |
| `mdl/blocks/attention.rs` 看 KV 环、日志和回滚 | `tests/architectures.rs` 看新家族的行为验收合同 |

## 16. 扩展点与常见坑

按成本从低到高。

加纯协议端点最轻：在 `server.rs` 或适配器转成 `Request`，复用 `Engine`。容易漏的是路由优先级、流结尾、专用错误形状和模型种类。

用现有计算块加家族，是第二轻的：`arch/<family>.rs` 加 `FAMILIES` 注册。容易漏 GGUF 输入维在前、norm/scale、head tying、RoPE 和未消费 tensor；我不建议用“跳过未知权重”来凑加载成功。

扩展绑定要碰 C header/ABI 或 Python 转换，容易漏所有权、回调取消、错误语义；库的 GPU 选择与 CLI 不完全相同。

加新计算块或后端最重：CPU 参考和状态语义先行，接 `Target`/`Backend`，再补上传校验、设备 kernel、checkpoint/rollback、快照和 batch identity。最容易漏的是窗口已绕回、含拒绝草稿或多序列时的回滚。

## 17. 限制与文档漂移

- `main` 还没有注册 `phi3`；Phi-3 目前只有 CPU 分支支持，尚未进入主线。
- OpenAI `n != 1` 返回 400；Ollama 的 `suffix/template/context` 显式拒绝；Vulkan 不支持 MoE、层切分和若干 mixer；分片 GGUF 和 `quantize --imatrix` 仍在待做。
- README 的 0.1 小节报告 split CUDA 后续请求不一致、Linux RX 9070 崩溃、thinking 吞工具调用；`main` 有 split 状态 reset 修补，但不能据此确认所有复现已消失；RX 9070 未验证。
- Gemma 4 26B A4B 是文档漂移的例子：旧模型表仍写“不支持”，`main` 已有 CPU 专家实现和 26B A4B 布局测试；整模未跑，CUDA 专家限制仍适用。
- roadmap 的“双 GPU”不能覆盖 `main` 已有的 `--gpus`。Issue 模板是报告入口，不是 bug 台账。
- CI 没有实体 GPU 和真实模型验收；CI 绿不等于设备行为通过。

## 18. PR 提交检查表

1. 先读 `CONTRIBUTING.md` 和 `docs/src/adding-a-family.md`。大于小修的改动先开 issue；附模型、GGUF 链接和真实 `general.architecture`。
2. 新家族同时补 header fixture、`tests/headers/SOURCES.md` 和结构 digest；再用真实权重跑 identity gates。先解释未读 tensor、shape 或 digest 的改变，再更新期望值。
3. 加速 PR 至少给同模型、同设备、同 seed 的开关对照。报告冷/热、prefill/decode、单槽/多槽中实际测过的条件；和 llama.cpp 的 conformance 输出单独附上，不用它代替仓库内逐位测试。
4. 协议改动同时测流与非流、停止、错误码和兼容路由。模型支持改 `docs/src/models.md`、README 和 `CHANGELOG.md`；端点改 `docs/src/server.md`；设备边界改 `docs/src/devices.md`。
5. 新 GPU 能力把检查放在上传或初始化处，给不支持的格式和布局明确报错。分别验证普通矩阵、专家、长 prompt、回滚与 warm restore；一次成功加载不能证明这些路径都成立。

## 19. 后续方向

这篇只建立了地图。往下可以选三个方向。

一是批处理与调度，连续批处理、PagedAttention，参考已有的 nano-vllm 笔记。二是内核与量化，读 `kinetic-cpu` 的整型矩阵乘，对照 `kinetic-quant` 的参考实现。三是服务与协议，读 `kinetic-server`，理解流式、工具调用和结构化输出怎样落地。

先把这条线走完，再看 vLLM 或 TensorRT-LLM，会容易很多。

# 附录

## A. 术语表

| 术语 | 含义 |
|---|---|
| Prefill | 一次处理整段提示，计算 KV Cache |
| Decode | 逐 token 生成 |
| TTFT / TPOT | 首 token 延迟 / 每 token 延迟 |
| KV Cache | 缓存历史 Key 和 Value，避免重算 |
| 量化 | 用低位宽表示权重，减少体积和带宽 |
| GGUF | llama.cpp 生态的模型文件格式 |
| MoE | 混合专家，每次只激活部分专家 |
| 投机解码 | 先猜多个 token，再并行验证 |
| CUDA Graph | 把一串 kernel 启动录下来重放，减少启动开销 |
| mmap | 把文件映射进地址空间，按需读页 |
| Roofline | 用算力和带宽两条上限判断瓶颈 |
| batch invariant | 同一算术路径中，改变同批其它行不改变该行结果 |
| MTP | 模型附带的多 token 预测块，可用作投机解码草稿 |
| GEMV / GEMM | 矩阵乘向量 / 矩阵乘矩阵 |
| NDJSON | 每行一个 JSON 的流 |
| checkpoint journal | 生成过程中可回滚的状态日志 |
| coopmat | Vulkan 的协同矩阵路径，算术顺序与 decode 可能不同 |
| D-Spark | 仓库里的一种 companion drafter；扫描未找到默认 CLI/HTTP 接线 |

## B. 参考资料

- 仓库：https://github.com/Veloresearch/kinetic
- 文档：https://veloresearch.github.io/kinetic/
- PR 复盘：`kinetic-pr.md`
- 新增家族指南：`docs/src/adding-a-family.md`
- 入门清单：`docs/good-first-issues.md`
- 基准：`docs/src/benchmarks.md`
- 精度与收据：`docs/src/receipts.md`

## C. 版本与基线

- 笔记基线：`main` 提交 `1b671b9`，2026-11-25。
- PR 复盘见 `kinetic-pr.md`。
- 本文的代码结论来自静态阅读；没有运行构建、测试、基准或真实模型。
- 性能、设备兼容性和逐位一致性，以对应设备、模型和配置的实测为准。
