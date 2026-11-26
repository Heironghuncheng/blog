---
title: "Kinetic PR 复盘：logprobs 与 Phi-3 接入"
date: 2026-11-25T22:00:00+08:00
draft: false
featuredImg: ""
description: '两个真实 PR 的复盘：流式 logprobs 一致性与 Phi-3 CPU 接入'
tags:
  - Kinetic
  - PR
  - AI Infra
  - 推理引擎
author: BLESS
scrolltotop: true
toc: true
mathjax: true
comments: false
---

# Kinetic PR 复盘

这篇是《从 Kinetic 认识 AI Infra》（`kinetic.md`）的配套文档，记录我参与的两个 PR。

写这篇的原因很简单。读源码是一回事，真正把一个改动送进仓库是另一回事。前者只要看懂，后者要面对边界、口径、评审和维护者的时间。第一个 PR 是 Issue #31 / PR #33，修流式与非流式 `logprobs` 不一致；第二个是 Issue #35 / PR #39，给 Phi-3 Mini 4K 加 CPU 支持。两个都不大，但正好覆盖了两类问题：API 一致性，和模型接入的数值一致性。

下面按“怎么发现、怎么定位、怎么定预期、最后怎么讲”的顺序写。机制细节在主文档，这里不重复。这些是我自己的准备笔记，不是标准答案；同一个问题，换个人可能有不同的讲法。

## 面试速记

面试前我会把每个话题压成两三句话，先自己讲一遍，再等追问。下面这张表是压缩版，详细解释在主文档和下面的复盘里。

| 话题 | 30 秒回答 | 常见追问 |
|---|---|---|
| AI Infra | 让模型在真实硬件上跑得动、跑得快、跑得便宜的工程；分训练和推理，入门从推理开始。 | 训练和推理的瓶颈有什么不同？ |
| 推理引擎 | 把模型文件变成 token 流的程序：加载、分词、prefill、decode、采样、流式输出。 | 一次请求经过哪些步骤？ |
| Prefill vs Decode | Prefill 一次处理整段提示，算力瓶颈，决定 TTFT；Decode 逐 token 生成，内存带宽瓶颈，决定 TPOT。 | 为什么 decode 是带宽瓶颈？ |
| KV Cache | 缓存历史 Key/Value，避免每步重算；大小是 2 × 层数 × KV 头数 × 头维 × 长度 × 字节。 | GQA 为什么能省 KV Cache？ |
| 量化 | 用低位宽和块缩放表示权重，减少模型体积和 decode 读取量；Q4_K_M 是常用折中。 | Q4_K 和 Q4_0 有什么区别？ |
| GGUF | llama.cpp 生态的单文件模型格式：元数据、词表、张量信息、量化权重，可 mmap。 | 为什么能零拷贝？ |
| OpenAI API | 一套 HTTP 约定，核心是 `/v1/chat/completions`、messages、SSE；兼容它就能接入现有生态。 | 流式怎么实现？兼容边界在哪？ |
| 投机解码 | 先便宜地猜几个 token，再用目标模型一次验证；猜对的接受，猜错的回退。 | 怎么保证输出分布不变？ |
| Batch invariant | 同一路径中，改变同批其它行不改变该行结果；靠固定浮点求和顺序实现。 | 为什么批处理会改变结果？ |
| Kinetic | Rust 写的本地推理引擎，读 GGUF，CPU/CUDA/Vulkan，单二进制，OpenAI 兼容。 | 和 llama.cpp 有什么区别？ |
| RAD-1 | Kinetic 的精确投机解码，草稿来自上下文、历史、模型自带 MTP。 | 为什么不需要 draft model？ |
| Warm Start | 复用相同前缀的引擎状态，跨请求、跨进程，按收益选择，不必然最长。 | 和普通前缀缓存有什么区别？ |
| Convoy | 多个请求共享一次权重读取，各自保留状态；有条件，不是所有设备都收益。 | 为什么 Vulkan 没有？ |
| 精确性 | 分逐位、逐 token、逐字节三层；同设备同路径可以逐位，跨设备通常不能。 | 为什么 CPU 和 GPU 会有差异？ |
| PR #33 | 修流式/非流式 logprobs 不一致；差分测试、合并 SSE、按可见文本裁剪。 | usage 为什么不变？ |
| PR #39 | Phi-3 CPU 接入；零拷贝拆融合权重；CUDA 因数值分叉撤回；默认 conformance 仍未过。 | 为什么撤回 CUDA？ |

# 案例复盘

## 1. bug 的发现过程：Issue #31 / PR #33

### 1.1 这个 bug 长什么样

这个 bug 不大，但它让我第一次认真想“流式和非流式应该一致”到底是什么意思。

`/v1/chat/completions` 有流式和非流式两种返回。同一个请求，除了 `stream` 字段，其他都一样，`logprobs.content` 的条目数却不同。可见文本、`finish_reason`、`usage` 都一致，只有 logprobs 不一致。

Issue #31 列了三个复现用例：

| 用例 | 可见文本 | `finish_reason` | `usage.completion_tokens` | 非流式条目 | 流式条目 |
|---|---|---|---:|---:|---:|
| `stop: [" Paris"]` | `"The capital of France is"` | stop | 6 | 6 | 5 |
| `stop: [" capital of France"]` | `"The"` | stop | 4 | 4 | 1 |
| `max_tokens: 3`，目标 `A🙂` | `"A"` | length | 3 | 3 | 1 |

差异都出在“生成了、但没有出现在返回文本里”的尾部 token 上。第一眼看过去，会以为是流式少发了；读代码之后才发现，是非流式多发了。

### 1.2 我是怎么发现的

方法其实很朴素：控制变量。

同一个请求体，只把 `stream` 从 false 改成 true；把 SSE 的所有 chunk 合并成一个逻辑响应；然后逐字段比较。为了让结果稳定，temperature 设成 0，用 grammar 强制输出固定文本，服务端加 `--parallel 1 --no-prompt-cache`，模型选一个小一点的，CPU 就能跑。

具体步骤是这样：

1. 发一次非流式请求，拿到完整 JSON。
2. 发一次流式请求，收集所有 `data:` 行，忽略 `[DONE]`，把 `choices[0].delta.content` 和 `choices[0].logprobs.content` 拼起来。
3. 比较四样东西：可见文本、`finish_reason`、`usage`、logprobs 数组。
4. 每次只改一个边界条件：加 stop、改 `max_tokens`、换含多字节字符的输出。

脚本骨架大概是这样（Issue 里的 `compare` 函数简化）：

```sh
url=http://127.0.0.1:8080/v1/chat/completions
common='"messages":[{"role":"user","content":"What is the capital of France? Answer in one word."}],"temperature":0,"logprobs":true,"top_logprobs":1'

curl -sS "$url" -H 'Content-Type: application/json' \
  -d "{$common,\"max_tokens\":16,\"stop\":[\" Paris\"],\"stream\":false}"

curl -sSN "$url" -H 'Content-Type: application/json' \
  -d "{$common,\"max_tokens\":16,\"stop\":[\" Paris\"],\"stream\":true,\"stream_options\":{\"include_usage\":true}}"
```

有一个细节很容易忽略：流式输出要合并后再比，不能逐 chunk 比。chunk 的切分由引擎决定，和 token 的边界不保证对齐。逐 chunk 比，会把“切分不同”误判成“内容不同”。

### 1.3 三个边界为什么能命中

要理解这个 bug，得先接受一件事：token、字符、字节不是一回事。

一个 token 可能是半个 UTF-8 字符。解码器会缓冲，等后续字节补齐再输出。stop 字符串在解码后的文本上匹配，匹配成功时，可能吞掉一个 token，也可能吞掉多个。`max_tokens` 按 token 计数，截断时，最后一个字符可能只到了一半，解码器会丢掉这个不完整字符。`usage.completion_tokens` 数的是生成的 token，包括被 stop 吞掉的、以及 EOS；它不是 logprobs 条目数。

三个用例正好打中三种“生成了但不可见”：

1. 整个 token 被 stop 吞掉。
2. 多个 token 被 stop 吞掉。
3. 字符被 `max_tokens` 截断，字节留在 token 里，文本里没有。

非流式实现拿着完整列表，直接全部返回；流式实现要等文本真正发出时才带 logprobs，尾部没有文本，就没有条目。两条路径对“尾部无文本 token”的处理不同，bug 就露出来了。

### 1.4 定位

读两个 handler 就够了。

非流式走 `complete_with_logprobs`，它读完整个 logprobs 列表，直接返回。流式走 SSE handler，把 pending 条目挂到下一个文本 chunk 上；没有下一个 chunk，就不发。

引擎两边都保留了记录，所以问题不在生成，而在“哪些记录应该出现在响应里”。这一点很重要：如果一上来就怀疑采样或内核，会绕很远。

维护者给出的规则很干脆：logprobs 只列“其字节出现在返回文本里”的 token。被 stop 删掉的、没拼完的字符，不算答案的一部分。流式已经符合这个规则，所以要改的是非流式。`usage.completion_tokens` 仍然统计所有生成的 token。

我后来记住的规则是：**logprobs 跟随可见文本，usage 跟随生成过程**。两者统计口径不同，不能为了对齐数组去改 usage。

### 1.5 修复

PR #33 的做法不复杂，关键是找对“可见前缀”的边界。

新增 `TextLogprobs`，按 token 记录原始字节和起始偏移。每次发出文本时，把文本长度映射回字节偏移，用 `partition_point` 截出可见的 token 前缀。stop 切在 token 中间时，保留这个 token 的 logprob。非流式和流式都走同一条裁剪路径。

`usage.completion_tokens` 不动；旧的 `/v1/completions` 不动；文档补一句规则。

测试分两层。单元测试覆盖“可见前缀停在 token 中间”和“替换字符 `�` 不能把缓冲字节算成可见”。HTTP 回归测试是 `chat_logprobs_after_string_stop` 和 `chat_logprobs_after_utf8_cutoff`：先用 grammar 固定输出，拿到完整 logprobs，再用 stop 或 `max_tokens` 截断，最后断言两种模式的数组合并后完全相等，且 usage 不变。

运行方式：

```sh
KINETIC_TEST_GGUF=/path/to/SmolLM2-135M-Instruct-f16.gguf \
  cargo test --release --locked -p kinetic-server --test server \
  chat_logprobs_after_ -- --ignored
```

### 1.6 复盘

这个 PR 最值得记的不是 `partition_point`，而是两件事。

第一，先找“应该一致”的两条路径。流式/非流式、单条/批处理、开缓存/关缓存、CPU/GPU，都是天然的差分对象。只要找到一对，构造边界输入，比盲读代码快得多。

第二，差异不一定是 bug，也可能是没写明的约定。Issue 里先问“这两边应该一致吗”，比直接改代码更稳。维护者一句“应该一致”，后面的方向就清楚了。

这类 bug 也适合入门：不用改内核，靠读协议、构造边界输入和对比输出就能定位。

## 2. 案例：Phi-3 接入与 PR #39

### 2.1 为什么选 Phi-3

#33 合入后，我想再做一次模型接入。最开始有人建议 Qwen3-30B-A3B，但我回头翻了仓库自己的 `docs/good-first-issues.md`，第 16 项是「Fused QKV weights (Phi-3 and others)」。Phi-3 Mini 4K 的官方量化文件小，CPU 就能跑完验收，反馈循环短。于是认领了 #35。

这件事让我改了一个习惯：先看项目自己的入门清单和验收标准，再决定做什么。模型大小直接影响你能多快拿到 conformance 结果；一个跑不动的模型，读得再多也没用。

### 2.2 维护者的三句话就是规格

#35 里维护者给了三条线索。

新家族放在 `crates/kinetic-models/src/arch/`，一个文件负责把文件里的张量和键映射到共享 block，外加必要的 tokenizer 和模板差异。

Phi-3 把 Q、K、V 放在一个 tensor 里，把 gate 和 up 放在另一个 tensor 里；先看共享 block 是直接接融合张量，还是需要在加载时拆。

方向不确定就早开 Draft，布局看不明白就在 issue 里问。

这三句话比 README 有用。写代码前先读维护者在 issue 里的最新回复，它通常就是当前规格。

### 2.3 权重怎么拆

摆在面前的有两条路。一是改共享 Attention 和 SwiGLU，让它们直接接受融合张量；二是在 `arch/phi3.rs` 加载时把融合张量拆成行视图，共享 block 保持不变。

草稿选了第二条。`Loader::matrix(name, cols, rows)` 会检查 GGUF 里的维度是 `[cols, rows]`；`Matrix::row_bytes()` 给出每个输出行的字节数；`rows()` 按 `start * stride` 切片，`Matrix::new` 再断言长度正好是 `count * row_bytes`。所以 QKV 被拆成 `[0, d)`、`[d, d + kv_dim)`、`[d + kv_dim, ...)`，gate/up 被拆成前 `n_ff` 行和后 `n_ff` 行，没有复制，也没有反量化。

这个方案的前提是：融合张量按输出行拼接，顺序是 Q/K/V 和 gate/up。注释里写了，但最好加一条测试，把偏移钉住：用一个小型量化矩阵，比较 `rows(qkv, 0, d).row(r)` 和 `qkv.row(r)`，gate/up 同理。这样遇到布局不同的变体，测试会先报错。

这里还有一个没解决的问题。入门清单第 16 项写的改动位置是 `arch/transformer.rs` 和 `loader.rs`，而草稿把逻辑放进了 `arch/phi3.rs`。维护者的评论支持“一个家族一个文件”，但清单暗示共享实现。这就是 Draft 里第一个问题的来源。我倾向于先问，不自己猜。

### 2.4 权重边界

草稿最初给 Phi-3 做全张量白名单，后来改成：保留必需张量的形状检查，只明确拒绝已知不支持的张量。拒绝列表包括 bias、QK norm、post norm、`rope_freqs`、LongRoPE 的 `rope_factors_long/short`；同时拒绝非 4K 上下文、sliding window、非 1.0 的 RoPE scaling、head_dim 不匹配。

两种做法各有利弊。白名单能抓住布局漂移：多出一个没见过的张量，加载就失败，不会悄悄算错。黑名单更宽容：可选张量、新元数据不会挡住加载，但未知张量会被静默忽略。

草稿选黑名单，是为了先跑通官方文件；维护者可能更想要白名单。这是一个应该在 PR 里问清楚的设计边界。

### 2.5 模板、分词、停止是三件事

模板这边，我用微软官方 GGUF 手动渲染了单轮和多轮对话，现有的 MiniJinja 路径直接可用，没有改模板代码。这一点挺省事，但也说明“模板能跑”和“模板正确”是两回事，至少要拿真实文件看一眼。

输入侧分词有一个 Phi-3 的特例：大多数特殊 token 后会吃掉紧跟的 ASCII 空白，例外是 `<unk>`、`<s>`、`<|endoftext|>`；`</s>` 不在例外里。实现放在 `spm.rs` 和 `special.rs`，用 `architecture == "phi3"` 收住作用域。

输出侧停止是另一件事。Phi-3 可能吐出 `</s>`，而 GGUF 命名的 EOS 是另一个 token。`engine.rs` 给 phi3 追加 `</s>` 作为停止候选，push 前检查重复。

长期看，输入规则应该跟词表或家族描述走，停止列表应该由模型元数据提供给 engine。草稿里的两个小特判可以先留，但要集中、命名、加测试。我倾向于不要都塞进 server。

### 2.6 CUDA：做了又撤

CUDA 最初的报错是每头 96 维不被接受，表面看像显存不够。原型补了短批分块和长提示的 Flash 路径，在本机 RTX 5070 Ti、另一台 RTX 3060 和一台 GTX 1660 SUPER 上跑过专项测试；还有一台机器卡在驱动初始化，没测成。

随后外部审阅指出：Kinetic CUDA 和 llama.cpp Vulkan 在第 5 个 token 分叉，参考侧前两名差 0.338 nat。改用同卡 CUDA 对 CUDA，仍在同一位置分叉，差 0.244 nat。两个数都超过仓库的 0.1 nat 近似平局阈值。

继续逐层排查：固定 token 后仍能复现；差异在进入 96 维 attention 之前就出现了；把 Kinetic 的首层 QKV 输出换进 llama.cpp CUDA，最终选择从 `That` 翻成 `Yes`；独立重算显示，两边量化缩放的精度不同，足以解释这个同卡样例；但只换首层 QKV 在 Vulkan 上没有翻转，所以结论不能跨后端套用，也不能据此宣布整个 CUDA 补丁正确。

最后决定：把整个 CUDA 目录撤回。这段是原型和诊断记录，不是 #39 的交付。

回头想，这段“白干”其实最值钱。它让我第一次认真区分“能跑”和“对得上”：能跑只说明路径通了，对得上要说明数值一致。同卡对照比跨后端对照更有说服力；首层 QKV 替换是一个好的二分手段，但它只定位到“进入 attention 之前”，不证明整个后端正确。我最后没有把未验证的后端随 PR 发；诊断留在独立记录或 issue 里。

### 2.7 CPU 的三种对照

草稿把验证拆成三层，这个拆分值得学。

| 检查 | 结果 | 说明 |
|---|---|---|
| Kinetic 自身 `architectures` | 4/4 通过 | 包括投机贪心与普通贪心逐 token 一致 |
| 输入 token ID 对 llama.cpp | 4/4 一致 | 分词和模板没有引入偏差 |
| `scripts/conformance.py` 默认设置 | exit 1 | 5/32（0.022 near-tie）、32/32、21/32（0.110）、17/18（0.917） |
| 只加 `--no-repack` | 第 3、4 条翻转 | 与 Kinetic 一致 |
| 只关 `-fa off` | 第 4 条翻转 | 与 Kinetic 一致 |
| `--no-repack -ctk f32 -ctv f32 -fa off` | exit 0 | 32/32、3/32（token 4，0.023 near-tie）、32/32、19/19 |

另外做了 F32 固定前缀实验：在两个非近似平局的分叉位置，两边选中的 top token 相同，候选 log-prob 的差距之差小于 0.00006 nat。

怎么读这组数？`0.022` 和 `0.023` 是近似平局，低于 0.1 nat 阈值，不能算 bug。`0.110` 刚过线，`0.917` 是决定性分叉，默认路径确实没对上。`--no-repack` 和 `-fa off` 能翻转，说明分歧更可能在默认 CPU 路径（repack / FlashAttention）的累加顺序，而不是 `phi3.rs` 的权重拆分。

入门清单的 done-when 是“匹配 llama.cpp 前 32 个贪心 token”。默认设置没过，就不能说任务完成；受控参数通过只能作为诊断证据。这就是 Draft 第四个问题，也是最重要的未决项：默认设置是不是硬门槛。如果是，要么修内核，要么让维护者明确豁免；我倾向于不把受控参数当成 conformance。

### 2.8 测试和文案的收敛

这一轮删掉了几样东西，理由都值得记。

分词测试从一条大用例拆成通用切分测试和 Phi-3 规则测试。单独测 `rows()` 抓不到 loader 接反权重的问题，所以删了。server 单测也撤下，保留真实模型的停止行为验证。补了无权重 header fixture，文档写明 `--device cpu`。没动 Issue 模板和任务清单。

changelog 一开始撤了；回看时发现新增家族指南明确要求改 changelog，应该补回去。这件事提醒我，贡献清单要逐项对，不能凭印象。

PR 文案不手动折行，新分支名不带 `codex`，现有分支不改。测试要针对失败模式，不能只镜像实现；这一点比多写几个测试更重要。

### 2.9 Draft 的四个问题

| 问题 | 草稿选择 | 建议 |
|---|---|---|
| 融合权重在加载时拆视图，还是共享块直接接融合张量？ | 加载时拆 | 保持局部拆视图；加布局测试；如果第二个家族也需要，再提 helper |
| 分词和停止特判放哪？ | `spm.rs` / `engine.rs` | 输入规则归 tokenizer，停止归 engine/模型元数据；加非 Phi-3 回归 |
| CPU 先落，`auto` 要不要主动跳过 GPU？ | 依赖初始化失败 + 文档 | 按能力表主动跳过；`--device cuda` 给明确错误 |
| CPU 对照用默认设置还是受控参数？ | 两者都给了 | 默认是验收线，受控是诊断；当前默认未过，先解决或申请豁免 |

这张表本身就是一次好的 PR 沟通：把不确定的地方写成可回答的问题，而不是把草稿包装成“完成”。

### 2.10 时间线和现状

- 11 月 21 日：#33 合入，开始找下一个模型接入任务。
- 11 月 23 日：认领 #35；维护者回复三条线索；CPU 原型和 CUDA 实验开始。
- 11 月 23 日：PR #39 以 Draft 打开，提交 `b4f7723`，11 个文件。
- 11 月 24–25 日：CUDA 对照出现分叉，撤回 CUDA 目录；CPU conformance 拆分验证；收敛测试和文案。
- 截至 11 月 25 日：PR 仍是 Draft，CI 全绿，0 评论、0 review。默认 CPU 对照仍未通过，这是取消 Draft 前需要解决或明确豁免的项。

注意：本地分支包含很多上游提交，`git diff main...HEAD` 会显示上百个文件；判断 PR 范围时，我看 GitHub 上的 11 个文件，不被本地分支吓到。

### 2.11 复盘

这次最值得记的，不是某一行代码，而是几个判断。

第一个家族尽量局部实现，等第二个案例再抽象。零拷贝视图要配一条布局测试，钉住行拼接假设。“能加载”不等于“支持”，权重边界要写清楚。模板、分词、停止分属三层，不要混在 server。“能跑”不等于“对得上”，同卡对照和逐层二分更有说服力。未验证的后端撤回，诊断记录与 PR 交付分开。默认设置是验收线，受控参数只是诊断。能力判断不要依赖初始化失败，`auto` 要主动跳过不支持的设备。测试要能抓住目标失败模式，抓不住就删。贡献清单逐项对，尤其是 changelog。

最后一条是心态：一个带精确问题的 Draft，比一个“看起来完成”的 PR 更有价值。

### 2.12 参考

- Issue #35：https://github.com/Veloresearch/kinetic/issues/35
- PR #39：https://github.com/Veloresearch/kinetic/pull/39
- 新增家族指南：`docs/src/adding-a-family.md`
- 入门清单第 16 项：`docs/good-first-issues.md`
- CUDA 诊断记录：`CROSS-AUDIT.md`（本地 conformance 目录）

## 3. 项目讲述方式：PR #33 和 PR #39

我准备项目时，尽量不按时间流水账讲，而是用 STAR：背景、任务、行动、结果；最后主动说边界和没做成的事。两个 PR 我都是这么准备的。

### PR #33：流式与非流式 logprobs 不一致

**S（背景）**：Kinetic 的 `/v1/chat/completions` 有流式和非流式两种返回。Issue #31 发现，同一个请求只把 `stream` 从 false 改成 true，可见文本、`finish_reason`、`usage` 都一样，但 `logprobs.content` 的条目数不同：非流式多出尾部 token，流式没有。

**T（任务）**：先判断这是 bug 还是未写明的约定；如果是 bug，统一两种模式，同时不破坏 `usage` 的语义。

**A（行动）**：

- 做差分测试：固定 temperature 0 和 grammar，只切换 `stream`。
- 合并 SSE 的所有 chunk，把 `delta.content` 和 `logprobs.content` 拼成完整响应，再和非流式逐字段比较。
- 构造三个边界：stop 吞掉一个 token、stop 吞掉多个 token、`max_tokens` 截断半个 UTF-8 字符。
- 定位到非流式 handler 返回了“生成了但没有文本”的尾部 token；流式 handler 因为没有下一个文本 chunk，不会发这些条目。
- 按维护者的规则修：logprobs 只列字节出现在返回文本里的 token；stop 切在 token 中间时保留该 token 的 logprob；`usage.completion_tokens` 不动。
- 加单元测试和两个 HTTP 回归测试，断言两种模式合并后的数组完全相等，usage 不变。

**R（结果）**：PR #33 合入；两种模式一致；文档补了规则；usage 语义保持。

**可能追问**：

- 为什么 `usage` 不能跟着数组改？——usage 统计的是生成过程，包含被 stop 吞掉的 token 和 EOS；logprobs 跟随可见文本。两者口径不同。
- 为什么流式是对的？——流式天然只给可见文本带 logprobs；非流式拿到了完整列表，多返回了尾部。
- stop 切在 token 中间怎么办？——保留这个 token 的 logprob，因为它的字节部分出现在文本里。

**边界**：这不是推理数学 bug，是 API 一致性和边界处理；改动集中在 server/logprobs，不涉及内核。

### PR #39：Phi-3 接入

**S（背景）**：#33 合入后，想做一次模型接入。项目入门清单第 16 项是 fused QKV（Phi-3 and others）；Phi-3 Mini 4K 的官方量化文件小，CPU 就能跑完验收。

**T（任务）**：让 Phi-3 Mini 4K 在 CPU 上加载、分词、停止、跑 conformance；同时确认融合权重、分词特判和停止特判的实现边界。

**A（行动）**：

- 在 `arch/phi3.rs` 加载时把 QKV 和 gate/up 拆成零拷贝行视图，复用现有 Attention 和 SwiGLU，不改共享 block、不反量化。
- 加 4K、无 RoPE scaling、无 sliding window、无 LongRoPE 的边界；拒绝已知不支持的张量。
- 模板不改；分词加 Phi-3 的特殊 token 后空白规则；engine 加 `</s>` 停止候选。
- 加无权重 header fixture 和文档。
- CPU 验证分三层：Kinetic 自身一致性 4/4；输入 token ID 对 llama.cpp 4/4；conformance 默认设置 exit 1，受控参数 exit 0 但仍有一个 near-tie。
- CUDA 原型打通了 96 维头，但同卡 CUDA 对 CUDA 在第 5 个 token 分叉 0.244 nat；首层 QKV 换进 llama.cpp 会翻转选择；Vulkan 不翻转。结论不能跨后端，最后撤回整个 CUDA 目录。

**R（结果）**：PR #39 以 Draft 打开，CI 全绿；CPU 自身一致性和 tokenizer 对齐通过；默认 conformance 仍有两处分叉；四个问题请维护者确认。PR 还没合并。

**可能追问**：

- 为什么加载时拆视图，而不是改共享 block？——共享 block 被所有家族和后端共用；为了一个 CPU-only 家族改契约，回归面太大。零拷贝视图没有额外内存。
- 零拷贝怎么保证正确？——`Loader::matrix` 校验 `[cols, rows]`，`Matrix::row_bytes` 按块算行宽，`rows()` 按行切片；前提是融合张量按输出行拼接。应该加布局测试。
- 为什么 CUDA 撤回？——“能跑”不等于“对得上”；同卡对照仍超过 0.1 nat 阈值；未验证的后端不应该随 PR 发。
- 默认 conformance 和受控 conformance 怎么选？——默认是用户实际路径，是验收线；受控参数是诊断，不是通关。

**边界**：PR 还是 Draft；默认 conformance 未过；CUDA 是诊断记录，不是交付；没有合并。

## 4. 面试题库

### 基础

**1. Prefill 和 decode 的区别？**
Prefill 一次处理整段提示，矩阵乘矩阵，算力瓶颈，决定 TTFT；decode 逐 token 生成，矩阵乘向量，内存带宽瓶颈，决定 TPOT。

**2. 为什么 decode 是带宽瓶颈？**
每生成一个 token 都要把权重读一遍，算术量小、搬运量大。速度上限约等于有效带宽除以每 token 读取的字节数。

**3. KV Cache 怎么算？GQA 的作用？**
大小是 `2 × 层数 × KV 头数 × 头维 × 序列长度 × 每元素字节`。GQA 让多个 query 头共享一组 KV 头，直接减少 KV 头数，从而缩小缓存。

**4. 量化为什么能提速？Q4_K 和 Q4_0 的区别？**
权重变小，decode 每步读的字节变少，所以更快；模型和 KV 占用也更小。Q4_K 用 256 元素块、6 位子块缩放和最小值，精度比 32 元素块、单一缩放的 Q4_0 好。

**5. GGUF 里有什么？为什么能 mmap？**
魔数、版本、元数据、张量信息、对齐和量化权重。数据区按固定对齐排布，张量偏移连续，所以可以直接映射文件、按需读页。

**6. OpenAI 兼容 API 的核心？流式怎么实现？**
核心是 `/v1/chat/completions` 的 `messages` 和 SSE。流式用 `text/event-stream`，每个 chunk 一个 `data: <JSON>`，最后 `[DONE]`；chunk 边界不等于 token 边界。

### 机制

**7. 投机解码的原理？怎么保证输出不变？**
先便宜地猜几个 token，再用目标模型一次前向验证；猜对的接受，猜错的回退。贪心要求逐 token 一致；采样用拒绝采样，保持目标分布。

**8. 什么是 batch invariant？为什么难？**
同一算术路径中，改变同批其它行不改变该行结果。难在浮点求和顺序：合并矩阵行、换 tile、换内核都会改变累加顺序，必须刻意保持。

**9. Kinetic 和 llama.cpp 的区别？**
文件和常用参数兼容，但 Kinetic 是从零用 Rust 重写的引擎；加速默认开启并强调同设备逐 token 一致；一个二进制覆盖 server、cli、bench、quantize、perplexity。

**10. RAD-1 的草稿从哪来？**
当前上下文的 lookup、会话历史索引、模型自带的 MTP 预测块。它不需要单独的 draft model。

**11. Warm Start 和普通前缀缓存的区别？**
它跨请求、跨进程保存引擎状态，支持 recurrent 模型；按“省下的 prefill 减恢复成本”选择，不必然取最长前缀。

**12. Zero Wait 改善什么？**
改善 TTFT，不是吞吐。它在用户打字时预填充稳定前缀，按回车时只剩最后几个词。

**13. Convoy 有条件吗？**
有。CPU 和 CUDA 的条件不同，Vulkan 没有；CUDA 需要单 GPU、无 CPU 前段、无专家 offload、mixer 可拆分等。多个槽不等于一定合批。

**14. Shortlist 怎么剪枝？**
用 Q4_K 头部副本和误差上界筛掉不可能进 top-k 的行，再用原始内核精确计算剩下的行；条件不满足时回退完整 head。

**15. Flywheel 适合什么模型？**
受支持的 CUDA MoE。专家放不下显存时，命中的留 GPU，未命中的比较 host 计算和拷贝成本；dense、Vulkan、专家全驻留不适用。

**16. 为什么 CPU 和 GPU 会有差异？**
浮点舍入和累加顺序不同。同设备同路径可以逐位一致；跨设备通常只能承诺逐 token 或更弱。

**17. `--device auto` 怎么回退？**
依次试 CUDA、Vulkan、CPU；GPU 初始化失败记录原因。显式指定 GPU 时，初始化失败直接报错；请求级回退另算。

### 精度与测试

**18. 为什么 logprobs 要和可见文本对齐？**
它描述的是返回文本的 token。被 stop 吞掉或没拼完的字符不属于答案，不应该出现在 logprobs 里。

**19. `usage.completion_tokens` 统计什么？**
生成的 token，包括被 stop 吞掉的 token 和 EOS。它不是可见 token 数，也不是 logprobs 条目数。

**20. 为什么 stop 会吞掉 token？**
stop 在解码后的文本上匹配。匹配成功时，可能正好吞掉一个 token，也可能吞掉多个；这些 token 生成了，但没有出现在返回文本里。

**21. UTF-8 截断为什么会产生半个字符？**
一个字符可能由多个字节组成，分词器又可能把它拆到多个 token 里。`max_tokens` 按 token 截断时，最后一个字符可能只到了一半，解码器会丢掉不完整字符。

**22. 怎么测试推理引擎的精确性？**
同设备逐位断言；identity gates（批内、tree、投机 vs 普通）；conformance 对 llama.cpp；真实模型回归；CI。还要分清逐位、逐 token、逐字节三种合同。

**23. header fixture 和 digest 是干什么的？**
fixture 只保留 GGUF 头和占位 tensor，走正式 loader，不需要权重；digest 记录结构、已读张量名、形状、类型，防止布局漂移。

**24. 为什么 CI 绿不等于 GPU 正确？**
CI 没有实体 CUDA/Vulkan 和真实模型验收；发布产物构建只证明能编译，不证明设备上的数值和状态正确。

**25. 加一个新模型家族要改什么？**
`arch/<family>.rs`、`FAMILIES` 注册、tokenizer/template 差异、header fixture 和 digest、真实权重 identity gates、文档和 CHANGELOG。

### 项目

**26. 为什么 PR #39 撤回 CUDA？**
CUDA 原型能跑，但同卡 CUDA 对 CUDA 在第 5 个 token 分叉 0.244 nat，超过 0.1 nat 阈值；首层 QKV 换进 llama.cpp 会翻转选择，Vulkan 不翻转，结论不能跨后端。未验证的后端不随 PR 发。

**27. 默认 conformance 和受控 conformance 的区别？**
默认是用户实际路径，是验收线；受控参数是诊断，用来定位分歧在哪。默认没过，就不能说任务完成。

**28. 你在这个项目里最大的收获？**
先定义合同再改代码：逐位、逐 token、逐字节是三种承诺；原型和交付分开；默认设置是验收线；有疑问的地方写成精确的 PR 问题，而不是包装成“完成”。

**29. 如果让你优化 decode，你会先做什么？**
先测带宽上限和实际 tokens/s，确认瓶颈；再看量化、KV 精度、投机解码接受率、批处理。我一般不会先改内核。

**30. 你怎么定位“流式和非流式不一致”的 bug？**
找两条应当一致的路径，只改一个变量；把流式输出归一化成一个完整响应；构造 stop、UTF-8、EOS 等边界；同时比较文本、`finish_reason`、usage 和结构化字段；先确定预期，再改代码。

## 5. 参考

- 主文档：`kinetic.md`
- 仓库：https://github.com/Veloresearch/kinetic
- Issue #31：https://github.com/Veloresearch/kinetic/issues/31
- PR #33：https://github.com/Veloresearch/kinetic/pull/33
- Issue #35：https://github.com/Veloresearch/kinetic/issues/35
- PR #39：https://github.com/Veloresearch/kinetic/pull/39
- 新增家族指南：`docs/src/adding-a-family.md`
- 入门清单：`docs/good-first-issues.md`
- CUDA 诊断记录：`CROSS-AUDIT.md`
