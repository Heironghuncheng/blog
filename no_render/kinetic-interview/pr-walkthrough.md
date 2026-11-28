# 面试讲述：PR #33 和 PR #39

我准备项目时，尽量不按时间流水账讲，而是用 STAR：背景、任务、行动、结果；最后主动说边界和没做成的事。两个 PR 我都是这么准备的。完整过程见 [PR 复盘](../kinetic-pr.md)。

## PR #33：接口一致性

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

## PR #39：Phi-3 接入

**S（背景）**：#33 合入后，想做一次模型接入。项目入门清单第 16 项是 fused QKV（Phi-3 and others）；Phi-3 Mini 4K 的官方量化文件小，CPU 就能跑完验收。

**T（任务）**：让 Phi-3 Mini 4K 在 CPU 上加载、分词、停止、跑 conformance；同时确认融合权重、分词特判和停止特判的实现边界。

**A（行动）**：

- 在 `arch/phi3.rs` 加载时把 QKV 和 gate/up 拆成零拷贝行视图，复用现有 Attention 和 SwiGLU，不改共享 block、不反量化。
- 加 4K、无 RoPE scaling、无 sliding window、无 LongRoPE 的边界；拒绝已知不支持的张量。
- 模板不改；在 tokenizer 里按架构选择特殊 token 后的空白规则，由 `stop_tokens()` 向服务端提供包括 `</s>` 在内的停止候选。加 54 条对照向量和停止标记回归单测。
- 加无权重 header fixture 和文档。
- CPU 验证分三层：Kinetic 自身一致性 4/4；输入 token ID 对 llama.cpp 4/4；conformance 默认设置 exit 1，受控参数 exit 0 但仍有一个 near-tie。
- CUDA 原型打通了 96 维头，但同卡 CUDA 对 CUDA 在第 5 个 token 分叉 0.244 nat；首层 QKV 换进 llama.cpp 会翻转选择；Vulkan 不翻转。结论不能跨后端，最后撤回整个 CUDA 目录。

**R（结果）**：PR #39 已转为 Ready for review。维护者认可 CPU 实现，建议把架构规则移到小表里，并计划在 `0.1.1` 之后合入。远端 head 是 `5912f54`，规则表调整目前只在本地，PR 还没合并。

**可能追问**：

- 为什么加载时拆视图，而不是改共享 block？——共享 block 被所有家族和后端共用；为了一个 CPU-only 家族改契约，回归面太大。零拷贝视图没有额外内存。
- 零拷贝怎么保证正确？——`Loader::matrix` 校验 `[cols, rows]`，`Matrix::row_bytes` 按块算行宽，`rows()` 按行切片；前提是融合张量按输出行拼接。应该加布局测试。
- 为什么 CUDA 撤回？——“能跑”不等于“对得上”；同卡对照仍超过 0.1 nat 阈值；未验证的后端不应该随 PR 发。
- 默认 conformance 和受控 conformance 怎么选？——默认设置有两处超过阈值；受控设置只剩一处容许的近似平局。两组都如实报告，项目日常使用哪组量化 CPU 设置还没有确认。

**边界**：默认 CPU conformance 仍有两处超过阈值的分叉，维护者的复测结果还没公布。CUDA 是诊断记录，CUDA 和 Vulkan 支持都不在这次交付里。
