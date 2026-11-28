# 面试速记

面试前我会把每个话题压成两三句话，先自己讲一遍，再等追问。下面这张表是压缩版，概念和机制在 [Kinetic 学习笔记](../kinetic.md)里展开，两个 PR 的细节见 [PR 复盘](../kinetic-pr.md)。

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
