# 面试题库

概念和机制见 [Kinetic 学习笔记](../kinetic.md)，项目经历见 [PR 面试讲述](pr-walkthrough.md)。

## 基础

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

## 机制

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

## 精度与测试

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

## 项目

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
