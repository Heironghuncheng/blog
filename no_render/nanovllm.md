---
title: "nano-vllm阅读"
date: 2026-11-20T23:00:00+08:00
draft: false
featuredImg: ""
description: 'nano-vllm学习'
tags:
  - vllm
author: BLESS
scrolltotop: true
toc: true
mathjax: true
comments: false
---

# 大语言模型推理基础

## 为什么要了解大语言模型推理基础

在深入阅读 **nano-vllm 源码解析** 之前，必须先厘清大语言模型（LLM）在执行自回归生成时的底层执行特征与硬件瓶颈。只有理解了 **Prefill（Prompt 阶段）** 与 **Decode（Token 阶段）** 截然不同的计算访存特性，以及 **KV Cache** 带来的显存碎片化问题，你才能看懂 `nano-vllm` 为什么需要设计专门的调度器（Scheduler）和分块内存管理器（Block Manager / PagedAttention）。

## 核心概念

### 自回归生成的两个阶段：Prompt 阶段与 Token 阶段

大语言模型基于 Decoder-only Transformer 架构，其文本生成本质上是“根据已有序列预测下一个 Token，并将其拼接到输入中继续预测”的自回归（Autoregressive）过程。在推理引擎（如 `nano-vllm`）的实现中，这一过程被严格划分为两个计算特性完全不同的阶段：

- **Prompt 阶段（Prefill / 预填充阶段）**：一次性并行处理用户输入的全部 Prompt Tokens。由于所有输入 Token 均已知，模型可以通过大规模矩阵-矩阵乘法（GEMM）并行计算所有位置的注意力（Attention），同时生成第一个输出 Token，并将所有输入 Token 的 Key 和 Value 向量写入显存。该阶段属于**计算密集型（Compute-Bound）**，能充分打满 GPU 的算力。
- **Token 阶段（Decode / 解码阶段）**：逐个自回归生成后续的 Output Tokens。每次前向传播（Forward Pass）仅输入上一步刚生成的 1 个 Token，将其与历史序列的 KV Cache 进行注意力计算。由于每次只处理单个 Token，矩阵乘法退化为矩阵-向量乘法（GEMV），GPU 需要频繁从高带宽显存（HBM）中读取庞大的模型权重和历史 KV Cache，但实际执行的浮点运算量极小。因此，该阶段属于**访存密集型（Memory-Bandwidth Bound）**。

| 维度 | Prompt 阶段（Prefill） | Token 阶段（Decode） |
| --- | --- | --- |
| **输入粒度** | 整个输入序列（长度为 L 的 Prompt） | 单个新生成的 Token（长度为 1） |
| **并行度** | 序列维度高度并行（一次处理所有输入 Token） | 序列维度串行（依赖前一步输出），仅能在 Batch 维度并行 |
| **核心算子形态** | 矩阵-矩阵乘法（GEMM） | 矩阵-向量乘法（GEMV） |
| **性能瓶颈** | 计算密集型（受限于 GPU Tensor Core 算力） | 访存密集型（受限于 GPU HBM 显存带宽） |
| **关键延迟指标** | TTFT（Time To First Token，首字延迟） | TPOT（Time Per Output Token，单字生成延迟） |

### KV Cache 的加速机制与显存痛点

在自回归生成的每一步中，当前 Token 的 Query 向量（Q）需要与之前所有历史 Token 的 Key（K）和 Value（V）向量计算注意力得分。由于因果掩码（Causal Mask）的存在，历史 Token 的 K 和 V 向量在后续步骤中永远不会发生变化。

**加速机制（以空间换时间）**：推理引擎将每层 Transformer 计算出的历史 K、V 张量缓存在显存中（即 **KV Cache**）。在 Token 阶段生成第 t 个 Token 时，模型只需计算当前这 1 个 Token 的 Q、K、V，将新的 K、V 追加到缓存末尾，并直接复用历史 KV Cache 完成注意力计算，从而将单步注意力计算复杂度从随序列长度平方增长降至线性增长。

**内存开销痛点**：KV Cache 虽然消除了重复计算，却带来了极高的显存占用与管理难题：

1. **显存容量巨大且动态增长**：KV Cache 的大小与 Batch Size、序列总长度、模型层数、注意力头数及隐藏层维度成正比。长上下文和大并发下，KV Cache 甚至会超过模型权重本身的显存占用。
2. **显存碎片化严重**：传统的推理框架按最大可能长度（`max_seq_len`）为每个请求预先分配连续显存，导致大量内部碎片（预留但未使用的空间）和外部碎片，严重限制了系统能同时处理的 Batch Size，进而拉低了 Token 阶段的吞吐量。这正是 `nano-vllm` 引入分页式 KV Cache 管理的核心动因。

[Prefill 与 Decoding 阶段的 KV Cache 复用机制, AI generated](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTVOaBJXS9jVJc8dw0d_JOS3m6ekMehqL3_TxtN6tSzAd6FznfjD3FtyhI\&s=10)

## 实战示例

将上述理论映射到具体的数值计算中，能直观感受到推理引擎面临的显存压力。下面通过单次推理的形状变化和显存占用计算，看看 `nano-vllm` 在底层究竟在管理什么数据。

### 示例 1：单序列自回归生成的张量形状演进

假设输入 Prompt 包含 128 个 Token，模型需要生成 3 个输出 Token。观察单条请求在 Prompt 阶段与 Token 阶段的输入形状及 KV Cache 状态变化：

1. **执行 Prompt 阶段（Prefill）**

   输入张量形状为 `[1, 128]`。模型一次性并行计算这 128 个 Token 的隐藏状态，输出形状为 `[1, 128, hidden_size]`。此时，每层网络将这 128 个 Token 对应的 K 和 V 写入显存，KV Cache 序列长度变为 `128`，并基于最后一个位置的隐藏状态采样出第 1 个输出 Token。

2. **执行第 1 步 Token 阶段（Decode Step 1）**

   输入张量形状缩减为 `[1, 1]`（仅包含刚生成的第 1 个 Token）。模型仅计算该 Token 的 Q、K、V，将新的 K、V 追加至缓存使 KV Cache 长度变为 `129`，随后用长度为 `1` 的 Q 与长度为 `129` 的 KV Cache 做注意力计算，产出第 2 个输出 Token。

3. **执行第 2 步 Token 阶段（Decode Step 2）**

   输入张量形状依然为 `[1, 1]`（第 2 个输出 Token）。KV Cache 长度更新为 `130`，计算注意力后产出第 3 个输出 Token。若命中结束符（EOS），该序列释放其占用的 KV Cache 空间。

### 示例 2：估算单请求的 KV Cache 显存占用

以一个采用标准多头注意力（MHA）的 7B 规模模型为例（32 层、32 个注意力头、每个头维度 `head_dim = 128`、使用 FP16/BF16 半精度存储，即每个元素占 2 字节），计算单条序列长度达到 2048 时的 KV Cache 显存消耗：

1. **计算单个 Token 在单层网络的 KV Cache 大小**

   每个 Token 需要同时保存 Key 和 Value 两个向量（因子为 2），单层单 Token 的字节数为：`2 (K和V) × 32 (头数) × 128 (头维度) × 2 字节 (FP16) = 16,384 字节 = 16 KB`。

2. **计算单个 Token 在全模型所有层的总大小**

   将单层开销乘以模型总层数 32：`16 KB × 32 层 = 512 KB / Token`。也就是说，序列每多生成一个 Token，显存就会净增加 0.5 MB。

3. **计算序列长度为 2048 时的总显存开销**

   将单 Token 全模型开销乘以序列总长度 2048：`512 KB × 2048 = 1,048,576 KB = 1 GB`。如果并发 Batch Size 为 32，仅 KV Cache 就需要消耗 32 GB 显存，凸显了精细化显存管理的必要性。

# 多进程基础

阅读 nano-vLLM 前，先理清 Python 多进程的任务分工与进程间通信（IPC）机制。多卡张量并行（TP）下，推理引擎由主进程负责调度，多个子进程分别驱动各张显卡执行计算，它们之间的交互全依赖这些基础原语。

## 进程与线程

磁盘上的 `main.py` 是静态程序文件，执行 `python main.py` 后，操作系统会创建一个进程来运行它。进程拥有独立的执行状态和内存空间，Python 中的变量、列表、字典等对象均存放在各自的内存空间中。

同一份代码运行两次，会得到两个互不干扰的进程。即便两个进程都定义了 `numbers = [1, 2, 3]`，其中一个执行 `numbers.append(4)`，另一个的列表也不会改变。

一个进程内部可以包含多个线程。线程共享所属进程的内存空间，可以直接读写进程内的同一份对象。多线程与多进程的核心差异在于内存边界与协作成本：

| 比较项 | 多线程 | 多进程 |
| --- | --- | --- |
| 内存隔离 | 共享同一进程的内存空间 | 各自拥有独立的内存空间 |
| 普通变量 | 可直接访问同一份对象 | 默认完全隔离，互不可见 |
| 数据交换 | 直接读写共享对象即可 | 必须通过显式的进程间通信（IPC）机制 |
| 核心难点 | 控制并发修改的顺序（防竞争） | 解决数据如何跨进程传输，以及同步访问顺序 |

## 并发、并行与 GIL

多个任务推进执行时，**并发**指任务在时间片上交错执行（单核 CPU 也可实现）；**并行**指多个任务在同一时刻真正同时执行（依赖多核 CPU 或多张 GPU）。

在启用了 GIL（全局解释器锁）的常规 CPython 中，同一进程内的多个线程无法同时执行 Python 字节码。因此，纯 Python 的 CPU 密集型任务无法通过多线程利用多核，必须通过多进程（每个进程拥有独立的解释器和 GIL）实现真正的并行。

**适用边界：** I/O 密集型任务（磁盘读写、网络请求）或底层释放了 GIL 的 C/CUDA 扩展（如 PyTorch 算子）不受此限；此外，Python 已引入可关闭 GIL 的自由线程构建版本。但进程创建与跨进程数据传输均有开销，切分粒度过细或通信过于频繁会抵消并行收益。

## 在 Python 中启动子进程

以下代码演示了如何启动两个子进程并等待它们结束（保存为 `process_demo.py` 运行）：

```python
import multiprocessing as mp
import os


def worker(worker_id):
    print(f"worker={worker_id}, pid={os.getpid()}")


if __name__ == "__main__":
    ctx = mp.get_context("spawn")

    processes = [
        ctx.Process(target=worker, args=(worker_id,))
        for worker_id in range(2)
    ]

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    print("两个子进程都已结束")
```

**`Process`、`start()` 和 `join()`：** `Process(target=..., args=...)` 仅创建进程描述对象，调用 `start()` 才真正创建系统进程并开始执行；`join()` 阻塞当前进程，直到目标子进程运行结束。必须先将所有进程 `start()` 再统一 `join()`，否则会退化为串行执行。子进程的实际执行与结束顺序由操作系统调度决定，不保证与创建顺序一致。

**`spawn` 启动方式：** `spawn` 会为子进程启动一个全新的 Python 解释器，并重新加载当前入口文件以获取函数定义。CUDA 上下文不支持通过 `fork`（直接复制父进程内存快照）继承，因此 PyTorch 多卡程序必须使用 `spawn` 或 `forkserver`。

**`if __name__ == "__main__":` 的作用：** 直接运行脚本时，`__name__ == "__main__"` 为真，主进程执行创建子进程的逻辑；子进程通过 `spawn` 重新加载该文件时，`__name__` 为模块名，条件不成立，从而跳过进程创建代码（避免子进程递归创建子进程触发 `RuntimeError`）。`worker` 函数必须定义在 `if` 之外的模块顶层，确保子进程加载时能正确导入该函数。

## 进程间通信（IPC）

子进程的函数返回值不会自动回传给主进程，传入 `args` 的普通对象也会在序列化复制后变成子进程内的独立副本。跨进程传递数据与协调状态主要依赖以下四类工具：

| 工具 | 核心职责 | 特点与适用场景 |
| --- | --- | --- |
| `Queue` | 传递消息或任务结果 | 底层自动完成对象的序列化与反序列化，适合低频、轻量级数据传递 |
| `SharedMemory` | 提供跨进程共享的字节内存 | 零拷贝直接读写同一块物理内存，适合高频或大数据量传输 |
| `Event` | 跨进程状态通知 | 传递布尔信号（条件是否满足），控制执行先后顺序 |
| `Lock` | 互斥访问控制 | 限制同一时刻只有一个进程进入临界区，防止并发写冲突 |

### 1. Queue：通过序列化传递对象

下面的代码通过 `Queue` 将子进程修改后的列表传回主进程。

```python
import multiprocessing as mp


def worker(numbers, result_queue):
    numbers.append(4)
    result_queue.put(numbers)


if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    numbers = [1, 2, 3]
    result_queue = ctx.Queue()

    process = ctx.Process(target=worker, args=(numbers, result_queue))
    process.start()

    result = result_queue.get()
    process.join()

    print("主进程的列表：", numbers)      # [1, 2, 3]
    print("子进程传回的结果：", result)  # [1, 2, 3, 4]

    result_queue.close()
    result_queue.join_thread()
```

传入的 `numbers` 经过序列化重建，子进程修改的是自己的副本；子进程通过 `put()` 将结果序列化写入队列，主进程通过 `get()` 取出反序列化后的新对象。

### 2. SharedMemory：直接读写同一块内存

对于推理循环中每步都要下发的调度指令，频繁序列化开销过高。`SharedMemory` 允许多个进程直接映射到同一块系统共享内存区域。创建、连接和清理共享内存时，可以使用以下接口：

| 接口 | 作用 |
| --- | --- |
| `SharedMemory(create=True, size=N)` | 创建一块至少 `N` 字节的共享内存，系统自动分配唯一名称 |
| `shm.name` | 获取该共享内存的名称字符串，可作为参数传给子进程 |
| `SharedMemory(name=...)` | 根据名称连接到已创建的共享内存 |
| `shm.buf` | 获取可直接按字节切片读写的 `memoryview` 视图 |
| `shm.close()` | 关闭当前进程对该共享内存的映射句柄 |
| `shm.unlink()` | 标记销毁底层共享内存块（全局只需由创建者调用一次） |

下面的代码演示主进程和子进程如何读写同一块共享内存。

```python
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory


def worker(name):
    shm = SharedMemory(name=name)
    try:
        print("子进程读到：", bytes(shm.buf[:5]))
        shm.buf[:5] = b"world"
    finally:
        shm.close()


if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    shm = SharedMemory(create=True, size=5)
    try:
        shm.buf[:5] = b"hello"
        process = ctx.Process(target=worker, args=(shm.name,))
        process.start()
        process.join()
        if process.exitcode != 0:
            raise RuntimeError("子进程执行失败")
        print("主进程读到：", bytes(shm.buf[:5]))
    finally:
        shm.close()
        shm.unlink()
```

`SharedMemory` 仅提供固定容量的裸字节区，不会自动扩容。数据的编码格式（如字符串编解码、对象序列化协议）以及读写时机，必须由通信双方自行约定。

### 3. Event 与 Lock：同步与互斥

共享内存本身不提供读写保护，多个进程直接操作同一块内存必须配合同步原语。

**`Event`（解决先后顺序）** 内部维护一个初始为 `False` 的跨进程布尔标志，可以通过三个方法等待或更新这个标志：

- `event.set()`：将标志置为 `True`，唤醒所有阻塞在 `wait()` 上的进程。
- `event.wait()`：若标志为 `False` 则阻塞等待，为 `True` 则立即放行。
- `event.clear()`：将标志重置为 `False`。

`set()` 会使标志持续保持为 `True`，不会因某个进程通过了 `wait()` 而自动复位，也不会累计触发次数。典型用法是：写入方写完共享内存后调用 `set()`，读取方通过 `wait()` 后读取数据，随后手动 `clear()` 以便下一轮同步。

**`Lock`（解决写冲突）** 维护“未锁定”与“已锁定”两种状态，保护同一时刻只能由一个进程执行的临界区代码。对于读取、修改、写回这类非原子操作（如 `counter.value += 1`），必须使用 `with lock:` 包裹，确保一个进程完整执行完写回后，另一个进程才能进入。`Lock` 仅对遵守约定去申请锁的进程有效，且不能将普通 Python 对象变成共享数据。

在实际工程中，`SharedMemory` 负责承载数据，`Event` 负责通知“数据已就绪”或“数据已消费”，`Lock` 负责在多写者场景下保证修改的原子性，三者组合即可实现低延迟的跨进程协同。

# PyTorch 与 CUDA 编程基础

## 为什么要了解 PyTorch 与 CUDA 编程基础

在阅读 **nano-vllm 源码解析** 时，你会频繁看到显存预分配（`pin_memory`、`.cuda(non_blocking=True)`）、自定义 Triton 算子（`store_kvcache_kernel`）以及图捕获（`CUDAGraph`）等底层操作。理解 PyTorch 如何调度 GPU 算子以及 GPU 的显存分级与线程块对齐机制，是看懂推理引擎如何消除 Python 调度瓶颈、压榨硬件带宽的关键前提。

## 核心概念

### PyTorch 自定义算子扩展与 GPU 异构调度

在标准 PyTorch 代码中，每一次张量运算（如加法、切片）都会由 CPU 上的 Python 解释器向 GPU 发送一条 CUDA Kernel 启动指令。当推理进入 Decode 阶段时，单步计算量极小，CPU 发射指令的延迟甚至会超过 GPU 实际计算的时间。为此，推理引擎通常采用三种手段打通底层调用接口：

- **C++/CUDA 扩展接口（`torch.utils.cpp_extension`）**：通过 `load`（即时编译 JIT）或 `CUDAExtension`（预编译），将编写好的 `.cu` 核函数绑定为 PyTorch 可直接调用的 Python 函数（如 nano-vllm 引入的 `flash_attn` 预编译二进制库）。
- **基于 Triton 的 Python 级 GPU 算子（`@triton.jit`）**：无需编写繁琐的 C++ 胶水代码，直接用 Python 语法编写 GPU 核函数并在运行时即时编译为 PTX 指令。例如 `nanovllm/layers/attention.py` 中的 `store_kvcache_kernel`，通过 `store_kvcache_kernel[(N,)](...)` 的网格语法直接接收 PyTorch 张量指针完成 KV 写入。
- **异步执行与内存锁页（Pinned Memory）**：CPU 与 GPU 之间的数据拷贝默认是同步阻塞的。nano-vllm 在 `prepare_decode` 中先在 CPU 上分配锁页内存（`pin_memory=True`），再调用 `.cuda(non_blocking=True)` 实现 DMA 异步传输，避免阻塞主线程。

| **扩展/优化方式** | **核心机制** | **nano-vllm 源码对应实现** | **解决的瓶颈** |
| --- | --- | --- | --- |
| **预编译 CUDA 扩展** | C++/CUDA 源码编译为 `.so` 动态库并注册至 PyTorch | `flash_attn_varlen_func` / `flash_attn_with_kvcache` | 融合注意力计算，减少 HBM 读写次数 |
| **Triton 核函数** | `@triton.jit` 装饰器将 Python 算子编译为 GPU 机器码 | `layers/attention.py` 中的 `store_kvcache_kernel` | 消除碎片化索引赋值带来的大量小算子开销 |
| **`torch.compile` 融合** | 捕获计算图并将逐元素操作融合为单一内核 | `RMSNorm`、`SiluAndMul`、`Sampler` | 减少中间张量的显存分配与内核启动次数 |
| **CUDA Graph** | 录制完整的 GPU 内核发射序列并在后续直接回放 | `model_runner.py` 中的 `capture_cudagraph` | 彻底消除 Decode 阶段的 CPU 端 Python 调度延迟 |

### GPU 内存分级缓存与线程块对齐

要写出高性能的自定义算子，必须顺应 GPU 的硬件存储层级和线程调度粒度。GPU 内部存在速度与容量呈反比的多级存储结构：

1. **全局显存（HBM / Global Memory）**：容量最大（数十 GB）但延迟最高（数百个时钟周期）。模型权重和 `self.kv_cache` 均存放在此。算子优化的核心目标就是减少对 HBM 的非合并访问（Uncoalesced Access）。
2. **共享内存与 L1 缓存（Shared Memory / SRAM）**：位于每个流式多处理器（SM）内部，容量极小（通常每块几十 KB）但带宽极高、延迟极低。FlashAttention 之所以快，正是因为将 Q、K、V 切块加载进 SRAM 中完成局部 Softmax 归约，避免中间结果写回 HBM。
3. **寄存器（Registers）**：每个 GPU 线程私有的最快存储空间。在 `store_kvcache_kernel` 中，通过 `tl.load` 读取的 `key` 和 `value` 向量正是暂存在寄存器中，随后直接写入目标缓存槽位。

此外，**线程块对齐与内存连续性（Stride）** 直接决定访存效率。GPU 以 32 个线程为一束（Warp）并行执行，当同一 Warp 内的线程访问连续且对齐的显存地址时，硬件会将多次请求合并为一次显存事务（合并访存）。如果张量经过切片或转置变得不连续，或维度未按 2 的幂次对齐，就会导致访存吞吐暴跌。

```text
+-------------------------------------------------------------------+
|                        GPU 硬件存储层级                            |
|                                                                   |
|  [ 寄存器 Registers ]  <-- 线程私有，最快 (每个线程)               |
|          ^                                                        |
|          v                                                        |
|  [ 共享内存 SRAM / L1 ] <-- 线程块 (Block) 共享，极高带宽 (~KB级)   |
|          ^                                                        |
|          v                                                        |
|  [ 全局显存 HBM ]       <-- 所有 SM 共享，大容量 (~GB级)，延迟高     |
+-------------------------------------------------------------------+
```

## 实战示例

将上述硬件与接口原理映射到 nano-vllm 的具体实现中，可以看到源码是如何通过步长校验和内存对齐来保证自定义算子高效运行的。

### 示例 1：分析 `store_kvcache` 中的内存连续性与对齐检查

在 `nanovllm/layers/attention.py` 中，宿主端包装函数 `store_kvcache` 在启动 Triton 核函数前执行了严格的步长（Stride）断言。下面逐步拆解这些断言如何保障合并访存：

1. **获取张量维度与总特征长度:**

   输入张量 `key` 的形状为 `[N, num_heads, head_dim]`，计算单个 Token 的总向量长度 `D = num_heads * head_dim`（例如 8 个 KV 头 × 128 维 = 1024）。

2. **校验最内层维度连续性:**

   检查 `key.stride(-1) == 1`。这保证了同一个注意力头内部的 `head_dim` 个元素在物理显存上是紧挨着存放的，没有任何间隔。

3. **校验头维度步长对齐:**

   检查 `key.stride(1) == head_dim` 且 `k_cache.stride(1) == D`。这确保了跳过一个注意力头或跳过一个缓存槽位时，地址偏移严格等于元素总数，使得 `tl.arange(0, D)` 能够一次性发起连续的合并读写。

4. **按 Token 粒度发射线程块:**

   执行 `store_kvcache_kernel[(N,)](...)`，为批次中的每个 Token 分配一个独立的 Program（线程块），并行将寄存器中的数据搬运到 `slot_mapping` 指定的物理块位置。

### 示例 2：计算 KV Cache 显存块大小与硬件对齐设计

在 `nanovllm/config.py` 和 `model_runner.py` 中，引擎对分页大小和显存块字节数做了严格的对齐约束。按以下步骤计算单个物理显存块的字节占用：

1. **确认块大小约束:**

   `Config.__post_init__` 强制要求 `assert self.kvcache_block_size % 256 == 0`（默认 256）。256 是 2 的幂次，完美契合 GPU 线程束（32 线程）和矩阵计算核心（Tensor Core）的访存对齐边界。

2. **提取单层单槽位的元素个数:**

   对于每个 Token 槽位，需要同时存储 Key 和 Value 两份数据，元素总数为 `2 * num_kv_heads * head_dim`。

3. **计算跨所有层的单块总字节数:**

   在 `allocate_kv_cache` 中，将单槽位大小乘以层数、块大小以及数据类型字节数（如 FP16 为 2 字节）：`block_bytes = 2 * num_hidden_layers * block_size * num_kv_heads * head_dim * dtype.itemsize`。

# 系统架构与组件划分

## 为什么要了解系统架构与组件划分

在阅读 **nano-vllm 源码解析** 时，最容易迷失在跨文件的类调用与进程通信细节中。先厘清顶层入口、控制平面（调度与显存管理）以及数据平面（模型执行与算子计算）的职责边界，能让你在后续拆解任意单个函数时，始终清楚它在全局流水线中所处的上下游位置。

## 核心概念

### 三层架构与核心组件边界

nano-vllm 将大模型推理系统清晰地拆分为用户接口层、引擎控制层与 GPU 执行层三个职责独立的模块，彼此之间通过明确的数据结构（如 `Sequence` 对象和 `Context` 全局上下文）解耦。

| **架构层级** | **核心文件与类** | **核心职责边界** |
| --- | --- | --- |
| **用户接口层** | `llm.py` (`LLM`)、`sampling_params.py` | 暴露极简的 `generate()` 同步批量推理接口，接收原始文本或 Token IDs 列表及采样超参数，屏蔽底层调度与多卡并行细节 |
| **引擎控制层（CPU）** | `engine/llm_engine.py` (`LLMEngine`)、`engine/scheduler.py` (`Scheduler`)、`engine/block_manager.py` (`BlockManager`) | 负责请求的生命周期管理（`Sequence` 状态流转）、Continuous Batching 动态组批、抢占重排以及 KV Cache 物理块的页表分配与哈希缓存回收 |
| **模型执行层（GPU）** | `engine/model_runner.py` (`ModelRunner`)、`models/qwen3.py`、`layers/*.py` | 负责张量准备与异步拷贝、多卡共享内存进程通信（TP）、CUDA Graph 录制回放、前向神经网络计算以及最终的 Token 采样 |

```text
+-------------------------------------------------------------------+
|                     用户接口层 (llm.py / LLM)                      |
+-------------------------------------------------------------------+
                                  | Prompts & SamplingParams
                                  v
+-------------------------------------------------------------------+
|               引擎控制层 (LLMEngine / Scheduler)                   |
|  +-----------------------+     +-------------------------------+  |
|  |   waiting / running   | <-> |   BlockManager (Paged KV)     |  |
|  +-----------------------+     +-------------------------------+  |
+-------------------------------------------------------------------+
                                  | Scheduled Seqs & is_prefill
                                  v
+-------------------------------------------------------------------+
|               模型执行层 (ModelRunner / Qwen3 / Layers)            |
|  [ 张量准备与异步拷贝 ] -> [ 前向计算与 KV 读写 ] -> [ 采样新词 ]   |
+-------------------------------------------------------------------+
```

### 客户端请求到前向计算的完整数据流

一个客户端请求从进入系统到完成单步推理，严格遵循“文本编码封装 $\rightarrow$ 队列准入与分页分配 $\rightarrow$ 张量打包与上下文注入 $\rightarrow$ 模型前向与采样回写”的闭环流转。

1. **请求接收与序列封装（`LLMEngine.add_request`）:**

   接收字符串或 Token IDs，调用分词器编码后实例化为独立追踪状态与页表的 `Sequence` 对象，压入调度器的 `waiting` 队列。

2. **动态组批与块分配（`Scheduler.schedule`）:**

   在 `LLMEngine.step()` 驱动下，调度器按 Prefill 优先策略检查空闲显存块，通过 `BlockManager` 为选中序列分配物理块编号并写入 `block_table`。

3. **元数据打包与上下文设置（`ModelRunner.prepare_*`）:**

   `ModelRunner` 将选中序列的输入 Token、位置索引、`slot_mapping` 与 `block_tables` 打包为锁页张量异步拷入 GPU，并注入全局 `Context` 单例。

4. **前向传播与状态回写（`run_model` 与 `postprocess`）:**

   模型读取 `Context` 完成注意力与 KV Cache 读写，经 `Sampler` 采样出新 Token ID 后，由 `Scheduler` 更新序列长度并在满足终止条件时释放显存块。

## 实战示例

将上述组件划分与数据流映射到具体的运行场景中，可以清晰看到一次 `llm.generate()` 调用是如何驱动各模块协同工作的。

### 示例 1：追踪双请求批量推理的组件协作路径

假设用户传入 `prompts = ["Hello", "Explain AI"]` 且 `max_tokens = 2`，观察控制层与执行层在首步（Prefill）与次步（Decode）的交互分工：

1. **初始化入队（`LLMEngine`）:**

   `add_request` 为两条提示词分别创建 `seq_id=0` 和 `seq_id=1` 的 `Sequence` 实例，初始状态均为 `WAITING`，全部进入 `scheduler.waiting` 队列。

2. **首步组批与显存建表（`Scheduler` 与 `BlockManager`）:**

   第 1 次调用 `step()` 时，`schedule()` 发现 `waiting` 非空，向 `BlockManager` 申请物理块并填充两条序列的 `block_table`，将状态改为 `RUNNING` 后移入 `running` 队列，返回 `(seqs, is_prefill=True)`。

3. **首步前向计算与首字采样（`ModelRunner`）:**

   `model_runner.call("run", seqs, True)` 调用 `prepare_prefill` 将两段变长输入拼接为一维张量，经 `Qwen3ForCausalLM` 计算并写入 KV Cache，采样出各自的第 1 个新 Token。

4. **次步解码与结束回收（`Scheduler.postprocess`）:**

   第 2 次调用 `step()` 时，`schedule()` 从 `running` 队列取出两条序列（`is_prefill=False`），`ModelRunner` 仅提取 `last_token` 执行 Decode 推理；生成第 2 个 Token 后达到 `max_tokens`，`postprocess` 将状态置为 `FINISHED` 并调用 `deallocate` 归还物理块。
