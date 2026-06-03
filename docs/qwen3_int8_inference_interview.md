# Qwen3-0.6B 推理框架与 INT8 优化整理

本文档用于面试和项目复盘，整理 KuiperLLama 在 Jetson Orin Nano / Docker 环境下接入 Qwen3-0.6B、支持 fp32 与 INT8 权重量化推理、定位性能瓶颈并优化 CUDA kernel 的过程。

## 1. 项目目标

本阶段目标不是简单跑通 demo，而是把 Qwen3 推理链路做成一个可解释、可复现、可扩展的边缘端推理框架：

- 支持 Qwen3-0.6B fp32 原始模型推理。
- 支持 Qwen3-0.6B INT8 weight-only 量化推理。
- 保证 fp32 路径不被量化改动破坏。
- 对模型文件做格式校验，避免错误模型导致 segmentation fault。
- 对 INT8 CUDA matmul 进行 profile，基于数据定位主要耗时点。
- 在 Jetson 端获得可观的速度提升，并保留后续 INT4 / AWQ / GPTQ 扩展空间。

## 2. 整体推理框架

当前 Qwen3 推理入口是：

```bash
./build-qwen3/demo/qwen3_infer \
  /models/qwen3-0.6b/qwen0.6.bin2 \
  /models/qwen3-0.6b/tokenizer.json \
  "你好"
```

INT8 路径通过显式参数打开：

```bash
./build-qwen3/demo/qwen3_infer \
  /models/qwen3-0.6b/qwen3-int8.bin \
  /models/qwen3-0.6b/tokenizer.json \
  "你好" \
  --int8
```

推理流程可以概括为：

```text
prompt
  -> tokenizer.json BPE 编码
  -> embedding
  -> N 层 Transformer block
       -> RMSNorm
       -> q/k/v projection
       -> q_norm / k_norm
       -> RoPE
       -> MHA attention
       -> o projection
       -> residual
       -> RMSNorm
       -> gate/up/down FFN
       -> residual
  -> final RMSNorm
  -> lm_head
  -> argmax sampler
  -> tokenizer decode
```

代码上主要分为几层：

- `demo/main_qwen3.cpp`：命令行入口，解析模型路径、tokenizer、prompt、`--int8`。
- `kuiper/source/model/qwen3.cpp`：Qwen3 模型结构、参数加载、forward、predict、post processing。
- `kuiper/source/op/*`：Embedding、RMSNorm、Matmul、RoPE、MHA、SwiGLU 等算子封装。
- `kuiper/source/op/kernels/cuda/*`：CUDA kernel 实现。
- `tools/export_qwen3/*`：Qwen3 模型导出和量化工具。

## 3. Qwen3 接入重点

Qwen3 和 LLaMA/Qwen2 的结构相近，但仍有一些必须单独处理的地方：

- tokenizer 使用 `tokenizer.json`，不是 SentencePiece 的 `tokenizer.model`。
- prompt 需要使用 Qwen chat template。
- Qwen3 attention 中有 `q_norm` 和 `k_norm`。
- `num_attention_heads` 与 `num_key_value_heads` 不一定相等，需要处理 GQA。
- Qwen3 的权重顺序和已有 LLaMA/Qwen2 导出格式不同，需要单独 loader。

当前 Qwen3 模型参数加载分两条路径：

- `create_param_layers()`：fp32 权重路径。
- `create_param_quant_layers()`：INT8 weight-only 量化路径。

这样做的好处是 fp32 和 INT8 各自有清晰边界，不会因为量化改动影响原本 fp32 推理。

## 4. 模型文件格式校验

早期直接运行错误的 Q4 / 不匹配 bin 文件会出现 segmentation fault。根因是 loader 按某种格式解释二进制文件，但文件实际 layout 不匹配，导致权重指针越界或错位。

因此加入了模型文件大小校验：

- fp32：按 Qwen3 config 计算所有 fp32 权重的理论字节数。
- INT8：按 Q8_0 layout 计算量化权重和 scale 的理论字节数。
- 加载前比较实际文件大小和期望大小。

这样错误从运行时崩溃变成了明确报错：

```text
The Qwen3 checkpoint size does not match the selected format.
Use qwen0.6.bin2 for fp32, or pass --int8 for Q8_0 int8 checkpoints.
```

这个改动在工程上很关键，因为推理框架经常会面对不同来源的模型文件。格式校验可以把“野指针/段错误”变成“可定位的输入错误”。

## 5. INT8 量化方案

当前使用的是 weight-only INT8，对激活保持 fp32，权重量化为 int8：

```text
fp32 activation x int8 weight -> fp32 output
```

量化粒度是 group-wise，对每 64 个权重保存一个 fp32 scale：

```text
group_size = 64
scale = max(abs(weight_group)) / 127
q = round(weight / scale), clamp 到 [-127, 127]
dequant(weight) = q * scale
```

当前保持 fp32 的权重：

- embedding
- RMSNorm 权重
- q_norm
- k_norm

当前量化为 INT8 的权重：

- q_proj
- k_proj
- v_proj
- o_proj
- gate_proj / w1
- down_proj / w2
- up_proj / w3
- lm_head

这样取舍的原因：

- Matmul 权重占模型绝大多数参数，也是主要计算热点。
- Norm 和 embedding 对体积/计算贡献相对较小，保持 fp32 可以降低精度风险。
- weight-only 方案改动较小，不需要重写整个图的激活量化和校准流程。

## 6. INT8 文件 layout

INT8 文件头部仍然保留原始 Qwen3 config，后面额外写入 `group_size`。

权重布局大致是：

```text
ModelConfig
group_size

fp32 rmsnorm weights
fp32 embedding

for each q_proj layer:
  int8 weight
  fp32 scales

fp32 q_norm

for each k_proj layer:
  int8 weight
  fp32 scales

fp32 k_norm

for each v/o/w1/w2/w3 layer:
  int8 weight
  fp32 scales

lm_head int8 weight
lm_head fp32 scales
```

这里踩过一个关键坑：不能把所有 int8 values 写完后再统一写所有 scales。C++ loader 是按“每层 int8 权重后紧跟该层 scales”的方式读取的。如果导出脚本和 C++ loader layout 不一致，模型虽然能加载，但输出会严重异常。

最终导出脚本 `tools/export_qwen3/quantize_bin2_int8.py` 改成逐层写入：

```text
layer_int8_values + layer_scales
```

这保证了导出和加载严格一致。

## 7. CUDA INT8 Matmul 实现

当前核心 kernel 做的是一维向量乘矩阵：

```text
input:  fp32[M]
weight: int8[K, M]
scale:  fp32[K * M / group_size]
output: fp32[K]
```

每个输出 row 对应一次 dot product：

```cpp
sum += input[i] * scales[group_idx] * float(weight[row * M + i]);
```

其中：

```cpp
group_idx = weight_idx / group_size
```

在 Qwen3 INT8 导出中，`group_size=64` 是固定热点路径，因此优化为：

```cpp
group_idx = weight_idx >> 6
```

这个优化很小，但对高频 kernel 有意义，因为 INT8 matmul 在每个 token、每层、多个 projection 中反复调用。

## 8. Profile 方法

为了避免盲目优化，加入了 INT8 matmul profile：

```bash
KUIPER_PROFILE_INT8_MATMUL=1 ./build-qwen3/demo/qwen3_infer \
  /models/qwen3-0.6b/qwen3-int8.bin \
  /models/qwen3-0.6b/tokenizer.json \
  "用一句话解释边缘计算" \
  --int8
```

输出示例：

```text
[KuiperProfile] INT8 matmul CUDA kernel profile
shape, calls, total_ms, avg_us
K=3072 M=1024 group=64, 17024, 2694.1, 158.253
K=151936 M=1024 group=64, 292, 1984.4, 6795.9
K=1024 M=3072 group=64, 8512, 1106.7, 130.016
K=1024 M=1024 group=64, 17024, 1051.51, 61.7664
K=2048 M=1024 group=64, 8512, 929.862, 109.241
K=1024 M=2048 group=64, 8512, 806.849, 94.7896
```

这些 shape 可以对应到模型结构：

- `K=1024 M=1024`：q_proj / o_proj 等 hidden 到 hidden 的投影。
- `K=2048 M=1024`、`K=1024 M=2048`：FFN 中 intermediate 相关投影。
- `K=3072 M=1024`：多个 FFN projection 热点。
- `K=151936 M=1024`：lm_head，词表非常大，单次耗时高。

通过 profile 可以看出，不应该只凭感觉优化。实际热点既包括每层反复调用的小中型 matmul，也包括调用次数少但单次极重的 lm_head。

## 9. 已做过的优化

### 9.1 prompt 阶段跳过 logits

自回归推理中，prompt prefill 阶段的中间 token 不需要立刻计算完整 vocab logits，只需要把 KV cache 建好，并让下一个 prompt token 继续前进。

因此新增了内部 forward 路径：

```text
forward_internal(input, pos, next, need_logits)
```

在 `predict()` 中：

- prompt 阶段：`need_logits=false`
- decode 阶段：`need_logits=true`

这样可以减少 prompt 阶段不必要的 lm_head 计算。

### 9.2 INT8 weight allocation 修复

原始 `LayerParam::set_weight()` 默认按 fp32 大小计算 buffer，量化层也会被错误当成 float 权重处理。

修复后：

```cpp
element_size = is_quant_layer_ ? sizeof(int8_t) : sizeof(float)
```

否则 INT8 权重指针和 scales 指针都会错位。

### 9.3 group_size=64 快路径

将：

```cpp
weight_idx / group_size
```

在 `group_size == 64` 时改为：

```cpp
weight_idx >> 6
```

这是一个低风险的热点优化。实际测试中，INT8 推理从约 20 steps/s 提升到约 30 steps/s 左右。

### 9.4 rows-per-block 调参

INT8 matmul 支持不同 row-per-block 策略：

- `KUIPER_INT8_FFN_ROWS`
- `KUIPER_INT8_LMHEAD_ROWS`

默认策略当前偏向：

```text
FFN / attention projection: 4 rows per block
lm_head: 4 rows per block
```

这样能减少 kernel launch/grid 相关开销，同时保持每个 row 的归约逻辑简单。

### 9.5 lm_head fused argmax 实验

尝试过把：

```text
lm_head matmul -> logits -> argmax
```

融合成：

```text
int8 lm_head + argmax
```

理论上可以避免写出完整 vocab logits。但当前实验版本因为内部还有临时分配和两阶段 reduction，实测不如默认路径，因此保留但默认关闭：

```bash
KUIPER_FUSED_INT8_LMHEAD=1
```

面试时可以说明：这个优化方向是正确的，但第一版实现不够好，所以没有默认启用。这体现了“用数据决定是否合入性能优化”，而不是看到 fused 就认为一定更快。

## 10. 性能结果

当前测试数据大致如下：

| 路径 | 速度 |
| --- | ---: |
| Qwen3 fp32 | 约 22 steps/s |
| Qwen3 INT8 初始跑通 | 约 20 steps/s |
| Qwen3 INT8 优化后 | 约 30 steps/s |

最终 INT8 相对 fp32 约有：

```text
30.9 / 22.1 ≈ 1.4x
```

需要注意，weight-only INT8 不一定天然比 fp32 快很多，因为当前 kernel 仍然是：

```text
fp32 activation x int8 weight -> fp32 accumulate
```

并没有使用 Tensor Core 的 int8 GEMM 路径，也没有把 activation 量化成 int8。因此它更像是“减少权重带宽 + 手写 dequant matmul”的方案。

## 11. 为什么没有直接用 cuBLAS

cuBLAS 的 INT8 GEMM 通常更适合：

```text
int8 activation x int8 weight -> int32 accumulate
```

或者使用 cuBLASLt 配合特定 layout、scale、epilogue 做高性能矩阵乘。

当前框架是：

```text
fp32 activation x int8 weight
```

每个权重 group 还要乘不同 scale。这个计算不是标准的单一 GEMM，因为 dequant scale 是 per group 的：

```text
input[i] * int8_weight[i] * scale[group_idx]
```

如果要上 cuBLAS，有两条路线：

- 运行时先 dequant 成 fp16/fp32，再调用 GEMM，但会增加显存和带宽开销。
- 把 activation 也量化成 int8，重构为真正的 int8 GEMM，再处理 scale/zero point/epilogue。

所以当前没有直接使用 cuBLAS，而是先做专用 CUDA kernel，保证框架可控、格式简单、能在 Jetson 上稳定运行。

## 12. 主要问题定位过程

这次工作里遇到的问题可以分为几类。

### 12.1 容器依赖问题

每次进入新容器都缺少 `re2`、`nlohmann_json` 等依赖，是因为容器不是持久环境。解决方式是把当前已经装好依赖的容器 commit 成镜像：

```bash
sudo docker commit 756d681ae5f7 kuiperllama-qwen3:fixed
```

以后从固定镜像启动：

```bash
sudo docker run -it \
  --name kuiper-qwen3-fixed \
  --runtime nvidia \
  --network host \
  -v /home/ubuntu/LLM/KuiperLLama-Q:/workspaces/KuiperLLama \
  -v /home/ubuntu/LLM/KuiperLLama-Q/models:/models \
  -w /workspaces/KuiperLLama \
  kuiperllama-qwen3:fixed \
  bash
```

### 12.2 CMake cache 路径问题

从宿主机路径和容器路径混用时，`CMakeCache.txt` 会记录旧 source/build 路径，导致：

```text
The current CMakeCache.txt directory is different...
The source ... does not match ...
```

正确做法是用容器内固定路径重新 configure：

```bash
cmake -S . -B build-qwen3 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DQWEN3_SUPPORT=ON \
  -DRE2_INCLUDE_DIR=/usr/include \
  -DRE2_LIBRARY=/usr/lib/aarch64-linux-gnu/libre2.so
```

### 12.3 absl / re2 / nlohmann_json 问题

Qwen3 tokenizer 相关代码依赖较多，系统包和源码包混用容易出现 ABI 或 CMake find_package 问题。

最终处理思路：

- 尽量减少对 absl 的直接依赖。
- `encode.cpp` 中替换 `absl::StrReplaceAll` 为本地简单实现。
- CMake 里显式查找 `re2` include 和 library。
- demo/test 不再额外强制 find absl/re2/json，避免重复依赖炸开。

### 12.4 错误模型导致 segfault

直接拿不匹配的 Q4 文件跑 `qwen3_infer` 会崩溃。解决方式是：

- 入口层拒绝当前不支持的 Q4 文件。
- 模型层做 fp32 / INT8 文件大小校验。

## 13. 面试讲法

可以按这个顺序讲：

1. 我先把 Qwen3-0.6B 接到 KuiperLLama 的 C++/CUDA 推理框架里，补齐 tokenizer、prompt template、Qwen3 特有 q_norm/k_norm 和 GQA 参数。
2. 然后保证 fp32 路径稳定，避免后面量化时破坏原功能。
3. 在此基础上做 INT8 weight-only 量化，设计了 Q8_0 文件格式，norm/embedding 保持 fp32，主要 matmul 权重量化为 int8。
4. 为了避免错误模型文件直接 segfault，我加了模型文件大小校验，把运行时崩溃变成明确的格式错误。
5. INT8 初始版本并不快，所以我加了 CUDA event profile，按 matmul shape 统计耗时，找到 lm_head 和 FFN projection 的热点。
6. 根据 profile 做了低风险优化，包括 group_size=64 的位移快路径、rows-per-block 调参、prompt 阶段跳过不必要 logits。
7. 最终在 Jetson 上 INT8 从约 20 steps/s 提升到约 30 steps/s，相比 fp32 约 1.4 倍。
8. 我也尝试过 fused lm_head argmax，但实测不如默认路径，所以保留为实验开关，没有默认启用。这说明优化不是写了 fused 就算成功，必须用数据验证。

## 14. 可以强调的工程能力

- 能把模型结构映射到工程代码：Qwen3 block、q/k/v/o、FFN、lm_head。
- 能设计二进制权重 layout，并保证 Python exporter 和 C++ loader 一致。
- 能处理容器、CMake、依赖、CUDA arch 等实际部署问题。
- 能用 profile 数据指导优化，而不是盲目改 kernel。
- 能在优化时保护已有 fp32 路径，避免性能改动破坏正确性。
- 能判断哪些优化应该合入，哪些实验应该默认关闭。

## 15. 后续优化方向

### 15.1 INT4 / AWQ / GPTQ

后续可以支持更多量化格式：

- INT4 Q4_0：权重打包成 4bit，每组 scale。
- AWQ：按激活重要性保护部分通道，精度通常好于普通 PTQ。
- GPTQ：基于二阶近似的逐层量化，精度更稳，但导出和 loader 更复杂。

当前不建议直接兼容未知来源的 `qwen3-q4.bin`，因为必须先确定它的 header、packing、scale layout、group size 和权重顺序。

### 15.2 更强的 CUDA kernel

当前 kernel 还是比较朴素的 fp32 x int8 dot product。后续可以尝试：

- 使用 fp16 activation，降低带宽和计算压力。
- 做 activation quantization，转成真正 int8 GEMM。
- 使用 cuBLASLt / CUTLASS 做 int8 GEMM。
- 对 lm_head 做更高效的 fused top1/topk，避免完整 logits 写回。
- 使用 warp-level reduction 替代 block-level CUB reduce，减少 shared memory 和同步开销。

### 15.3 常驻服务

当前 `qwen3_infer` 是命令行 demo。实际应用中更适合做常驻服务：

- 模型只加载一次。
- HTTP / ROS2 请求复用同一个进程。
- 避免每次请求重复初始化 CUDA、加载权重和构建缓存。

## 16. 常用命令

构建：

```bash
cmake -S . -B build-qwen3 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DQWEN3_SUPPORT=ON \
  -DRE2_INCLUDE_DIR=/usr/include \
  -DRE2_LIBRARY=/usr/lib/aarch64-linux-gnu/libre2.so

cmake --build build-qwen3 --target qwen3_infer
```

fp32 推理：

```bash
./build-qwen3/demo/qwen3_infer \
  /models/qwen3-0.6b/qwen0.6.bin2 \
  /models/qwen3-0.6b/tokenizer.json \
  "用一句话解释边缘计算"
```

INT8 推理：

```bash
./build-qwen3/demo/qwen3_infer \
  /models/qwen3-0.6b/qwen3-int8.bin \
  /models/qwen3-0.6b/tokenizer.json \
  "用一句话解释边缘计算" \
  --int8
```

INT8 profile：

```bash
KUIPER_PROFILE_INT8_MATMUL=1 ./build-qwen3/demo/qwen3_infer \
  /models/qwen3-0.6b/qwen3-int8.bin \
  /models/qwen3-0.6b/tokenizer.json \
  "用一句话解释边缘计算" \
  --int8
```

从 fp32 bin2 量化导出 INT8：

```bash
python3 tools/export_qwen3/quantize_bin2_int8.py \
  --input_file models/qwen3-0.6b/qwen0.6.bin2 \
  --output_file models/qwen3-0.6b/qwen3-int8.bin \
  --group_size 64
```

