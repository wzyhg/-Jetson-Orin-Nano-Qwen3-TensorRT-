# Qwen3 Inference

本文档说明如何在 Jetson Orin Nano 上使用 KuiperLLama 编译并运行 Qwen3-0.6B 推理。

## 文件准备

Qwen3 推理需要两个核心文件：

```text
qwen0.6.bin2
tokenizer.json
```

推荐放置位置：

```text
~/LLM/KuiperLLama/models/qwen3-0.6b/
├── qwen0.6.bin2
└── tokenizer.json
```

进入 Docker 容器后，对应路径为：

```text
/models/qwen3-0.6b/
├── qwen0.6.bin2
└── tokenizer.json
```

检查文件：

```bash
ls -lh /models/qwen3-0.6b
```

正常可以看到类似：

```text
2.9G qwen0.6.bin2
11M  tokenizer.json
```

## 进入容器

在 Jetson 主机执行：

```bash
cd ~/LLM/KuiperLLama

sudo docker run --rm -it --runtime nvidia --network host \
  -v ~/LLM/KuiperLLama:/workspaces/KuiperLLama \
  -v ~/LLM/KuiperLLama/models:/models \
  -w /workspaces/KuiperLLama \
  kuiperllama:jetson bash
```

设置动态库路径：

```bash
export LD_LIBRARY_PATH=/workspaces/KuiperLLama/lib:$LD_LIBRARY_PATH
```

## 编译 Qwen3

在容器内执行：

```bash
cmake -S . -B build-qwen3 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DQWEN3_SUPPORT=ON

cmake --build build-qwen3
```

编译成功后会生成：

```bash
./build-qwen3/demo/qwen3_infer
```

如果已经编译过，后续只要源码和编译目录没有被删除，不需要每次重新编译。

## 运行推理

基础命令：

```bash
./build-qwen3/demo/qwen3_infer \
  /models/qwen3-0.6b/qwen0.6.bin2 \
  /models/qwen3-0.6b/tokenizer.json
```

带自定义 prompt：

```bash
./build-qwen3/demo/qwen3_infer \
  /models/qwen3-0.6b/qwen0.6.bin2 \
  /models/qwen3-0.6b/tokenizer.json \
  "什么是人工智能"
```

输出示例：

```text
什么是人工智能
<think>
...
</think>

人工智能是指让机器模拟、延伸或辅助人类智能的技术...

steps:344
duration:10.65
steps/s:32.28
```

其中：

- `steps`：生成步数
- `duration`：推理耗时，单位秒
- `steps/s`：每秒生成步数，可近似理解为 tokens/s

## Qwen3 Prompt 模板

当前 `demo/main_qwen3.cpp` 内部会将用户输入包装成 Qwen chat 模板：

```text
<|im_start|>user
用户输入
<|im_end|>
<|im_start|>assistant
```

因此运行命令时直接传普通文本即可：

```bash
./build-qwen3/demo/qwen3_infer \
  /models/qwen3-0.6b/qwen0.6.bin2 \
  /models/qwen3-0.6b/tokenizer.json \
  "请用一句话解释边缘计算"
```

## 与 TinyLlama 的区别

TinyLlama / stories110M 使用：

```bash
./build/demo/llama_infer /models/stories110M.bin /models/tokenizer.model "hello"
```

Qwen3 使用：

```bash
./build-qwen3/demo/qwen3_infer /models/qwen3-0.6b/qwen0.6.bin2 /models/qwen3-0.6b/tokenizer.json "hello"
```

主要区别：

- Qwen3 使用 BPE tokenizer，文件是 `tokenizer.json`
- TinyLlama 使用 SentencePiece tokenizer，文件是 `tokenizer.model`
- Qwen3 模型更大，推理速度低于 stories110M
- Qwen3 支持中文和英文对话能力更好

## 常见问题

### `No such file or directory`

检查程序路径和编译目录：

```bash
ls -lh ./build-qwen3/demo/qwen3_infer
```

如果不存在，重新编译：

```bash
cmake -S . -B build-qwen3 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DQWEN3_SUPPORT=ON

cmake --build build-qwen3
```

### `libllama.so: cannot open shared object file`

设置动态库路径：

```bash
export LD_LIBRARY_PATH=/workspaces/KuiperLLama/lib:$LD_LIBRARY_PATH
```

### `libglog.so.2` 找不到

通常是把容器里编译出的程序拿到宿主机直接运行导致的。推荐在容器内运行，或者在宿主机安装对应依赖。

容器内运行：

```bash
sudo docker run --rm -it --runtime nvidia --network host \
  -v ~/LLM/KuiperLLama:/workspaces/KuiperLLama \
  -v ~/LLM/KuiperLLama/models:/models \
  -w /workspaces/KuiperLLama \
  kuiperllama:jetson bash
```

### CMake 找不到 absl/re2

Qwen3 编译链路可能需要 `absl` 和 `re2`。如果 CMake 报：

```text
Could not find a package configuration file provided by "absl"
Could not find a package configuration file provided by "re2"
```

可以安装系统依赖，或让 CMake 自动拉取依赖。网络不稳定时建议使用 host network 构建，或者提前配置镜像源。

### 生成很慢

Qwen3-0.6B 比 stories110M 大很多。Jetson Orin Nano 上实际速度约为：

```text
Qwen3-0.6B: 约 32 steps/s
```

如果速度明显偏低，检查性能模式：

```bash
sudo nvpmodel -q
sudo jetson_clocks --show
```

## 展示重点

讲 Qwen3 部署时重点强调：

```text
1. 模型权重和 tokenizer 文件如何组织。
2. 为什么 Qwen3 需要单独打开 QWEN3_SUPPORT。
3. Jetson Orin Nano 编译时为什么指定 CUDA 架构 87。
4. 推理程序如何加载 tokenizer 和 bin2 权重。
5. 如何解释 steps/s、duration、prompt 长度对性能的影响。
6. 解决过动态库、依赖、内存和模型格式问题。
```

