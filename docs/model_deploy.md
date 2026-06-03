# Model Deploy On Jetson

本文档整理 KuiperLLama 在 Jetson Orin Nano 上部署轻量大模型的完整流程，重点是讲清楚模型文件、Docker 环境、C++/CUDA 编译、推理运行和性能结果。

## 项目目标

本项目基于 KuiperLLama，在 Jetson Orin Nano 上完成轻量大模型的本地部署和推理。当前验证过的模型包括：

- TinyLlama / stories110M：使用 `stories110M.bin` 和 `tokenizer.model`
- Qwen3-0.6B：使用 `qwen0.6.bin2` 和 `tokenizer.json`

整体流程：

```text
准备模型文件 -> 构建 Jetson Docker 环境 -> 编译 C++/CUDA 推理程序 -> 挂载模型目录 -> 运行推理 -> 记录性能
```

## 目录约定

Jetson 主机上的项目路径示例：

```bash
~/LLM/KuiperLLama
```

模型统一放在：

```bash
~/LLM/KuiperLLama/models
```

推荐结构：

```text
KuiperLLama/
├── build/
├── build-qwen3/
├── demo/
├── kuiper/
├── models/
│   ├── stories110M.bin
│   ├── tokenizer.model
│   └── qwen3-0.6b/
│       ├── qwen0.6.bin2
│       └── tokenizer.json
├── Dockerfile.jetson
└── docker-compose.jetson.yml
```

进入 Docker 容器后，宿主机的 `models` 目录会挂载为：

```bash
/models
```

## 模型文件说明

推理时至少需要两个文件：

- 权重文件：保存模型参数，例如 `stories110M.bin`、`qwen0.6.bin2`
- tokenizer 文件：负责文本和 token 之间的转换，例如 `tokenizer.model`、`tokenizer.json`

推理过程可以概括为：

```text
输入文本 -> tokenizer 编码成 token -> 模型逐 token 预测 -> tokenizer 解码成文本
```

## Docker 环境

Jetson 上 CUDA、TensorRT、系统库版本绑定较强。为了减少宿主机环境污染，本项目使用 NVIDIA L4T JetPack 镜像构建 Docker 环境。

构建镜像：

```bash
cd ~/LLM/KuiperLLama

sudo docker build \
  --network host \
  --build-arg MODEL_SUPPORT=LLAMA2 \
  --build-arg CUDA_ARCHITECTURES=87 \
  -f Dockerfile.jetson \
  -t kuiperllama:jetson \
  .
```

Orin Nano 的 CUDA 架构为 `87`，因此编译时使用：

```bash
-DCMAKE_CUDA_ARCHITECTURES=87
```

启动容器：

```bash
cd ~/LLM/KuiperLLama

sudo docker run --rm -it --runtime nvidia --network host \
  -v ~/LLM/KuiperLLama:/workspaces/KuiperLLama \
  -v ~/LLM/KuiperLLama/models:/models \
  -w /workspaces/KuiperLLama \
  kuiperllama:jetson bash
```

参数说明：

- `--runtime nvidia`：让容器访问 Jetson GPU
- `--network host`：使用宿主机网络，避免 Jetson 上 Docker bridge/iptables 问题
- `-v ~/LLM/KuiperLLama:/workspaces/KuiperLLama`：把源码挂载进容器，方便修改后直接编译
- `-v ~/LLM/KuiperLLama/models:/models`：把模型目录挂载进容器，模型不需要打进镜像
- `-w /workspaces/KuiperLLama`：进入项目工作目录

## 编译

### LLaMA2 / TinyLlama 类模型

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=87

cmake --build build
```

生成程序：

```bash
./build/demo/llama_infer
```

### Qwen3

```bash
cmake -S . -B build-qwen3 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DQWEN3_SUPPORT=ON

cmake --build build-qwen3
```

生成程序：

```bash
./build-qwen3/demo/qwen3_infer
```

如果运行时报动态库找不到，可以先设置：

```bash
export LD_LIBRARY_PATH=/workspaces/KuiperLLama/lib:$LD_LIBRARY_PATH
```

## 推理运行

### TinyLlama / stories110M

```bash
./build/demo/llama_infer \
  /models/stories110M.bin \
  /models/tokenizer.model \
  "hello"
```

### Qwen3-0.6B

```bash
./build-qwen3/demo/qwen3_infer \
  /models/qwen3-0.6b/qwen0.6.bin2 \
  /models/qwen3-0.6b/tokenizer.json \
  "什么是人工智能"
```

## 性能结果

实际测试中：

```text
TinyLlama / stories110M: 约 120 steps/s
Qwen3-0.6B: 约 32 steps/s
```

这里的 `steps/s` 可以近似理解为每秒生成 token 数。速度会受到以下因素影响：

- 模型参数量
- prompt 长度
- 最大生成长度
- 是否输出思考内容
- Jetson 功耗模式
- CPU/GPU/EMC 频率
- 是否使用 Debug/Release 编译

建议运行前检查 Jetson 性能模式：

```bash
sudo nvpmodel -q
sudo jetson_clocks --show
```

## 常见问题

### Docker build 时 apt-get 失败

如果报 iptables/raw table 相关错误，Jetson 的 Docker bridge 网络可能有问题。可以使用 host network 构建：

```bash
sudo docker build --network host -f Dockerfile.jetson -t kuiperllama:jetson .
```

### `libllama.so` 或 `libglog.so.2` 找不到

设置动态库路径：

```bash
export LD_LIBRARY_PATH=/workspaces/KuiperLLama/lib:$LD_LIBRARY_PATH
```

如果是在宿主机运行容器里编译出的程序，可能还会缺少容器里的系统库。推荐在同一个容器中编译和运行。

### `compute_86+PTX` 不支持

Jetson Orin Nano 应固定指定：

```bash
-DCMAKE_CUDA_ARCHITECTURES=87
```

避免 CMake 自动检测失败后生成不适合 Jetson 的架构列表。

### Qwen3 编译缺少 absl/re2/json

Qwen3 相关 tokenizer/解析依赖可能需要 `absl`、`re2`、`nlohmann_json` 等库。优先使用项目 CMake 自动拉取或系统包安装；如果网络不稳定，建议提前准备依赖或换源。

## 面试/答辩讲法

可以按这个顺序讲：

```text
1. 我选择 Jetson Orin Nano 作为边缘端部署平台。
2. 使用 Docker 固化 CUDA/JetPack 环境，避免宿主机依赖污染。
3. 模型文件和 tokenizer 通过挂载目录传入容器，不打进镜像。
4. 针对 Orin Nano 指定 CUDA_ARCHITECTURES=87 编译 C++/CUDA 推理程序。
5. 分别验证 TinyLlama/stories110M 和 Qwen3-0.6B 的本地推理。
6. 记录推理速度，分析模型大小、prompt 长度、功耗模式对性能的影响。
7. 解决过依赖缺失、动态库路径、Docker 网络、CUDA 架构等部署问题。
```

