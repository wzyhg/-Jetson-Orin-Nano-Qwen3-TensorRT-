# Jetson Docker

本文档说明 KuiperLLama 在 Jetson Orin Nano 上使用 Docker 构建和运行的流程。

## 为什么使用 Docker

Jetson 上的 CUDA、TensorRT、cuDNN、OpenCV、系统库和 JetPack 版本关系紧密。直接在宿主机安装依赖容易造成环境污染，也不方便迁移。

使用 Docker 的好处：

- 固定 JetPack/L4T 基础镜像
- 编译环境和运行环境一致
- 模型文件通过 volume 挂载，不需要打进镜像
- 方便删除和重建环境
- 适合展示边缘端部署流程

## 基础镜像

当前使用：

```dockerfile
nvcr.io/nvidia/l4t-jetpack:r36.4.0
```

该镜像适合 JetPack 6.x 系列。不同 JetPack 版本可以通过 `BASE_IMAGE` 替换。

## 构建镜像

在 Jetson 主机执行：

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

参数说明：

- `--network host`：构建阶段使用宿主机网络，减少 Docker bridge/iptables 问题
- `MODEL_SUPPORT=LLAMA2`：默认编译 LLaMA2/TinyLlama 类 demo
- `MODEL_SUPPORT=QWEN3`：构建时打开 Qwen3 支持
- `CUDA_ARCHITECTURES=87`：Jetson Orin Nano 的 CUDA compute capability
- `-t kuiperllama:jetson`：镜像名称

如果要直接构建 Qwen3 版本：

```bash
sudo docker build \
  --network host \
  --build-arg MODEL_SUPPORT=QWEN3 \
  --build-arg CUDA_ARCHITECTURES=87 \
  -f Dockerfile.jetson \
  -t kuiperllama:jetson \
  .
```

## 使用 docker compose

也可以使用 compose：

```bash
cd ~/LLM/KuiperLLama

sudo docker compose -f docker-compose.jetson.yml build
```

如果 Jetson 的 Docker bridge 网络报 iptables/raw table 错误，优先使用 `docker build --network host`。

## 启动容器

推荐开发模式：源码和模型都从宿主机挂载进去。

```bash
cd ~/LLM/KuiperLLama

sudo docker run --rm -it --runtime nvidia --network host \
  -v ~/LLM/KuiperLLama:/workspaces/KuiperLLama \
  -v ~/LLM/KuiperLLama/models:/models \
  -w /workspaces/KuiperLLama \
  kuiperllama:jetson bash
```

进入容器后：

```bash
pwd
ls
ls /models
```

应该看到：

```text
/workspaces/KuiperLLama
```

以及模型目录：

```text
/models
```

## 在容器内编译

因为源码被挂载进容器，所以可以在 VS Code 或宿主机修改代码，然后在容器内直接编译。

LLaMA2/TinyLlama：

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=87

cmake --build build
```

Qwen3：

```bash
cmake -S . -B build-qwen3 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DQWEN3_SUPPORT=ON

cmake --build build-qwen3
```

如果你改了源码，只需要重新执行：

```bash
cmake --build build
```

或：

```bash
cmake --build build-qwen3
```

不需要每次重建 Docker 镜像。

## 在容器内运行

TinyLlama / stories110M：

```bash
./build/demo/llama_infer \
  /models/stories110M.bin \
  /models/tokenizer.model \
  "hello"
```

Qwen3：

```bash
./build-qwen3/demo/qwen3_infer \
  /models/qwen3-0.6b/qwen0.6.bin2 \
  /models/qwen3-0.6b/tokenizer.json \
  "什么是人工智能"
```

如果动态库找不到：

```bash
export LD_LIBRARY_PATH=/workspaces/KuiperLLama/lib:$LD_LIBRARY_PATH
```

## 删除镜像

查看镜像：

```bash
sudo docker images
```

删除项目镜像：

```bash
sudo docker rmi kuiperllama:jetson
```

如果有容器占用镜像，先查看容器：

```bash
sudo docker ps -a
```

删除对应容器后再删镜像。

## 常见问题

### 没有 Docker 权限

报错：

```text
permission denied while trying to connect to the docker API
```

临时方案是在命令前加 `sudo`：

```bash
sudo docker images
```

长期方案是把当前用户加入 docker 组：

```bash
sudo usermod -aG docker $USER
```

然后重新登录。

### Docker build 阶段 iptables 报错

报错示例：

```text
iptables: can't initialize iptables table `raw'
failed to create endpoint ... on network bridge
```

使用 host network 构建：

```bash
sudo docker build --network host -f Dockerfile.jetson -t kuiperllama:jetson .
```

运行容器时也建议：

```bash
--network host
```

### Dockerfile 第一行拉取失败

如果出现：

```text
failed to resolve source metadata for docker.io/docker/dockerfile:1
```

可以删除 Dockerfile 顶部的：

```dockerfile
# syntax=docker/dockerfile:1
```

避免额外访问 Docker Hub frontend 镜像。

### `build` 目录不存在

如果使用 volume 挂载源码，镜像构建时产生的 `build` 目录可能被宿主机目录覆盖。进入容器后重新配置：

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=87

cmake --build build
```

### 容器里编译，宿主机运行报动态库错误

容器和宿主机的系统库可能不同。推荐保持：

```text
在哪里编译，就在哪里运行。
```

如果必须宿主机运行，需要安装对应的 `glog`、`absl`、`re2` 等依赖，并设置 `LD_LIBRARY_PATH`。

## 推荐开发流程

日常开发推荐三步：

```text
1. VS Code 在宿主机打开 ~/LLM/KuiperLLama 修改代码。
2. Docker 容器挂载源码目录，在容器内 cmake --build。
3. 在同一个容器内运行推理程序。
```

这样既能保留 Docker 演示，又不需要每次改代码都重建镜像。

