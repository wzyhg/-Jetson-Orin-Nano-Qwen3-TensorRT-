# Jetson Docker

This project can be built on Jetson Orin Nano with an NVIDIA L4T container.

## Build

Run these commands on the Jetson:

```bash
cd ~/KuiperLLama
docker compose -f docker-compose.jetson.yml build
```

The default build targets Qwen2/Qwen2.5. To build another demo:

```bash
MODEL_SUPPORT=LLAMA3 docker compose -f docker-compose.jetson.yml build
MODEL_SUPPORT=QWEN3 docker compose -f docker-compose.jetson.yml build
```

If your JetPack version needs a different base image, override it:

```bash
BASE_IMAGE=nvcr.io/nvidia/l4t-jetpack:r36.4.0 docker compose -f docker-compose.jetson.yml build
```

## Run

Start a shell with CUDA access:

```bash
docker compose -f docker-compose.jetson.yml run --rm kuiperllama
```

Model files should be placed in `./models` on the host. The container mounts that
directory as `/models`.

For Qwen2.5:

```bash
./build/demo/qwen_infer /models/Qwen2.5-0.5B.bin /models/Qwen2.5-0.5B/tokenizer.json
```

For Llama 3:

```bash
./build/demo/llama_infer /models/Llama-3.2-1B.bin /models/Llama-3.2-1B/tokenizer.json
```

## Notes

Jetson Orin Nano uses compute capability 8.7, so the default
`CUDA_ARCHITECTURES` is `87`.

The repository's demos initialize CUDA by default. INT8 quantized models are
CUDA-only in this implementation.
