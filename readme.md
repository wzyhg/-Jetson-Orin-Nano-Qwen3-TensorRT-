
# Jetson Qwen3 + YOLOv8 TensorRT

## Quick Start

This section records the verified commands. Use these first.

### 0. Host Performance Mode

Run on the Jetson host before testing speed:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
sudo jetson_clocks --show
```

Expected clock state:

```text
GPU CurrentFreq=612000000
EMC CurrentFreq=2133000000
NV Power Mode: 15W
```

### 1. Run Qwen3 INT8, About 32 steps/s

Use the saved good container:

```bash
sudo docker start -ai kuiper-qwen3-fixed
```

Run inside the container:

```bash
cd /workspaces/KuiperLLama

export LD_LIBRARY_PATH=/workspaces/KuiperLLama/lib:$LD_LIBRARY_PATH
unset KUIPER_PROFILE_INT8_MATMUL
export KUIPER_INT8_FFN_ROWS=4
export KUIPER_INT8_LMHEAD_ROWS=4
export KUIPER_FUSED_INT8_LMHEAD=1

./build-qwen3/demo/qwen3_infer \
  ./models/qwen3-0.6b/qwen3-int8.bin \
  ./models/qwen3-0.6b/tokenizer.json \
  "hello" \
  --int8
```

Expected result:

```text
steps/s: about 32
```

Important notes:

```text
qwen3-int8.bin must be used with --int8.
qwen0.6.bin2 is the fp32 checkpoint and is slower.
Do not enable KUIPER_PROFILE_INT8_MATMUL for normal speed tests.
Profiling inserts CUDA event timing and slows inference.
```

### 2. Run YOLO Gimbal Demo With Window

Run on the Jetson host. The current binary uses `--trt_model` and `--target-class`; do not use `--engine` or `--target`.

CSI camera:

```bash
cd /home/ubuntu/LLM/KuiperLLama-Q/yolo/build

export DISPLAY=:0

./object_tracking_gimbal \
  --trt_model ../models/yolov8n.engine.Orin.fp16.1.1 \
  --input csi \
  --target-class person \
  --servo-port /dev/ttyUSB0
```

USB camera:

```bash
cd /home/ubuntu/LLM/KuiperLLama-Q/yolo/build

./object_tracking_gimbal \
  --trt_model ../models/yolov8n.engine.Orin.fp16.1.1 \
  --input 0 \
  --target-class person \
  --servo-port /dev/ttyUSB0
```

Window display is enabled by default. If running on the Jetson local desktop:

```bash
export DISPLAY=:0
```

If the window is all green, the camera input format is usually wrong. Try `--input csi` for CSI camera and `--input 0` for USB camera.

Restart CSI camera service:

```bash
sudo systemctl restart nvargus-daemon
```

Validate CSI camera alone:

```bash
gst-launch-1.0 nvarguscamerasrc ! \
  'video/x-raw(memory:NVMM),width=1280,height=720,framerate=60/1' ! \
  nvvidconv ! 'video/x-raw,format=BGRx' ! videoconvert ! autovideosink
```

Check USB camera formats:

```bash
v4l2-ctl --device=/dev/video0 --list-formats-ext
```

### 3. Run Qwen3 HTTP Parser

Use the same saved container:

```bash
sudo docker start -ai kuiper-qwen3-fixed
```

Run inside the container:

```bash
cd /workspaces/KuiperLLama
export LD_LIBRARY_PATH=/workspaces/KuiperLLama/lib:$LD_LIBRARY_PATH
unset KUIPER_PROFILE_INT8_MATMUL
export KUIPER_INT8_FFN_ROWS=4
export KUIPER_INT8_LMHEAD_ROWS=4
export KUIPER_FUSED_INT8_LMHEAD=1

cmake --build build-qwen3 --target qwen3_infer

python3 apps/qwen3_server/qwen3_http_server.py \
  --host 127.0.0.1 \
  --port 18080 \
  --model /models/qwen3-0.6b/qwen3-int8.bin \
  --tokenizer /models/qwen3-0.6b/tokenizer.json \
  --int8 \
  --max-steps 128 \
  --timeout 120
```

Test from host:

```bash
curl -X POST http://127.0.0.1:18080/parse \
  -H "Content-Type: application/json" \
  -d '{"command":"find person"}'
```

The Qwen3 HTTP service is INT8 and persistent by default in this project. It starts one `qwen3_infer --server` worker, loads the model once, and reuses it for later HTTP requests. The command above also passes `--int8` explicitly so the runtime flag always matches `qwen3-int8.bin`.

### 4. Combined Run: YOLO + Qwen3 + ROS2 Command

Use this when running the full vision-language gimbal demo.

Terminal 0, Jetson host performance mode:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
sudo jetson_clocks --show
```

Terminal 1, Jetson host, start YOLO ROS2 node:

```bash
cd /home/ubuntu/LLM/KuiperLLama-Q/yolo/ros2

source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run yolov8_gimbal_ros2 yolov8_gimbal_node \
  --ros-args \
  -p engine_path:=/home/ubuntu/LLM/KuiperLLama-Q/yolo/models/yolov8n.engine.Orin.fp16.1.1 \
  -p camera_input:=csi \
  -p target_class:=person \
  -p servo_port:=/dev/ttyUSB0 \
  -p display_window:=false


停止 YOLO 节点：

sudo pkill -f yolov8_gimbal_node


```

Terminal 2, Qwen3 INT8 HTTP service in the saved container:

```bash
sudo docker start -ai kuiper-qwen3-fixed

cd /workspaces/KuiperLLama
export LD_LIBRARY_PATH=/workspaces/KuiperLLama/lib:$LD_LIBRARY_PATH
unset KUIPER_PROFILE_INT8_MATMUL
export KUIPER_INT8_FFN_ROWS=4
export KUIPER_INT8_LMHEAD_ROWS=4
export KUIPER_FUSED_INT8_LMHEAD=1

cmake --build build-qwen3 --target qwen3_infer

python3 apps/qwen3_server/qwen3_http_server.py \
  --host 127.0.0.1 \
  --port 18080 \
  --model /models/qwen3-0.6b/qwen3-int8.bin \
  --tokenizer /models/qwen3-0.6b/tokenizer.json \
  --int8 \
  --max-steps 128 \
  --timeout 120
```

Terminal 3, Jetson host, start the ROS2 command parser client:

```bash
cd /home/ubuntu/LLM/KuiperLLama-Q/yolo/ros2

source /opt/ros/humble/setup.bash
colcon build --packages-select gimbal_command_ros2 --symlink-install
source install/setup.bash

ros2 run gimbal_command_ros2 command_cli \
  --ros-args \
  -p qwen_server_url:=http://127.0.0.1:18080/parse
```

HTTP service smoke test:

```bash
curl -X POST http://127.0.0.1:18080/parse \
  -H "Content-Type: application/json" \
  -d '{"command":"find person"}'
```

