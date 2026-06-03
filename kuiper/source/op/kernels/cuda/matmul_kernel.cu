#include <tensor/tensor.h>
#include <cub/block/block_reduce.cuh>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include "../kernels_interface.h"
#include "matmul_kernel.cuh"
namespace kernel {

namespace {
struct MatmulProfileStats {
  uint64_t calls = 0;
  double total_ms = 0.0;
};

std::map<std::string, MatmulProfileStats>& int8_matmul_profile_stats() {
  static auto* stats = new std::map<std::string, MatmulProfileStats>();
  return *stats;
}

std::mutex& int8_matmul_profile_mutex() {
  static auto* mutex = new std::mutex();
  return *mutex;
}

bool int8_matmul_profile_enabled() {
  static bool enabled = []() {
    const char* value = std::getenv("KUIPER_PROFILE_INT8_MATMUL");
    return value != nullptr && std::string(value) != "0";
  }();
  return enabled;
}

int int8_lmhead_rows_per_block() {
  static int rows = []() {
    const char* value = std::getenv("KUIPER_INT8_LMHEAD_ROWS");
    if (value == nullptr) {
      return 4;
    }
    const int parsed = std::atoi(value);
    if (parsed == 1 || parsed == 2 || parsed == 4 || parsed == 8) {
      return parsed;
    }
    return 4;
  }();
  return rows;
}

int int8_ffn_rows_per_block() {
  static int rows = []() {
    const char* value = std::getenv("KUIPER_INT8_FFN_ROWS");
    if (value == nullptr) {
      return 4;
    }
    const int parsed = std::atoi(value);
    if (parsed == 1 || parsed == 2 || parsed == 4) {
      return parsed;
    }
    return 4;
  }();
  return rows;
}

void print_int8_matmul_profile() {
  if (!int8_matmul_profile_enabled()) {
    return;
  }

  std::lock_guard<std::mutex> lock(int8_matmul_profile_mutex());
  auto& stats = int8_matmul_profile_stats();
  if (stats.empty()) {
    return;
  }

  std::vector<std::pair<std::string, MatmulProfileStats>> rows(stats.begin(), stats.end());
  std::sort(rows.begin(), rows.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.second.total_ms > rhs.second.total_ms;
  });

  std::cerr << "\n[KuiperProfile] INT8 matmul CUDA kernel profile\n";
  std::cerr << "shape, calls, total_ms, avg_us\n";
  for (const auto& item : rows) {
    const auto& shape = item.first;
    const auto& stat = item.second;
    const double avg_us = stat.total_ms * 1000.0 / static_cast<double>(stat.calls);
    std::cerr << shape << ", " << stat.calls << ", " << stat.total_ms << ", " << avg_us << "\n";
  }
}

void ensure_int8_matmul_profile_registered() {
  static bool registered = []() {
    std::atexit(print_int8_matmul_profile);
    return true;
  }();
  UNUSED(registered);
}

std::string matmul_profile_key(int32_t M, int32_t K, int32_t group_size) {
  std::ostringstream oss;
  oss << "K=" << K << " M=" << M << " group=" << group_size;
  return oss.str();
}

void add_int8_matmul_profile(int32_t M, int32_t K, int32_t group_size, float elapsed_ms) {
  ensure_int8_matmul_profile_registered();
  std::lock_guard<std::mutex> lock(int8_matmul_profile_mutex());
  auto& stat = int8_matmul_profile_stats()[matmul_profile_key(M, K, group_size)];
  stat.calls += 1;
  stat.total_ms += elapsed_ms;
}
}  // namespace

template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32(const float* input, const float* weight, float* output, int M,
                                      int K) {
  __shared__ float sdata[THREAD_PER_BLOCK];
  unsigned int tid = threadIdx.x;

  int start_row = blockIdx.x * ROW_PER_BLOCK;
  int end_row = start_row + ROW_PER_BLOCK;
  if (start_row >= K) {
    return;
  }

  constexpr int pack_size = 4;
  const int pack_num = M / pack_size;
  const int pack_off = pack_size * pack_num;

#pragma unroll
  for (int p = start_row; p < end_row; ++p) {
    sdata[tid] = 0;
    int row_offset = p * M;
    float4* input_float4_ptr = (float4*)input;
    float4* weight_float4_ptr = (float4*)(weight + row_offset);

#pragma unroll
    for (int i = tid; i < pack_num; i += blockDim.x) {
      float4 input_float4 = *(input_float4_ptr + i);
      float4 weight_float4 = *(weight_float4_ptr + i);
      float part_sum = input_float4.x * weight_float4.x + input_float4.y * weight_float4.y +
                       input_float4.z * weight_float4.z + input_float4.w * weight_float4.w;
      sdata[tid] += part_sum;
    }

    for (int i = pack_off + tid; i < M; i += blockDim.x) {
      sdata[tid] += input[i] * weight[row_offset + i];
    }

    __syncthreads();

    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();

    if (tid == 0) {
      output[p] = part_sum;
    }
    __syncthreads();
  }
}

template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32int8(const float* input, const int8_t* weight,
                                          const float* scales, const int32_t group_size,
                                          float* output, int M, int K) {
  __shared__ float sdata[THREAD_PER_BLOCK];
  unsigned int tid = threadIdx.x;

  int start_row = blockIdx.x * ROW_PER_BLOCK;
  int end_row = start_row + ROW_PER_BLOCK;
  if (start_row >= K) {
    return;
  }
  const bool group_size_is_64 = group_size == 64;
  for (int p = start_row; p < end_row && p < K; ++p) {
    sdata[tid] = 0;
    for (int i = tid; i < M; i += THREAD_PER_BLOCK) {
      const int weight_idx = p * M + i;
      const int group_idx = group_size_is_64 ? (weight_idx >> 6) : (weight_idx / group_size);
      sdata[tid] += input[i] * scales[group_idx] * static_cast<float>(weight[weight_idx]);
    }
    __syncthreads();

    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();

    if (tid == 0) {
      output[p] = part_sum;
    }
    __syncthreads();
  }
}

template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32int8_g64(const float* input, const int8_t* weight,
                                              const float* scales, float* output, int M, int K);

template <int ROW_PER_BLOCK>
void launch_matmul_kernel_cu_fp32int8(const tensor::Tensor& input, const tensor::Tensor& weight,
                                      const tensor::Tensor& output, int32_t group_size,
                                      const tensor::Tensor& scale, int32_t M, int32_t K,
                                      cudaStream_t stream);

void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                      const tensor::Tensor& output, const float scale, const CudaConfig* config) {
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  const int32_t K = weight.get_dim(0);  // row
  const int32_t M = weight.get_dim(1);  // col
  int packet_size = 4;
  // CHECK_EQ(M % packet_size, 0);

  CHECK_EQ(M, input.get_dim(0));
  if (config && config->stream) {
    matmul_kernel_cu_fp32<128, 1><<<K, 128, 0, config->stream>>>(
        input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()), M, K);
  } else {
    matmul_kernel_cu_fp32<128, 1><<<K, 128>>>(input.ptr<float>(), weight.ptr<float>(),
                                              const_cast<float*>(output.ptr<float>()), M, K);
  }
}

void matmul_kernel_cu_qint8(const tensor::Tensor& input, const tensor::Tensor& weight,
                            const tensor::Tensor& output, int32_t group_size,
                            const tensor::Tensor& scale, const CudaConfig* config) {
  CHECK(config != nullptr);
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  const int32_t K = weight.get_dim(0);  // row
  const int32_t M = weight.get_dim(1);  // col
  int packet_size = 4;
  CHECK_EQ(M % packet_size, 0);
  CHECK_EQ(M, input.get_dim(0));
  cudaEvent_t start_event = nullptr;
  cudaEvent_t stop_event = nullptr;
  const bool profile = int8_matmul_profile_enabled();
  if (profile) {
    ensure_int8_matmul_profile_registered();
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaEventRecord(start_event, config->stream);
  }
  if (config->stream) {
    if (K >= 32768) {
      const int rows_per_block = int8_lmhead_rows_per_block();
      if (rows_per_block == 1) {
        launch_matmul_kernel_cu_fp32int8<1>(input, weight, output, group_size, scale, M, K,
                                            config->stream);
      } else if (rows_per_block == 2) {
        launch_matmul_kernel_cu_fp32int8<2>(input, weight, output, group_size, scale, M, K,
                                            config->stream);
      } else if (rows_per_block == 8) {
        launch_matmul_kernel_cu_fp32int8<8>(input, weight, output, group_size, scale, M, K,
                                            config->stream);
      } else {
        launch_matmul_kernel_cu_fp32int8<4>(input, weight, output, group_size, scale, M, K,
                                            config->stream);
      }
    } else {
      const int rows_per_block = int8_ffn_rows_per_block();
      if (rows_per_block == 2) {
        launch_matmul_kernel_cu_fp32int8<2>(input, weight, output, group_size, scale, M, K,
                                            config->stream);
      } else if (rows_per_block == 4) {
        launch_matmul_kernel_cu_fp32int8<4>(input, weight, output, group_size, scale, M, K,
                                            config->stream);
      } else {
        launch_matmul_kernel_cu_fp32int8<1>(input, weight, output, group_size, scale, M, K,
                                            config->stream);
      }
    }
  } else {
    if (K >= 32768) {
      const int rows_per_block = int8_lmhead_rows_per_block();
      if (rows_per_block == 1) {
        launch_matmul_kernel_cu_fp32int8<1>(input, weight, output, group_size, scale, M, K,
                                            nullptr);
      } else if (rows_per_block == 2) {
        launch_matmul_kernel_cu_fp32int8<2>(input, weight, output, group_size, scale, M, K,
                                            nullptr);
      } else if (rows_per_block == 8) {
        launch_matmul_kernel_cu_fp32int8<8>(input, weight, output, group_size, scale, M, K,
                                            nullptr);
      } else {
        launch_matmul_kernel_cu_fp32int8<4>(input, weight, output, group_size, scale, M, K,
                                            nullptr);
      }
    } else {
      const int rows_per_block = int8_ffn_rows_per_block();
      if (rows_per_block == 2) {
        launch_matmul_kernel_cu_fp32int8<2>(input, weight, output, group_size, scale, M, K,
                                            nullptr);
      } else if (rows_per_block == 4) {
        launch_matmul_kernel_cu_fp32int8<4>(input, weight, output, group_size, scale, M, K,
                                            nullptr);
      } else {
        launch_matmul_kernel_cu_fp32int8<1>(input, weight, output, group_size, scale, M, K,
                                            nullptr);
      }
    }
  }
  if (profile) {
    cudaEventRecord(stop_event, config->stream);
    cudaEventSynchronize(stop_event);
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start_event, stop_event);
    add_int8_matmul_profile(M, K, group_size, elapsed_ms);
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
  }
}

template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32int8_g64(const float* input, const int8_t* weight,
                                              const float* scales, float* output, int M, int K) {
  __shared__ float sdata[THREAD_PER_BLOCK];
  unsigned int tid = threadIdx.x;

  int start_row = blockIdx.x * ROW_PER_BLOCK;
  int end_row = start_row + ROW_PER_BLOCK;
  if (start_row >= K) {
    return;
  }
  for (int p = start_row; p < end_row && p < K; ++p) {
    sdata[tid] = 0;
    for (int i = tid; i < M; i += THREAD_PER_BLOCK) {
      const int weight_idx = p * M + i;
      sdata[tid] += input[i] * scales[weight_idx >> 6] * static_cast<float>(weight[weight_idx]);
    }
    __syncthreads();

    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();

    if (tid == 0) {
      output[p] = part_sum;
    }
    __syncthreads();
  }
}

template <int ROW_PER_BLOCK>
void launch_matmul_kernel_cu_fp32int8(const tensor::Tensor& input, const tensor::Tensor& weight,
                                      const tensor::Tensor& output, int32_t group_size,
                                      const tensor::Tensor& scale, int32_t M, int32_t K,
                                      cudaStream_t stream) {
  const int grid = (K + ROW_PER_BLOCK - 1) / ROW_PER_BLOCK;
  if (group_size == 64) {
    matmul_kernel_cu_fp32int8_g64<128, ROW_PER_BLOCK><<<grid, 128, 0, stream>>>(
        input.ptr<float>(), weight.ptr<int8_t>(), scale.ptr<float>(),
        const_cast<float*>(output.ptr<float>()), M, K);
  } else {
    matmul_kernel_cu_fp32int8<128, ROW_PER_BLOCK><<<grid, 128, 0, stream>>>(
        input.ptr<float>(), weight.ptr<int8_t>(), scale.ptr<float>(), group_size,
        const_cast<float*>(output.ptr<float>()), M, K);
  }
}
}  // namespace kernel
