#include "int8_lmhead_argmax_kernel.cuh"
#include <cuda_runtime.h>
#include <cfloat>

namespace kernel {
namespace {

__forceinline__ __device__ void warp_reduce_argmax_float(float& val, size_t& idx) {
  unsigned int mask = __ballot_sync(0xFFFFFFFF, true);
  for (unsigned int offset = warpSize >> 1; offset > 0; offset >>= 1) {
    float other_val = __shfl_down_sync(mask, val, offset, warpSize);
    size_t other_idx = __shfl_down_sync(mask, idx, offset, warpSize);
    if (other_val > val || (other_val == val && other_idx < idx)) {
      val = other_val;
      idx = other_idx;
    }
  }
}

__forceinline__ __device__ void block_reduce_argmax_float(float& val, size_t& idx,
                                                          float* shared_val,
                                                          size_t* shared_idx) {
  int lane_id = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;

  warp_reduce_argmax_float(val, idx);

  __syncthreads();
  if (lane_id == 0) {
    shared_val[warp_id] = val;
    shared_idx[warp_id] = idx;
  }

  __syncthreads();
  if (threadIdx.x < blockDim.x / warpSize) {
    val = shared_val[lane_id];
    idx = shared_idx[lane_id];
  } else {
    val = -FLT_MAX;
    idx = SIZE_MAX;
  }

  if (warp_id == 0) {
    warp_reduce_argmax_float(val, idx);
  }
}

__global__ void lmhead_partial_argmax_kernel(const float* input, const int8_t* weight,
                                             const float* scales, int32_t group_size, int32_t M,
                                             int32_t K, float* partial_values,
                                             size_t* partial_indices) {
  __shared__ float shared_sum[128];
  __shared__ float shared_max_val[32];
  __shared__ size_t shared_max_idx[32];

  const int rows_per_block = 4;
  const int row_base = blockIdx.x * rows_per_block;
  const unsigned int tid = threadIdx.x;

  float best_val = -FLT_MAX;
  size_t best_idx = SIZE_MAX;

  for (int row_offset = 0; row_offset < rows_per_block; ++row_offset) {
    const int row = row_base + row_offset;
    if (row >= K) {
      continue;
    }

    float sum = 0.0f;
    for (int col = tid; col < M; col += blockDim.x) {
      const int weight_idx = row * M + col;
      const int scale_idx = group_size == 64 ? (weight_idx >> 6) : (weight_idx / group_size);
      sum += input[col] * scales[scale_idx] * static_cast<float>(weight[weight_idx]);
    }

    shared_sum[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        shared_sum[tid] += shared_sum[tid + stride];
      }
      __syncthreads();
    }

    if (tid == 0) {
      const float row_val = shared_sum[0];
      if (row_val > best_val || (row_val == best_val && static_cast<size_t>(row) < best_idx)) {
        best_val = row_val;
        best_idx = static_cast<size_t>(row);
      }
    }
    __syncthreads();
  }

  if (tid != 0) {
    best_val = -FLT_MAX;
    best_idx = SIZE_MAX;
  }

  block_reduce_argmax_float(best_val, best_idx, shared_max_val, shared_max_idx);
  if (tid == 0) {
    partial_values[blockIdx.x] = best_val;
    partial_indices[blockIdx.x] = best_idx;
  }
}

__global__ void final_argmax_kernel(const float* partial_values, const size_t* partial_indices,
                                    int32_t count, size_t* output) {
  __shared__ float shared_val[32];
  __shared__ size_t shared_idx[32];

  float best_val = -FLT_MAX;
  size_t best_idx = SIZE_MAX;
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    const float val = partial_values[i];
    const size_t idx = partial_indices[i];
    if (val > best_val || (val == best_val && idx < best_idx)) {
      best_val = val;
      best_idx = idx;
    }
  }

  block_reduce_argmax_float(best_val, best_idx, shared_val, shared_idx);
  if (threadIdx.x == 0) {
    *output = best_idx;
  }
}

}  // namespace

size_t int8_lmhead_argmax_kernel_cu(const float* input, const int8_t* weight, const float* scales,
                                    int32_t group_size, int32_t M, int32_t K, void* stream) {
  const int rows_per_block = 4;
  const int blocks = (K + rows_per_block - 1) / rows_per_block;
  float* partial_values = nullptr;
  size_t* partial_indices = nullptr;
  size_t* output_index = nullptr;
  cudaMalloc(&partial_values, blocks * sizeof(float));
  cudaMalloc(&partial_indices, blocks * sizeof(size_t));
  cudaMalloc(&output_index, sizeof(size_t));
  size_t host_index = 0;

  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    lmhead_partial_argmax_kernel<<<blocks, 128, 0, stream_>>>(
        input, weight, scales, group_size, M, K, partial_values, partial_indices);
    final_argmax_kernel<<<1, 512, 0, stream_>>>(partial_values, partial_indices, blocks,
                                                output_index);
    cudaMemcpyAsync(&host_index, output_index, sizeof(size_t), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
  } else {
    lmhead_partial_argmax_kernel<<<blocks, 128>>>(
        input, weight, scales, group_size, M, K, partial_values, partial_indices);
    final_argmax_kernel<<<1, 512>>>(partial_values, partial_indices, blocks, output_index);
    cudaMemcpy(&host_index, output_index, sizeof(size_t), cudaMemcpyDeviceToHost);
  }
  cudaFree(partial_values);
  cudaFree(partial_indices);
  cudaFree(output_index);
  return host_index;
}

}  // namespace kernel
