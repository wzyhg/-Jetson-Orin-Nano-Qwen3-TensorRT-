#ifndef INT8_LMHEAD_ARGMAX_KERNEL_CUH
#define INT8_LMHEAD_ARGMAX_KERNEL_CUH
#include <cstddef>
#include <cstdint>
namespace kernel {
size_t int8_lmhead_argmax_kernel_cu(const float* input, const int8_t* weight, const float* scales,
                                    int32_t group_size, int32_t M, int32_t K, void* stream);
}
#endif  // INT8_LMHEAD_ARGMAX_KERNEL_CUH
