#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Optional.h>

#include "cuda_utils.h"
#include "cuda_compat.h"
#include "dispatch_utils.h"
#include "quantization/vectorization_utils.cuh"

#ifdef USE_ROCM
  #include "quantization/w8a8/fp8/amd/quant_utils.cuh"
#else
  #include "quantization/w8a8/fp8/nvidia/quant_utils.cuh"
#endif

#include <algorithm>
#include <cassert>
#include <cfloat>

#ifdef USE_ROCM
  #include <hip/hip_bf16.h>
typedef __hip_bfloat16 __nv_bfloat16;
#endif

void swap_blocks(torch::Tensor& src, torch::Tensor& dst,
                 const torch::Tensor& block_mapping) {
  torch::Device src_device = src.device();
  torch::Device dst_device = dst.device();
  cudaMemcpyKind memcpy_type;
  if (src_device.is_cuda() && dst_device.is_cuda()) {
    TORCH_CHECK(src_device.index() == dst_device.index(),
                "src and dst must be on the same GPU");
    memcpy_type = cudaMemcpyDeviceToDevice;
  } else if (src_device.is_cuda() && dst_device.is_cpu()) {
    memcpy_type = cudaMemcpyDeviceToHost;
  } else if (src_device.is_cpu() && dst_device.is_cuda()) {
    memcpy_type = cudaMemcpyHostToDevice;
  } else {
    TORCH_CHECK(false, "Invalid device combination");
  }

  // NOTE(youkaichao): keep in mind that `block_mapping` should be
  // a cpu tensor, otherwise every `item` call will require a gpu-cpu
  // synchronization.
  TORCH_CHECK(block_mapping.device().is_cpu(), "block_mapping must be on CPU");

  char* src_ptr = static_cast<char*>(src.data_ptr());
  char* dst_ptr = static_cast<char*>(dst.data_ptr());

  // We use the stride instead of numel in case the cache is padded for memory
  // alignment reasons, we assume the blocks data (inclusive of any padding)
  // is contiguous in memory
  const int64_t block_size_in_bytes = src.element_size() * src.stride(0);
  const at::cuda::OptionalCUDAGuard device_guard(
      src_device.is_cuda() ? src_device : dst_device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // NOTE(woosuk): This can be slow if the number of blocks is large.
  const int64_t num_blocks = block_mapping.size(0);
  for (size_t i = 0; i < num_blocks; i++) {
    int64_t src_block_number = block_mapping[i][0].item<int64_t>();
    int64_t dst_block_number = block_mapping[i][1].item<int64_t>();
    int64_t src_offset = src_block_number * block_size_in_bytes;
    int64_t dst_offset = dst_block_number * block_size_in_bytes;
    cudaMemcpyAsync(dst_ptr + dst_offset, src_ptr + src_offset,
                    block_size_in_bytes, memcpy_type, stream);
  }
}

namespace vllm {

// Grid: (num_layers, num_pairs)
template <typename scalar_t>
__global__ void copy_blocks_kernel(int64_t* key_cache_ptrs,
                                   int64_t* value_cache_ptrs,
                                   const int64_t* __restrict__ block_mapping,
                                   const int numel_per_block) {
  const int layer_idx = blockIdx.x;
  const int pair_idx = blockIdx.y;

  scalar_t* key_cache = reinterpret_cast<scalar_t*>(key_cache_ptrs[layer_idx]);
  scalar_t* value_cache =
      reinterpret_cast<scalar_t*>(value_cache_ptrs[layer_idx]);
  int64_t src_block_number = block_mapping[2 * pair_idx];
  int64_t dst_block_number = block_mapping[2 * pair_idx + 1];

  const int64_t src_block_offset = src_block_number * numel_per_block;
  const int64_t dst_block_offset = dst_block_number * numel_per_block;
  for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
    int64_t src_offset = src_block_offset + i;
    int64_t dst_offset = dst_block_offset + i;
    key_cache[dst_offset] = key_cache[src_offset];
  }
  for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
    int64_t src_offset = src_block_offset + i;
    int64_t dst_offset = dst_block_offset + i;
    value_cache[dst_offset] = value_cache[src_offset];
  }
}

// Kernel for MLA, which works on a single joint kv_cache
// Grid: (num_layers, num_pairs)
template <typename scalar_t>
__global__ void copy_blocks_mla_kernel(
    int64_t* cache_ptrs, const int64_t* __restrict__ block_mapping,
    const int mem_footprint_per_block) {
  const int layer_idx = blockIdx.x;
  const int pair_idx = blockIdx.y;
  scalar_t* cache = reinterpret_cast<scalar_t*>(cache_ptrs[layer_idx]);
  int64_t src_block = block_mapping[2 * pair_idx];
  int64_t dst_block = block_mapping[2 * pair_idx + 1];
  int64_t src_offset = src_block * mem_footprint_per_block;
  int64_t dst_offset = dst_block * mem_footprint_per_block;
  for (int i = threadIdx.x; i < mem_footprint_per_block; i += blockDim.x) {
    cache[dst_offset + i] = cache[src_offset + i];
  }
}

}  // namespace vllm

namespace vllm {

// Used to copy/convert one element
template <typename OutT, typename InT, Fp8KVCacheDataType kv_dt>
struct CopyWithScaleOp {
  float scale;

  __device__ __forceinline__ void operator()(OutT& dst, const InT src) const {
    if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
      dst = static_cast<OutT>(src);
    } else {
      dst = fp8::scaled_convert<OutT, InT, kv_dt>(src, scale);
    }
  }
};

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void reshape_and_cache_kernel(
    const scalar_t* __restrict__ key,    // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,  // [num_tokens, num_heads, head_size]
    cache_t* __restrict__ key_cache,     // [num_blocks, num_heads, head_size/x,
                                         // block_size, x]
    cache_t* __restrict__ value_cache,   // [num_blocks, num_heads, head_size,
                                         // block_size]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int key_stride, const int value_stride, const int num_heads,
    const int head_size, const int block_size, const int x,
    const float* k_scale, const float* v_scale) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  const int h_block_count = head_size / x;  // head_size//x

  const int h_block_idx = threadIdx.x;
  if (h_block_idx >= num_heads * h_block_count) {
    return;
  }

  const int head_idx = h_block_idx / h_block_count;
  const int h_block = h_block_idx % h_block_count;

  const scalar_t* __restrict__ key_src =
      key + token_idx * key_stride + head_idx * head_size + h_block * x;
  const int64_t src_value_start =
      token_idx * value_stride + head_idx * head_size + h_block * x;

  cache_t* __restrict__ key_dst =
      key_cache + block_idx * num_heads * h_block_count * block_size * x +
      head_idx * h_block_count * block_size * x + h_block * block_size * x +
      block_offset * x;
  const int64_t tgt_value_start =
      block_idx * num_heads * h_block_count * x * block_size +
      head_idx * h_block_count * x * block_size + h_block * x * block_size +
      block_offset;

  constexpr int VEC_SIZE = (sizeof(scalar_t) == 2) ? 8 : 4;
  float k_scale_val = (kv_dt == Fp8KVCacheDataType::kAuto) ? 0.f : *k_scale;
  CopyWithScaleOp<cache_t, scalar_t, kv_dt> k_op{k_scale_val};
  float v_scale_val = (kv_dt == Fp8KVCacheDataType::kAuto) ? 0.f : *v_scale;
  CopyWithScaleOp<cache_t, scalar_t, kv_dt> v_op{v_scale_val};

  vectorize_with_alignment<VEC_SIZE>(key_src, key_dst, x, 0, 1, k_op);

  const scalar_t* __restrict__ value_src = value + src_value_start;
  cache_t* __restrict__ value_dst = value_cache + tgt_value_start;
#pragma unroll
  for (int i = 0; i < x; i++) {
    v_op(value_dst[i * block_size], value_src[i]);
  }
}

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void reshape_and_cache_flash_kernel(
    const scalar_t* __restrict__ key,    // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,  // [num_tokens, num_heads, head_size]
    cache_t* __restrict__ key_cache,     // NHD or HND, shape see comments below
    cache_t* __restrict__ value_cache,   // same above
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int64_t block_stride, const int64_t page_stride,
    const int64_t head_stride, const int64_t key_stride,
    const int64_t value_stride, const int num_heads, const int head_size,
    const int block_size, const int num_blocks, const float* k_scale,
    const float* v_scale) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  // NOTE: slot_idx can be -1 if the token is padded
  if (slot_idx < 0) {
    return;
  }
  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  // Bounds check to avoid GPU memory access fault from invalid slot_mapping
  // (e.g. during CUDA graph capture/replay or race).
  if (block_idx >= num_blocks || block_offset >= block_size) {
    return;
  }
  const int n_elems = num_heads * head_size;

  // pointers to the beginning of the source row for this token.
  const scalar_t* __restrict__ key_src = key + token_idx * key_stride;
  const scalar_t* __restrict__ value_src = value + token_idx * value_stride;

  // find the start position inside the kv-cache for this token.
  cache_t* __restrict__ key_dst =
      key_cache + block_idx * block_stride + block_offset * page_stride;
  cache_t* __restrict__ value_dst =
      value_cache + block_idx * block_stride + block_offset * page_stride;

  // this is true for the NHD layout where `head_stride == head_size`
  const bool is_contiguous_heads = (head_stride == head_size);

  float k_scale_val = (kv_dt == Fp8KVCacheDataType::kAuto) ? 0.f : *k_scale;
  float v_scale_val = (kv_dt == Fp8KVCacheDataType::kAuto) ? 0.f : *v_scale;
  constexpr int VEC_SIZE = (sizeof(scalar_t) == 2) ? 8 : 4;
  CopyWithScaleOp<cache_t, scalar_t, kv_dt> k_op{k_scale_val};
  CopyWithScaleOp<cache_t, scalar_t, kv_dt> v_op{v_scale_val};

#ifdef USE_ROCM
  // FP4 uses 2 values per byte (packed); write n_elems/2 bytes.
  // Caller must ensure head_size is even and cache last dim is head_size/2.
  if constexpr (kv_dt == Fp8KVCacheDataType::kFp4E2M1) {
    if (head_size & 1) {
      return;  // Should not happen if C++ validation is used.
    }
    const int half_head = head_size / 2;
    const int n_elems_packed = num_heads * half_head;
    for (int i = threadIdx.x; i < n_elems_packed; i += blockDim.x) {
      const int head_idx = i / half_head;
      const int j = i % half_head;
      const int64_t dst_off = is_contiguous_heads
          ? static_cast<int64_t>(i)
          : (static_cast<int64_t>(head_idx) * head_stride + j);
      if (dst_off >= page_stride) {
        continue;  // Guard against out-of-bounds write (avoid GPU memory fault)
      }
      const int src_base = 2 * i;
      uint8_t k_lo = fp8::scaled_convert<uint8_t, scalar_t, kv_dt>(
          key_src[src_base], k_scale_val);
      uint8_t k_hi = fp8::scaled_convert<uint8_t, scalar_t, kv_dt>(
          key_src[src_base + 1], k_scale_val);
      key_dst[dst_off] = static_cast<cache_t>((k_lo & 0xF) | ((k_hi & 0xF) << 4));
      uint8_t v_lo = fp8::scaled_convert<uint8_t, scalar_t, kv_dt>(
          value_src[src_base], v_scale_val);
      uint8_t v_hi = fp8::scaled_convert<uint8_t, scalar_t, kv_dt>(
          value_src[src_base + 1], v_scale_val);
      value_dst[dst_off] =
          static_cast<cache_t>((v_lo & 0xF) | ((v_hi & 0xF) << 4));
    }
    return;
  }
#endif

  if (is_contiguous_heads) {
    // NHD layout
    // kv cache: [num_blocks, block_size, num_heads, head_size]
    vectorize_with_alignment<VEC_SIZE>(key_src, key_dst, n_elems, threadIdx.x,
                                       blockDim.x, k_op);

    vectorize_with_alignment<VEC_SIZE>(value_src, value_dst, n_elems,
                                       threadIdx.x, blockDim.x, v_op);

  } else {
    // HND layout: heads are strided, but each head_size segment is contiguous
    // kv cache: [num_blocks, num_heads, block_size, head_size]
    const int lane = threadIdx.x & 31;     // 0..31 within warp
    const int warp_id = threadIdx.x >> 5;  // warp index within block
    const int warps_per_block = blockDim.x >> 5;

    for (int head = warp_id; head < num_heads; head += warps_per_block) {
      const scalar_t* __restrict__ k_src_h = key_src + head * head_size;
      const scalar_t* __restrict__ v_src_h = value_src + head * head_size;

      cache_t* __restrict__ k_dst_h =
          key_dst + static_cast<int64_t>(head) * head_stride;
      cache_t* __restrict__ v_dst_h =
          value_dst + static_cast<int64_t>(head) * head_stride;

      // within each head, let the 32 threads of the warp perform the vector
      // copy
      vectorize_with_alignment<VEC_SIZE>(k_src_h, k_dst_h, head_size, lane, 32,
                                         k_op);

      vectorize_with_alignment<VEC_SIZE>(v_src_h, v_dst_h, head_size, lane, 32,
                                         v_op);
    }
  }
}

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void concat_and_cache_mla_kernel(
    const scalar_t* __restrict__ kv_c,  // [num_tokens, kv_lora_rank]
    const scalar_t* __restrict__ k_pe,  // [num_tokens, pe_dim]
    cache_t* __restrict__ kv_cache,  // [num_blocks, block_size, (kv_lora_rank
                                     // + pe_dim)]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int block_stride,                    //
    const int entry_stride,                    //
    const int kv_c_stride,                     //
    const int k_pe_stride,                     //
    const int kv_lora_rank,                    //
    const int pe_dim,                          //
    const int block_size,                      //
    const float* scale                         //
) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  // NOTE: slot_idx can be -1 if the token is padded
  if (slot_idx < 0) {
    return;
  }
  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;

  auto copy = [&](const scalar_t* __restrict__ src, cache_t* __restrict__ dst,
                  int src_stride, int dst_stride, int size, int offset) {
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
      const int64_t src_idx = token_idx * src_stride + i;
      const int64_t dst_idx =
          block_idx * block_stride + block_offset * entry_stride + i + offset;
      if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
        dst[dst_idx] = src[src_idx];
      } else {
        dst[dst_idx] =
            fp8::scaled_convert<cache_t, scalar_t, kv_dt>(src[src_idx], *scale);
      }
    }
  };

  copy(kv_c, kv_cache, kv_c_stride, block_stride, kv_lora_rank, 0);
  copy(k_pe, kv_cache, k_pe_stride, block_stride, pe_dim, kv_lora_rank);
}

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void concat_and_cache_ds_mla_kernel(
    const scalar_t* __restrict__ kv_c,  // [num_tokens, kv_lora_rank]
    const scalar_t* __restrict__ k_pe,  // [num_tokens, pe_dim]
    cache_t* __restrict__ kv_cache,  // [num_blocks, block_size, (kv_lora_rank
                                     // + pe_dim)]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int block_stride,                    //
    const int entry_stride,                    //
    const int kv_c_stride,                     //
    const int k_pe_stride,                     //
    const int kv_lora_rank,                    //
    const int pe_dim,                          //
    const int block_size,                      //
    const float* scale                         //
) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  // NOTE: slot_idx can be -1 if the token is padded
  if (slot_idx < 0) {
    return;
  }
  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  const int64_t dst_idx_start =
      block_idx * block_stride + block_offset * entry_stride;

  // For the NoPE part, each tile of 128 elements is handled by half of one warp
  // (16 threads). There are 4 total tiles, so 2 warps (64 threads).
  // Lanes 0 and 16 of each warp write the scale values for that warp's tiles.
  // The RoPE part (last 64 elements) is handled by another 1 warp (32 threads).
  // So in total, we use 3 warps (96 threads) per block.

  // Cast kv_cache to 16_bit for RoPE values
  scalar_t* kv_cache_16bit =
      reinterpret_cast<scalar_t*>(&kv_cache[dst_idx_start]);

  // The last warp handles the RoPE part
  if (threadIdx.x >= 64) {
    // Each thread handles two elements of RoPE
    const int8_t pe_idx_start = (threadIdx.x - 64) * 2;
    const int64_t src_idx = token_idx * k_pe_stride + pe_idx_start;
    // Vectorized load of two 16-bit values, performed as one 32-bit load
    const int32_t vals = *reinterpret_cast<const int32_t*>(&k_pe[src_idx]);
    // RoPE values start after the packed 8-bit NoPE values and the
    // 32-bit scales
    const int64_t dst_idx = kv_lora_rank / 2 + 8 + pe_idx_start;
    // Vectorized store of two 16-bit values, performed as one 32-bit store
    *reinterpret_cast<int32_t*>(&kv_cache_16bit[dst_idx]) = vals;
    return;
  }

  // The first two warps handle the NoPE part
  const int8_t warp_idx = threadIdx.x >> 5;
  const int8_t lane_idx = threadIdx.x & 31;
  const int8_t tile_idx = warp_idx * 2 + (lane_idx >> 4);

  // Each thread handles 8 elements of NoPE
  // Load the NoPE elements for this thread into registers
  const int64_t src_idx_start = token_idx * kv_c_stride + (threadIdx.x * 8);
  // Vectorized load of eight 16-bit values, performed as an int4 load
  const int4 vals_i4 = *reinterpret_cast<const int4*>(&kv_c[src_idx_start]);
  const scalar_t* vals = reinterpret_cast<const scalar_t*>(&vals_i4);

  // Max absolute value of this thread's elements
  float max_abs = fmaxf(fmaxf(fmaxf(fabsf(vals[0]), fabsf(vals[1])),
                              fmaxf(fabsf(vals[2]), fabsf(vals[3]))),
                        fmaxf(fmaxf(fabsf(vals[4]), fabsf(vals[5])),
                              fmaxf(fabsf(vals[6]), fabsf(vals[7]))));

  // Warp-level reduction to find the max absolute value in each half-warp
#pragma unroll
  for (int offset = 8; offset > 0; offset /= 2) {
    max_abs = fmaxf(max_abs, VLLM_SHFL_XOR_SYNC_WIDTH(max_abs, offset, 16));
  }

  // Compute the scale for the tile
  float tile_scale = max_abs / 448.f;
  tile_scale = fmaxf(tile_scale, FLT_MIN);

  // The first lane of each half-warp writes the scale to kv_cache
  if ((lane_idx == 0) || (lane_idx == 16)) {
    float* kv_cache_32bit = reinterpret_cast<float*>(&kv_cache[dst_idx_start]);
    const uint64_t dst_idx = kv_lora_rank / 4 + tile_idx;
    kv_cache_32bit[dst_idx] = tile_scale;
  }

  // Now all threads in the block scale and write their elements
  // NoPE data is packed in the first kv_lora_rank/2 bytes (first 256 bytes)
  const int64_t dst_idx_base = dst_idx_start + (threadIdx.x * 8);

  uint8_t result[8];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    result[i] =
        fp8::scaled_convert<uint8_t, scalar_t, Fp8KVCacheDataType::kFp8E4M3>(
            vals[i], tile_scale);
  }

  // Store as aligned 64-bit writes
  *reinterpret_cast<uint64_t*>(&kv_cache[dst_idx_base]) =
      *reinterpret_cast<const uint64_t*>(result);
}

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void indexer_k_quant_and_cache_kernel(
    const scalar_t* __restrict__ k,  // [num_tokens, head_dim]
    cache_t* __restrict__ kv_cache,  // [num_blocks, block_size, cache_stride]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int head_dim,                        // dimension of each head
    const int quant_block_size,                // quantization block size
    const int cache_block_size,                // cache block size
    const int cache_stride,  // stride for each token in kv_cache

    const bool use_ue8m0  // use ue8m0 scale format
) {
  constexpr int VEC_SIZE = 4;
  const int64_t token_idx = blockIdx.x;
  const int64_t head_dim_idx = (blockIdx.y * blockDim.y * blockDim.x +
                                threadIdx.y * blockDim.x + threadIdx.x) *
                               VEC_SIZE;
  const int64_t slot_idx = slot_mapping[token_idx];
  const int64_t block_idx = slot_idx / cache_block_size;
  const int64_t block_offset = slot_idx % cache_block_size;

  // NOTE: slot_idx can be -1 if the token is padded
  if (slot_idx < 0 || (head_dim_idx >= head_dim)) {
    return;
  }

  float2 k_val = (reinterpret_cast<const float2*>(
      k))[(token_idx * head_dim + head_dim_idx) / VEC_SIZE];
  scalar_t* k_val_ptr = reinterpret_cast<scalar_t*>(&k_val);
  float amax = 0.0f;
  for (int i = 0; i < VEC_SIZE; i++) {
    amax = fmaxf(amax, fabsf(float(k_val_ptr[i])));
  }

  // Reduced amax
  for (int mask = 16; mask > 0; mask /= 2) {
#ifdef USE_ROCM
    amax = fmaxf(amax, __shfl_xor_sync(uint64_t(-1), amax, mask));
#else
    amax = fmaxf(amax, __shfl_xor_sync(unsigned(-1), amax, mask));
#endif
  }

#if defined(__gfx942__)
  float scale = fmaxf(amax, 1e-4) / 224.0f;
#else
  float scale = fmaxf(amax, 1e-4) / 448.0f;
#endif
  if (use_ue8m0) {
    scale = exp2f(ceilf(log2f(scale)));
  }

  const int64_t dst_offset = block_idx * cache_block_size * cache_stride +
                             block_offset * head_dim + head_dim_idx;
  for (int i = 0; i < VEC_SIZE; i++) {
    kv_cache[dst_offset + i] =
        fp8::scaled_convert<cache_t, scalar_t, kv_dt>(k_val_ptr[i], scale);
  }
  if (threadIdx.x == 0) {
    const int64_t dst_scale_idx =
        block_idx * cache_block_size * cache_stride +
        cache_block_size * head_dim +
        (block_offset * head_dim + head_dim_idx) * 4 / quant_block_size;
    reinterpret_cast<float*>(kv_cache)[dst_scale_idx / 4] = scale;
  }
}

template <int BLOCK_Y_SIZE>
__global__ void cp_gather_indexer_k_quant_cache_kernel(
    const char* __restrict__ kv_cache,  // [num_blocks, block_size,
                                        // cache_stride]
    char* __restrict__ dst_k,           // [num_tokens, head_dim]
    char* __restrict__ dst_scale,  // [num_tokens, head_dim / quant_block_size *
                                   // 4]
    const int* __restrict__ block_table,  // [batch_size, num_blocks]
    const int* __restrict__ cu_seq_lens,  // [batch_size + 1]
    const int batch_size,                 // batch size
    const int64_t token_stride,           // stride for each token in dst_k
    const int64_t head_dim,               // dimension of each head
    const int64_t block_stride,           // stride for each block in kv_cache
    const int64_t cache_token_stride,     // stride for each token in kv_cache
    const int64_t cache_block_size,  // num_tokens for each block in kv_cache
    const int num_blocks,            // number of blocks
    const int num_tokens,            // number of tokens
    const int quant_block_size       // quantization block size
) {
  constexpr int VEC_SIZE = sizeof(float4) / sizeof(char);
  const int token_idx = blockIdx.x * blockDim.y + threadIdx.y;
  const int head_idx = (blockIdx.y * blockDim.x + threadIdx.x) * VEC_SIZE;
  // Find batch index within a block
  __shared__ int batch_idx[BLOCK_Y_SIZE];
  for (int iter = 0; iter < cuda_utils::ceil_div(batch_size, int(blockDim.x));
       iter++) {
    int tid = iter * blockDim.x + threadIdx.x;
    if (tid < batch_size) {
      const int seq_start = cu_seq_lens[tid];
      const int seq_end = cu_seq_lens[tid + 1];
      if (token_idx >= seq_start && token_idx < seq_end) {
        batch_idx[threadIdx.y] = tid;
      }
    }
  }

#ifndef USE_ROCM
  __syncwarp();
#endif

  if (head_idx >= head_dim || token_idx >= num_tokens) {
    return;
  }
  const int inbatch_seq_idx = token_idx - cu_seq_lens[batch_idx[threadIdx.y]];
  const int block_idx = block_table[batch_idx[threadIdx.y] * num_blocks +
                                    inbatch_seq_idx / cache_block_size];
  const int64_t src_block_offset = block_idx * block_stride;
  const int64_t cache_inblock_offset =
      (inbatch_seq_idx % cache_block_size) * head_dim + head_idx;
  const int64_t src_inblock_offset = src_block_offset + cache_inblock_offset;
  const int64_t dst_inblock_offset = token_idx * token_stride + head_idx;

  reinterpret_cast<float4*>(dst_k)[dst_inblock_offset / VEC_SIZE] =
      reinterpret_cast<const float4*>(kv_cache)[src_inblock_offset / VEC_SIZE];
  ;
  if (threadIdx.x == 0) {
    const int64_t src_scale_offset =
        src_block_offset + cache_block_size * head_dim +
        cache_inblock_offset * 4 / quant_block_size;
    reinterpret_cast<float*>(dst_scale)[dst_inblock_offset / quant_block_size] =
        reinterpret_cast<const float*>(kv_cache)[src_scale_offset / 4];
  }
}

}  // namespace vllm

// KV_T is the data type of key and value tensors.
// CACHE_T is the stored data type of kv-cache.
// KV_DTYPE is the real data type of kv-cache.
#define CALL_RESHAPE_AND_CACHE(KV_T, CACHE_T, KV_DTYPE)               \
  vllm::reshape_and_cache_kernel<KV_T, CACHE_T, KV_DTYPE>             \
      <<<grid, block, 0, stream>>>(                                   \
          reinterpret_cast<KV_T*>(key.data_ptr()),                    \
          reinterpret_cast<KV_T*>(value.data_ptr()),                  \
          reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),           \
          reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),         \
          slot_mapping.data_ptr<int64_t>(), key_stride, value_stride, \
          num_heads, head_size, block_size, x,                        \
          reinterpret_cast<const float*>(k_scale.data_ptr()),         \
          reinterpret_cast<const float*>(v_scale.data_ptr()));

void reshape_and_cache(
    torch::Tensor& key,    // [num_tokens, num_heads, head_size]
    torch::Tensor& value,  // [num_tokens, num_heads, head_size]
    torch::Tensor&
        key_cache,  // [num_blocks, num_heads, head_size/x, block_size, x]
    torch::Tensor&
        value_cache,  // [num_blocks, num_heads, head_size, block_size]
    torch::Tensor& slot_mapping,  // [num_tokens]
    const std::string& kv_cache_dtype, torch::Tensor& k_scale,
    torch::Tensor& v_scale) {
  int num_tokens = slot_mapping.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(3);
  int x = key_cache.size(4);

  int key_stride = key.stride(0);
  int value_stride = value.stride(0);
  int head_div_x = head_size / x;

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_div_x, 512));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_BY_KV_CACHE_DTYPE(key.dtype(), kv_cache_dtype,
                             CALL_RESHAPE_AND_CACHE);
}

// KV_T is the data type of key and value tensors.
// CACHE_T is the stored data type of kv-cache.
// KV_DTYPE is the real data type of kv-cache.
#define CALL_RESHAPE_AND_CACHE_FLASH(KV_T, CACHE_T, KV_DTYPE)             \
  vllm::reshape_and_cache_flash_kernel<KV_T, CACHE_T, KV_DTYPE>           \
      <<<grid, block, 0, stream>>>(                                       \
          reinterpret_cast<KV_T*>(key.data_ptr()),                        \
          reinterpret_cast<KV_T*>(value.data_ptr()),                      \
          reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),               \
          reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),             \
          slot_mapping.data_ptr<int64_t>(), block_stride, page_stride,    \
          head_stride, key_stride, value_stride, num_heads, head_size,    \
          block_size, num_blocks,                                         \
          reinterpret_cast<const float*>(k_scale.data_ptr()),              \
          reinterpret_cast<const float*>(v_scale.data_ptr()));

void reshape_and_cache_flash(
    torch::Tensor& key,        // [num_tokens, num_heads, head_size]
    torch::Tensor& value,      // [num_tokens, num_heads, head_size]
    torch::Tensor& key_cache,  // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor&
        value_cache,  // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor& slot_mapping,  // [num_tokens] or [num_actual_tokens]
    const std::string& kv_cache_dtype, torch::Tensor& k_scale,
    torch::Tensor& v_scale) {
  // NOTE(woosuk): In vLLM V1, key.size(0) can be different from
  // slot_mapping.size(0) because of padding for CUDA graphs.
  // In vLLM V0, key.size(0) is always equal to slot_mapping.size(0) because
  // both include padding.
  // In vLLM V1, however, key.size(0) can be larger than slot_mapping.size(0)
  // since key includes padding for CUDA graphs, while slot_mapping does not.
  // In this case, slot_mapping.size(0) represents the actual number of tokens
  // before padding.
  // For compatibility with both cases, we use slot_mapping.size(0) as the
  // number of tokens.
  int num_tokens = slot_mapping.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(1);
  int num_blocks = key_cache.size(0);

  // FP4 uses packed layout: cache last dim is head_size/2; validate to avoid
  // out-of-bounds writes and GPU memory access faults (e.g. "Memory access
  // fault by GPU" when aiter or other consumers read the cache).
  if (kv_cache_dtype == "fp4" || kv_cache_dtype == "fp4_e2m1") {
    TORCH_CHECK(key_cache.dim() == 4,
                "FP4 KV cache key_cache must be 4D [num_blocks, block_size, "
                "num_heads, head_size/2]");
    TORCH_CHECK(value_cache.dim() == 4 && value_cache.sizes() == key_cache.sizes(),
                "FP4 KV cache value_cache must have same shape as key_cache");
    TORCH_CHECK(head_size % 2 == 0,
                "FP4 KV cache requires even head_size, got ", head_size);
    int cache_head_dim = key_cache.size(3);
    TORCH_CHECK(cache_head_dim == head_size / 2,
                "FP4 KV cache last dim must be head_size/2: expected ",
                head_size / 2, ", got ", cache_head_dim);
  }

  int64_t key_stride = key.stride(0);
  int64_t value_stride = value.stride(0);
  int64_t block_stride = key_cache.stride(0);
  int64_t page_stride = key_cache.stride(1);
  int64_t head_stride = key_cache.stride(2);
  TORCH_CHECK(key_cache.stride(0) == value_cache.stride(0));

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size, 512));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_BY_KV_CACHE_DTYPE(key.dtype(), kv_cache_dtype,
                             CALL_RESHAPE_AND_CACHE_FLASH);
}

#ifdef USE_ROCM
namespace vllm {

// ---------------------------------------------------------------------------
// FP4 outlier-clipping tunable.
//
// When FP4_OUTLIER_CLIP_SIGMA > 0 the quantisation scale is computed as
//   effective_max = min(absmax, rms * FP4_OUTLIER_CLIP_SIGMA)
//   scale         = effective_max / 6.0
// instead of the default  scale = absmax / 6.0.
//
// This trades pinpoint accuracy for the very few outlier elements
// (those beyond CLIP_SIGMA standard-deviations) in exchange for tighter
// quantisation steps — and therefore lower MSE — for the vast majority
// of values.  Typical recommended range: 4.0 – 6.0.
//
// Set to 0 (the default) to disable clipping and keep the original
// absmax behaviour.
//
// NOTE: FP4_OUTLIER_CLIP_SIGMA must be an integer constant (not 0.0f)
// because it is used in preprocessor #if expressions, which only
// support integer arithmetic.  The float sigma value is derived from
// this integer at runtime via static_cast<float>.
// ---------------------------------------------------------------------------
#ifndef FP4_OUTLIER_CLIP_SIGMA
#define FP4_OUTLIER_CLIP_SIGMA 0
#endif

// ---------------------------------------------------------------------------
// FP4 MSE-optimal scale refinement.
//
// When FP4_MSE_REFINE_ITERS > 0, after the initial absmax-based
// quantization, the scale is refined using the closed-form MSE-optimal
// formula  s* = sum(x_i * q_i) / sum(q_i^2)   where q_i is the E2M1
// quantization level assigned to x_i.  Each refinement re-quantizes
// with the updated scale and re-computes s*.  Typical: 1 iteration is
// sufficient for near-optimal MSE.
//
// Set to 0 (the default) to disable refinement and keep the absmax scale.
// ---------------------------------------------------------------------------
#ifndef FP4_MSE_REFINE_ITERS
#define FP4_MSE_REFINE_ITERS 0
#endif

// ---------------------------------------------------------------------------
// Per-block-specific accuracy tuning.
//
// These control outlier clipping and MSE refinement for per-block-32 paths
// independently from the per-token flags.  Per-block quantization benefits
// MORE from these optimizations because each block has only 32 elements:
// a single outlier can waste 1/32 of the dynamic range vs 1/128 for
// per-token.  Defaults are enabled for better out-of-the-box accuracy.
//
// FP4_PER_BLOCK_MSE_REFINE_ITERS: MSE-optimal scale refinement iterations
//   applied only in the per-block path (num_quant_blocks > 1).
//   Default: 1  (one iteration is sufficient for near-optimal MSE).
//   Overridden by FP4_MSE_REFINE_ITERS if that is nonzero.
//
// FP4_PER_BLOCK_CLIP_SIGMA: outlier clip sigma for per-block paths.
//   Default: 0 (disabled).  Typical range: 4 – 6.
//   Overridden by FP4_OUTLIER_CLIP_SIGMA if that is nonzero.
// ---------------------------------------------------------------------------
#ifndef FP4_PER_BLOCK_MSE_REFINE_ITERS
#define FP4_PER_BLOCK_MSE_REFINE_ITERS 1
#endif
#ifndef FP4_PER_BLOCK_CLIP_SIGMA
#define FP4_PER_BLOCK_CLIP_SIGMA 0
#endif

// Effective per-block flags: global override takes priority
#if FP4_OUTLIER_CLIP_SIGMA > 0
#define FP4_BLOCK_CLIP_EFF FP4_OUTLIER_CLIP_SIGMA
#elif FP4_PER_BLOCK_CLIP_SIGMA > 0
#define FP4_BLOCK_CLIP_EFF FP4_PER_BLOCK_CLIP_SIGMA
#else
#define FP4_BLOCK_CLIP_EFF 0
#endif

#if FP4_MSE_REFINE_ITERS > 0
#define FP4_BLOCK_MSE_EFF FP4_MSE_REFINE_ITERS
#elif FP4_PER_BLOCK_MSE_REFINE_ITERS > 0
#define FP4_BLOCK_MSE_EFF FP4_PER_BLOCK_MSE_REFINE_ITERS
#else
#define FP4_BLOCK_MSE_EFF 0
#endif

// ---------------------------------------------------------------------------
// Per-channel-K kernel V-path specific compile-time flags.
//
// FP4_PCK_V_MSE_REFINE_ITERS: MSE-optimal scale refinement for V per-token
//   in the per-channel-K kernel.  Default: 1 (enabled).
//
// FP4_PCK_V_CLIP_SIGMA: outlier clipping sigma for V per-token in the
//   per-channel-K kernel.  Default: 0 (disabled).
//
// Global flags (FP4_MSE_REFINE_ITERS / FP4_OUTLIER_CLIP_SIGMA) override
// these if set to a nonzero value.
// ---------------------------------------------------------------------------
#ifndef FP4_PCK_V_MSE_REFINE_ITERS
#define FP4_PCK_V_MSE_REFINE_ITERS 1
#endif
#ifndef FP4_PCK_V_CLIP_SIGMA
#define FP4_PCK_V_CLIP_SIGMA 0
#endif

#if FP4_MSE_REFINE_ITERS > 0
#define FP4_PCK_V_MSE_EFF FP4_MSE_REFINE_ITERS
#elif FP4_PCK_V_MSE_REFINE_ITERS > 0
#define FP4_PCK_V_MSE_EFF FP4_PCK_V_MSE_REFINE_ITERS
#else
#define FP4_PCK_V_MSE_EFF 0
#endif

#if FP4_OUTLIER_CLIP_SIGMA > 0
#define FP4_PCK_V_CLIP_EFF FP4_OUTLIER_CLIP_SIGMA
#elif FP4_PCK_V_CLIP_SIGMA > 0
#define FP4_PCK_V_CLIP_EFF FP4_PCK_V_CLIP_SIGMA
#else
#define FP4_PCK_V_CLIP_EFF 0
#endif

// ---------------------------------------------------------------------------
// In-kernel Walsh-Hadamard Transform with QuIP#-style random sign flipping.
//
// When FP4_USE_IN_KERNEL_WHT is 1, the kernel applies a randomized
// Hadamard rotation to K/V *in float32* before computing absmax and
// quantizing.  This eliminates the BF16 roundtrip that occurs when WHT
// is performed in Python, giving slightly better precision.
//
// The random ±1 signs use Knuth's multiplicative hash on the dimension
// index, matching the Python _get_fp4_hadamard_signs() function exactly.
//
// When this is enabled, the Python-side cache write should NOT apply
// WHT (set VLLM_FP4_IN_KERNEL_WHT=1 alongside VLLM_FP4_HADAMARD=1).
// The Python-side WHT is still used for Q rotation at decode time and
// K/V inverse-rotation in the extend dequant path.
// ---------------------------------------------------------------------------
#ifndef FP4_USE_IN_KERNEL_WHT
#define FP4_USE_IN_KERNEL_WHT 0
#endif

#if FP4_USE_IN_KERNEL_WHT
__inline__ __device__ float fp4_wht_sign(int dim_idx) {
  unsigned int hash = static_cast<unsigned int>(dim_idx) * 2654435761u;
  return (hash & 0x80000000u) ? -1.0f : 1.0f;
}
#endif

// FP4 E2M1 per-token quantization nibble encoder matching aiter's standard
// E2M1 encoding used by its paged attention kernel.  The nibble layout is
// sign(bit3) | magnitude(bits 2:0) mapping to
// {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}.
__inline__ __device__ uint8_t fp4_e2m1_quantize_nibble(float x) {
  uint8_t sign = (x < 0.0f) ? 8 : 0;
  float a = fabsf(x);
  uint8_t mag;
  if      (a < 0.25f) mag = 0;
  else if (a < 0.75f) mag = 1;
  else if (a < 1.25f) mag = 2;
  else if (a < 1.75f) mag = 3;
  else if (a < 2.5f)  mag = 4;
  else if (a < 3.5f)  mag = 5;
  else if (a < 5.0f)  mag = 6;
  else                mag = 7;
  return sign | mag;
}

__constant__ float fp4_e2m1_lut[8] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f
};

__inline__ __device__ float fp4_e2m1_dequant_value(uint8_t nibble) {
  float mag = fp4_e2m1_lut[nibble & 0x7];
  return (nibble & 0x8) ? -mag : mag;
}

// ---------------------------------------------------------------------------
// E8M0 (OCP MX shared-exponent) conversion helpers for MXFP4 block scaling.
//
// E8M0 format: 8-bit unsigned exponent with bias 127.
//   value = 2^(e8m0 - 127)
//   e8m0 = round(log2(value)) + 127
// Range: 2^(-127) to 2^(127).  e8m0=255 is NaN (mapped to 0 for safety).
// ---------------------------------------------------------------------------
__inline__ __device__ uint8_t float_to_e8m0(float x) {
  if (x <= 0.0f) return 0;

  // Pure IEEE 754 bit extraction — avoids log2f/roundf entirely.
  // log2f is broken on some ROCm/HIP builds (returns NaN or wrong
  // values), and -ffast-math can defeat NaN guards.  Bit extraction
  // is always correct regardless of math library bugs.
  unsigned int bits;
  memcpy(&bits, &x, sizeof(bits));
  int ieee_exp = static_cast<int>((bits >> 23) & 0xFFu);
  unsigned int mantissa = bits & 0x7FFFFFu;

  if (ieee_exp == 0) return 1;    // subnormal → smallest E8M0
  if (ieee_exp == 255) return 254; // inf/NaN → largest E8M0

  // E8M0 byte = biased exponent of nearest power-of-2.
  // IEEE 754 bias (127) matches E8M0 bias, so ieee_exp is the
  // E8M0 byte for the lower bound.  Round to nearest: if the
  // mantissa fraction >= sqrt(2)-1 ≈ 0.4142 (i.e. the value is
  // closer to the next power-of-2), round up.
  // sqrt(2)-1 in 23-bit mantissa = 0x3504F3.
  int e8m0 = ieee_exp;
  if (mantissa >= 0x3504F3u) e8m0++;

  if (e8m0 < 1) e8m0 = 1;
  if (e8m0 > 254) e8m0 = 254;
  return static_cast<uint8_t>(e8m0);
}

__inline__ __device__ float e8m0_to_float(uint8_t e8m0) {
  if (e8m0 == 0 || e8m0 == 255) return 0.0f;
  return exp2f(static_cast<float>(e8m0) - 127.0f);
}

// ---------------------------------------------------------------------------
// FP8 E4M3 (FNUZ, AMD-style) conversion helpers for NVFP4 block scaling.
//
// E4M3 FNUZ format: 1 sign + 4 exponent + 3 mantissa bits, bias = 8.
//   Normal:     value = (-1)^s * 2^(E-8) * (1 + M/8)   for E > 0
//   Subnormal:  value = (-1)^s * 2^(-7)  * (M/8)        for E = 0
//   Zero:       0x00 (positive zero), 0x80 (negative zero / NaN in FNUZ)
//   Max value:  240.0  (E=15, M=7 → 2^7 * 1.875 = 240)
//
// For quantization scales we only need positive values.
// Compared to E8M0, E4M3 has 3 mantissa bits of precision within each
// binade, reducing round-trip quantization error by ~5% on typical LLM
// benchmarks (per NVIDIA NVFP4 findings).
// ---------------------------------------------------------------------------
__inline__ __device__ float fp8_e4m3_to_float(uint8_t x);  // fwd decl

__inline__ __device__ uint8_t float_to_fp8_e4m3(float x) {
  if (x <= 0.0f) return 0;
  constexpr float FP8_E4M3_MAX = 240.0f;
  x = fminf(x, FP8_E4M3_MAX);

  // Pure IEEE 754 bit extraction — avoids log2f/floorf/exp2f entirely.
  // These math functions are broken on some ROCm/HIP builds (return
  // NaN), and -ffast-math defeats NaN guards like !(x==x).
  unsigned int bits;
  memcpy(&bits, &x, sizeof(bits));
  int ieee_exp = static_cast<int>((bits >> 23) & 0xFFu);
  unsigned int ieee_man = bits & 0x7FFFFFu;

  // FP8 E4M3 FNUZ: bias=8.  biased_exp = ieee_exp - 127 + 8 = ieee_exp - 119
  int biased_exp = ieee_exp - 119;

  if (ieee_exp == 0) {
    // IEEE subnormal → FP8 subnormal (very small value)
    return 1;
  }

  if (biased_exp <= 0) {
    // Subnormal in FP8: value = 2^(-7) * (mantissa/8)
    // x = 2^(ieee_exp - 127) * (1 + ieee_man/2^23)
    // mantissa = round(x * 128 * 8) = round(x * 1024)
    // Use bit shifts: x * 1024 = 2^(ieee_exp-127+10) * (1 + ieee_man/2^23)
    // = 2^(ieee_exp-117) * (1 + ieee_man/2^23)
    int shift = 117 - ieee_exp;  // >= 1 since biased_exp <= 0 → ieee_exp <= 119
    unsigned int full = (0x800000u | ieee_man);  // 1.mantissa in Q23
    unsigned int val;
    if (shift < 23) {
      val = full >> shift;
      unsigned int half = 1u << (shift - 1);
      if ((full & ((half << 1) - 1)) > half) val++;  // round up
      else if ((full & ((half << 1) - 1)) == half && (val & 1)) val++;
    } else {
      val = 0;
    }
    int mantissa = static_cast<int>(val);
    if (mantissa > 7) mantissa = 7;
    if (mantissa < 1) mantissa = 1;
    return static_cast<uint8_t>(mantissa);
  }
  if (biased_exp >= 16) {
    return 0x7F;
  }

  // Normal FP8: extract 3-bit mantissa from 23-bit IEEE mantissa
  // Shift right by (23-3)=20, then round.
  int mantissa = static_cast<int>(ieee_man >> 20);
  unsigned int remainder = ieee_man & 0xFFFFFu;
  if (remainder > 0x80000u) {
    mantissa++;
  } else if (remainder == 0x80000u && (mantissa & 1)) {
    mantissa++;  // round to even
  }
  if (mantissa >= 8) {
    mantissa = 0;
    biased_exp++;
    if (biased_exp >= 16) return 0x7F;
  }
  uint8_t result = static_cast<uint8_t>((biased_exp << 3) | (mantissa & 0x7));
  return (result == 0 && x > 0.0f) ? static_cast<uint8_t>(1) : result;
}

__inline__ __device__ float fp8_e4m3_to_float(uint8_t x) {
  if (x == 0) return 0.0f;
  int exp_bits = (x >> 3) & 0xF;
  int mantissa = x & 0x7;
  if (exp_bits == 0) {
    return exp2f(-7.0f) * (static_cast<float>(mantissa) / 8.0f);
  }
  return exp2f(static_cast<float>(exp_bits - 8))
         * (1.0f + static_cast<float>(mantissa) / 8.0f);
}

// ---------------------------------------------------------------------------
// AMXFP4 (Asymmetric Microscaling FP4) helpers.
//
// AMXFP4 repurposes the Block Maximum (BM) element's 2 exponent bits as
// extra mantissa bits (E0M3 encoding: 1 sign + 3 mantissa), giving 8
// representable magnitudes {4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5} instead of
// just {4.0,6.0} in standard E2M1.  Non-BM elements use standard E2M1.
// A 1-byte metadata per block stores the BM index (0-31).
// Reference: arXiv:2411.09909 "AMXFP4: Taming Activation Outliers with
// Asymmetric Microscaling Floating-Point for 4-bit LLM Inference"
// ---------------------------------------------------------------------------
__constant__ float amxfp4_bm_lut[8] = {
    4.0f, 4.5f, 5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.5f
};

__inline__ __device__ uint8_t amxfp4_bm_quantize_nibble(float x) {
  uint8_t sign = (x < 0.0f) ? 8 : 0;
  float a = fabsf(x);
  uint8_t mantissa;
  if      (a < 4.25f)  mantissa = 0;
  else if (a < 4.75f)  mantissa = 1;
  else if (a < 5.25f)  mantissa = 2;
  else if (a < 5.75f)  mantissa = 3;
  else if (a < 6.25f)  mantissa = 4;
  else if (a < 6.75f)  mantissa = 5;
  else if (a < 7.25f)  mantissa = 6;
  else                 mantissa = 7;
  return sign | mantissa;
}

__inline__ __device__ float amxfp4_bm_dequant_value(uint8_t nibble) {
  float mag = amxfp4_bm_lut[nibble & 0x7];
  return (nibble & 0x8) ? -mag : mag;
}

// Per-token / per-block FP4 E2M1 quantization kernel for KV cache (NHD layout).
//
// Grid : (num_tokens, num_heads)  – one block per (token, head) pair.
// Block: 64 threads (one wavefront on MI300X).
//
// Both K and V support per-block quantization: each group of
// FP4_QUANT_BLOCK_SIZE (32) head elements gets its own scale.
// When num_k_quant_blocks == 1 or num_v_quant_blocks == 1 the kernel
// falls back to per-token scaling for the respective tensor.
//
// Both per-block and per-token paths support the optional accuracy
// optimizations controlled by the compile-time flags:
//   - FP4_OUTLIER_CLIP_SIGMA: clamp scale via per-block/per-token RMS
//   - FP4_MSE_REFINE_ITERS:  MSE-optimal scale refinement iterations
// Per-block paths use partial warp reductions (32 lanes) to keep each
// block's statistics independent.
template <typename scalar_t>
__global__ void reshape_and_cache_flash_fp4_pertoken_quant_kernel(
    const scalar_t* __restrict__ key,         // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,       // [num_tokens, num_heads, head_size]
    uint8_t* __restrict__ key_cache,          // [num_blocks, block_size, num_heads, head_size/2]
    uint8_t* __restrict__ value_cache,        // [num_blocks, block_size, num_heads, head_size/2]
    float* __restrict__ k_dequant_scales,     // [num_heads * num_k_quant_blocks, max_kv_tokens]
    float* __restrict__ v_dequant_scales,     // [num_heads * num_v_quant_blocks, max_kv_tokens]
    const int64_t* __restrict__ slot_mapping, // [num_tokens]
    const int64_t key_stride,
    const int64_t value_stride,
    const int head_size,
    const int block_size,
    const int num_blocks,
    const int64_t block_stride,
    const int64_t page_stride,
    const int64_t head_stride,
    const int64_t k_scale_stride_h,
    const int64_t v_scale_stride_h,
    const int num_k_quant_blocks,
    const int num_v_quant_blocks) {

  constexpr float FP4_MAX = 6.0f;
  constexpr int LOCAL_DIM_ELEMS = 8;  // supports head_size up to 64*8 = 512

  const int64_t token_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int lane_id = threadIdx.x;

  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) return;

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  if (block_idx >= num_blocks || block_offset >= block_size) return;

  const int half_head = head_size / 2;

  const scalar_t* k_src =
      key + token_idx * key_stride + head_idx * head_size;
  const scalar_t* v_src =
      value + token_idx * value_stride + head_idx * head_size;

  // --- Phase A: load K/V elements into registers ----------------------------
  float k_local[LOCAL_DIM_ELEMS];
  float v_local[LOCAL_DIM_ELEMS];

#pragma unroll
  for (int i = 0; i < LOCAL_DIM_ELEMS; i++) {
    int d = lane_id + i * warpSize;
    if (d < head_size) {
      k_local[i] = static_cast<float>(k_src[d]);
      v_local[i] = static_cast<float>(v_src[d]);
    } else {
      k_local[i] = 0.0f;
      v_local[i] = 0.0f;
    }
  }

#if FP4_USE_IN_KERNEL_WHT
  // --- Phase B: in-kernel randomized Hadamard rotation ----------------------
  // Apply QuIP#-style random signs (D matrix)
#pragma unroll
  for (int i = 0; i < LOCAL_DIM_ELEMS; i++) {
    int d = lane_id + i * warpSize;
    if (d < head_size) {
      float sign = fp4_wht_sign(d);
      k_local[i] *= sign;
      v_local[i] *= sign;
    }
  }

  // Fast Walsh-Hadamard Transform via warp shuffles
  // Inter-lane butterflies (stages for h = 1, 2, 4, ..., warpSize/2)
  for (int h = 1; h < warpSize && h < head_size; h *= 2) {
    bool is_lower = (lane_id & h) == 0;
#pragma unroll
    for (int j = 0; j < LOCAL_DIM_ELEMS; j++) {
      if (lane_id + j * warpSize < head_size) {
        float kp = __shfl_xor_sync(uint64_t(-1), k_local[j], h);
        float vp = __shfl_xor_sync(uint64_t(-1), v_local[j], h);
        k_local[j] = is_lower ? (k_local[j] + kp) : (kp - k_local[j]);
        v_local[j] = is_lower ? (v_local[j] + vp) : (vp - v_local[j]);
      }
    }
  }

  // Intra-thread butterflies (stages for h = warpSize, 2·warpSize, ...)
  for (int s = 1; s * warpSize < head_size; s *= 2) {
#pragma unroll
    for (int j = 0; j < LOCAL_DIM_ELEMS; j++) {
      int pj = j ^ s;
      if (pj > j && pj < LOCAL_DIM_ELEMS
          && lane_id + j  * warpSize < head_size
          && lane_id + pj * warpSize < head_size) {
        float ka = k_local[j] + k_local[pj];
        float kb = k_local[j] - k_local[pj];
        k_local[j]  = ka;
        k_local[pj] = kb;
        float va = v_local[j] + v_local[pj];
        float vb = v_local[j] - v_local[pj];
        v_local[j]  = va;
        v_local[pj] = vb;
      }
    }
  }

  // Normalize
  float wht_norm = rsqrtf(static_cast<float>(head_size));
#pragma unroll
  for (int j = 0; j < LOCAL_DIM_ELEMS; j++) {
    if (lane_id + j * warpSize < head_size) {
      k_local[j] *= wht_norm;
      v_local[j] *= wht_norm;
    }
  }
#endif  // FP4_USE_IN_KERNEL_WHT

  // --- Phase C: compute absmax on (possibly WHT'd) data ---------------------
  // V uses per-token absmax (full warp reduction).
  // K uses per-block-32 absmax when num_k_quant_blocks > 1: a partial warp
  // reduction of FP4_QUANT_BLOCK_SIZE elements, giving one scale per block.
  constexpr int FP4_QUANT_BLOCK_SIZE = 32;
  constexpr int MAX_K_BLOCKS = LOCAL_DIM_ELEMS * 2;  // max blocks we can track

  // --- V: per-block-32 or per-token absmax -----------------------------------
  constexpr int MAX_V_BLOCKS = LOCAL_DIM_ELEMS * 2;
  float v_block_scale[MAX_V_BLOCKS];
  float v_block_scale_inv[MAX_V_BLOCKS];

  if (num_v_quant_blocks > 1) {
    // Per-block-32 V scales: absmax + outlier clipping + MSE refinement
#pragma unroll
    for (int i = 0; i < LOCAL_DIM_ELEMS; i++) {
      int d = lane_id + i * warpSize;
      if (d < head_size) {
        float v_abs = fabsf(v_local[i]);
        for (int off = FP4_QUANT_BLOCK_SIZE / 2; off > 0; off /= 2)
          v_abs = fmaxf(v_abs,
                        __shfl_xor_sync(uint64_t(-1), v_abs, off));
#if FP4_BLOCK_CLIP_EFF > 0
        float v_sq = v_local[i] * v_local[i];
        for (int off = FP4_QUANT_BLOCK_SIZE / 2; off > 0; off /= 2)
          v_sq += __shfl_xor_sync(uint64_t(-1), v_sq, off);
        float sq_lo = __shfl_sync(uint64_t(-1), v_sq, 0);
        float sq_hi = __shfl_sync(uint64_t(-1), v_sq, FP4_QUANT_BLOCK_SIZE);
#endif
        float blk_lo = __shfl_sync(uint64_t(-1), v_abs, 0);
        float blk_hi = __shfl_sync(uint64_t(-1), v_abs, FP4_QUANT_BLOCK_SIZE);
        int b0 = (i * warpSize) / FP4_QUANT_BLOCK_SIZE;
        int b1 = b0 + 1;
        if (b0 < num_v_quant_blocks) {
          float eff_max = blk_lo;
#if FP4_BLOCK_CLIP_EFF > 0
          float rms = sqrtf(sq_lo / static_cast<float>(FP4_QUANT_BLOCK_SIZE));
          eff_max = fminf(blk_lo, rms * static_cast<float>(FP4_BLOCK_CLIP_EFF));
#endif
          v_block_scale[b0] = fmaxf(eff_max / FP4_MAX, 1e-3f);
          v_block_scale_inv[b0] = 1.0f / v_block_scale[b0];
        }
        if (b1 < num_v_quant_blocks) {
          float eff_max = blk_hi;
#if FP4_BLOCK_CLIP_EFF > 0
          float rms = sqrtf(sq_hi / static_cast<float>(FP4_QUANT_BLOCK_SIZE));
          eff_max = fminf(blk_hi, rms * static_cast<float>(FP4_BLOCK_CLIP_EFF));
#endif
          v_block_scale[b1] = fmaxf(eff_max / FP4_MAX, 1e-3f);
          v_block_scale_inv[b1] = 1.0f / v_block_scale[b1];
        }
      }
    }
#if FP4_BLOCK_MSE_EFF > 0
    for (int refine = 0; refine < FP4_BLOCK_MSE_EFF; refine++) {
#pragma unroll
      for (int i = 0; i < LOCAL_DIM_ELEMS; i++) {
        int d = lane_id + i * warpSize;
        if (d < head_size) {
          int blk = d / FP4_QUANT_BLOCK_SIZE;
          float s = v_block_scale[blk];
          float vx = v_local[i];
          float vq = fp4_e2m1_dequant_value(
              fp4_e2m1_quantize_nibble(vx / s));
          float xq_val = vx * vq;
          float qq_val = vq * vq;
          for (int off = FP4_QUANT_BLOCK_SIZE / 2; off > 0; off /= 2) {
            xq_val += __shfl_xor_sync(uint64_t(-1), xq_val, off);
            qq_val += __shfl_xor_sync(uint64_t(-1), qq_val, off);
          }
          float xq_lo = __shfl_sync(uint64_t(-1), xq_val, 0);
          float qq_lo = __shfl_sync(uint64_t(-1), qq_val, 0);
          float xq_hi = __shfl_sync(uint64_t(-1), xq_val, FP4_QUANT_BLOCK_SIZE);
          float qq_hi = __shfl_sync(uint64_t(-1), qq_val, FP4_QUANT_BLOCK_SIZE);
          int b0 = (i * warpSize) / FP4_QUANT_BLOCK_SIZE;
          int b1 = b0 + 1;
          if (b0 < num_v_quant_blocks && qq_lo > 0.0f) {
            v_block_scale[b0] = fmaxf(xq_lo / qq_lo, 1e-3f);
            v_block_scale_inv[b0] = 1.0f / v_block_scale[b0];
          }
          if (b1 < num_v_quant_blocks && qq_hi > 0.0f) {
            v_block_scale[b1] = fmaxf(xq_hi / qq_hi, 1e-3f);
            v_block_scale_inv[b1] = 1.0f / v_block_scale[b1];
          }
        }
      }
    }
#endif
  } else {
    // Per-token V scale (backward compatible)
    float v_local_max = 0.0f;
    float v_local_sum_sq = 0.0f;
#pragma unroll
    for (int i = 0; i < LOCAL_DIM_ELEMS; i++) {
      if (lane_id + i * warpSize < head_size) {
        v_local_max = fmaxf(v_local_max, fabsf(v_local[i]));
        v_local_sum_sq += v_local[i] * v_local[i];
      }
    }
    float v_max = v_local_max;
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
      v_max = fmaxf(v_max, __shfl_xor_sync(uint64_t(-1), v_max, offset));
    float v_effective_max = v_max;
#if FP4_OUTLIER_CLIP_SIGMA > 0
    {
      float v_sum_sq = v_local_sum_sq;
#pragma unroll
      for (int offset = warpSize / 2; offset > 0; offset /= 2)
        v_sum_sq += __shfl_xor_sync(uint64_t(-1), v_sum_sq, offset);
      float v_rms = sqrtf(v_sum_sq / static_cast<float>(head_size));
      float clip_sigma = static_cast<float>(FP4_OUTLIER_CLIP_SIGMA);
      v_effective_max = fminf(v_max, v_rms * clip_sigma);
    }
#endif
    float v_scale = fmaxf(v_effective_max / FP4_MAX, 1e-3f);
#if FP4_MSE_REFINE_ITERS > 0
    for (int refine = 0; refine < FP4_MSE_REFINE_ITERS; refine++) {
      float v_xq = 0.0f, v_qq = 0.0f;
#pragma unroll
      for (int i = 0; i < LOCAL_DIM_ELEMS; i++) {
        int d = lane_id + i * warpSize;
        if (d < head_size) {
          float vx = v_local[i];
          float vq = fp4_e2m1_dequant_value(
              fp4_e2m1_quantize_nibble(vx / v_scale));
          v_xq += vx * vq;
          v_qq += vq * vq;
        }
      }
#pragma unroll
      for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        v_xq += __shfl_xor_sync(uint64_t(-1), v_xq, offset);
        v_qq += __shfl_xor_sync(uint64_t(-1), v_qq, offset);
      }
      if (v_qq > 0.0f) v_scale = fmaxf(v_xq / v_qq, 1e-3f);
    }
#endif
    v_block_scale[0] = v_scale;
    v_block_scale_inv[0] = 1.0f / v_scale;
  }

  // --- K: per-block-32 absmax ------------------------------------------------
  // Each k_local[i] covers warpSize elements.  With FP4_QUANT_BLOCK_SIZE=32
  // and warpSize=64, each k_local[i] spans exactly 2 blocks.  A partial warp
  // reduction with max offset = FP4_QUANT_BLOCK_SIZE/2 - 1 keeps each 32-lane
  // half independent.
  float k_block_scale[MAX_K_BLOCKS];
  float k_block_scale_inv[MAX_K_BLOCKS];

  if (num_k_quant_blocks > 1) {
    // Per-block-32 K scales: absmax + outlier clipping + MSE refinement
#pragma unroll
    for (int i = 0; i < LOCAL_DIM_ELEMS; i++) {
      int d = lane_id + i * warpSize;
      if (d < head_size) {
        float k_abs = fabsf(k_local[i]);
        for (int off = FP4_QUANT_BLOCK_SIZE / 2; off > 0; off /= 2)
          k_abs = fmaxf(k_abs,
                        __shfl_xor_sync(uint64_t(-1), k_abs, off));
#if FP4_BLOCK_CLIP_EFF > 0
        float k_sq = k_local[i] * k_local[i];
        for (int off = FP4_QUANT_BLOCK_SIZE / 2; off > 0; off /= 2)
          k_sq += __shfl_xor_sync(uint64_t(-1), k_sq, off);
        float k_sq_lo = __shfl_sync(uint64_t(-1), k_sq, 0);
        float k_sq_hi = __shfl_sync(uint64_t(-1), k_sq, FP4_QUANT_BLOCK_SIZE);
#endif
        float blk_lo = __shfl_sync(uint64_t(-1), k_abs, 0);
        float blk_hi = __shfl_sync(uint64_t(-1), k_abs, FP4_QUANT_BLOCK_SIZE);
        int b0 = (i * warpSize) / FP4_QUANT_BLOCK_SIZE;
        int b1 = b0 + 1;
        if (b0 < num_k_quant_blocks) {
          float eff_max = blk_lo;
#if FP4_BLOCK_CLIP_EFF > 0
          float rms = sqrtf(k_sq_lo / static_cast<float>(FP4_QUANT_BLOCK_SIZE));
          eff_max = fminf(blk_lo, rms * static_cast<float>(FP4_BLOCK_CLIP_EFF));
#endif
          // 1e-3f ensures the scale round-trips through FP8 E4M3 FNUZ as
          // byte >= 1 (boundary ≈ 4.88e-4).  1e-12f would encode as byte 0,
          // making the PA kernel dequantize all K values in this block as 0.
          k_block_scale[b0] = fmaxf(eff_max / FP4_MAX, 1e-3f);
          k_block_scale_inv[b0] = 1.0f / k_block_scale[b0];
        }
        if (b1 < num_k_quant_blocks) {
          float eff_max = blk_hi;
#if FP4_BLOCK_CLIP_EFF > 0
          float rms = sqrtf(k_sq_hi / static_cast<float>(FP4_QUANT_BLOCK_SIZE));
          eff_max = fminf(blk_hi, rms * static_cast<float>(FP4_BLOCK_CLIP_EFF));
#endif
          k_block_scale[b1] = fmaxf(eff_max / FP4_MAX, 1e-3f);
          k_block_scale_inv[b1] = 1.0f / k_block_scale[b1];
        }
      }
    }
#if FP4_BLOCK_MSE_EFF > 0
    for (int refine = 0; refine < FP4_BLOCK_MSE_EFF; refine++) {
#pragma unroll
      for (int i = 0; i < LOCAL_DIM_ELEMS; i++) {
        int d = lane_id + i * warpSize;
        if (d < head_size) {
          int blk = d / FP4_QUANT_BLOCK_SIZE;
          float s = k_block_scale[blk];
          float kx = k_local[i];
          float kq = fp4_e2m1_dequant_value(
              fp4_e2m1_quantize_nibble(kx / s));
          float xq_val = kx * kq;
          float qq_val = kq * kq;
          for (int off = FP4_QUANT_BLOCK_SIZE / 2; off > 0; off /= 2) {
            xq_val += __shfl_xor_sync(uint64_t(-1), xq_val, off);
            qq_val += __shfl_xor_sync(uint64_t(-1), qq_val, off);
          }
          float xq_lo = __shfl_sync(uint64_t(-1), xq_val, 0);
          float qq_lo = __shfl_sync(uint64_t(-1), qq_val, 0);
          float xq_hi = __shfl_sync(uint64_t(-1), xq_val, FP4_QUANT_BLOCK_SIZE);
          float qq_hi = __shfl_sync(uint64_t(-1), qq_val, FP4_QUANT_BLOCK_SIZE);
          int b0 = (i * warpSize) / FP4_QUANT_BLOCK_SIZE;
          int b1 = b0 + 1;
          if (b0 < num_k_quant_blocks && qq_lo > 0.0f) {
            k_block_scale[b0] = fmaxf(xq_lo / qq_lo, 1e-3f);
            k_block_scale_inv[b0] = 1.0f / k_block_scale[b0];
          }
          if (b1 < num_k_quant_blocks && qq_hi > 0.0f) {
            k_block_scale[b1] = fmaxf(xq_hi / qq_hi, 1e-3f);
            k_block_scale_inv[b1] = 1.0f / k_block_scale[b1];
          }
        }
      }
    }
#endif
  } else {
    // Per-token K scale (backward compatible: num_k_quant_blocks == 1)
    float k_local_max = 0.0f;
#pragma unroll
    for (int i = 0; i < LOCAL_DIM_ELEMS; i++) {
      if (lane_id + i * warpSize < head_size)
        k_local_max = fmaxf(k_local_max, fabsf(k_local[i]));
    }
    float k_max = k_local_max;
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
      k_max = fmaxf(k_max,
                    __shfl_xor_sync(uint64_t(-1), k_max, offset));
    k_block_scale[0] = fmaxf(k_max / FP4_MAX, 1e-3f);
    k_block_scale_inv[0] = 1.0f / k_block_scale[0];
  }

  // --- Store scales ----------------------------------------------------------
  const int64_t k_blk_stride =
      (num_k_quant_blocks > 1) ? (k_scale_stride_h / num_k_quant_blocks) : 0;
  const int64_t v_blk_stride =
      (num_v_quant_blocks > 1) ? (v_scale_stride_h / num_v_quant_blocks) : 0;
  if (lane_id == 0) {
    for (int b = 0; b < num_k_quant_blocks; b++) {
      k_dequant_scales[head_idx * k_scale_stride_h
                       + b * k_blk_stride + slot_idx] = k_block_scale[b];
    }
    for (int b = 0; b < num_v_quant_blocks; b++) {
      v_dequant_scales[head_idx * v_scale_stride_h
                       + b * v_blk_stride + slot_idx] = v_block_scale[b];
    }
  }

  // --- Pass 2: quantize to FP4 E2M1, pack pairs, write to cache -------------
  // Scatter float32 values (possibly WHT'd) to shared memory so each lane
  // can read adjacent element-pairs for nibble packing.  This eliminates a
  // second global-memory read and BF16→float conversion that the non-WHT
  // path previously required.  The kernel is single-wavefront (64 threads)
  // so no explicit barrier is needed between the smem write and read.
  __shared__ float k_smem[512];
  __shared__ float v_smem[512];
#pragma unroll
  for (int i = 0; i < LOCAL_DIM_ELEMS; i++) {
    int d = lane_id + i * warpSize;
    if (d < head_size) {
      k_smem[d] = k_local[i];
      v_smem[d] = v_local[i];
    }
  }

  uint8_t* k_dst = key_cache + block_idx * block_stride
                    + block_offset * page_stride
                    + head_idx * head_stride;
  uint8_t* v_dst = value_cache + block_idx * block_stride
                    + block_offset * page_stride
                    + head_idx * head_stride;

  for (int j = lane_id; j < half_head; j += warpSize) {
    int d0 = 2 * j;
    int d1 = d0 + 1;

    int kb = (num_k_quant_blocks > 1) ? (d0 / FP4_QUANT_BLOCK_SIZE) : 0;
    float k_sinv = k_block_scale_inv[kb];
    float kf0 = k_smem[d0] * k_sinv;
    float kf1 = k_smem[d1] * k_sinv;
    uint8_t kn0 = fp4_e2m1_quantize_nibble(kf0);
    uint8_t kn1 = fp4_e2m1_quantize_nibble(kf1);
    k_dst[j] = (kn0 & 0xF) | ((kn1 & 0xF) << 4);

    int vb = (num_v_quant_blocks > 1) ? (d0 / FP4_QUANT_BLOCK_SIZE) : 0;
    float v_sinv = v_block_scale_inv[vb];
    float vf0 = v_smem[d0] * v_sinv;
    float vf1 = v_smem[d1] * v_sinv;
    uint8_t vn0 = fp4_e2m1_quantize_nibble(vf0);
    uint8_t vn1 = fp4_e2m1_quantize_nibble(vf1);
    v_dst[j] = (vn0 & 0xF) | ((vn1 & 0xF) << 4);
  }
}

// ---------------------------------------------------------------------------
// Per-channel-K / per-token-V FP4 E2M1 quantization kernel.
//
// K quantization uses externally-provided STATIC per-channel scales
// (one scale per head-dimension element, shared across all tokens).
// V quantization computes DYNAMIC per-token scales at write time
// (one scale per token, shared across all head-dimension elements).
//
// During the PA decode, K channel scales are absorbed into Q before
// the QK^T computation: Q_eff[d] = Q[d] * k_ch_scale[d], then
// QK^T = Q_eff * K_fp4_dequant^T.  This eliminates per-token K scale
// lookups entirely.
//
// Grid : (num_tokens, num_heads)  – one block per (token, head).
// Block: 64 threads (one wavefront on MI300X).
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void reshape_and_cache_flash_fp4_perchannel_k_pertoken_v_kernel(
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    uint8_t* __restrict__ key_cache,
    uint8_t* __restrict__ value_cache,
    const float* __restrict__ k_channel_scales,  // [num_heads, head_size]
    float* __restrict__ v_dequant_scales,         // [num_heads, max_kv_tokens]
    const int64_t* __restrict__ slot_mapping,
    const int64_t key_stride,
    const int64_t value_stride,
    const int head_size,
    const int block_size,
    const int num_blocks,
    const int64_t block_stride,
    const int64_t page_stride,
    const int64_t head_stride,
    const int64_t v_scale_stride_h) {

  constexpr float FP4_MAX = 6.0f;
  constexpr int LOCAL_DIM_ELEMS = 8;

  const int64_t token_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int lane_id = threadIdx.x;

  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) return;

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  if (block_idx >= num_blocks || block_offset >= block_size) return;

  const int half_head = head_size / 2;

  const scalar_t* k_src =
      key + token_idx * key_stride + head_idx * head_size;
  const scalar_t* v_src =
      value + token_idx * value_stride + head_idx * head_size;
  const float* k_ch_scale =
      k_channel_scales + head_idx * head_size;

  // --- Load V elements and compute per-token absmax -------------------------
  float v_local[LOCAL_DIM_ELEMS];
  float v_local_max = 0.0f;
  float v_local_sum_sq = 0.0f;

#pragma unroll
  for (int i = 0; i < LOCAL_DIM_ELEMS; i++) {
    int d = lane_id + i * warpSize;
    if (d < head_size) {
      v_local[i] = static_cast<float>(v_src[d]);
      v_local_max = fmaxf(v_local_max, fabsf(v_local[i]));
      v_local_sum_sq += v_local[i] * v_local[i];
    } else {
      v_local[i] = 0.0f;
    }
  }

  float v_max = v_local_max;
#pragma unroll
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    v_max = fmaxf(v_max, __shfl_xor_sync(uint64_t(-1), v_max, offset));

  float v_effective_max = v_max;
#if FP4_PCK_V_CLIP_EFF > 0
  {
    float v_sum_sq = v_local_sum_sq;
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
      v_sum_sq += __shfl_xor_sync(uint64_t(-1), v_sum_sq, offset);
    float v_rms = sqrtf(v_sum_sq / static_cast<float>(head_size));
    float clip_sigma = static_cast<float>(FP4_PCK_V_CLIP_EFF);
    v_effective_max = fminf(v_max, v_rms * clip_sigma);
  }
#endif

  float v_scale = fmaxf(v_effective_max / FP4_MAX, 1e-3f);

#if FP4_PCK_V_MSE_EFF > 0
  for (int refine = 0; refine < FP4_PCK_V_MSE_EFF; refine++) {
    float v_xq = 0.0f, v_qq = 0.0f;
#pragma unroll
    for (int i = 0; i < LOCAL_DIM_ELEMS; i++) {
      int d = lane_id + i * warpSize;
      if (d < head_size) {
        float vx = v_local[i];
        float vq = fp4_e2m1_dequant_value(
            fp4_e2m1_quantize_nibble(vx / v_scale));
        v_xq += vx * vq;
        v_qq += vq * vq;
      }
    }
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      v_xq += __shfl_xor_sync(uint64_t(-1), v_xq, offset);
      v_qq += __shfl_xor_sync(uint64_t(-1), v_qq, offset);
    }
    if (v_qq > 0.0f) v_scale = fmaxf(v_xq / v_qq, 1e-3f);
  }
#endif

  float v_scale_inv = 1.0f / v_scale;

  if (lane_id == 0) {
    v_dequant_scales[head_idx * v_scale_stride_h + slot_idx] = v_scale;
  }

  // --- Quantize and write K (per-channel) and V (per-token) -----------------
  uint8_t* k_dst = key_cache + block_idx * block_stride
                    + block_offset * page_stride
                    + head_idx * head_stride;
  uint8_t* v_dst = value_cache + block_idx * block_stride
                    + block_offset * page_stride
                    + head_idx * head_stride;

  for (int j = lane_id; j < half_head; j += warpSize) {
    int d0 = 2 * j;
    int d1 = d0 + 1;

    float kf0 = static_cast<float>(k_src[d0]) / k_ch_scale[d0];
    float kf1 = static_cast<float>(k_src[d1]) / k_ch_scale[d1];
    uint8_t kn0 = fp4_e2m1_quantize_nibble(kf0);
    uint8_t kn1 = fp4_e2m1_quantize_nibble(kf1);
    k_dst[j] = (kn0 & 0xF) | ((kn1 & 0xF) << 4);

    float vf0 = static_cast<float>(v_src[d0]) * v_scale_inv;
    float vf1 = static_cast<float>(v_src[d1]) * v_scale_inv;
    uint8_t vn0 = fp4_e2m1_quantize_nibble(vf0);
    uint8_t vn1 = fp4_e2m1_quantize_nibble(vf1);
    v_dst[j] = (vn0 & 0xF) | ((vn1 & 0xF) << 4);
  }
}

// ---------------------------------------------------------------------------
// MXFP4 (OCP MX) block-scale FP4 E2M1 quantization kernel for KV cache.
//
// Uses per-block-32 quantization with E8M0 (power-of-2) scales stored as
// uint8.  Each group of 32 head-dimension elements shares one E8M0 scale
// byte.  This matches the OCP Microscaling FP4 specification:
//   - Data: FP4 E2M1 (same as other FP4 kernels)
//   - Scale: E8M0 — 8-bit exponent-only, value = 2^(e8m0 - 127)
//   - Block size: 32
//
// Scale storage is 4× smaller than FP32 per-block scales (1 byte vs 4).
// Power-of-2 scales enable efficient dequantization via exponent adjustment.
//
// Grid : (num_tokens, num_heads)  – one block per (token, head) pair.
// Block: 64 threads (one wavefront on MI300X).
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void reshape_and_cache_flash_fp4_mxfp4_kernel(
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    uint8_t* __restrict__ key_cache,
    uint8_t* __restrict__ value_cache,
    uint8_t* __restrict__ k_e8m0_scales,
    uint8_t* __restrict__ v_e8m0_scales,
    const int64_t* __restrict__ slot_mapping,
    const int64_t key_stride,
    const int64_t value_stride,
    const int head_size,
    const int block_size,
    const int num_blocks,
    const int64_t block_stride,
    const int64_t page_stride,
    const int64_t head_stride,
    const int64_t k_scale_stride_h,
    const int64_t v_scale_stride_h,
    const int num_k_quant_blocks,
    const int num_v_quant_blocks) {

  constexpr float FP4_MAX = 6.0f;
  constexpr int LOCAL_DIM_ELEMS = 8;
  constexpr int FP4_QUANT_BLOCK_SIZE = 32;
  constexpr int MAX_BLOCKS = LOCAL_DIM_ELEMS * 2;

  const int64_t token_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int lane_id = threadIdx.x;

  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) return;

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  if (block_idx >= num_blocks || block_offset >= block_size) return;

  const int half_head = head_size / 2;

  const scalar_t* k_src =
      key + token_idx * key_stride + head_idx * head_size;
  const scalar_t* v_src =
      value + token_idx * value_stride + head_idx * head_size;

  // --- Phase A: load K/V elements into registers ----------------------------
  float k_local[LOCAL_DIM_ELEMS];
  float v_local[LOCAL_DIM_ELEMS];

#pragma unroll
  for (int i = 0; i < LOCAL_DIM_ELEMS; i++) {
    int d = lane_id + i * warpSize;
    if (d < head_size) {
      k_local[i] = static_cast<float>(k_src[d]);
      v_local[i] = static_cast<float>(v_src[d]);
    } else {
      k_local[i] = 0.0f;
      v_local[i] = 0.0f;
    }
  }

#if FP4_USE_IN_KERNEL_WHT
  // WHT rotation (same as per-block kernel)
#pragma unroll
  for (int i = 0; i < LOCAL_DIM_ELEMS; i++) {
    int d = lane_id + i * warpSize;
    if (d < head_size) {
      float sign = fp4_wht_sign(d);
      k_local[i] *= sign;
      v_local[i] *= sign;
    }
  }
  for (int h = 1; h < warpSize && h < head_size; h *= 2) {
    bool is_lower = (lane_id & h) == 0;
#pragma unroll
    for (int j = 0; j < LOCAL_DIM_ELEMS; j++) {
      if (lane_id + j * warpSize < head_size) {
        float kp = __shfl_xor_sync(uint64_t(-1), k_local[j], h);
        float vp = __shfl_xor_sync(uint64_t(-1), v_local[j], h);
        k_local[j] = is_lower ? (k_local[j] + kp) : (kp - k_local[j]);
        v_local[j] = is_lower ? (v_local[j] + vp) : (vp - v_local[j]);
      }
    }
  }
  for (int s = 1; s * warpSize < head_size; s *= 2) {
#pragma unroll
    for (int j = 0; j < LOCAL_DIM_ELEMS; j++) {
      int pj = j ^ s;
      if (pj > j && pj < LOCAL_DIM_ELEMS
          && lane_id + j  * warpSize < head_size
          && lane_id + pj * warpSize < head_size) {
        float ka = k_local[j] + k_local[pj];
        float kb = k_local[j] - k_local[pj];
        k_local[j]  = ka;
        k_local[pj] = kb;
        float va = v_local[j] + v_local[pj];
        float vb = v_local[j] - v_local[pj];
        v_local[j]  = va;
        v_local[pj] = vb;
      }
    }
  }
  float wht_norm = rsqrtf(static_cast<float>(head_size));
#pragma unroll
  for (int j = 0; j < LOCAL_DIM_ELEMS; j++) {
    if (lane_id + j * warpSize < head_size) {
      k_local[j] *= wht_norm;
      v_local[j] *= wht_norm;
    }
  }
#endif  // FP4_USE_IN_KERNEL_WHT

  // --- Phase B: compute per-block-32 E8M0 scales ----------------------------
  // Partial warp reductions keep each 32-lane half independent.
  float k_block_scale_inv[MAX_BLOCKS];
  float v_block_scale_inv[MAX_BLOCKS];
  uint8_t k_e8m0_local[MAX_BLOCKS];
  uint8_t v_e8m0_local[MAX_BLOCKS];

  // K blocks
#pragma unroll
  for (int i = 0; i < LOCAL_DIM_ELEMS; i++) {
    int d = lane_id + i * warpSize;
    if (d < head_size) {
      float k_abs = fabsf(k_local[i]);
      for (int off = FP4_QUANT_BLOCK_SIZE / 2; off > 0; off /= 2)
        k_abs = fmaxf(k_abs,
                      __shfl_xor_sync(uint64_t(-1), k_abs, off));
      float blk_lo = __shfl_sync(uint64_t(-1), k_abs, 0);
      float blk_hi = __shfl_sync(uint64_t(-1), k_abs, FP4_QUANT_BLOCK_SIZE);
      int b0 = (i * warpSize) / FP4_QUANT_BLOCK_SIZE;
      int b1 = b0 + 1;
      if (b0 < num_k_quant_blocks) {
        float ideal = fmaxf(blk_lo / FP4_MAX, 1e-10f);
        k_e8m0_local[b0] = float_to_e8m0(ideal);
        if (k_e8m0_local[b0] == 0) k_e8m0_local[b0] = 1;
        float scale = e8m0_to_float(k_e8m0_local[b0]);
        k_block_scale_inv[b0] = 1.0f / fmaxf(scale, 1e-38f);
      }
      if (b1 < num_k_quant_blocks) {
        float ideal = fmaxf(blk_hi / FP4_MAX, 1e-10f);
        k_e8m0_local[b1] = float_to_e8m0(ideal);
        if (k_e8m0_local[b1] == 0) k_e8m0_local[b1] = 1;
        float scale = e8m0_to_float(k_e8m0_local[b1]);
        k_block_scale_inv[b1] = 1.0f / fmaxf(scale, 1e-38f);
      }
    }
  }

  // V blocks
#pragma unroll
  for (int i = 0; i < LOCAL_DIM_ELEMS; i++) {
    int d = lane_id + i * warpSize;
    if (d < head_size) {
      float v_abs = fabsf(v_local[i]);
      for (int off = FP4_QUANT_BLOCK_SIZE / 2; off > 0; off /= 2)
        v_abs = fmaxf(v_abs,
                      __shfl_xor_sync(uint64_t(-1), v_abs, off));
      float blk_lo = __shfl_sync(uint64_t(-1), v_abs, 0);
      float blk_hi = __shfl_sync(uint64_t(-1), v_abs, FP4_QUANT_BLOCK_SIZE);
      int b0 = (i * warpSize) / FP4_QUANT_BLOCK_SIZE;
      int b1 = b0 + 1;
      if (b0 < num_v_quant_blocks) {
        float ideal = fmaxf(blk_lo / FP4_MAX, 1e-10f);
        v_e8m0_local[b0] = float_to_e8m0(ideal);
        if (v_e8m0_local[b0] == 0) v_e8m0_local[b0] = 1;
        float scale = e8m0_to_float(v_e8m0_local[b0]);
        v_block_scale_inv[b0] = 1.0f / fmaxf(scale, 1e-38f);
      }
      if (b1 < num_v_quant_blocks) {
        float ideal = fmaxf(blk_hi / FP4_MAX, 1e-10f);
        v_e8m0_local[b1] = float_to_e8m0(ideal);
        if (v_e8m0_local[b1] == 0) v_e8m0_local[b1] = 1;
        float scale = e8m0_to_float(v_e8m0_local[b1]);
        v_block_scale_inv[b1] = 1.0f / fmaxf(scale, 1e-38f);
      }
    }
  }

  // --- Store E8M0 scales (uint8) --------------------------------------------
  const int64_t k_blk_stride =
      (num_k_quant_blocks > 1) ? (k_scale_stride_h / num_k_quant_blocks) : 0;
  const int64_t v_blk_stride =
      (num_v_quant_blocks > 1) ? (v_scale_stride_h / num_v_quant_blocks) : 0;
  if (lane_id == 0) {
    for (int b = 0; b < num_k_quant_blocks; b++) {
      k_e8m0_scales[head_idx * k_scale_stride_h
                     + b * k_blk_stride + slot_idx] = k_e8m0_local[b];
    }
    for (int b = 0; b < num_v_quant_blocks; b++) {
      v_e8m0_scales[head_idx * v_scale_stride_h
                     + b * v_blk_stride + slot_idx] = v_e8m0_local[b];
    }
  }

  // --- Phase C: quantize to FP4 E2M1, pack pairs, write to cache -----------
#if FP4_USE_IN_KERNEL_WHT
  __shared__ float k_wht_smem_mx[512];
  __shared__ float v_wht_smem_mx[512];
#pragma unroll
  for (int i = 0; i < LOCAL_DIM_ELEMS; i++) {
    int d = lane_id + i * warpSize;
    if (d < head_size) {
      k_wht_smem_mx[d] = k_local[i];
      v_wht_smem_mx[d] = v_local[i];
    }
  }
#endif

  uint8_t* k_dst = key_cache + block_idx * block_stride
                    + block_offset * page_stride
                    + head_idx * head_stride;
  uint8_t* v_dst = value_cache + block_idx * block_stride
                    + block_offset * page_stride
                    + head_idx * head_stride;

  for (int j = lane_id; j < half_head; j += warpSize) {
    int d0 = 2 * j;
    int d1 = d0 + 1;

    int kb = d0 / FP4_QUANT_BLOCK_SIZE;
    float k_sinv = k_block_scale_inv[kb];

#if FP4_USE_IN_KERNEL_WHT
    float kf0 = k_wht_smem_mx[d0] * k_sinv;
    float kf1 = k_wht_smem_mx[d1] * k_sinv;
#else
    float kf0 = static_cast<float>(k_src[d0]) * k_sinv;
    float kf1 = static_cast<float>(k_src[d1]) * k_sinv;
#endif
    uint8_t kn0 = fp4_e2m1_quantize_nibble(kf0);
    uint8_t kn1 = fp4_e2m1_quantize_nibble(kf1);
    k_dst[j] = (kn0 & 0xF) | ((kn1 & 0xF) << 4);

    int vb = d0 / FP4_QUANT_BLOCK_SIZE;
    float v_sinv = v_block_scale_inv[vb];

#if FP4_USE_IN_KERNEL_WHT
    float vf0 = v_wht_smem_mx[d0] * v_sinv;
    float vf1 = v_wht_smem_mx[d1] * v_sinv;
#else
    float vf0 = static_cast<float>(v_src[d0]) * v_sinv;
    float vf1 = static_cast<float>(v_src[d1]) * v_sinv;
#endif
    uint8_t vn0 = fp4_e2m1_quantize_nibble(vf0);
    uint8_t vn1 = fp4_e2m1_quantize_nibble(vf1);
    v_dst[j] = (vn0 & 0xF) | ((vn1 & 0xF) << 4);
  }
}

// ---------------------------------------------------------------------------
// NVFP4 kernel: per-block-32 quantization with FP8 E4M3 scales (1 byte each).
//
// Identical structure to the MXFP4 kernel but stores scales as FP8 E4M3
// instead of E8M0, providing ~3 mantissa bits of precision per scale
// (vs 0 for E8M0).  Same 1-byte-per-scale storage cost.
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void reshape_and_cache_flash_fp4_nvfp4_kernel(
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    uint8_t* __restrict__ key_cache,
    uint8_t* __restrict__ value_cache,
    uint8_t* __restrict__ k_fp8_scales,
    uint8_t* __restrict__ v_fp8_scales,
    const int64_t* __restrict__ slot_mapping,
    const int64_t key_stride,
    const int64_t value_stride,
    const int head_size,
    const int block_size,
    const int num_blocks,
    const int64_t block_stride,
    const int64_t page_stride,
    const int64_t head_stride,
    const int64_t k_scale_stride_h,
    const int64_t v_scale_stride_h,
    const int num_k_quant_blocks,
    const int num_v_quant_blocks) {

  constexpr float FP4_MAX = 6.0f;
  constexpr int LOCAL_DIM_ELEMS = 8;
  constexpr int FP4_QUANT_BLOCK_SIZE = 32;
  constexpr int MAX_BLOCKS = LOCAL_DIM_ELEMS * 2;

  const int64_t token_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int lane_id = threadIdx.x;

  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) return;

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  if (block_idx >= num_blocks || block_offset >= block_size) return;

  const int half_head = head_size / 2;

  const scalar_t* k_src =
      key + token_idx * key_stride + head_idx * head_size;
  const scalar_t* v_src =
      value + token_idx * value_stride + head_idx * head_size;

  // --- Phase A: load K/V elements into registers ---
  float k_local[LOCAL_DIM_ELEMS];
  float v_local[LOCAL_DIM_ELEMS];

#pragma unroll
  for (int i = 0; i < LOCAL_DIM_ELEMS; i++) {
    int d = lane_id + i * warpSize;
    if (d < head_size) {
      k_local[i] = static_cast<float>(k_src[d]);
      v_local[i] = static_cast<float>(v_src[d]);
    } else {
      k_local[i] = 0.0f;
      v_local[i] = 0.0f;
    }
  }

#if FP4_USE_IN_KERNEL_WHT
#pragma unroll
  for (int i = 0; i < LOCAL_DIM_ELEMS; i++) {
    int d = lane_id + i * warpSize;
    if (d < head_size) {
      float sign = fp4_wht_sign(d);
      k_local[i] *= sign;
      v_local[i] *= sign;
    }
  }
  for (int h = 1; h < warpSize && h < head_size; h *= 2) {
    bool is_lower = (lane_id & h) == 0;
#pragma unroll
    for (int j = 0; j < LOCAL_DIM_ELEMS; j++) {
      if (lane_id + j * warpSize < head_size) {
        float kp = __shfl_xor_sync(uint64_t(-1), k_local[j], h);
        float vp = __shfl_xor_sync(uint64_t(-1), v_local[j], h);
        k_local[j] = is_lower ? (k_local[j] + kp) : (kp - k_local[j]);
        v_local[j] = is_lower ? (v_local[j] + vp) : (vp - v_local[j]);
      }
    }
  }
  for (int s = 1; s * warpSize < head_size; s *= 2) {
#pragma unroll
    for (int j = 0; j < LOCAL_DIM_ELEMS; j++) {
      int pj = j ^ s;
      if (pj > j && pj < LOCAL_DIM_ELEMS
          && lane_id + j  * warpSize < head_size
          && lane_id + pj * warpSize < head_size) {
        float ka = k_local[j] + k_local[pj];
        float kb = k_local[j] - k_local[pj];
        k_local[j]  = ka;
        k_local[pj] = kb;
        float va = v_local[j] + v_local[pj];
        float vb = v_local[j] - v_local[pj];
        v_local[j]  = va;
        v_local[pj] = vb;
      }
    }
  }
  float wht_norm = rsqrtf(static_cast<float>(head_size));
#pragma unroll
  for (int j = 0; j < LOCAL_DIM_ELEMS; j++) {
    if (lane_id + j * warpSize < head_size) {
      k_local[j] *= wht_norm;
      v_local[j] *= wht_norm;
    }
  }
#endif  // FP4_USE_IN_KERNEL_WHT

  // --- Phase B: compute per-block-32 FP8 E4M3 scales ---
  float k_block_scale_inv[MAX_BLOCKS];
  float v_block_scale_inv[MAX_BLOCKS];
  uint8_t k_fp8_local[MAX_BLOCKS];
  uint8_t v_fp8_local[MAX_BLOCKS];

#pragma unroll
  for (int i = 0; i < LOCAL_DIM_ELEMS; i++) {
    int d = lane_id + i * warpSize;
    if (d < head_size) {
      float k_abs = fabsf(k_local[i]);
      for (int off = FP4_QUANT_BLOCK_SIZE / 2; off > 0; off /= 2)
        k_abs = fmaxf(k_abs,
                      __shfl_xor_sync(uint64_t(-1), k_abs, off));
      float blk_lo = __shfl_sync(uint64_t(-1), k_abs, 0);
      float blk_hi = __shfl_sync(uint64_t(-1), k_abs, FP4_QUANT_BLOCK_SIZE);
      int b0 = (i * warpSize) / FP4_QUANT_BLOCK_SIZE;
      int b1 = b0 + 1;
      if (b0 < num_k_quant_blocks) {
        // 1e-3f lower bound ensures float_to_fp8_e4m3 produces byte >= 1
        // (FP8 E4M3 FNUZ round-to-nearest boundary ≈ 4.88e-4; 1e-3 > that).
        float ideal = fmaxf(blk_lo / FP4_MAX, 1e-3f);
        k_fp8_local[b0] = float_to_fp8_e4m3(ideal);
        float scale = fp8_e4m3_to_float(k_fp8_local[b0]);
        k_block_scale_inv[b0] = 1.0f / fmaxf(scale, 1e-30f);
      }
      if (b1 < num_k_quant_blocks) {
        float ideal = fmaxf(blk_hi / FP4_MAX, 1e-3f);
        k_fp8_local[b1] = float_to_fp8_e4m3(ideal);
        float scale = fp8_e4m3_to_float(k_fp8_local[b1]);
        k_block_scale_inv[b1] = 1.0f / fmaxf(scale, 1e-30f);
      }
    }
  }

#pragma unroll
  for (int i = 0; i < LOCAL_DIM_ELEMS; i++) {
    int d = lane_id + i * warpSize;
    if (d < head_size) {
      float v_abs = fabsf(v_local[i]);
      for (int off = FP4_QUANT_BLOCK_SIZE / 2; off > 0; off /= 2)
        v_abs = fmaxf(v_abs,
                      __shfl_xor_sync(uint64_t(-1), v_abs, off));
      float blk_lo = __shfl_sync(uint64_t(-1), v_abs, 0);
      float blk_hi = __shfl_sync(uint64_t(-1), v_abs, FP4_QUANT_BLOCK_SIZE);
      int b0 = (i * warpSize) / FP4_QUANT_BLOCK_SIZE;
      int b1 = b0 + 1;
      if (b0 < num_v_quant_blocks) {
        float ideal = fmaxf(blk_lo / FP4_MAX, 1e-3f);
        v_fp8_local[b0] = float_to_fp8_e4m3(ideal);
        float scale = fp8_e4m3_to_float(v_fp8_local[b0]);
        v_block_scale_inv[b0] = 1.0f / fmaxf(scale, 1e-30f);
      }
      if (b1 < num_v_quant_blocks) {
        float ideal = fmaxf(blk_hi / FP4_MAX, 1e-3f);
        v_fp8_local[b1] = float_to_fp8_e4m3(ideal);
        float scale = fp8_e4m3_to_float(v_fp8_local[b1]);
        v_block_scale_inv[b1] = 1.0f / fmaxf(scale, 1e-30f);
      }
    }
  }

  // --- Store FP8 E4M3 scales (uint8) ---
  // Safety: ideal >= 1e-3 guarantees float_to_fp8_e4m3 returns >= 1,
  // but add belt-and-suspenders check in case of GPU math anomalies.
  const int64_t k_blk_stride =
      (num_k_quant_blocks > 1) ? (k_scale_stride_h / num_k_quant_blocks) : 0;
  const int64_t v_blk_stride =
      (num_v_quant_blocks > 1) ? (v_scale_stride_h / num_v_quant_blocks) : 0;
  if (lane_id == 0) {
    for (int b = 0; b < num_k_quant_blocks; b++) {
      uint8_t kbyte = k_fp8_local[b];
      if (kbyte == 0) kbyte = 1;
      k_fp8_scales[head_idx * k_scale_stride_h
                    + b * k_blk_stride + slot_idx] = kbyte;
    }
    for (int b = 0; b < num_v_quant_blocks; b++) {
      uint8_t vbyte = v_fp8_local[b];
      if (vbyte == 0) vbyte = 1;
      v_fp8_scales[head_idx * v_scale_stride_h
                    + b * v_blk_stride + slot_idx] = vbyte;
    }
  }

  // --- Phase C: quantize to FP4 E2M1, pack pairs, write to cache ---
#if FP4_USE_IN_KERNEL_WHT
  __shared__ float k_wht_smem_nv[512];
  __shared__ float v_wht_smem_nv[512];
#pragma unroll
  for (int i = 0; i < LOCAL_DIM_ELEMS; i++) {
    int d = lane_id + i * warpSize;
    if (d < head_size) {
      k_wht_smem_nv[d] = k_local[i];
      v_wht_smem_nv[d] = v_local[i];
    }
  }
#endif

  uint8_t* k_dst = key_cache + block_idx * block_stride
                    + block_offset * page_stride
                    + head_idx * head_stride;
  uint8_t* v_dst = value_cache + block_idx * block_stride
                    + block_offset * page_stride
                    + head_idx * head_stride;

  for (int j = lane_id; j < half_head; j += warpSize) {
    int d0 = 2 * j;
    int d1 = d0 + 1;

    int kb = d0 / FP4_QUANT_BLOCK_SIZE;
    float k_sinv = k_block_scale_inv[kb];

#if FP4_USE_IN_KERNEL_WHT
    float kf0 = k_wht_smem_nv[d0] * k_sinv;
    float kf1 = k_wht_smem_nv[d1] * k_sinv;
#else
    float kf0 = static_cast<float>(k_src[d0]) * k_sinv;
    float kf1 = static_cast<float>(k_src[d1]) * k_sinv;
#endif
    uint8_t kn0 = fp4_e2m1_quantize_nibble(kf0);
    uint8_t kn1 = fp4_e2m1_quantize_nibble(kf1);
    k_dst[j] = (kn0 & 0xF) | ((kn1 & 0xF) << 4);

    int vb = d0 / FP4_QUANT_BLOCK_SIZE;
    float v_sinv = v_block_scale_inv[vb];

#if FP4_USE_IN_KERNEL_WHT
    float vf0 = v_wht_smem_nv[d0] * v_sinv;
    float vf1 = v_wht_smem_nv[d1] * v_sinv;
#else
    float vf0 = static_cast<float>(v_src[d0]) * v_sinv;
    float vf1 = static_cast<float>(v_src[d1]) * v_sinv;
#endif
    uint8_t vn0 = fp4_e2m1_quantize_nibble(vf0);
    uint8_t vn1 = fp4_e2m1_quantize_nibble(vf1);
    v_dst[j] = (vn0 & 0xF) | ((vn1 & 0xF) << 4);
  }
}

// ---------------------------------------------------------------------------
// AMXFP4 (Asymmetric Microscaling FP4) cache write kernel.
//
// Uses E8M0 shared scales (like MXFP4) plus a 1-byte BM index per block.
// The Block Maximum (BM) element is encoded with E0M3 (3 mantissa bits)
// for higher precision, while all other elements use standard E2M1.
// Achieves ~90% of the MXFP4→BF16 accuracy gap recovery with only
// 0.25 bits/element overhead for the BM index metadata.
//
// Tensors stored:
//   - FP4 packed data:  same layout as MXFP4
//   - E8M0 scales:      [num_heads * num_blocks, total_tokens] uint8
//   - BM indices:        [num_heads * num_blocks, total_tokens] uint8
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void reshape_and_cache_flash_fp4_amxfp4_kernel(
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    uint8_t* __restrict__ key_cache,
    uint8_t* __restrict__ value_cache,
    uint8_t* __restrict__ k_e8m0_scales,
    uint8_t* __restrict__ v_e8m0_scales,
    uint8_t* __restrict__ k_bm_indices,
    uint8_t* __restrict__ v_bm_indices,
    const int64_t* __restrict__ slot_mapping,
    const int64_t key_stride,
    const int64_t value_stride,
    const int head_size,
    const int block_size,
    const int num_blocks,
    const int64_t block_stride,
    const int64_t page_stride,
    const int64_t head_stride,
    const int64_t k_scale_stride_h,
    const int64_t v_scale_stride_h,
    const int num_k_quant_blocks,
    const int num_v_quant_blocks) {

  constexpr float FP4_MAX = 6.0f;
  constexpr int LOCAL_DIM_ELEMS = 8;
  constexpr int FP4_QUANT_BLOCK_SIZE = 32;
  constexpr int MAX_BLOCKS = LOCAL_DIM_ELEMS * 2;

  const int64_t token_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int lane_id = threadIdx.x;

  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) return;

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  if (block_idx >= num_blocks || block_offset >= block_size) return;

  const int half_head = head_size / 2;

  const scalar_t* k_src =
      key + token_idx * key_stride + head_idx * head_size;
  const scalar_t* v_src =
      value + token_idx * value_stride + head_idx * head_size;

  // --- Phase A: load K/V elements into registers ---
  float k_local[LOCAL_DIM_ELEMS];
  float v_local[LOCAL_DIM_ELEMS];

#pragma unroll
  for (int i = 0; i < LOCAL_DIM_ELEMS; i++) {
    int d = lane_id + i * warpSize;
    if (d < head_size) {
      k_local[i] = static_cast<float>(k_src[d]);
      v_local[i] = static_cast<float>(v_src[d]);
    } else {
      k_local[i] = 0.0f;
      v_local[i] = 0.0f;
    }
  }

#if FP4_USE_IN_KERNEL_WHT
#pragma unroll
  for (int i = 0; i < LOCAL_DIM_ELEMS; i++) {
    int d = lane_id + i * warpSize;
    if (d < head_size) {
      float sign = fp4_wht_sign(d);
      k_local[i] *= sign;
      v_local[i] *= sign;
    }
  }
  for (int h = 1; h < warpSize && h < head_size; h *= 2) {
    bool is_lower = (lane_id & h) == 0;
#pragma unroll
    for (int j = 0; j < LOCAL_DIM_ELEMS; j++) {
      if (lane_id + j * warpSize < head_size) {
        float kp = __shfl_xor_sync(uint64_t(-1), k_local[j], h);
        float vp = __shfl_xor_sync(uint64_t(-1), v_local[j], h);
        k_local[j] = is_lower ? (k_local[j] + kp) : (kp - k_local[j]);
        v_local[j] = is_lower ? (v_local[j] + vp) : (vp - v_local[j]);
      }
    }
  }
  for (int s = 1; s * warpSize < head_size; s *= 2) {
#pragma unroll
    for (int j = 0; j < LOCAL_DIM_ELEMS; j++) {
      int pj = j ^ s;
      if (pj > j && pj < LOCAL_DIM_ELEMS
          && lane_id + j  * warpSize < head_size
          && lane_id + pj * warpSize < head_size) {
        float ka = k_local[j] + k_local[pj];
        float kb = k_local[j] - k_local[pj];
        k_local[j]  = ka;
        k_local[pj] = kb;
        float va = v_local[j] + v_local[pj];
        float vb = v_local[j] - v_local[pj];
        v_local[j]  = va;
        v_local[pj] = vb;
      }
    }
  }
  float wht_norm = rsqrtf(static_cast<float>(head_size));
#pragma unroll
  for (int j = 0; j < LOCAL_DIM_ELEMS; j++) {
    if (lane_id + j * warpSize < head_size) {
      k_local[j] *= wht_norm;
      v_local[j] *= wht_norm;
    }
  }
#endif  // FP4_USE_IN_KERNEL_WHT

  // --- Phase B: compute E8M0 scales and find BM indices per block ---
  float k_block_scale_inv[MAX_BLOCKS];
  float v_block_scale_inv[MAX_BLOCKS];
  uint8_t k_e8m0_local[MAX_BLOCKS];
  uint8_t v_e8m0_local[MAX_BLOCKS];
  uint8_t k_bm_local[MAX_BLOCKS];
  uint8_t v_bm_local[MAX_BLOCKS];

#pragma unroll
  for (int i = 0; i < LOCAL_DIM_ELEMS; i++) {
    int d = lane_id + i * warpSize;
    if (d < head_size) {
      float k_abs = fabsf(k_local[i]);
      int k_bm_pos = lane_id % FP4_QUANT_BLOCK_SIZE;
      for (int off = FP4_QUANT_BLOCK_SIZE / 2; off > 0; off /= 2) {
        float other_abs = __shfl_xor_sync(uint64_t(-1), k_abs, off);
        int other_pos = __shfl_xor_sync(uint64_t(-1), k_bm_pos, off);
        if (other_abs > k_abs ||
            (other_abs == k_abs && other_pos < k_bm_pos)) {
          k_abs = other_abs;
          k_bm_pos = other_pos;
        }
      }
      float blk_lo_abs = __shfl_sync(uint64_t(-1), k_abs, 0);
      float blk_hi_abs = __shfl_sync(
          uint64_t(-1), k_abs, FP4_QUANT_BLOCK_SIZE);
      int bm_lo = __shfl_sync(uint64_t(-1), k_bm_pos, 0);
      int bm_hi = __shfl_sync(uint64_t(-1), k_bm_pos, FP4_QUANT_BLOCK_SIZE);

      int b0 = (i * warpSize) / FP4_QUANT_BLOCK_SIZE;
      int b1 = b0 + 1;
      if (b0 < num_k_quant_blocks) {
        float ideal = fmaxf(blk_lo_abs / FP4_MAX, 1e-10f);
        k_e8m0_local[b0] = float_to_e8m0(ideal);
        if (k_e8m0_local[b0] == 0) k_e8m0_local[b0] = 1;
        float scale = e8m0_to_float(k_e8m0_local[b0]);
        k_block_scale_inv[b0] = 1.0f / fmaxf(scale, 1e-38f);
        k_bm_local[b0] = static_cast<uint8_t>(bm_lo);
      }
      if (b1 < num_k_quant_blocks) {
        float ideal = fmaxf(blk_hi_abs / FP4_MAX, 1e-10f);
        k_e8m0_local[b1] = float_to_e8m0(ideal);
        if (k_e8m0_local[b1] == 0) k_e8m0_local[b1] = 1;
        float scale = e8m0_to_float(k_e8m0_local[b1]);
        k_block_scale_inv[b1] = 1.0f / fmaxf(scale, 1e-38f);
        k_bm_local[b1] = static_cast<uint8_t>(bm_hi);
      }
    }
  }

#pragma unroll
  for (int i = 0; i < LOCAL_DIM_ELEMS; i++) {
    int d = lane_id + i * warpSize;
    if (d < head_size) {
      float v_abs = fabsf(v_local[i]);
      int v_bm_pos = lane_id % FP4_QUANT_BLOCK_SIZE;
      for (int off = FP4_QUANT_BLOCK_SIZE / 2; off > 0; off /= 2) {
        float other_abs = __shfl_xor_sync(uint64_t(-1), v_abs, off);
        int other_pos = __shfl_xor_sync(uint64_t(-1), v_bm_pos, off);
        if (other_abs > v_abs ||
            (other_abs == v_abs && other_pos < v_bm_pos)) {
          v_abs = other_abs;
          v_bm_pos = other_pos;
        }
      }
      float blk_lo_abs = __shfl_sync(uint64_t(-1), v_abs, 0);
      float blk_hi_abs = __shfl_sync(
          uint64_t(-1), v_abs, FP4_QUANT_BLOCK_SIZE);
      int bm_lo = __shfl_sync(uint64_t(-1), v_bm_pos, 0);
      int bm_hi = __shfl_sync(uint64_t(-1), v_bm_pos, FP4_QUANT_BLOCK_SIZE);

      int b0 = (i * warpSize) / FP4_QUANT_BLOCK_SIZE;
      int b1 = b0 + 1;
      if (b0 < num_v_quant_blocks) {
        float ideal = fmaxf(blk_lo_abs / FP4_MAX, 1e-10f);
        v_e8m0_local[b0] = float_to_e8m0(ideal);
        if (v_e8m0_local[b0] == 0) v_e8m0_local[b0] = 1;
        float scale = e8m0_to_float(v_e8m0_local[b0]);
        v_block_scale_inv[b0] = 1.0f / fmaxf(scale, 1e-38f);
        v_bm_local[b0] = static_cast<uint8_t>(bm_lo);
      }
      if (b1 < num_v_quant_blocks) {
        float ideal = fmaxf(blk_hi_abs / FP4_MAX, 1e-10f);
        v_e8m0_local[b1] = float_to_e8m0(ideal);
        if (v_e8m0_local[b1] == 0) v_e8m0_local[b1] = 1;
        float scale = e8m0_to_float(v_e8m0_local[b1]);
        v_block_scale_inv[b1] = 1.0f / fmaxf(scale, 1e-38f);
        v_bm_local[b1] = static_cast<uint8_t>(bm_hi);
      }
    }
  }

  // --- Store E8M0 scales and BM indices (uint8) ---
  const int64_t k_blk_stride =
      (num_k_quant_blocks > 1) ? (k_scale_stride_h / num_k_quant_blocks) : 0;
  const int64_t v_blk_stride =
      (num_v_quant_blocks > 1) ? (v_scale_stride_h / num_v_quant_blocks) : 0;
  if (lane_id == 0) {
    for (int b = 0; b < num_k_quant_blocks; b++) {
      int64_t off = head_idx * k_scale_stride_h + b * k_blk_stride + slot_idx;
      k_e8m0_scales[off] = k_e8m0_local[b];
      k_bm_indices[off] = k_bm_local[b];
    }
    for (int b = 0; b < num_v_quant_blocks; b++) {
      int64_t off = head_idx * v_scale_stride_h + b * v_blk_stride + slot_idx;
      v_e8m0_scales[off] = v_e8m0_local[b];
      v_bm_indices[off] = v_bm_local[b];
    }
  }

  // --- Phase C: quantize to FP4 with AMXFP4 encoding and write to cache ---
#if FP4_USE_IN_KERNEL_WHT
  __shared__ float k_wht_smem_amx[512];
  __shared__ float v_wht_smem_amx[512];
#pragma unroll
  for (int i = 0; i < LOCAL_DIM_ELEMS; i++) {
    int d = lane_id + i * warpSize;
    if (d < head_size) {
      k_wht_smem_amx[d] = k_local[i];
      v_wht_smem_amx[d] = v_local[i];
    }
  }
#endif

  uint8_t* k_dst = key_cache + block_idx * block_stride
                    + block_offset * page_stride
                    + head_idx * head_stride;
  uint8_t* v_dst = value_cache + block_idx * block_stride
                    + block_offset * page_stride
                    + head_idx * head_stride;

  for (int j = lane_id; j < half_head; j += warpSize) {
    int d0 = 2 * j;
    int d1 = d0 + 1;

    int kb = d0 / FP4_QUANT_BLOCK_SIZE;
    float k_sinv = k_block_scale_inv[kb];
    int k_bm_idx = static_cast<int>(k_bm_local[kb]);
    int d0_in_blk = d0 % FP4_QUANT_BLOCK_SIZE;
    int d1_in_blk = d1 % FP4_QUANT_BLOCK_SIZE;

#if FP4_USE_IN_KERNEL_WHT
    float kf0 = k_wht_smem_amx[d0] * k_sinv;
    float kf1 = k_wht_smem_amx[d1] * k_sinv;
#else
    float kf0 = static_cast<float>(k_src[d0]) * k_sinv;
    float kf1 = static_cast<float>(k_src[d1]) * k_sinv;
#endif
    uint8_t kn0 = (d0_in_blk == k_bm_idx)
        ? amxfp4_bm_quantize_nibble(kf0)
        : fp4_e2m1_quantize_nibble(kf0);
    uint8_t kn1 = (d1_in_blk == k_bm_idx)
        ? amxfp4_bm_quantize_nibble(kf1)
        : fp4_e2m1_quantize_nibble(kf1);
    k_dst[j] = (kn0 & 0xF) | ((kn1 & 0xF) << 4);

    int vb = d0 / FP4_QUANT_BLOCK_SIZE;
    float v_sinv = v_block_scale_inv[vb];
    int v_bm_idx = static_cast<int>(v_bm_local[vb]);

#if FP4_USE_IN_KERNEL_WHT
    float vf0 = v_wht_smem_amx[d0] * v_sinv;
    float vf1 = v_wht_smem_amx[d1] * v_sinv;
#else
    float vf0 = static_cast<float>(v_src[d0]) * v_sinv;
    float vf1 = static_cast<float>(v_src[d1]) * v_sinv;
#endif
    uint8_t vn0 = (d0_in_blk == v_bm_idx)
        ? amxfp4_bm_quantize_nibble(vf0)
        : fp4_e2m1_quantize_nibble(vf0);
    uint8_t vn1 = (d1_in_blk == v_bm_idx)
        ? amxfp4_bm_quantize_nibble(vf1)
        : fp4_e2m1_quantize_nibble(vf1);
    v_dst[j] = (vn0 & 0xF) | ((vn1 & 0xF) << 4);
  }
}

}  // namespace vllm
#endif  // USE_ROCM

// ---------------------------------------------------------------------------
// Per-channel-K / per-token-V FP4 cache write dispatch.
// ---------------------------------------------------------------------------
void reshape_and_cache_flash_fp4_per_channel_k_per_token_v(
    torch::Tensor& key,              // [num_tokens, num_heads, head_size]
    torch::Tensor& value,            // [num_tokens, num_heads, head_size]
    torch::Tensor& key_cache,        // [num_blocks, block_size, num_heads, head_size/2]
    torch::Tensor& value_cache,      // [num_blocks, block_size, num_heads, head_size/2]
    torch::Tensor& k_channel_scales, // [num_heads, head_size]  (static per-channel K)
    torch::Tensor& v_dequant_scales, // [num_heads, max_kv_tokens]  (dynamic per-token V)
    torch::Tensor& slot_mapping,     // [num_tokens]
    const std::string& kv_cache_dtype) {
#ifdef USE_ROCM
  TORCH_CHECK(kv_cache_dtype == "fp4" || kv_cache_dtype == "fp4_e2m1",
              "per_channel_k_per_token_v only supports fp4/fp4_e2m1, got: ",
              kv_cache_dtype);

  int num_tokens = slot_mapping.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(1);
  int num_blocks = key_cache.size(0);

  TORCH_CHECK(head_size % 2 == 0,
              "FP4 per-channel-K requires even head_size, got ", head_size);
  TORCH_CHECK(key_cache.dim() == 4,
              "key_cache must be 4D [num_blocks, block_size, num_heads, head_size/2]");
  TORCH_CHECK(value_cache.dim() == 4 &&
              value_cache.sizes() == key_cache.sizes(),
              "value_cache must have same shape as key_cache");
  TORCH_CHECK(key_cache.size(3) == head_size / 2,
              "cache last dim must be head_size/2");
  TORCH_CHECK(k_channel_scales.dim() == 2 &&
              k_channel_scales.size(0) == num_heads &&
              k_channel_scales.size(1) == head_size,
              "k_channel_scales must be [num_heads, head_size], got [",
              k_channel_scales.size(0), ", ", k_channel_scales.size(1), "]");
  TORCH_CHECK(v_dequant_scales.dim() == 2 &&
              v_dequant_scales.size(0) == num_heads,
              "v_dequant_scales must be [num_heads, max_kv_tokens]");

  int64_t key_stride = key.stride(0);
  int64_t value_stride = value.stride(0);
  int64_t block_stride = key_cache.stride(0);
  int64_t page_stride = key_cache.stride(1);
  int64_t head_stride = key_cache.stride(2);
  int64_t v_scale_stride_h = v_dequant_scales.stride(0);

  dim3 grid(num_tokens, num_heads);
  dim3 block(64);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH_FP4_PCK_PTV_KERNEL(scalar_type) \
  vllm::reshape_and_cache_flash_fp4_perchannel_k_pertoken_v_kernel<scalar_type> \
      <<<grid, block, 0, stream>>>( \
          reinterpret_cast<scalar_type*>(key.data_ptr()), \
          reinterpret_cast<scalar_type*>(value.data_ptr()), \
          reinterpret_cast<uint8_t*>(key_cache.data_ptr()), \
          reinterpret_cast<uint8_t*>(value_cache.data_ptr()), \
          k_channel_scales.data_ptr<float>(), \
          v_dequant_scales.data_ptr<float>(), \
          slot_mapping.data_ptr<int64_t>(), \
          key_stride, value_stride, head_size, block_size, num_blocks, \
          block_stride, page_stride, head_stride, v_scale_stride_h)

  if (key.dtype() == at::ScalarType::Half) {
    LAUNCH_FP4_PCK_PTV_KERNEL(uint16_t);
  } else if (key.dtype() == at::ScalarType::BFloat16) {
    LAUNCH_FP4_PCK_PTV_KERNEL(__nv_bfloat16);
  } else if (key.dtype() == at::ScalarType::Float) {
    LAUNCH_FP4_PCK_PTV_KERNEL(float);
  } else {
    TORCH_CHECK(false, "Unsupported key/value dtype: ", key.dtype());
  }
#undef LAUNCH_FP4_PCK_PTV_KERNEL
#else
  TORCH_CHECK(false,
              "reshape_and_cache_flash_fp4_per_channel_k_per_token_v is only "
              "supported on ROCm");
#endif
}

// ---------------------------------------------------------------------------
// MXFP4 (OCP MX) block-scale FP4 cache write dispatch.
//
// Scales are E8M0 (power-of-2) stored as uint8.  Both K and V use
// per-block-32 quantization with E8M0 scales.
// ---------------------------------------------------------------------------
void reshape_and_cache_flash_fp4_mxfp4(
    torch::Tensor& key,              // [num_tokens, num_heads, head_size]
    torch::Tensor& value,            // [num_tokens, num_heads, head_size]
    torch::Tensor& key_cache,        // [num_blocks, block_size, num_heads, head_size/2]
    torch::Tensor& value_cache,      // [num_blocks, block_size, num_heads, head_size/2]
    torch::Tensor& k_e8m0_scales,    // [num_heads * num_k_blocks, max_kv_tokens] uint8
    torch::Tensor& v_e8m0_scales,    // [num_heads * num_v_blocks, max_kv_tokens] uint8
    torch::Tensor& slot_mapping,     // [num_tokens]
    const std::string& kv_cache_dtype) {
#ifdef USE_ROCM
  TORCH_CHECK(kv_cache_dtype == "fp4" || kv_cache_dtype == "fp4_e2m1",
              "reshape_and_cache_flash_fp4_mxfp4 only supports "
              "fp4/fp4_e2m1, got: ", kv_cache_dtype);

  int num_tokens = slot_mapping.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(1);
  int num_blocks = key_cache.size(0);
  constexpr int FP4_QUANT_BLOCK_SIZE = 32;

  TORCH_CHECK(head_size % 2 == 0,
              "MXFP4 requires even head_size, got ", head_size);
  TORCH_CHECK(head_size % FP4_QUANT_BLOCK_SIZE == 0,
              "MXFP4 requires head_size divisible by 32, got ", head_size);
  TORCH_CHECK(key_cache.dim() == 4,
              "key_cache must be 4D [num_blocks, block_size, num_heads, "
              "head_size/2]");
  TORCH_CHECK(value_cache.dim() == 4 &&
              value_cache.sizes() == key_cache.sizes(),
              "value_cache must have same shape as key_cache");
  TORCH_CHECK(key_cache.size(3) == head_size / 2,
              "cache last dim must be head_size/2");
  TORCH_CHECK(k_e8m0_scales.scalar_type() == at::ScalarType::Byte,
              "k_e8m0_scales must be uint8 for MXFP4 E8M0 scales");
  TORCH_CHECK(v_e8m0_scales.scalar_type() == at::ScalarType::Byte,
              "v_e8m0_scales must be uint8 for MXFP4 E8M0 scales");

  int num_k_quant_blocks = head_size / FP4_QUANT_BLOCK_SIZE;
  int num_v_quant_blocks = head_size / FP4_QUANT_BLOCK_SIZE;

  TORCH_CHECK(k_e8m0_scales.dim() == 2 &&
              k_e8m0_scales.size(0) == num_heads * num_k_quant_blocks,
              "k_e8m0_scales must be [num_heads*num_k_blocks, total_tokens]");
  TORCH_CHECK(v_e8m0_scales.dim() == 2 &&
              v_e8m0_scales.size(0) == num_heads * num_v_quant_blocks,
              "v_e8m0_scales must be [num_heads*num_v_blocks, total_tokens]");

  int64_t key_stride = key.stride(0);
  int64_t value_stride = value.stride(0);
  int64_t block_stride = key_cache.stride(0);
  int64_t page_stride = key_cache.stride(1);
  int64_t head_stride = key_cache.stride(2);
  int64_t k_scale_stride_h = k_e8m0_scales.stride(0) * num_k_quant_blocks;
  int64_t v_scale_stride_h = v_e8m0_scales.stride(0) * num_v_quant_blocks;

  dim3 grid(num_tokens, num_heads);
  dim3 block(64);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH_FP4_MXFP4_KERNEL(scalar_type) \
  vllm::reshape_and_cache_flash_fp4_mxfp4_kernel<scalar_type> \
      <<<grid, block, 0, stream>>>( \
          reinterpret_cast<scalar_type*>(key.data_ptr()), \
          reinterpret_cast<scalar_type*>(value.data_ptr()), \
          reinterpret_cast<uint8_t*>(key_cache.data_ptr()), \
          reinterpret_cast<uint8_t*>(value_cache.data_ptr()), \
          k_e8m0_scales.data_ptr<uint8_t>(), \
          v_e8m0_scales.data_ptr<uint8_t>(), \
          slot_mapping.data_ptr<int64_t>(), \
          key_stride, value_stride, head_size, block_size, num_blocks, \
          block_stride, page_stride, head_stride, \
          k_scale_stride_h, v_scale_stride_h, \
          num_k_quant_blocks, num_v_quant_blocks)

  if (key.dtype() == at::ScalarType::Half) {
    LAUNCH_FP4_MXFP4_KERNEL(uint16_t);
  } else if (key.dtype() == at::ScalarType::BFloat16) {
    LAUNCH_FP4_MXFP4_KERNEL(__nv_bfloat16);
  } else if (key.dtype() == at::ScalarType::Float) {
    LAUNCH_FP4_MXFP4_KERNEL(float);
  } else {
    TORCH_CHECK(false, "Unsupported key/value dtype: ", key.dtype());
  }
#undef LAUNCH_FP4_MXFP4_KERNEL
#else
  TORCH_CHECK(false,
              "reshape_and_cache_flash_fp4_mxfp4 is only supported on ROCm");
#endif
}

// ---------------------------------------------------------------------------
// NVFP4-style block-scale FP4 cache write dispatch.
//
// Scales are FP8 E4M3 FNUZ stored as uint8.  Both K and V use
// per-block-32 quantization with FP8 E4M3 scales.
// Provides ~5% better accuracy than MXFP4 E8M0 at same storage cost.
// ---------------------------------------------------------------------------
void reshape_and_cache_flash_fp4_nvfp4(
    torch::Tensor& key,              // [num_tokens, num_heads, head_size]
    torch::Tensor& value,            // [num_tokens, num_heads, head_size]
    torch::Tensor& key_cache,        // [num_blocks, block_size, num_heads, head_size/2]
    torch::Tensor& value_cache,      // [num_blocks, block_size, num_heads, head_size/2]
    torch::Tensor& k_fp8_scales,     // [num_heads * num_k_blocks, max_kv_tokens] uint8
    torch::Tensor& v_fp8_scales,     // [num_heads * num_v_blocks, max_kv_tokens] uint8
    torch::Tensor& slot_mapping,     // [num_tokens]
    const std::string& kv_cache_dtype) {
#ifdef USE_ROCM
  TORCH_CHECK(kv_cache_dtype == "fp4" || kv_cache_dtype == "fp4_e2m1",
              "reshape_and_cache_flash_fp4_nvfp4 only supports "
              "fp4/fp4_e2m1, got: ", kv_cache_dtype);

  int num_tokens = slot_mapping.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(1);
  int num_blocks = key_cache.size(0);
  constexpr int FP4_QUANT_BLOCK_SIZE = 32;

  TORCH_CHECK(head_size % 2 == 0,
              "NVFP4 requires even head_size, got ", head_size);
  TORCH_CHECK(head_size % FP4_QUANT_BLOCK_SIZE == 0,
              "NVFP4 requires head_size divisible by 32, got ", head_size);
  TORCH_CHECK(key_cache.dim() == 4,
              "key_cache must be 4D [num_blocks, block_size, num_heads, "
              "head_size/2]");
  TORCH_CHECK(value_cache.dim() == 4 &&
              value_cache.sizes() == key_cache.sizes(),
              "value_cache must have same shape as key_cache");
  TORCH_CHECK(key_cache.size(3) == head_size / 2,
              "cache last dim must be head_size/2");
  TORCH_CHECK(k_fp8_scales.scalar_type() == at::ScalarType::Byte,
              "k_fp8_scales must be uint8 for NVFP4 E4M3 scales");
  TORCH_CHECK(v_fp8_scales.scalar_type() == at::ScalarType::Byte,
              "v_fp8_scales must be uint8 for NVFP4 E4M3 scales");

  int num_k_quant_blocks = head_size / FP4_QUANT_BLOCK_SIZE;
  int num_v_quant_blocks = head_size / FP4_QUANT_BLOCK_SIZE;

  TORCH_CHECK(k_fp8_scales.dim() == 2 &&
              k_fp8_scales.size(0) == num_heads * num_k_quant_blocks,
              "k_fp8_scales must be [num_heads*num_k_blocks, total_tokens]");
  TORCH_CHECK(v_fp8_scales.dim() == 2 &&
              v_fp8_scales.size(0) == num_heads * num_v_quant_blocks,
              "v_fp8_scales must be [num_heads*num_v_blocks, total_tokens]");

  int64_t key_stride = key.stride(0);
  int64_t value_stride = value.stride(0);
  int64_t block_stride = key_cache.stride(0);
  int64_t page_stride = key_cache.stride(1);
  int64_t head_stride = key_cache.stride(2);
  int64_t k_scale_stride_h = k_fp8_scales.stride(0) * num_k_quant_blocks;
  int64_t v_scale_stride_h = v_fp8_scales.stride(0) * num_v_quant_blocks;

  dim3 grid(num_tokens, num_heads);
  dim3 block(64);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH_FP4_NVFP4_KERNEL(scalar_type) \
  vllm::reshape_and_cache_flash_fp4_nvfp4_kernel<scalar_type> \
      <<<grid, block, 0, stream>>>( \
          reinterpret_cast<scalar_type*>(key.data_ptr()), \
          reinterpret_cast<scalar_type*>(value.data_ptr()), \
          reinterpret_cast<uint8_t*>(key_cache.data_ptr()), \
          reinterpret_cast<uint8_t*>(value_cache.data_ptr()), \
          k_fp8_scales.data_ptr<uint8_t>(), \
          v_fp8_scales.data_ptr<uint8_t>(), \
          slot_mapping.data_ptr<int64_t>(), \
          key_stride, value_stride, head_size, block_size, num_blocks, \
          block_stride, page_stride, head_stride, \
          k_scale_stride_h, v_scale_stride_h, \
          num_k_quant_blocks, num_v_quant_blocks)

  if (key.dtype() == at::ScalarType::Half) {
    LAUNCH_FP4_NVFP4_KERNEL(uint16_t);
  } else if (key.dtype() == at::ScalarType::BFloat16) {
    LAUNCH_FP4_NVFP4_KERNEL(__nv_bfloat16);
  } else if (key.dtype() == at::ScalarType::Float) {
    LAUNCH_FP4_NVFP4_KERNEL(float);
  } else {
    TORCH_CHECK(false, "Unsupported key/value dtype: ", key.dtype());
  }
#undef LAUNCH_FP4_NVFP4_KERNEL
#else
  TORCH_CHECK(false,
              "reshape_and_cache_flash_fp4_nvfp4 is only supported on ROCm");
#endif
}

// ---------------------------------------------------------------------------
// AMXFP4 (Asymmetric Microscaling FP4) cache write dispatch.
//
// Uses E8M0 shared scales (like MXFP4) plus a 1-byte BM index per block.
// The Block Maximum element gets E0M3 encoding (3 mantissa bits) for
// higher outlier precision.
// ---------------------------------------------------------------------------
void reshape_and_cache_flash_fp4_amxfp4(
    torch::Tensor& key,              // [num_tokens, num_heads, head_size]
    torch::Tensor& value,            // [num_tokens, num_heads, head_size]
    torch::Tensor& key_cache,        // [num_blocks, block_size, num_heads, head_size/2]
    torch::Tensor& value_cache,      // [num_blocks, block_size, num_heads, head_size/2]
    torch::Tensor& k_e8m0_scales,    // [num_heads * num_k_blocks, max_kv_tokens] uint8
    torch::Tensor& v_e8m0_scales,    // [num_heads * num_v_blocks, max_kv_tokens] uint8
    torch::Tensor& k_bm_indices,     // [num_heads * num_k_blocks, max_kv_tokens] uint8
    torch::Tensor& v_bm_indices,     // [num_heads * num_v_blocks, max_kv_tokens] uint8
    torch::Tensor& slot_mapping,     // [num_tokens]
    const std::string& kv_cache_dtype) {
#ifdef USE_ROCM
  TORCH_CHECK(kv_cache_dtype == "fp4" || kv_cache_dtype == "fp4_e2m1",
              "reshape_and_cache_flash_fp4_amxfp4 only supports "
              "fp4/fp4_e2m1, got: ", kv_cache_dtype);

  int num_tokens = slot_mapping.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(1);
  int num_blocks = key_cache.size(0);
  constexpr int FP4_QUANT_BLOCK_SIZE = 32;

  TORCH_CHECK(head_size % 2 == 0,
              "AMXFP4 requires even head_size, got ", head_size);
  TORCH_CHECK(head_size % FP4_QUANT_BLOCK_SIZE == 0,
              "AMXFP4 requires head_size divisible by 32, got ", head_size);
  TORCH_CHECK(key_cache.dim() == 4,
              "key_cache must be 4D [num_blocks, block_size, num_heads, "
              "head_size/2]");
  TORCH_CHECK(value_cache.dim() == 4 &&
              value_cache.sizes() == key_cache.sizes(),
              "value_cache must have same shape as key_cache");
  TORCH_CHECK(key_cache.size(3) == head_size / 2,
              "cache last dim must be head_size/2");
  TORCH_CHECK(k_e8m0_scales.scalar_type() == at::ScalarType::Byte,
              "k_e8m0_scales must be uint8 for AMXFP4 E8M0 scales");
  TORCH_CHECK(v_e8m0_scales.scalar_type() == at::ScalarType::Byte,
              "v_e8m0_scales must be uint8 for AMXFP4 E8M0 scales");
  TORCH_CHECK(k_bm_indices.scalar_type() == at::ScalarType::Byte,
              "k_bm_indices must be uint8 for AMXFP4 BM indices");
  TORCH_CHECK(v_bm_indices.scalar_type() == at::ScalarType::Byte,
              "v_bm_indices must be uint8 for AMXFP4 BM indices");

  int num_k_quant_blocks = head_size / FP4_QUANT_BLOCK_SIZE;
  int num_v_quant_blocks = head_size / FP4_QUANT_BLOCK_SIZE;

  TORCH_CHECK(k_e8m0_scales.dim() == 2 &&
              k_e8m0_scales.size(0) == num_heads * num_k_quant_blocks,
              "k_e8m0_scales must be [num_heads*num_k_blocks, total_tokens]");
  TORCH_CHECK(v_e8m0_scales.dim() == 2 &&
              v_e8m0_scales.size(0) == num_heads * num_v_quant_blocks,
              "v_e8m0_scales must be [num_heads*num_v_blocks, total_tokens]");
  TORCH_CHECK(k_bm_indices.dim() == 2 &&
              k_bm_indices.size(0) == num_heads * num_k_quant_blocks,
              "k_bm_indices must be [num_heads*num_k_blocks, total_tokens]");
  TORCH_CHECK(v_bm_indices.dim() == 2 &&
              v_bm_indices.size(0) == num_heads * num_v_quant_blocks,
              "v_bm_indices must be [num_heads*num_v_blocks, total_tokens]");

  int64_t key_stride = key.stride(0);
  int64_t value_stride = value.stride(0);
  int64_t block_stride = key_cache.stride(0);
  int64_t page_stride = key_cache.stride(1);
  int64_t head_stride = key_cache.stride(2);
  int64_t k_scale_stride_h = k_e8m0_scales.stride(0) * num_k_quant_blocks;
  int64_t v_scale_stride_h = v_e8m0_scales.stride(0) * num_v_quant_blocks;

  dim3 grid(num_tokens, num_heads);
  dim3 block(64);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH_FP4_AMXFP4_KERNEL(scalar_type) \
  vllm::reshape_and_cache_flash_fp4_amxfp4_kernel<scalar_type> \
      <<<grid, block, 0, stream>>>( \
          reinterpret_cast<scalar_type*>(key.data_ptr()), \
          reinterpret_cast<scalar_type*>(value.data_ptr()), \
          reinterpret_cast<uint8_t*>(key_cache.data_ptr()), \
          reinterpret_cast<uint8_t*>(value_cache.data_ptr()), \
          k_e8m0_scales.data_ptr<uint8_t>(), \
          v_e8m0_scales.data_ptr<uint8_t>(), \
          k_bm_indices.data_ptr<uint8_t>(), \
          v_bm_indices.data_ptr<uint8_t>(), \
          slot_mapping.data_ptr<int64_t>(), \
          key_stride, value_stride, head_size, block_size, num_blocks, \
          block_stride, page_stride, head_stride, \
          k_scale_stride_h, v_scale_stride_h, \
          num_k_quant_blocks, num_v_quant_blocks)

  if (key.dtype() == at::ScalarType::Half) {
    LAUNCH_FP4_AMXFP4_KERNEL(uint16_t);
  } else if (key.dtype() == at::ScalarType::BFloat16) {
    LAUNCH_FP4_AMXFP4_KERNEL(__nv_bfloat16);
  } else if (key.dtype() == at::ScalarType::Float) {
    LAUNCH_FP4_AMXFP4_KERNEL(float);
  } else {
    TORCH_CHECK(false, "Unsupported key/value dtype: ", key.dtype());
  }
#undef LAUNCH_FP4_AMXFP4_KERNEL
#else
  TORCH_CHECK(false,
              "reshape_and_cache_flash_fp4_amxfp4 is only supported on ROCm");
#endif
}

void reshape_and_cache_flash_with_pertoken_quant(
    torch::Tensor& key,              // [num_tokens, num_heads, head_size]
    torch::Tensor& value,            // [num_tokens, num_heads, head_size]
    torch::Tensor& key_cache,        // [num_blocks, block_size, num_heads, head_size/2]
    torch::Tensor& value_cache,      // [num_blocks, block_size, num_heads, head_size/2]
    torch::Tensor& k_dequant_scales, // [num_heads * num_k_blocks, max_kv_tokens]
    torch::Tensor& v_dequant_scales, // [num_heads * num_v_blocks, max_kv_tokens]
    torch::Tensor& slot_mapping,     // [num_tokens]
    const std::string& kv_cache_dtype) {
#ifdef USE_ROCM
  TORCH_CHECK(kv_cache_dtype == "fp4" || kv_cache_dtype == "fp4_e2m1",
              "reshape_and_cache_flash_with_pertoken_quant only supports "
              "fp4/fp4_e2m1 kv_cache_dtype, got: ", kv_cache_dtype);

  int num_tokens = slot_mapping.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(1);
  int num_blocks = key_cache.size(0);

  TORCH_CHECK(head_size % 2 == 0,
              "FP4 per-token quant requires even head_size, got ", head_size);
  TORCH_CHECK(key_cache.dim() == 4,
              "FP4 per-token quant key_cache must be 4D "
              "[num_blocks, block_size, num_heads, head_size/2]");
  TORCH_CHECK(value_cache.dim() == 4 &&
              value_cache.sizes() == key_cache.sizes(),
              "FP4 per-token quant value_cache must have same shape as "
              "key_cache");
  int cache_head_dim = key_cache.size(3);
  TORCH_CHECK(cache_head_dim == head_size / 2,
              "FP4 per-token quant cache last dim must be head_size/2: "
              "expected ", head_size / 2, ", got ", cache_head_dim);
  TORCH_CHECK(v_dequant_scales.dim() == 2,
              "V dequant scale tensor must be 2D "
              "[num_heads * num_v_blocks, max_kv_tokens]");
  TORCH_CHECK(k_dequant_scales.dim() == 2,
              "K dequant scale tensor must be 2D "
              "[num_heads * num_k_blocks, max_kv_tokens]");

  constexpr int FP4_QUANT_BLOCK_SIZE = 32;
  int num_k_quant_blocks = head_size / FP4_QUANT_BLOCK_SIZE;
  if (num_k_quant_blocks < 1) num_k_quant_blocks = 1;
  int num_v_quant_blocks = head_size / FP4_QUANT_BLOCK_SIZE;
  if (num_v_quant_blocks < 1) num_v_quant_blocks = 1;

  TORCH_CHECK(k_dequant_scales.size(0) == (int64_t)num_heads * num_k_quant_blocks,
              "K dequant scale tensor dim 0 must be num_heads * num_k_blocks = ",
              num_heads * num_k_quant_blocks, ", got ", k_dequant_scales.size(0));
  TORCH_CHECK(v_dequant_scales.size(0) == (int64_t)num_heads * num_v_quant_blocks,
              "V dequant scale tensor dim 0 must be num_heads * num_v_blocks = ",
              num_heads * num_v_quant_blocks, ", got ", v_dequant_scales.size(0));

  int64_t key_stride = key.stride(0);
  int64_t value_stride = value.stride(0);
  int64_t block_stride = key_cache.stride(0);
  int64_t page_stride = key_cache.stride(1);
  int64_t head_stride = key_cache.stride(2);
  int64_t k_scale_stride_h = k_dequant_scales.stride(0) * num_k_quant_blocks;
  int64_t v_scale_stride_h = v_dequant_scales.stride(0) * num_v_quant_blocks;
  TORCH_CHECK(key_cache.stride(0) == value_cache.stride(0));

  dim3 grid(num_tokens, num_heads);
  dim3 block(64);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH_FP4_KERNEL(scalar_type) \
  vllm::reshape_and_cache_flash_fp4_pertoken_quant_kernel<scalar_type> \
      <<<grid, block, 0, stream>>>( \
          reinterpret_cast<scalar_type*>(key.data_ptr()), \
          reinterpret_cast<scalar_type*>(value.data_ptr()), \
          reinterpret_cast<uint8_t*>(key_cache.data_ptr()), \
          reinterpret_cast<uint8_t*>(value_cache.data_ptr()), \
          k_dequant_scales.data_ptr<float>(), \
          v_dequant_scales.data_ptr<float>(), \
          slot_mapping.data_ptr<int64_t>(), \
          key_stride, value_stride, head_size, block_size, num_blocks, \
          block_stride, page_stride, head_stride, \
          k_scale_stride_h, v_scale_stride_h, num_k_quant_blocks, \
          num_v_quant_blocks)

  if (key.dtype() == at::ScalarType::Half) {
    LAUNCH_FP4_KERNEL(uint16_t);
  } else if (key.dtype() == at::ScalarType::BFloat16) {
    LAUNCH_FP4_KERNEL(__nv_bfloat16);
  } else if (key.dtype() == at::ScalarType::Float) {
    LAUNCH_FP4_KERNEL(float);
  } else {
    TORCH_CHECK(false, "Unsupported key/value dtype: ", key.dtype());
  }
#undef LAUNCH_FP4_KERNEL
#else
  TORCH_CHECK(false,
              "reshape_and_cache_flash_with_pertoken_quant is only supported "
              "on ROCm");
#endif
}

// KV_T is the data type of key and value tensors.
// CACHE_T is the stored data type of kv-cache.
// KV_DTYPE is the real data type of kv-cache.
#define CALL_CONCAT_AND_CACHE_MLA(KV_T, CACHE_T, KV_DTYPE)              \
  vllm::concat_and_cache_mla_kernel<KV_T, CACHE_T, KV_DTYPE>            \
      <<<grid, block, 0, stream>>>(                                     \
          reinterpret_cast<KV_T*>(kv_c.data_ptr()),                     \
          reinterpret_cast<KV_T*>(k_pe.data_ptr()),                     \
          reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()),              \
          slot_mapping.data_ptr<int64_t>(), block_stride, entry_stride, \
          kv_c_stride, k_pe_stride, kv_lora_rank, pe_dim, block_size,   \
          reinterpret_cast<const float*>(scale.data_ptr()));

// KV_T is the data type of key and value tensors.
// CACHE_T is the stored data type of kv-cache.
#define CALL_CONCAT_AND_CACHE_DS_MLA(KV_T, CACHE_T, KV_DTYPE)           \
  vllm::concat_and_cache_ds_mla_kernel<KV_T, CACHE_T, KV_DTYPE>         \
      <<<grid, block, 0, stream>>>(                                     \
          reinterpret_cast<KV_T*>(kv_c.data_ptr()),                     \
          reinterpret_cast<KV_T*>(k_pe.data_ptr()),                     \
          reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()),              \
          slot_mapping.data_ptr<int64_t>(), block_stride, entry_stride, \
          kv_c_stride, k_pe_stride, kv_lora_rank, pe_dim, block_size,   \
          reinterpret_cast<const float*>(scale.data_ptr()));

void concat_and_cache_mla(
    torch::Tensor& kv_c,          // [num_tokens, kv_lora_rank]
    torch::Tensor& k_pe,          // [num_tokens, pe_dim]
    torch::Tensor& kv_cache,      // [num_blocks, block_size, (kv_lora_rank +
                                  // pe_dim)]
    torch::Tensor& slot_mapping,  // [num_tokens] or [num_actual_tokens]
    const std::string& kv_cache_dtype, torch::Tensor& scale) {
  // NOTE(woosuk): In vLLM V1, key.size(0) can be different from
  // slot_mapping.size(0) because of padding for CUDA graphs.
  // In vLLM V0, key.size(0) is always equal to slot_mapping.size(0) because
  // both include padding.
  // In vLLM V1, however, key.size(0) can be larger than slot_mapping.size(0)
  // since key includes padding for CUDA graphs, while slot_mapping does not.
  // In this case, slot_mapping.size(0) represents the actual number of tokens
  // before padding.
  // For compatibility with both cases, we use slot_mapping.size(0) as the
  // number of tokens.
  int num_tokens = slot_mapping.size(0);
  int kv_lora_rank = kv_c.size(1);
  int pe_dim = k_pe.size(1);
  int block_size = kv_cache.size(1);

  if (kv_cache_dtype == "fp8_ds_mla") {
    TORCH_CHECK(kv_lora_rank == 512, "kv_lora_rank must be 512 for fp8_ds_mla");
    TORCH_CHECK(pe_dim == 64, "pe_dim must be 64 for fp8_ds_mla");
    TORCH_CHECK(kv_cache.size(2) == 656 / kv_cache.itemsize(),
                "kv_cache.size(2) must be 656 bytes for fp8_ds_mla");
    TORCH_CHECK(kv_c.itemsize() == 2,
                "kv_c.itemsize() must be 2 for fp8_ds_mla");
    TORCH_CHECK(k_pe.itemsize() == 2,
                "k_pe.itemsize() must be 2 for fp8_ds_mla");
  } else {
    TORCH_CHECK(kv_cache.size(2) == kv_lora_rank + pe_dim);
  }

  int kv_c_stride = kv_c.stride(0);
  int k_pe_stride = k_pe.stride(0);
  int block_stride = kv_cache.stride(0);
  int entry_stride = kv_cache.stride(1);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(kv_c));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (kv_cache_dtype == "fp8_ds_mla") {
    dim3 grid(num_tokens);
    // For the NoPE part, each tile of 128 elements is handled by half of one
    // warp (16 threads). There are 4 total tiles, so 2 warps (64 threads).
    // Lanes 0 and 16 of each warp write the scale values for that warp's tiles.
    // The RoPE part (last 64 elements) is handled by another 1 warp (32
    // threads). So in total, we use 3 warps (96 threads) per block.
    dim3 block(96);
    DISPATCH_BY_KV_CACHE_DTYPE(kv_c.dtype(), kv_cache_dtype,
                               CALL_CONCAT_AND_CACHE_DS_MLA);
  } else {
    dim3 grid(num_tokens);
    dim3 block(std::min(kv_lora_rank, 512));
    DISPATCH_BY_KV_CACHE_DTYPE(kv_c.dtype(), kv_cache_dtype,
                               CALL_CONCAT_AND_CACHE_MLA);
  }
}

namespace vllm {

template <typename Tout, typename Tin, Fp8KVCacheDataType kv_dt>
__global__ void convert_fp8_kernel(const Tin* __restrict__ src_cache,
                                   Tout* __restrict__ dst_cache,
                                   const float scale,
                                   const int64_t block_stride) {
  const int64_t block_idx = blockIdx.x;
  for (int i = threadIdx.x; i < block_stride; i += blockDim.x) {
    int64_t idx = block_idx * block_stride + i;
    dst_cache[idx] =
        fp8::scaled_convert<Tout, Tin, kv_dt>(src_cache[idx], scale);
  }
}

}  // namespace vllm

#define CALL_CONVERT_FP8(Tout, Tin, KV_DTYPE)                                \
  vllm::convert_fp8_kernel<Tout, Tin, KV_DTYPE><<<grid, block, 0, stream>>>( \
      reinterpret_cast<Tin*>(src_cache.data_ptr()),                          \
      reinterpret_cast<Tout*>(dst_cache.data_ptr()), scale, block_stride);

// Only for testing.
void convert_fp8(torch::Tensor& dst_cache, torch::Tensor& src_cache,
                 const double scale, const std::string& kv_cache_dtype) {
  torch::Device src_device = src_cache.device();
  torch::Device dst_device = dst_cache.device();
  TORCH_CHECK(src_device.is_cuda(), "src must be on a GPU")
  TORCH_CHECK(dst_device.is_cuda(), "dst must be on a GPU")
  TORCH_CHECK(src_device.index() == dst_device.index(),
              "src and dst must be on the same GPU");
  at::cuda::OptionalCUDAGuard device_guard(src_device);

  int64_t num_blocks = src_cache.size(0);
  int64_t block_stride = src_cache.stride(0);

  dim3 grid(num_blocks);
  dim3 block(std::min(block_stride, int64_t(512)));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (kv_cache_dtype == "auto") {
    if (src_cache.dtype() == at::ScalarType::Float) {
      CALL_CONVERT_FP8(uint8_t, float, vllm::Fp8KVCacheDataType::kAuto);
    } else if (src_cache.dtype() == at::ScalarType::Half) {
      CALL_CONVERT_FP8(uint8_t, uint16_t, vllm::Fp8KVCacheDataType::kAuto);
    } else if (src_cache.dtype() == at::ScalarType::BFloat16) {
      CALL_CONVERT_FP8(uint8_t, __nv_bfloat16, vllm::Fp8KVCacheDataType::kAuto);
    } else if (dst_cache.dtype() == at::ScalarType::Float) {
      CALL_CONVERT_FP8(float, uint8_t, vllm::Fp8KVCacheDataType::kAuto);
    } else if (dst_cache.dtype() == at::ScalarType::Half) {
      CALL_CONVERT_FP8(uint16_t, uint8_t, vllm::Fp8KVCacheDataType::kAuto);
    } else if (dst_cache.dtype() == at::ScalarType::BFloat16) {
      CALL_CONVERT_FP8(__nv_bfloat16, uint8_t, vllm::Fp8KVCacheDataType::kAuto);
    }
  } else if (kv_cache_dtype == "fp8" || kv_cache_dtype == "fp8_e4m3") {
    if (src_cache.dtype() == at::ScalarType::Float) {
      CALL_CONVERT_FP8(uint8_t, float, vllm::Fp8KVCacheDataType::kFp8E4M3);
    } else if (src_cache.dtype() == at::ScalarType::Half) {
      CALL_CONVERT_FP8(uint8_t, uint16_t, vllm::Fp8KVCacheDataType::kFp8E4M3);
    } else if (src_cache.dtype() == at::ScalarType::BFloat16) {
      CALL_CONVERT_FP8(uint8_t, __nv_bfloat16,
                       vllm::Fp8KVCacheDataType::kFp8E4M3);
    } else if (dst_cache.dtype() == at::ScalarType::Float) {
      CALL_CONVERT_FP8(float, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);
    } else if (dst_cache.dtype() == at::ScalarType::Half) {
      CALL_CONVERT_FP8(uint16_t, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);
    } else if (dst_cache.dtype() == at::ScalarType::BFloat16) {
      CALL_CONVERT_FP8(__nv_bfloat16, uint8_t,
                       vllm::Fp8KVCacheDataType::kFp8E4M3);
    }
#ifdef USE_ROCM
  } else if (kv_cache_dtype == "fp4" || kv_cache_dtype == "fp4_e2m1") {
    if (src_cache.dtype() == at::ScalarType::Float) {
      CALL_CONVERT_FP8(uint8_t, float, vllm::Fp8KVCacheDataType::kFp4E2M1);
    } else if (src_cache.dtype() == at::ScalarType::Half) {
      CALL_CONVERT_FP8(uint8_t, uint16_t, vllm::Fp8KVCacheDataType::kFp4E2M1);
    } else if (src_cache.dtype() == at::ScalarType::BFloat16) {
      CALL_CONVERT_FP8(uint8_t, __nv_bfloat16,
                       vllm::Fp8KVCacheDataType::kFp4E2M1);
    } else if (dst_cache.dtype() == at::ScalarType::Float) {
      CALL_CONVERT_FP8(float, uint8_t, vllm::Fp8KVCacheDataType::kFp4E2M1);
    } else if (dst_cache.dtype() == at::ScalarType::Half) {
      CALL_CONVERT_FP8(uint16_t, uint8_t, vllm::Fp8KVCacheDataType::kFp4E2M1);
    } else if (dst_cache.dtype() == at::ScalarType::BFloat16) {
      CALL_CONVERT_FP8(__nv_bfloat16, uint8_t,
                       vllm::Fp8KVCacheDataType::kFp4E2M1);
    }
  }
#endif
  else {
    TORCH_CHECK(false, "Unsupported data type: ", kv_cache_dtype);
  }
}

namespace vllm {

// grid is launched with dimensions (batch, num_splits)
template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt,
          int ENTRY_SIZE, int CTA_SIZE>
__global__ void gather_and_maybe_dequant_cache(
    const cache_t* __restrict__ src_cache,     // [NUM_BLOCKS, BLOCK_SIZE,
                                               // ENTRIES...]
    scalar_t* __restrict__ dst,                // [TOT_TOKENS, ENTRIES...]
    const int32_t* __restrict__ block_table,   // [BATCH, BLOCK_INDICES]
    const int32_t* __restrict__ cu_seq_lens,   // [BATCH+1]
    const int32_t* __restrict__ token_to_seq,  // [MAX_TOKEN_ACROSS_CHUNK]
    const int32_t num_tokens, const int32_t block_size,
    const int64_t block_table_stride, const int64_t cache_block_stride,
    const int64_t cache_entry_stride, const int64_t dst_entry_stride,
    const float* __restrict__ scale,
    const int32_t* __restrict__ seq_starts) {  // Optional: starting offsets per
                                               // batch
  constexpr int vec_size = sizeof(float4) / sizeof(scalar_t);
  using ltype = vllm::vec_n_t<cache_t, vec_size>;
  using stype = vllm::vec_n_t<scalar_t, vec_size>;
  // We are adding this for code readability which will be optimized out when
  // build in release.
  assert(CTA_SIZE == blockDim.x);

#pragma unroll
  for (int token_id = blockIdx.x; token_id < num_tokens;
       token_id += gridDim.x) {
    int64_t batch_id = token_to_seq[token_id];
    int64_t batch_start = cu_seq_lens[batch_id];
    int64_t batch_end = cu_seq_lens[batch_id + 1];
    int32_t batch_offset = token_id - batch_start;

    if (token_id >= batch_end) return;
    int32_t offset = 0;
    if (seq_starts != nullptr) {
      offset = seq_starts[batch_id];
    }
    batch_offset += offset;
    int32_t block_table_id = batch_offset / block_size;
    int32_t slot_id = batch_offset % block_size;
    int32_t block_table_offset = batch_id * block_table_stride + block_table_id;
    int32_t block_id = block_table[block_table_offset];
    int64_t cache_offset =
        block_id * cache_block_stride + slot_id * cache_entry_stride;
    constexpr int32_t vec_iter_cnt = ENTRY_SIZE / vec_size;
    scalar_t* dst_ = dst + token_id * dst_entry_stride;
    cache_t* src_ = const_cast<cache_t*>(src_cache) + cache_offset;

#pragma unroll
    for (int idx = threadIdx.x; idx < vec_iter_cnt; idx += CTA_SIZE) {
      if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
        reinterpret_cast<stype*>(dst_)[idx] =
            static_cast<stype>(reinterpret_cast<ltype*>(src_)[idx]);
      } else {
        ltype loaded_val = reinterpret_cast<ltype*>(src_)[idx];
        stype store_val;
#pragma unroll
        for (int j = 0; j < vec_size; ++j) {
          store_val.val[j] = fp8::scaled_convert<scalar_t, cache_t, kv_dt>(
              loaded_val.val[j], *scale);
        }
        reinterpret_cast<stype*>(dst_)[idx] = store_val;
      }
    }
    // process tail
    constexpr int32_t tail_cnt = ENTRY_SIZE % vec_size;
    dst_ = dst_ + ENTRY_SIZE - tail_cnt;
    src_ = src_ + ENTRY_SIZE - tail_cnt;
#pragma unroll
    for (int idx = threadIdx.x; idx < tail_cnt; idx += CTA_SIZE) {
      if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
        dst_[idx] = static_cast<scalar_t>(src_[idx]);
      } else {
        dst_[idx] =
            fp8::scaled_convert<scalar_t, cache_t, kv_dt>(src_[idx], *scale);
      }
    }
  }
}

}  // namespace vllm

// Macro to dispatch the kernel based on the data type.
// SCALAR_T is the data type of the destination tensor.
// CACHE_T is the stored data type of kv-cache.
// KV_DTYPE is the real data type of kv-cache.
#define CALL_GATHER_CACHE(SCALAR_T, CACHE_T, KV_DTYPE)                        \
  vllm::gather_and_maybe_dequant_cache<SCALAR_T, CACHE_T, KV_DTYPE, 576,      \
                                       thread_block_size>                     \
      <<<grid, block, 0, stream>>>(                                           \
          reinterpret_cast<CACHE_T*>(src_cache.data_ptr()),                   \
          reinterpret_cast<SCALAR_T*>(dst.data_ptr()),                        \
          block_table.data_ptr<int32_t>(), cu_seq_lens.data_ptr<int32_t>(),   \
          token_to_seq.data_ptr<int32_t>(), num_tokens, block_size,           \
          block_table_stride, cache_block_stride, cache_entry_stride,         \
          dst_entry_stride, reinterpret_cast<const float*>(scale.data_ptr()), \
          seq_starts_ptr);

// Gather sequences from the cache into the destination tensor.
//  - cu_seq_lens contains the cumulative sequence lengths for each batch
//  - block_table contains the cache block indices for each sequence
//  - token_to_seq contains the back mapping from token_id to batch_id
//  - Optionally, seq_starts (if provided) offsets the starting block index by
//  (seq_starts[bid] / page_size)
void gather_and_maybe_dequant_cache(
    torch::Tensor const& src_cache,     // [NUM_BLOCKS, BLOCK_SIZE, ENTRIES...]
    torch::Tensor const& dst,           // [TOT_TOKENS, ENTRIES...]
    torch::Tensor const& block_table,   // [BATCH, BLOCK_INDICES]
    torch::Tensor const& cu_seq_lens,   // [BATCH+1]
    torch::Tensor const& token_to_seq,  // [MAX_TOKEN_ACROSS_CHUNKS]
    int64_t num_tokens, const std::string& kv_cache_dtype,
    torch::Tensor const& scale,
    std::optional<torch::Tensor> seq_starts = std::nullopt) {
  at::cuda::OptionalCUDAGuard device_guard(src_cache.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int32_t block_size = src_cache.size(1);
  int32_t head_dim = dst.size(-1);

  TORCH_CHECK(block_table.dtype() == torch::kInt32,
              "block_table must be int32");
  TORCH_CHECK(cu_seq_lens.dtype() == torch::kInt32,
              "cu_seq_lens must be int32");
  if (seq_starts.has_value()) {
    TORCH_CHECK(seq_starts.value().dtype() == torch::kInt32,
                "seq_starts must be int32");
  }
  TORCH_CHECK(head_dim == 576,
              "gather_and_maybe_dequant_cache only support the head_dim to 576 "
              "for better performance")

  TORCH_CHECK(src_cache.device() == dst.device(),
              "src_cache and dst must be on the same device");
  TORCH_CHECK(src_cache.device() == block_table.device(),
              "src_cache and block_table must be on the same device");
  TORCH_CHECK(src_cache.device() == cu_seq_lens.device(),
              "src_cache and cu_seq_lens must be on the same device");
  if (seq_starts.has_value()) {
    TORCH_CHECK(src_cache.device() == seq_starts.value().device(),
                "src_cache and seq_starts must be on the same device");
  }

  int64_t block_table_stride = block_table.stride(0);
  int64_t cache_block_stride = src_cache.stride(0);
  int64_t cache_entry_stride = src_cache.stride(1);
  int64_t dst_entry_stride = dst.stride(0);

  constexpr int32_t thread_block_size = 64;
  dim3 grid(num_tokens);
  dim3 block(thread_block_size);

  const int32_t* seq_starts_ptr =
      seq_starts.has_value() ? seq_starts.value().data_ptr<int32_t>() : nullptr;

  DISPATCH_BY_KV_CACHE_DTYPE(dst.dtype(), kv_cache_dtype, CALL_GATHER_CACHE);
}

namespace vllm {

// Gather and upconvert FP8 KV cache tokens to BF16 workspace
// Similar to cp_gather_cache but specifically for FP8->BF16 conversion
__global__ void cp_gather_and_upconvert_fp8_kv_cache(
    const uint8_t* __restrict__ src_cache,    // [NUM_BLOCKS, BLOCK_SIZE, 656]
    __nv_bfloat16* __restrict__ dst,          // [TOT_TOKENS, 576]
    const int32_t* __restrict__ block_table,  // [BATCH, BLOCK_INDICES]
    const int32_t* __restrict__ seq_lens,     // [BATCH]
    const int32_t* __restrict__ workspace_starts,  // [BATCH]
    const int32_t block_size, const int32_t head_dim,
    const int64_t block_table_stride, const int64_t cache_block_stride,
    const int64_t cache_entry_stride, const int64_t dst_entry_stride) {
  const int64_t bid = blockIdx.x;  // Batch ID
  const int32_t num_splits = gridDim.y;
  const int32_t split = blockIdx.y;
  const int32_t seq_start = workspace_starts[bid];
  const int32_t seq_len = seq_lens[bid];
  const int32_t tot_slots = seq_len;
  const int32_t split_slots = cuda_utils::ceil_div(tot_slots, num_splits);

  const int32_t split_start = split * split_slots;
  const int32_t split_end = min((split + 1) * split_slots, tot_slots);

  const bool is_active_split = (split_start < tot_slots);

  if (!is_active_split) return;

  // Adjust the pointer for the block_table for this batch
  const int32_t batch_offset = bid * block_table_stride;
  int32_t offset = split_start;
  int32_t offset_div = offset / block_size;
  offset = offset % block_size;
  const int32_t* batch_block_table = block_table + batch_offset;

  // Adjust dst pointer based on the cumulative sequence lengths
  dst += seq_start * dst_entry_stride;

  const int tid = threadIdx.x;

  // Process each token in this split
  for (int pid = split_start; pid < split_end; ++pid) {
    auto block_id = batch_block_table[offset_div];
    const uint8_t* token_ptr =
        src_cache + block_id * cache_block_stride + offset * cache_entry_stride;
    __nv_bfloat16* dst_ptr = dst + pid * dst_entry_stride;

    // FP8 format: 512 bytes fp8 + 16 bytes scales + 128 bytes rope (64 bf16)
    const uint8_t* no_pe_ptr = token_ptr;
    const float* scales_ptr = reinterpret_cast<const float*>(token_ptr + 512);
    const __nv_bfloat16* rope_ptr =
        reinterpret_cast<const __nv_bfloat16*>(token_ptr + 512 + 16);

    // Parallelize fp8 dequant (512 elements) and rope copy (64 elements)
    if (tid < 512) {
      // FP8 dequantization
      const int tile = tid >> 7;  // each tile is 128 elements
      const float scale = scales_ptr[tile];
      const uint8_t val = no_pe_ptr[tid];
      dst_ptr[tid] =
          fp8::scaled_convert<__nv_bfloat16, uint8_t,
                              vllm::Fp8KVCacheDataType::kFp8E4M3>(val, scale);
    } else if (tid < 576) {
      // Rope copy (64 bf16 elements)
      const int rope_idx = tid - 512;
      dst_ptr[512 + rope_idx] = rope_ptr[rope_idx];
    }

    // Move to next token
    offset += 1;
    if (offset == block_size) {
      offset_div += 1;
      offset = 0;
    }
  }
}

template <typename scalar_t>
// Note(hc): The cp_gather_cache allows seq_starts to no longer be divisible by
// block_size.
__global__ void cp_gather_cache(
    const scalar_t* __restrict__ src_cache,   // [NUM_BLOCKS, BLOCK_SIZE,
                                              // ENTRY_SIZE]
    scalar_t* __restrict__ dst,               // [TOT_TOKENS, ENTRY_SIZE]
    const int32_t* __restrict__ block_table,  // [BATCH, BLOCK_INDICES]
    const int32_t* __restrict__ cu_seq_lens,  // [BATCH+1]
    const int32_t block_size, const int32_t entry_size,
    const int64_t block_table_stride, const int64_t cache_block_stride,
    const int64_t cache_entry_stride, const int64_t dst_entry_stride,
    const int32_t* __restrict__ seq_starts  // Optional: starting offsets per
                                            // batch
) {
  const int64_t bid = blockIdx.x;  // Batch ID
  const int32_t num_splits = gridDim.y;
  const int32_t split = blockIdx.y;
  const int32_t seq_start = cu_seq_lens[bid];
  const int32_t seq_end = cu_seq_lens[bid + 1];
  const int32_t seq_len = seq_end - seq_start;
  const int32_t tot_slots = seq_len;
  const int32_t split_slots = cuda_utils::ceil_div(tot_slots, num_splits);

  const int32_t split_start = split * split_slots;
  const int32_t split_end = min((split + 1) * split_slots, tot_slots);

  const bool is_active_split = (split_start < tot_slots);

  if (!is_active_split) return;

  // Adjust the pointer for the block_table for this batch.
  // If seq_starts is provided, compute an offset based on it
  const int32_t batch_offset = bid * block_table_stride;
  int32_t offset = split_start;
  if (seq_starts != nullptr) {
    offset += seq_starts[bid];
  }
  int32_t offset_div = offset / block_size;
  offset = offset % block_size;
  const int32_t* batch_block_table = block_table + batch_offset;

  // Adjust dst pointer based on the cumulative sequence lengths.
  dst += seq_start * dst_entry_stride;

  auto copy_entry = [&](const scalar_t* __restrict__ _src,
                        scalar_t* __restrict__ _dst) {
    for (int i = threadIdx.x; i < entry_size; i += blockDim.x)
      _dst[i] = _src[i];
  };

  for (int pid = split_start; pid < split_end; ++pid) {
    auto block_id = batch_block_table[offset_div];
    auto block_start_ptr = src_cache + block_id * cache_block_stride;
    auto block_dst_ptr = dst + pid * dst_entry_stride;
    copy_entry(block_start_ptr + offset * cache_entry_stride, block_dst_ptr);
    offset += 1;
    // bump to next block
    if (offset == block_size) {
      offset_div += 1;
      offset = 0;
    }
  }
}
}  // namespace vllm

// Macro to dispatch the kernel based on the data type.
#define CALL_CP_GATHER_CACHE(CPY_DTYPE)                                 \
  vllm::cp_gather_cache<CPY_DTYPE><<<grid, block, 0, stream>>>(         \
      reinterpret_cast<CPY_DTYPE*>(src_cache.data_ptr()),               \
      reinterpret_cast<CPY_DTYPE*>(dst.data_ptr()),                     \
      block_table.data_ptr<int32_t>(), cu_seq_lens.data_ptr<int32_t>(), \
      block_size, entry_size, block_table_stride, cache_block_stride,   \
      cache_entry_stride, dst_entry_stride, seq_starts_ptr);

// Gather sequences from the cache into the destination tensor.
//  - cu_seq_lens contains the cumulative sequence lengths for each batch
//  - block_table contains the cache block indices for each sequence
//  - Optionally, seq_starts (if provided) offsets the starting slot index by
//  seq_starts[bid]
void cp_gather_cache(
    torch::Tensor const& src_cache,    // [NUM_BLOCKS, BLOCK_SIZE, ENTRIES...]
    torch::Tensor const& dst,          // [TOT_TOKENS, ENTRIES...]
    torch::Tensor const& block_table,  // [BATCH, BLOCK_INDICES]
    torch::Tensor const& cu_seq_lens,  // [BATCH+1]
    int64_t batch_size,
    std::optional<torch::Tensor> seq_starts = std::nullopt) {
  at::cuda::OptionalCUDAGuard device_guard(src_cache.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int32_t block_size = src_cache.size(1);
  int32_t entry_size = src_cache.flatten(2, -1).size(2);

  TORCH_CHECK(block_table.dtype() == torch::kInt32,
              "block_table must be int32");
  TORCH_CHECK(cu_seq_lens.dtype() == torch::kInt32,
              "cu_seq_lens must be int32");
  if (seq_starts.has_value()) {
    TORCH_CHECK(seq_starts.value().dtype() == torch::kInt32,
                "seq_starts must be int32");
  }

  TORCH_CHECK(src_cache.device() == dst.device(),
              "src_cache and dst must be on the same device");
  TORCH_CHECK(src_cache.device() == block_table.device(),
              "src_cache and block_table must be on the same device");
  TORCH_CHECK(src_cache.device() == cu_seq_lens.device(),
              "src_cache and cu_seq_lens must be on the same device");
  if (seq_starts.has_value()) {
    TORCH_CHECK(src_cache.device() == seq_starts.value().device(),
                "src_cache and seq_starts must be on the same device");
  }

  int64_t block_table_stride = block_table.stride(0);
  int64_t cache_block_stride = src_cache.stride(0);
  int64_t cache_entry_stride = src_cache.stride(1);
  int64_t dst_entry_stride = dst.stride(0);

  // Decide on the number of splits based on the batch size.
  int num_splits = batch_size > 128 ? 2 : batch_size > 64 ? 4 : 16;
  dim3 grid(batch_size, num_splits);
  dim3 block(1024);

  TORCH_CHECK(src_cache.dtype() == dst.dtype(),
              "src_cache and dst must have the same dtype");

  const int dtype_bits = src_cache.element_size() * 8;
  const int32_t* seq_starts_ptr =
      seq_starts.has_value() ? seq_starts.value().data_ptr<int32_t>() : nullptr;

  if (dtype_bits == 32) {
    CALL_CP_GATHER_CACHE(uint32_t);
  } else if (dtype_bits == 16) {
    CALL_CP_GATHER_CACHE(uint16_t);
  } else if (dtype_bits == 8) {
    CALL_CP_GATHER_CACHE(uint8_t);
  } else {
    TORCH_CHECK(false, "Unsupported data type width: ", dtype_bits);
  }
}

void cp_gather_and_upconvert_fp8_kv_cache(
    torch::Tensor const& src_cache,         // [NUM_BLOCKS, BLOCK_SIZE, 656]
    torch::Tensor const& dst,               // [TOT_TOKENS, 576]
    torch::Tensor const& block_table,       // [BATCH, BLOCK_INDICES]
    torch::Tensor const& seq_lens,          // [BATCH]
    torch::Tensor const& workspace_starts,  // [BATCH]
    int64_t batch_size) {
  at::cuda::OptionalCUDAGuard device_guard(src_cache.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int32_t block_size = src_cache.size(1);
  int32_t head_dim = dst.size(1);

  TORCH_CHECK(block_table.dtype() == torch::kInt32,
              "block_table must be int32");
  TORCH_CHECK(seq_lens.dtype() == torch::kInt32, "seq_lens must be int32");
  TORCH_CHECK(workspace_starts.dtype() == torch::kInt32,
              "workspace_starts must be int32");

  TORCH_CHECK(src_cache.device() == dst.device(),
              "src_cache and dst must be on the same device");
  TORCH_CHECK(src_cache.device() == block_table.device(),
              "src_cache and block_table must be on the same device");
  TORCH_CHECK(src_cache.device() == seq_lens.device(),
              "src_cache and seq_lens must be on the same device");
  TORCH_CHECK(src_cache.device() == workspace_starts.device(),
              "src_cache and workspace_starts must be on the same device");

  TORCH_CHECK(src_cache.dtype() == torch::kUInt8, "src_cache must be uint8");
  TORCH_CHECK(dst.dtype() == torch::kBFloat16, "dst must be bfloat16");
  TORCH_CHECK(head_dim == 576, "head_dim must be 576 for MLA");

  int64_t block_table_stride = block_table.stride(0);
  int64_t cache_block_stride = src_cache.stride(0);
  int64_t cache_entry_stride = src_cache.stride(1);
  int64_t dst_entry_stride = dst.stride(0);

  // Decide on the number of splits based on the batch size
  int num_splits = batch_size > 128 ? 2 : batch_size > 64 ? 4 : 16;
  dim3 grid(batch_size, num_splits);
  dim3 block(576);

  vllm::cp_gather_and_upconvert_fp8_kv_cache<<<grid, block, 0, stream>>>(
      src_cache.data_ptr<uint8_t>(),
      reinterpret_cast<__nv_bfloat16*>(dst.data_ptr()),
      block_table.data_ptr<int32_t>(), seq_lens.data_ptr<int32_t>(),
      workspace_starts.data_ptr<int32_t>(), block_size, head_dim,
      block_table_stride, cache_block_stride, cache_entry_stride,
      dst_entry_stride);
}

// Macro to dispatch the kernel based on the data type.
#define CALL_INDEXER_K_QUANT_AND_CACHE(KV_T, CACHE_T, KV_DTYPE)         \
  vllm::indexer_k_quant_and_cache_kernel<KV_T, CACHE_T, KV_DTYPE>       \
      <<<grid, block, 0, stream>>>(                                     \
          reinterpret_cast<KV_T*>(k.data_ptr()),                        \
          reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()),              \
          slot_mapping.data_ptr<int64_t>(), head_dim, quant_block_size, \
          cache_block_size, cache_stride, use_ue8m0);

void indexer_k_quant_and_cache(
    torch::Tensor& k,             // [num_tokens, head_dim]
    torch::Tensor& kv_cache,      // [num_blocks, block_size, cache_stride]
    torch::Tensor& slot_mapping,  // [num_tokens]
    int64_t quant_block_size,     // quantization block size
    const std::string& scale_fmt) {
  int num_tokens = k.size(0);
  int head_dim = k.size(1);
  int cache_block_size = kv_cache.size(1);
  int cache_stride = kv_cache.size(2);
  bool use_ue8m0 = scale_fmt == "ue8m0";

  TORCH_CHECK(k.device() == kv_cache.device(),
              "k and kv_cache must be on the same device");
  TORCH_CHECK(k.device() == slot_mapping.device(),
              "k and slot_mapping must be on the same device");
  TORCH_CHECK(head_dim % quant_block_size == 0,
              "head_dim must be divisible by quant_block_size");

  constexpr int vec_size = 4;
  dim3 grid(num_tokens, (head_dim + quant_block_size * vec_size - 1) /
                            (quant_block_size * vec_size));
  dim3 block(32, vec_size);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(k));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_BY_KV_CACHE_DTYPE(k.dtype(), "fp8_e4m3",
                             CALL_INDEXER_K_QUANT_AND_CACHE);
}

// Macro to dispatch the kernel based on the data amount.
#define CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(BLOCK_Y_SIZE)                  \
  vllm::cp_gather_indexer_k_quant_cache_kernel<BLOCK_Y_SIZE>                \
      <<<dim3((num_tokens + BLOCK_Y_SIZE - 1) / BLOCK_Y_SIZE,               \
              (head_dim + 8 * vec_size - 1) / (8 * vec_size)),              \
         dim3(8, BLOCK_Y_SIZE), 0, stream>>>(                               \
          reinterpret_cast<char*>(kv_cache.data_ptr()),                     \
          reinterpret_cast<char*>(dst_k.data_ptr()),                        \
          reinterpret_cast<char*>(dst_scale.data_ptr()),                    \
          block_table.data_ptr<int32_t>(), cu_seq_lens.data_ptr<int32_t>(), \
          batch_size, dst_k.stride(0), dst_k.size(1), kv_cache.stride(0),   \
          kv_cache.stride(1), kv_cache.size(1), block_table.size(1),        \
          num_tokens, quant_block_size);

void cp_gather_indexer_k_quant_cache(
    const torch::Tensor& kv_cache,  // [num_blocks, block_size, cache_stride]
    torch::Tensor& dst_k,           // [num_tokens, head_dim]
    torch::Tensor& dst_scale,  // [num_tokens, head_dim / quant_block_size * 4]
    const torch::Tensor& block_table,  // [batch_size, num_blocks]
    const torch::Tensor& cu_seq_lens   // [batch_size + 1]
) {
  int batch_size = block_table.size(0);
  int num_tokens = dst_k.size(0);
  int head_dim = dst_k.size(1);
  int quant_block_size = head_dim * 4 / dst_scale.size(1);

  TORCH_CHECK(kv_cache.device() == dst_k.device(),
              "kv_cache and dst_k must be on the same device");
  TORCH_CHECK(kv_cache.device() == dst_scale.device(),
              "kv_cache and dst_scale must be on the same device");
  TORCH_CHECK(kv_cache.device() == block_table.device(),
              "kv_cache and block_table must be on the same device");
  TORCH_CHECK(kv_cache.device() == cu_seq_lens.device(),
              "kv_cache and cu_seq_lens must be on the same device");
  TORCH_CHECK(head_dim % quant_block_size == 0,
              "head_dim must be divisible by quant_block_size");

  constexpr int vec_size = 16;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(kv_cache));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (num_tokens < 32) {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(1);
  } else if (num_tokens < 64) {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(2);
  } else if (num_tokens < 128) {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(4);
  } else if (num_tokens < 256) {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(8);
  } else if (num_tokens < 512) {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(16);
  } else {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(32);
  }
}

