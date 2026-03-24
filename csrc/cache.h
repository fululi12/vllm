#pragma once

#include <torch/all.h>
#include <c10/util/Optional.h>

#include <map>
#include <vector>

void swap_blocks(torch::Tensor& src, torch::Tensor& dst,
                 const torch::Tensor& block_mapping);

void reshape_and_cache(torch::Tensor& key, torch::Tensor& value,
                       torch::Tensor& key_cache, torch::Tensor& value_cache,
                       torch::Tensor& slot_mapping,
                       const std::string& kv_cache_dtype,
                       torch::Tensor& k_scale, torch::Tensor& v_scale);

void reshape_and_cache_flash(torch::Tensor& key, torch::Tensor& value,
                             torch::Tensor& key_cache,
                             torch::Tensor& value_cache,
                             torch::Tensor& slot_mapping,
                             const std::string& kv_cache_dtype,
                             torch::Tensor& k_scale, torch::Tensor& v_scale);

void reshape_and_cache_flash_with_pertoken_quant(
    torch::Tensor& key,              // [num_tokens, num_heads, head_size]
    torch::Tensor& value,            // [num_tokens, num_heads, head_size]
    torch::Tensor& key_cache,        // [num_blocks, block_size, num_heads,
                                     //  head_size/2]
    torch::Tensor& value_cache,      // [num_blocks, block_size, num_heads,
                                     //  head_size/2]
    torch::Tensor& k_dequant_scales, // [num_heads * num_k_blocks, max_kv_tokens]
    torch::Tensor& v_dequant_scales, // [num_heads * num_v_blocks, max_kv_tokens]
    torch::Tensor& slot_mapping,     // [num_tokens]
    const std::string& kv_cache_dtype);

void reshape_and_cache_flash_fp4_per_channel_k_per_token_v(
    torch::Tensor& key,              // [num_tokens, num_heads, head_size]
    torch::Tensor& value,            // [num_tokens, num_heads, head_size]
    torch::Tensor& key_cache,        // [num_blocks, block_size, num_heads,
                                     //  head_size/2]
    torch::Tensor& value_cache,      // [num_blocks, block_size, num_heads,
                                     //  head_size/2]
    torch::Tensor& k_channel_scales, // [num_heads, head_size] static per-channel
    torch::Tensor& v_dequant_scales, // [num_heads, max_kv_tokens] per-token
    torch::Tensor& slot_mapping,     // [num_tokens]
    const std::string& kv_cache_dtype);

void reshape_and_cache_flash_fp4_mxfp4(
    torch::Tensor& key,              // [num_tokens, num_heads, head_size]
    torch::Tensor& value,            // [num_tokens, num_heads, head_size]
    torch::Tensor& key_cache,        // [num_blocks, block_size, num_heads,
                                     //  head_size/2]
    torch::Tensor& value_cache,      // [num_blocks, block_size, num_heads,
                                     //  head_size/2]
    torch::Tensor& k_e8m0_scales,    // [num_heads*num_k_blocks, total] uint8
    torch::Tensor& v_e8m0_scales,    // [num_heads*num_v_blocks, total] uint8
    torch::Tensor& slot_mapping,     // [num_tokens]
    const std::string& kv_cache_dtype);

void reshape_and_cache_flash_fp4_nvfp4(
    torch::Tensor& key,              // [num_tokens, num_heads, head_size]
    torch::Tensor& value,            // [num_tokens, num_heads, head_size]
    torch::Tensor& key_cache,        // [num_blocks, block_size, num_heads,
                                     //  head_size/2]
    torch::Tensor& value_cache,      // [num_blocks, block_size, num_heads,
                                     //  head_size/2]
    torch::Tensor& k_fp8_scales,     // [num_heads*num_k_blocks, total] uint8
    torch::Tensor& v_fp8_scales,     // [num_heads*num_v_blocks, total] uint8
    torch::Tensor& slot_mapping,     // [num_tokens]
    const std::string& kv_cache_dtype);

void reshape_and_cache_flash_fp4_amxfp4(
    torch::Tensor& key,              // [num_tokens, num_heads, head_size]
    torch::Tensor& value,            // [num_tokens, num_heads, head_size]
    torch::Tensor& key_cache,        // [num_blocks, block_size, num_heads,
                                     //  head_size/2]
    torch::Tensor& value_cache,      // [num_blocks, block_size, num_heads,
                                     //  head_size/2]
    torch::Tensor& k_e8m0_scales,    // [num_heads*num_k_blocks, total] uint8
    torch::Tensor& v_e8m0_scales,    // [num_heads*num_v_blocks, total] uint8
    torch::Tensor& k_bm_indices,     // [num_heads*num_k_blocks, total] uint8
    torch::Tensor& v_bm_indices,     // [num_heads*num_v_blocks, total] uint8
    torch::Tensor& slot_mapping,     // [num_tokens]
    const std::string& kv_cache_dtype);

void concat_and_cache_mla(torch::Tensor& kv_c, torch::Tensor& k_pe,
                          torch::Tensor& kv_cache, torch::Tensor& slot_mapping,
                          const std::string& kv_cache_dtype,
                          torch::Tensor& scale);

// NOTE: k_pe and kv_c order is flipped compared to concat_and_cache_mla
void concat_and_cache_mla_rope_fused(
    torch::Tensor& positions, torch::Tensor& q_pe, torch::Tensor& k_pe,
    torch::Tensor& kv_c, torch::Tensor& rope_cos_sin_cache, bool rope_is_neox,
    torch::Tensor& kv_cache_slot_mapping, torch::Tensor& kv_cache,
    const std::string& kv_cache_dtype, torch::Tensor& kv_cache_quant_scale);

// Just for unittest
void convert_fp8(torch::Tensor& dst_cache, torch::Tensor& src_cache,
                 const double scale, const std::string& kv_cache_dtype);

void gather_and_maybe_dequant_cache(
    torch::Tensor const& src_cache,     // [NUM_BLOCKS, BLOCK_SIZE, ENTRIES...]
    torch::Tensor const& dst,           // [TOT_TOKENS, ENTRIES...]
    torch::Tensor const& block_table,   // [BATCH, BLOCK_INDICES]
    torch::Tensor const& cu_seq_lens,   // [BATCH+1]
    torch::Tensor const& token_to_seq,  // [MAX_TOKEN_ACROSS_CHUNKS]
    int64_t num_tokens, const std::string& kv_cache_dtype,
    torch::Tensor const& scale,
    std::optional<torch::Tensor> seq_starts = std::nullopt);

// TODO(hc): cp_gather_cache need support scaled kvcahe in the future.
void cp_gather_cache(
    torch::Tensor const& src_cache,    // [NUM_BLOCKS, BLOCK_SIZE, ENTRIES...]
    torch::Tensor const& dst,          // [TOT_TOKENS, ENTRIES...]
    torch::Tensor const& block_table,  // [BATCH, BLOCK_INDICES]
    torch::Tensor const& cu_seq_lens,  // [BATCH+1]
    int64_t batch_size, std::optional<torch::Tensor> seq_starts = std::nullopt);

// Gather and upconvert FP8 KV cache to BF16 workspace
void cp_gather_and_upconvert_fp8_kv_cache(
    torch::Tensor const& src_cache,         // [NUM_BLOCKS, BLOCK_SIZE, 656]
    torch::Tensor const& dst,               // [TOT_TOKENS, 576]
    torch::Tensor const& block_table,       // [BATCH, BLOCK_INDICES]
    torch::Tensor const& seq_lens,          // [BATCH]
    torch::Tensor const& workspace_starts,  // [BATCH]
    int64_t batch_size);

// Indexer K quantization and cache function
void indexer_k_quant_and_cache(
    torch::Tensor& k,             // [num_tokens, head_dim]
    torch::Tensor& kv_cache,      // [num_blocks, block_size, cache_stride]
    torch::Tensor& slot_mapping,  // [num_tokens]
    int64_t quant_block_size,     // quantization block size
    const std::string& scale_fmt);

// Extract function to gather quantized K cache
void cp_gather_indexer_k_quant_cache(
    const torch::Tensor& kv_cache,  // [num_blocks, block_size, cache_stride]
    torch::Tensor& dst_k,           // [num_tokens, head_dim]
    torch::Tensor& dst_scale,  // [num_tokens, head_dim / quant_block_size * 4]
    const torch::Tensor& block_table,   // [batch_size, num_blocks]
    const torch::Tensor& cu_seq_lens);  // [batch_size + 1]

