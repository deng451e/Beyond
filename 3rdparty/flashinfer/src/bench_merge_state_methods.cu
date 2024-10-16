/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <thrust/device_vector.h>

#include <cstddef>
#include <flashinfer/attention/cascade.cuh>
#include <nvbench/nvbench.cuh>

#include "flashinfer_ops.cuh"
#include "utils.h"

using namespace flashinfer;

 

template <typename T>
void bench_merge_states(nvbench::state& state) {
  cudaSetDevice(0);
  const auto num_index_sets = state.get_int64("num_index_sets");
  const auto seq_len = state.get_int64("seq_len");
  const auto num_heads = state.get_int64("num_heads");
  const auto head_dim = state.get_int64("head_dim");

  std::vector<T> V_a_host(seq_len * num_index_sets * num_heads * head_dim);
  std::vector<float> S_a_host(seq_len * num_index_sets * num_heads);

  utils::vec_normal_(V_a_host);
  utils::vec_uniform_(S_a_host, 5, 10);

  thrust::device_vector<T> V_a_device(V_a_host);
  thrust::device_vector<float> S_a_device(S_a_host);
  thrust::device_vector<T> V_merged(seq_len * num_heads * head_dim);
  thrust::device_vector<float> S_merged(seq_len * num_heads);

  state.add_global_memory_reads<T>(V_a_host.size(), "Read");
  state.add_global_memory_writes<T>(V_merged.size(), "Write");

  state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
    timer.start();
    cudaError_t status = MergeStates(
        thrust::raw_pointer_cast(V_a_device.data()), thrust::raw_pointer_cast(S_a_device.data()),
        thrust::raw_pointer_cast(V_merged.data()), thrust::raw_pointer_cast(S_merged.data()),
        num_index_sets, seq_len, num_heads, head_dim);
    timer.stop();
  });
}



/*!
 * \brief Merge the self-attention state with another state in place.
 * \tparam DType The data type of v and v_other.
 * \param v The partial v to be updated in-place. (n, h, d)
 * \param s The logsumexp value to be updated in-place. (n, h)
 * \param v_other The other v to be merged. (n, h, d)
 * \param s_other The other logsumexp value to be merged. (n, h)
 * \param seq_len The sequence length.
 * \param num_heads The number of heads of v and v_other.
 * \param head_dim The dimension of each head.
 * \param mask Optional mask of whether to merge given sequences or not. (n)
 * \param stream The CUDA stream to execute the kernel.
 * \return status Indicates whether CUDA calls are successful
 * \note Both s and s_other are logsumexp values with base 2.
 */

template <typename T>
void bench_merge_state(nvbench::state& state) {
  cudaSetDevice(0);
  const auto seq_len = state.get_int64("seq_len");
  const auto num_heads = state.get_int64("num_heads");
  const auto head_dim = state.get_int64("head_dim");

  std::vector<T> V_a_host(seq_len  * num_heads * head_dim);
  std::vector<float> S_a_host(seq_len  * num_heads);

  std::vector<T> V_b_host(seq_len  * num_heads * head_dim);
  std::vector<float> S_b_host(seq_len  * num_heads);

  utils::vec_normal_(V_a_host);
  utils::vec_uniform_(S_a_host, 5, 10);
  utils::vec_normal_(V_b_host);
  utils::vec_uniform_(S_b_host, 5, 10);

  thrust::device_vector<T> V_a_device(V_a_host);
  thrust::device_vector<float> S_a_device(S_a_host);
  thrust::device_vector<T> V_b_device(V_b_host);
  thrust::device_vector<float> S_b_device(S_b_host);
  thrust::device_vector<T> V_merged(seq_len * num_heads * head_dim);
  thrust::device_vector<float> S_merged(seq_len * num_heads);

  state.add_global_memory_reads<T>(V_a_host.size(), "Read");
  state.add_global_memory_writes<T>(V_merged.size(), "Write");

  state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
    timer.start();
    cudaError_t status = MergeState(
        thrust::raw_pointer_cast(V_a_device.data()), thrust::raw_pointer_cast(S_a_device.data()),
        thrust::raw_pointer_cast(V_b_device.data()), thrust::raw_pointer_cast(S_b_device.data()),
        thrust::raw_pointer_cast(V_merged.data()), thrust::raw_pointer_cast(S_merged.data()),
        seq_len, num_heads, head_dim);
    timer.stop();
  });
}
  



/*!
 * \brief Merge the self-attention state with another state in place.
 * \tparam DType The data type of v and v_other.
 * \param v The partial v to be updated in-place. (n, h, d)
 * \param s The logsumexp value to be updated in-place. (n, h)
 * \param v_other The other v to be merged. (n, h, d)
 * \param s_other The other logsumexp value to be merged. (n, h)
 * \param seq_len The sequence length.
 * \param num_heads The number of heads of v and v_other.
 * \param head_dim The dimension of each head.
 * \param mask Optional mask of whether to merge given sequences or not. (n)
 * \param stream The CUDA stream to execute the kernel.
 * \return status Indicates whether CUDA calls are successful
 * \note Both s and s_other are logsumexp values with base 2.
 */
template <typename T>
void bench_merge_state_InPlace(nvbench::state& state) {
  cudaSetDevice(0);
  const auto seq_len = state.get_int64("seq_len");
  const auto num_heads = state.get_int64("num_heads");
  const auto head_dim = state.get_int64("head_dim");

  std::vector<T> V_a_host(seq_len  * num_heads * head_dim);
  std::vector<float> S_a_host(seq_len  * num_heads);

  std::vector<T> V_other_host(seq_len  * num_heads * head_dim);
  std::vector<float> S_other_host(seq_len  * num_heads);

  utils::vec_normal_(V_a_host);
  utils::vec_uniform_(S_a_host, 5, 10);
  utils::vec_normal_(V_other_host);
  utils::vec_uniform_(S_other_host, 5, 10);

  thrust::device_vector<T> V_a_device(V_a_host);
  thrust::device_vector<float> S_a_device(S_a_host);
  thrust::device_vector<T> V_other_device(V_other_host);
  thrust::device_vector<float> S_other_device(S_other_host);
 

  state.add_global_memory_reads<T>(V_a_host.size(), "Read");
  state.add_global_memory_writes<T>(V_other_device.size(), "Write");

  state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
    timer.start();
    cudaError_t status = MergeStateInPlace(
        thrust::raw_pointer_cast(V_a_device.data()), thrust::raw_pointer_cast(S_a_device.data()),
        thrust::raw_pointer_cast(V_other_device.data()), thrust::raw_pointer_cast(S_other_device.data()),
        seq_len, num_heads, head_dim);
    timer.stop();
  });
}
  
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)


#define BENCH_FLASHINFER_MERGE_STATES_KERNELS(T)                            \
  auto bench_flashinfer_merge_states_##T##_ = bench_merge_states<T>; \
  NVBENCH_BENCH(bench_flashinfer_merge_states_##T##_)                \
      .set_name("flashinfer_merge_states_" STR(T))                   \
      .add_int64_axis("num_index_sets", {2, 16, 64, 128, 256})       \
      .add_int64_axis("seq_len", {1, 2, 4, 8, 16, 32, 64, 128, 256}) \
      .add_int64_axis("num_heads", {32})                             \
      .add_int64_axis("head_dim", {128})



#define BENCH_FLASHINFER_MERGE_STATE_KERNELS(T)                            \
auto bench_flashinfer_merge_state_##T##_ = bench_merge_state<T>; \
NVBENCH_BENCH(bench_flashinfer_merge_state_##T##_)                \
    .set_name("flashinfer_merge_state_" STR(T))                   \
    .add_int64_axis("seq_len", {1, 2, 4, 8, 16, 32, 64, 128, 256}) \
    .add_int64_axis("num_heads", {32,56})                             \
    .add_int64_axis("head_dim", {64,128})




#define BENCH_FLASHINFER_MERGE_STATE_InPlace_KERNELS(T)                            \
auto bench_flashinfer_merge_state_InPlace_##T##_ = bench_merge_state_InPlace<T>; \
NVBENCH_BENCH(bench_flashinfer_merge_state_InPlace_##T##_)                \
    .set_name("flashinfer_merge_state_InPlace_" STR(T))                   \
    .add_int64_axis("seq_len", {1, 2, 4, 8, 16, 32, 64, 128, 256}) \
    .add_int64_axis("num_heads", {32,56})                             \
    .add_int64_axis("head_dim", {64,128})


  
// BENCH_FLASHINFER_MERGE_STATES_KERNELS(half); 
BENCH_FLASHINFER_MERGE_STATE_KERNELS(half); 
BENCH_FLASHINFER_MERGE_STATE_InPlace_KERNELS(half); 