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
 
  
#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
    }                                                                          \
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
 void bench_merge_state_zeroCopy(nvbench::state& state) {
   cudaSetDevice(0);
   const auto seq_len = state.get_int64("seq_len");
   const auto num_heads = state.get_int64("num_heads");
   const auto head_dim = state.get_int64("head_dim");
 
   std::vector<T> V_a_host_(seq_len  * num_heads * head_dim);
   std::vector<float> S_a_host_(seq_len  * num_heads);
 
   std::vector<T> V_b_host_(seq_len  * num_heads * head_dim);
   std::vector<float> S_b_host_(seq_len  * num_heads);
 
   utils::vec_normal_(V_a_host_);
   utils::vec_uniform_(S_a_host_, 5, 10);
   utils::vec_normal_(V_b_host_);
   utils::vec_uniform_(S_b_host_, 5, 10);
    
   thrust::host_vector<T> V_a_host(V_a_host_);
   thrust::host_vector<float> S_a_host(S_a_host_);
   thrust::host_vector<T> V_b_host(V_b_host_);
   thrust::host_vector<float> S_b_host(S_b_host_);
   CHECK(cudaHostRegister(V_a_host.data(), sizeof(T) * seq_len  * num_heads * head_dim, cudaHostRegisterDefault));
   CHECK(cudaHostRegister(S_a_host.data(), sizeof(float) * seq_len  * num_heads, cudaHostRegisterDefault));
   CHECK(cudaHostRegister(V_b_host.data(), sizeof(T) * seq_len  * num_heads * head_dim, cudaHostRegisterDefault));
   CHECK(cudaHostRegister(S_b_host.data(), sizeof(float) * seq_len  * num_heads, cudaHostRegisterDefault));
   
   thrust::device_vector<T> V_merged(seq_len * num_heads * head_dim);
   thrust::device_vector<float> S_merged(seq_len * num_heads);
 
   //  state.add_global_memory_reads<T>(V_a_host.size(), "Read");
   state.add_global_memory_reads<T>(V_merged.size(), "Write");
   state.add_global_memory_writes<T>(S_merged.size(), "Write");
 
   state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
     timer.start();
     cudaError_t status = MergeState(
        V_a_host.data(), S_a_host.data(),
        V_b_host.data(), S_b_host.data(),
        thrust::raw_pointer_cast(V_merged.data()), thrust::raw_pointer_cast(S_merged.data()),
        seq_len, num_heads, head_dim);
     timer.stop();
   });
   CHECK(cudaDeviceSynchronize());
   CHECK(cudaHostUnregister(V_a_host.data() ));
   CHECK(cudaHostUnregister(S_a_host.data() ));
   CHECK(cudaHostUnregister(V_b_host.data() ));
   CHECK(cudaHostUnregister(S_b_host.data() ));
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
 void bench_merge_state_InPlace_zeroCopy(nvbench::state& state) {
    cudaSetDevice(0);
    const auto seq_len = state.get_int64("seq_len");
    const auto num_heads = state.get_int64("num_heads");
    const auto head_dim = state.get_int64("head_dim");

    std::vector<T> V_a_host_(seq_len  * num_heads * head_dim);
    std::vector<float> S_a_host_(seq_len  * num_heads);

    std::vector<T> V_other_host(seq_len  * num_heads * head_dim);
    std::vector<float> S_other_host(seq_len  * num_heads);

    utils::vec_normal_(V_a_host_);
    utils::vec_uniform_(S_a_host_, 5, 10);
    utils::vec_normal_(V_other_host);
    utils::vec_uniform_(S_other_host, 5, 10);
      
    thrust::host_vector<T> V_a_host(V_a_host_);
    thrust::host_vector<float> S_a_host(S_a_host_);
    CHECK(cudaHostRegister(V_a_host.data(), sizeof(T) * seq_len  * num_heads * head_dim, cudaHostRegisterDefault));
    CHECK(cudaHostRegister(S_a_host.data(), sizeof(float) * seq_len  * num_heads, cudaHostRegisterDefault));
    thrust::device_vector<T> V_other_device(V_other_host);
    thrust::device_vector<float> S_other_device(S_other_host);


    //  state.add_global_memory_reads<T>(V_a_host.size(), "Read");
    state.add_global_memory_writes<T>(V_other_device.size(), "Write");
    state.add_global_memory_writes<T>(S_other_device.size(), "Write");

    // state.exec(nvbench::exec_tag::sync, <KernelLauncher>); // Safe
    state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      timer.start();
      cudaError_t status = MergeStateInPlace(
          V_a_host.data(), S_a_host.data(),
          thrust::raw_pointer_cast(V_other_device.data()), thrust::raw_pointer_cast(S_other_device.data()),
          seq_len, num_heads, head_dim);
      timer.stop();
   });
   CHECK(cudaDeviceSynchronize());
   CHECK(cudaHostUnregister(V_a_host.data() ));
   CHECK(cudaHostUnregister(S_a_host.data() ));
 }
   


 #define STR_HELPER(x) #x
 #define STR(x) STR_HELPER(x)
  
 
 #define BENCH_FLASHINFER_MERGE_STATE_KERNELS(T)                            \
 auto bench_flashinfer_merge_state_zeroCopy_##T##_ = bench_merge_state_zeroCopy<T>; \
 NVBENCH_BENCH(bench_flashinfer_merge_state_zeroCopy_##T##_)                \
     .set_name("flashinfer_merge_state_zeroCopy_" STR(T))                   \
     .add_int64_axis("seq_len", {1, 2, 4, 8, 16, 32, 64, 128, 256}) \
     .add_int64_axis("num_heads", {32,56})                             \
     .add_int64_axis("head_dim", {64,128})
 
 
 
 
 #define BENCH_FLASHINFER_MERGE_STATE_InPlace_KERNELS(T)                            \
 auto bench_flashinfer_merge_state_InPlace_zeroCopy_##T##_ = bench_merge_state_InPlace_zeroCopy<T>; \
 NVBENCH_BENCH(bench_flashinfer_merge_state_InPlace_zeroCopy_##T##_)                \
     .set_name("flashinfer_merge_state_InPlace_zeroCopy_" STR(T))                   \
     .add_int64_axis("seq_len", {1, 2, 4, 8, 16, 32, 64, 128, 256}) \
     .add_int64_axis("num_heads", {32,56})                             \
     .add_int64_axis("head_dim", {64,128})


BENCH_FLASHINFER_MERGE_STATE_KERNELS(half); 
BENCH_FLASHINFER_MERGE_STATE_InPlace_KERNELS(half); 