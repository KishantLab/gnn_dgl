/**
 *  Copyright (c) 2021 by Contributors
 * @file array/cuda/rowwise_sampling.cu
 * @brief uniform rowwise sampling
 */

#include <curand_kernel.h>
#include <dgl/random.h>
#include <dgl/runtime/device_api.h>
#include <dgl/runtime/tensordispatch.h>

#include <cstdint>
#include <iostream>
#include <numeric>

#include "../../array/cuda/atomic.cuh"
#include "../../runtime/cuda/cuda_common.h"
#include "./dgl_cub.cuh"
#include "./utils.h"

#include "../../array/cuda/global_array.h"

using namespace dgl::cuda;
using namespace dgl::aten::cuda;
using TensorDispatcher = dgl::runtime::TensorDispatcher;

namespace dgl {
namespace aten {
namespace impl {

namespace {

constexpr int BLOCK_SIZE = 512;
int flag = 0;
int64_t* d_part_array;
float sampling_time = 0.0;
/**
 * @brief Compute the size of each row in the sampled CSR, without replacement.
 *
 * @tparam IdType The type of node and edge indexes.
 * @param num_picks The number of non-zero entries to pick per row.
 * @param num_rows The number of rows to pick.
 * @param in_rows The set of rows to pick.
 * @param in_ptr The index where each row's edges start.
 * @param out_deg The size of each row in the sampled matrix, as indexed by
 * `in_rows` (output).
 */
template <typename IdType>
__global__ void _CSRRowWiseSampleDegreeKernel(
  const int64_t num_picks, const int64_t num_rows,
  const IdType* const in_rows, const IdType* const in_ptr,
  IdType* const out_deg) {
  const int tIdx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tIdx < num_rows) {
    const int in_row = in_rows[tIdx];
    const int out_row = tIdx;
    out_deg[out_row] = min(
      static_cast<IdType>(num_picks), in_ptr[in_row + 1] - in_ptr[in_row]);

    if (out_row == num_rows - 1) {
      // make the prefixsum work
      out_deg[num_rows] = 0;
    }
  }
}

/**
 * @brief Compute the size of each row in the sampled CSR, with replacement.
 *
 * @tparam IdType The type of node and edge indexes.
 * @param num_picks The number of non-zero entries to pick per row.
 * @param num_rows The number of rows to pick.
 * @param in_rows The set of rows to pick.
 * @param in_ptr The index where each row's edges start.
 * @param out_deg The size of each row in the sampled matrix, as indexed by
 * `in_rows` (output).
 */
template <typename IdType>
__global__ void _CSRRowWiseSampleDegreeReplaceKernel(
  const int64_t num_picks, const int64_t num_rows,
  const IdType* const in_rows, const IdType* const in_ptr,
  IdType* const out_deg) {
  const int tIdx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tIdx < num_rows) {
    const int64_t in_row = in_rows[tIdx];
    const int64_t out_row = tIdx;

    if (in_ptr[in_row + 1] - in_ptr[in_row] == 0) {
      out_deg[out_row] = 0;
    } else {
      out_deg[out_row] = static_cast<IdType>(num_picks);
    }

    if (out_row == num_rows - 1) {
      // make the prefixsum work
      out_deg[num_rows] = 0;
    }
  }
}
/**
 * @brief Perform row-wise uniform sampling on a CSR matrix,
 * and generate a COO matrix, without replacement.
 *
 * @tparam IdType The ID type used for matrices.
 * @tparam TILE_SIZE The number of rows covered by each threadblock.
 * @param rand_seed The random seed to use.
 * @param num_picks The number of non-zeros to pick per row.
 * @param num_rows The number of rows to pick.
 * @param in_rows The set of rows to pick.
 * @param in_ptr The indptr array of the input CSR.
 * @param in_index The indices array of the input CSR.
 * @param data The data array of the input CSR.
 * @param out_ptr The offset to write each row to in the output COO.
 * @param out_rows The rows of the output COO (output).
 * @param out_cols The columns of the output COO (output).
 * @param out_idxs The data array of the output COO (output).
 */
template <typename IdType, int TILE_SIZE>
__global__ void _CSRRowWiseSampleUniformKernel(
  const uint64_t rand_seed, const int64_t num_picks, const int64_t num_rows,
  const IdType* const in_rows, const IdType* const in_ptr,
  const IdType* const in_index, const IdType* const data,
  const IdType* const out_ptr, IdType* const out_rows, IdType* const out_cols,
  IdType* const out_idxs) {
  // we assign one warp per row
  assert(blockDim.x == BLOCK_SIZE);

  int64_t out_row = blockIdx.x * TILE_SIZE;
  const int64_t last_row =
    min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);

  curandStatePhilox4_32_10_t rng;
  curand_init(rand_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);

  while (out_row < last_row) {
    const int64_t row = in_rows[out_row];
    const int64_t in_row_start = in_ptr[row];
    const int64_t deg = in_ptr[row + 1] - in_row_start;
    const int64_t out_row_start = out_ptr[out_row];

    if (deg <= num_picks) {
      // just copy row when there is not enough nodes to sample.
      for (int idx = threadIdx.x; idx < deg; idx += BLOCK_SIZE) {
        const IdType in_idx = in_row_start + idx;
        out_rows[out_row_start + idx] = row;
        out_cols[out_row_start + idx] = in_index[in_idx];
        out_idxs[out_row_start + idx] = data ? data[in_idx] : in_idx;
      }
    } else {
      // generate permutation list via reservoir algorithm
      for (int idx = threadIdx.x; idx < num_picks; idx += BLOCK_SIZE) {
        out_idxs[out_row_start + idx] = idx;
      }
      __syncthreads();

      for (int idx = num_picks + threadIdx.x; idx < deg; idx += BLOCK_SIZE) {
        const int num = curand(&rng) % (idx + 1);
        if (num < num_picks) {
          // use max so as to achieve the replacement order the serial
          // algorithm would have
          AtomicMax(out_idxs + out_row_start + num, idx);
        }
      }
      __syncthreads();

      // copy permutation over
      for (int idx = threadIdx.x; idx < num_picks; idx += BLOCK_SIZE) {
        const IdType perm_idx = out_idxs[out_row_start + idx] + in_row_start;
        out_rows[out_row_start + idx] = row;
        out_cols[out_row_start + idx] = in_index[perm_idx];
        out_idxs[out_row_start + idx] = data ? data[perm_idx] : perm_idx;
      }
    }
    out_row += 1;
  }
}



template <typename IdType, int TILE_SIZE>
__global__ void _CSRRowWiseSampleUniformKernel1(
  const uint64_t rand_seed, const int64_t num_picks, const int64_t num_rows,
  const IdType* const in_rows, const IdType* const in_ptr,
  const IdType* const in_index, const IdType* const data,
  const IdType* const out_ptr, IdType* const out_rows, IdType* const out_cols,
  IdType* const out_idxs, int64_t* d_part_array) {
  // we assign one warp per row
  assert(blockDim.x == BLOCK_SIZE);

  int64_t out_row = blockIdx.x * TILE_SIZE;
  const int64_t last_row =
    min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);

  curandStatePhilox4_32_10_t rng;
  curand_init(rand_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);

  // if (blockIdx.x==0 && threadIdx.x == 0)
  // {
  //   // printf("Kishan define function that passed d_part_array and 1st data is :%ld\n",d_part_array[0]);
  //   // printf("out_row: %ld\n", out_row);
  // }
  while (out_row < last_row) {
    const int64_t row = in_rows[out_row];
    const int64_t in_row_start = in_ptr[row];
    const int64_t deg = in_ptr[row + 1] - in_row_start;
    const int64_t out_row_start = out_ptr[out_row];

    if(blockIdx.x ==0 && threadIdx.x ==0)
    {
      // printf("row: %ld, in_row_start: %ld, deg: %ld, out_row_start: %ld\n", row, in_row_start, deg, out_row_start);
    }
    if (deg <= num_picks) {
      // just copy row when there is not enough nodes to sample.
      for (int idx = threadIdx.x; idx < deg; idx += BLOCK_SIZE) {
        const IdType in_idx = in_row_start + idx;
        out_rows[out_row_start + idx] = row;
        out_cols[out_row_start + idx] = in_index[in_idx];
        out_idxs[out_row_start + idx] = data ? data[in_idx] : in_idx;
      }
    } 
    else 
  {
      // Shared memory to store counts for each value and their index
      extern __shared__ int counts[];
      extern __shared__ int index_array[];
      // __shared__ int index_array[10];
      __shared__ int maxCount;
      __shared__ int maxValue;
      __shared__ int pointer_position;
      // int pointer_position;
      __shared__ int neighbours[BLOCK_SIZE];
      __shared__ int part_id[BLOCK_SIZE];

      // initilize 0 
      if ( deg < BLOCK_SIZE)
      {
        if(threadIdx.x < deg)
        {
          neighbours[threadIdx.x] = in_index[in_row_start + threadIdx.x];
          part_id[threadIdx.x] = d_part_array[in_index[in_row_start + threadIdx.x]];
        }
        if (threadIdx.x < num_picks) {
          counts[threadIdx.x] = 0;
        }

        if(threadIdx.x == 0)
        {
          maxValue = d_part_array[row];
        }
        __syncthreads();

        if( threadIdx.x < deg)
        {
          int val = part_id[threadIdx.x];
          if( val == maxValue)
          {
            atomicAdd(&counts[val], 1);
          }
          // atomicAdd(&counts[val], 1);
        }
        // __syncthreads();
        // Find maximum count and corresponding value
        if (threadIdx.x == 0) {
          maxCount = 0;
          pointer_position = 0; 
          // maxValue = d_part_array[row];
          maxCount = counts[maxValue];
          // printf("maxCount: %d, maxValue: %d, row: %ld\n",maxCount, maxValue, d_part_array[row]);
        }
        __syncthreads();
        // //   // // Find indices of maximum repeated value
        if (threadIdx.x == 0)
        {
          for(int kk = 0; kk < deg; kk++)
          // for(int kk = threadIdx.x; kk < deg; kk++)
          {
            if(part_id[kk] == maxValue && pointer_position < num_picks)
            // if(part_id[threadIdx.x] == maxValue && pointer_position < num_picks)
            // if(d_part_array[in_index[in_row_start + kk]] == maxValue && pointer_position < num_picks)
            {
              index_array[pointer_position] = threadIdx.x;
              pointer_position++;
            }
            if( pointer_position >= num_picks)
            {
              break;
            }
          }
        }
        // //   // generate permutation list via reservoir algorithm
        for (int idx = threadIdx.x; idx < num_picks; idx += BLOCK_SIZE) {
          out_idxs[out_row_start + idx] = idx;
        }
        __syncthreads();
        if (num_picks < maxCount)
        {
          for (int idx = threadIdx.x; idx < num_picks; idx += BLOCK_SIZE)
          {
            const IdType perm_idx = out_idxs[out_row_start + idx] + in_row_start;
            out_rows[out_row_start + idx] = row;
            out_cols[out_row_start + idx] = neighbours[index_array[idx]];
            out_idxs[out_row_start + idx] = data ? data[perm_idx] : perm_idx;
          }
        }
        else {
          // copy maximum same partition vertex avalible 
          for (int idx = threadIdx.x; idx < maxCount; idx += BLOCK_SIZE) {
            const IdType perm_idx = out_idxs[out_row_start + idx] + in_row_start;
            out_rows[out_row_start + idx] = row;
            out_cols[out_row_start + idx] = neighbours[index_array[idx]];
            out_idxs[out_row_start + idx] = data ? data[perm_idx] : perm_idx;
          }
          int remining_size = num_picks - maxCount;
          // printf("the remining size: %d maxCount: %d\n",remining_size,maxCount);
          for (int idx = threadIdx.x + maxCount; idx < num_picks; idx += BLOCK_SIZE) {
            const IdType perm_idx = out_idxs[out_row_start + idx] + in_row_start;
            out_rows[out_row_start + idx] = row;
            int counter = 0;
            for(int i = 0; i < deg && counter < remining_size; i++)
            {
              int flag = 0;
              for (int j = 0; j < maxCount; j++)
              {
                if ( neighbours[i] == neighbours[index_array[j]])
                {
                  flag = 1;
                  break;
                }
              }
              if (flag == 0)
              {
                out_cols[out_row_start + idx] = neighbours[i];
                counter++;
              }
            }
            out_idxs[out_row_start + idx] = data ? data[perm_idx] : perm_idx;
          }
        }
      }
      else
    {
        if (threadIdx.x < num_picks) {
          counts[threadIdx.x] = 0;
        }
        // __syncthreads();

        //count the total repeated values of same partition
        if( threadIdx.x == 0)
        {
          maxValue = d_part_array[row];
        }
        for( int kk = threadIdx.x; kk < deg; kk += BLOCK_SIZE)
        {

          int val = d_part_array[in_index[in_row_start + kk]];
          if( val == maxValue)
          {
            atomicAdd(&counts[val], 1);
          }
          // atomicAdd(&counts[val], 1);
        }
        __syncthreads();

        // Find maximum count and corresponding value
        if (threadIdx.x == 0) {
          maxCount = 0;
          pointer_position = 0; 
          maxCount = counts[maxValue];
        }
        __syncthreads();

        // Find indices of maximum repeated value
        if (threadIdx.x == 0)
        {
          for(int kk = 0; kk < deg; kk++)
          {
            if(d_part_array[in_index[in_row_start + kk]] == maxValue && pointer_position < num_picks)
            {
              index_array[pointer_position] = in_row_start + kk;
              // index_array[pointer_position] = in_index[in_row_start + kk];
              pointer_position++;
            }
            if( pointer_position >= num_picks)
            {
              break;
            }
          }
        }

        // generate permutation list via reservoir algorithm
        for (int idx = threadIdx.x; idx < num_picks; idx += BLOCK_SIZE) {
          out_idxs[out_row_start + idx] = idx;
        }
        __syncthreads();

        // copy all vertex if avalible of same partition
        if (num_picks < maxCount)
        {
          for (int idx = threadIdx.x; idx < num_picks; idx += BLOCK_SIZE)
          {
            const IdType perm_idx = out_idxs[out_row_start + idx] + in_row_start;
            out_rows[out_row_start + idx] = row;
            out_cols[out_row_start + idx] = in_index[index_array[idx]];
            out_idxs[out_row_start + idx] = data ? data[perm_idx] : perm_idx;
          }
        }
        else {
          // copy maximum same partition vertex avalible 
          for (int idx = threadIdx.x; idx < maxCount; idx += BLOCK_SIZE) {
            const IdType perm_idx = out_idxs[out_row_start + idx] + in_row_start;
            out_rows[out_row_start + idx] = row;
            out_cols[out_row_start + idx] = in_index[index_array[idx]];
            out_idxs[out_row_start + idx] = data ? data[perm_idx] : perm_idx;
          }
          int remining_size = num_picks - maxCount;
          for (int idx = threadIdx.x + maxCount; idx < num_picks; idx += BLOCK_SIZE) {
            const IdType perm_idx = out_idxs[out_row_start + idx] + in_row_start;
            out_rows[out_row_start + idx] = row;
            int counter = 0;
            for(int i = 0; i < deg && counter < remining_size; i++)
            {
              int flag = 0;
              for (int j = 0; j < maxCount; j++)
              {
                if ( in_index[i + in_row_start] == in_index[index_array[j]])
                {
                  flag = 1;
                  break;
                }

              }
              if (flag == 0)
              {
                out_cols[out_row_start + idx] = in_index[i + in_row_start];
                counter++;
              }
            }
            out_idxs[out_row_start + idx] = data ? data[perm_idx] : perm_idx;
          }
        }
      }
    }
    out_row += 1;
  }
}

/**
 * @brief Perform row-wise uniform sampling on a CSR matrix,
 * and generate a COO matrix, with replacement.
 *
 * @tparam IdType The ID type used for matrices.
 * @tparam TILE_SIZgE The number of rows covered by each threadblock.
 * @param rand_seed The random seed to use.
 * @param num_picks The number of non-zeros to pick per row.
 * @param num_rows The number of rows to pick.
 * @param in_rows The set of rows to pick.
 * @param in_ptr The indptr array of the input CSR.
 * @param in_index The indices array of the input CSR.
 * @param data The data array of the input CSR.
 * @param out_ptr The offset to write each row to in the output COO.
 * @param out_rows The rows of the output COO (output).
 * @param out_cols The columns of the output COO (output).
 * @param out_idxs The data array of the output COO (output).
 */
template <typename IdType, int TILE_SIZE>
__global__ void _CSRRowWiseSampleUniformReplaceKernel(
  const uint64_t rand_seed, const int64_t num_picks, const int64_t num_rows,
  const IdType* const in_rows, const IdType* const in_ptr,
  const IdType* const in_index, const IdType* const data,
  const IdType* const out_ptr, IdType* const out_rows, IdType* const out_cols,
  IdType* const out_idxs) {
  // we assign one warp per row
  assert(blockDim.x == BLOCK_SIZE);

  int64_t out_row = blockIdx.x * TILE_SIZE;
  const int64_t last_row =
    min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);

  curandStatePhilox4_32_10_t rng;
  curand_init(rand_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);

  while (out_row < last_row) {
    const int64_t row = in_rows[out_row];
    const int64_t in_row_start = in_ptr[row];
    const int64_t out_row_start = out_ptr[out_row];
    const int64_t deg = in_ptr[row + 1] - in_row_start;

    if (deg > 0) {
      // each thread then blindly copies in rows only if deg > 0.
      for (int idx = threadIdx.x; idx < num_picks; idx += BLOCK_SIZE) {
        const int64_t edge = curand(&rng) % deg;
        const int64_t out_idx = out_row_start + idx;
        out_rows[out_idx] = row;
        out_cols[out_idx] = in_index[in_row_start + edge];
        out_idxs[out_idx] =
          data ? data[in_row_start + edge] : in_row_start + edge;
      }
    }
    out_row += 1;
  }
}

}  // namespace

///////////////////////////// CSR sampling //////////////////////////

template <DGLDeviceType XPU, typename IdType>
COOMatrix _CSRRowWiseSamplingUniform(
  CSRMatrix mat, IdArray rows, const int64_t num_picks, const bool replace) {
  const auto& ctx = rows->ctx;
  auto device = runtime::DeviceAPI::Get(ctx);
  cudaStream_t stream = runtime::getCurrentCUDAStream();

  const int64_t num_rows = rows->shape[0];
  const IdType* const slice_rows = static_cast<const IdType*>(rows->data);

  IdArray picked_row =
    NewIdArray(num_rows * num_picks, ctx, sizeof(IdType) * 8);
  IdArray picked_col =
    NewIdArray(num_rows * num_picks, ctx, sizeof(IdType) * 8);
  IdArray picked_idx =
    NewIdArray(num_rows * num_picks, ctx, sizeof(IdType) * 8);
  IdType* const out_rows = static_cast<IdType*>(picked_row->data);
  IdType* const out_cols = static_cast<IdType*>(picked_col->data);
  IdType* const out_idxs = static_cast<IdType*>(picked_idx->data);

  const IdType* in_ptr = static_cast<IdType*>(GetDevicePointer(mat.indptr));
  const IdType* in_cols = static_cast<IdType*>(GetDevicePointer(mat.indices));
  const IdType* data = CSRHasData(mat)
    ? static_cast<IdType*>(GetDevicePointer(mat.data))
    : nullptr;


  // size_t size = parts_array->shape[0];
  // int64_t* part_array = static_cast<int64_t*>(parts_array->data);
  // 
  // printf("data from rowwise_sampling.cu line 265\n")
  // for(int i=0; i< size; i++)
  // {
  //     std::cout << part_array[i] << " ";
  // }
  // // int64_t* d_part_array;
  // cudaMalloc(&d_part_array, size * sizeof(int64_t));

  // cudaMemcpy(cuda_array, data_ptr, size * sizeof(int64_t), cudaMemcpyHostToDevice);


  // compute degree
  IdType* out_deg = static_cast<IdType*>(
    device->AllocWorkspace(ctx, (num_rows + 1) * sizeof(IdType)));
  if (replace) {
    const dim3 block(512);
    const dim3 grid((num_rows + block.x - 1) / block.x);
    CUDA_KERNEL_CALL(
      _CSRRowWiseSampleDegreeReplaceKernel, grid, block, 0, stream, num_picks,
      num_rows, slice_rows, in_ptr, out_deg);
  } else {
    const dim3 block(512);
    const dim3 grid((num_rows + block.x - 1) / block.x);
    CUDA_KERNEL_CALL(
      _CSRRowWiseSampleDegreeKernel, grid, block, 0, stream, num_picks,
      num_rows, slice_rows, in_ptr, out_deg);
  }

  // fill out_ptr
  IdType* out_ptr = static_cast<IdType*>(
    device->AllocWorkspace(ctx, (num_rows + 1) * sizeof(IdType)));
  size_t prefix_temp_size = 0;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
    nullptr, prefix_temp_size, out_deg, out_ptr, num_rows + 1, stream));
  void* prefix_temp = device->AllocWorkspace(ctx, prefix_temp_size);
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
    prefix_temp, prefix_temp_size, out_deg, out_ptr, num_rows + 1, stream));
  device->FreeWorkspace(ctx, prefix_temp);
  device->FreeWorkspace(ctx, out_deg);

  cudaEvent_t copyEvent;
  CUDA_CALL(cudaEventCreate(&copyEvent));

  NDArray new_len_tensor;
  if (TensorDispatcher::Global()->IsAvailable()) {
    new_len_tensor = NDArray::PinnedEmpty(
      {1}, DGLDataTypeTraits<IdType>::dtype, DGLContext{kDGLCPU, 0});
  } else {
    // use pageable memory, it will unecessarily block but be functional
    new_len_tensor = NDArray::Empty(
      {1}, DGLDataTypeTraits<IdType>::dtype, DGLContext{kDGLCPU, 0});
  }

  // copy using the internal current stream
  CUDA_CALL(cudaMemcpyAsync(
    new_len_tensor->data, out_ptr + num_rows, sizeof(IdType),
    cudaMemcpyDeviceToHost, stream));
  CUDA_CALL(cudaEventRecord(copyEvent, stream));

  const uint64_t random_seed = RandomEngine::ThreadLocal()->RandInt(1000000000);

  // select edges
  // the number of rows each thread block will cover
  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  // constexpr int TILE_SIZE = 128 / BLOCK_SIZE;
  constexpr int TILE_SIZE = BLOCK_SIZE / BLOCK_SIZE;
  cudaEventRecord(start);
  if (replace) {  // with replacement
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);

    // cudaEventRecord(start);
    CUDA_KERNEL_CALL(
      (_CSRRowWiseSampleUniformReplaceKernel<IdType, TILE_SIZE>), grid, block,
      0, stream, random_seed, num_picks, num_rows, slice_rows, in_ptr,
      in_cols, data, out_ptr, out_rows, out_cols, out_idxs);
    // cudaEventRecord(stop);
  } else {  // without replacement
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);
    CUDA_KERNEL_CALL(
      (_CSRRowWiseSampleUniformKernel<IdType, TILE_SIZE>), grid, block, 0,
      stream, random_seed, num_picks, num_rows, slice_rows, in_ptr, in_cols,
      data, out_ptr, out_rows, out_cols, out_idxs);
  }
  cudaDeviceSynchronize();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  device->FreeWorkspace(ctx, out_ptr);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  // printf("cuda sapmling time: %.6f\n", milliseconds/1000);

  // wait for copying `new_len` to finish
  CUDA_CALL(cudaEventSynchronize(copyEvent));
  CUDA_CALL(cudaEventDestroy(copyEvent));

  const IdType new_len = static_cast<const IdType*>(new_len_tensor->data)[0];
  picked_row = picked_row.CreateView({new_len}, picked_row->dtype);
  picked_col = picked_col.CreateView({new_len}, picked_col->dtype);
  picked_idx = picked_idx.CreateView({new_len}, picked_idx->dtype);

  return COOMatrix(
    mat.num_rows, mat.num_cols, picked_row, picked_col, picked_idx);
}

template <DGLDeviceType XPU, typename IdType>
COOMatrix _CSRRowWiseSamplingUniform1(
  CSRMatrix mat, IdArray rows, const int64_t num_picks, NDArray part_array, const bool replace) {
  const auto& ctx = rows->ctx;
  auto device = runtime::DeviceAPI::Get(ctx);
  cudaStream_t stream = runtime::getCurrentCUDAStream();

  size_t size = part_array->shape[0];
  int64_t* parts_array = static_cast<int64_t*>(part_array->data);

  // printf("data from _CSRRowWiseSamplingUniform1 function line 448");
  if(flag == 0)
  {
    // printf("flag activated");
    flag = 1;
    // allocate gpu memory
    // IdType* d_part_array = static_cast<IdType*>(device->AllocWorkspace(ctx, (size) * sizeof(IdType)));
    cudaMalloc(&d_part_array, size * sizeof(int64_t));

    cudaMemcpy(d_part_array, parts_array, size * sizeof(int64_t), cudaMemcpyHostToDevice);
    // printf("Cudamemcpy called");
    // flag = 1;
  }
  const int64_t num_rows = rows->shape[0];
  const IdType* const slice_rows = static_cast<const IdType*>(rows->data);

  IdArray picked_row =
    NewIdArray(num_rows * num_picks, ctx, sizeof(IdType) * 8);
  IdArray picked_col =
    NewIdArray(num_rows * num_picks, ctx, sizeof(IdType) * 8);
  IdArray picked_idx =
    NewIdArray(num_rows * num_picks, ctx, sizeof(IdType) * 8);
  IdType* const out_rows = static_cast<IdType*>(picked_row->data);
  IdType* const out_cols = static_cast<IdType*>(picked_col->data);
  IdType* const out_idxs = static_cast<IdType*>(picked_idx->data);

  const IdType* in_ptr = static_cast<IdType*>(GetDevicePointer(mat.indptr));
  const IdType* in_cols = static_cast<IdType*>(GetDevicePointer(mat.indices));
  const IdType* data = CSRHasData(mat)
    ? static_cast<IdType*>(GetDevicePointer(mat.data))
    : nullptr;


  // size_t size = parts_array->shape[0];
  // int64_t* part_array = static_cast<int64_t*>(parts_array->data);
  // 
  // printf("data from rowwise_sampling.cu line 478\n");
  // for(int i=0; i< size; i++)
  // {
  //     std::cout << part_array[i] << " ";
  // }
  // // int64_t* d_part_array;
  // cudaMalloc(&d_part_array, size * sizeof(int64_t));

  // cudaMemcpy(cuda_array, data_ptr, size * sizeof(int64_t), cudaMemcpyHostToDevice);


  // compute degree
  IdType* out_deg = static_cast<IdType*>(device->AllocWorkspace(ctx, (num_rows + 1) * sizeof(IdType)));
  if (replace) {
    const dim3 block(512);
    const dim3 grid((num_rows + block.x - 1) / block.x);
    CUDA_KERNEL_CALL(
      _CSRRowWiseSampleDegreeReplaceKernel, grid, block, 0, stream, num_picks,
      num_rows, slice_rows, in_ptr, out_deg);
  } else {
    const dim3 block(512);
    const dim3 grid((num_rows + block.x - 1) / block.x);
    CUDA_KERNEL_CALL(
      _CSRRowWiseSampleDegreeKernel, grid, block, 0, stream, num_picks,
      num_rows, slice_rows, in_ptr, out_deg);
  }

  // fill out_ptr
  IdType* out_ptr = static_cast<IdType*>(
    device->AllocWorkspace(ctx, (num_rows + 1) * sizeof(IdType)));
  size_t prefix_temp_size = 0;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
    nullptr, prefix_temp_size, out_deg, out_ptr, num_rows + 1, stream));
  void* prefix_temp = device->AllocWorkspace(ctx, prefix_temp_size);
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
    prefix_temp, prefix_temp_size, out_deg, out_ptr, num_rows + 1, stream));
  device->FreeWorkspace(ctx, prefix_temp);
  device->FreeWorkspace(ctx, out_deg);

  cudaEvent_t copyEvent;
  CUDA_CALL(cudaEventCreate(&copyEvent));

  NDArray new_len_tensor;
  if (TensorDispatcher::Global()->IsAvailable()) {
    new_len_tensor = NDArray::PinnedEmpty(
      {1}, DGLDataTypeTraits<IdType>::dtype, DGLContext{kDGLCPU, 0});
  } else {
    // use pageable memory, it will unecessarily block but be functional
    new_len_tensor = NDArray::Empty(
      {1}, DGLDataTypeTraits<IdType>::dtype, DGLContext{kDGLCPU, 0});
  }

  // copy using the internal current stream
  CUDA_CALL(cudaMemcpyAsync(
    new_len_tensor->data, out_ptr + num_rows, sizeof(IdType),
    cudaMemcpyDeviceToHost, stream));
  CUDA_CALL(cudaEventRecord(copyEvent, stream));

  const uint64_t random_seed = RandomEngine::ThreadLocal()->RandInt(1000000000);

  // select edges
  // the number of rows each thread block will cover
  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  constexpr int TILE_SIZE = BLOCK_SIZE / BLOCK_SIZE;
  // constexpr int TILE_SIZE = 128 / BLOCK_SIZE;
  cudaEventRecord(start);
  if (replace) {  // with replacement
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);

    // cudaEventRecord(start);
    CUDA_KERNEL_CALL(
      (_CSRRowWiseSampleUniformReplaceKernel<IdType, TILE_SIZE>), grid, block,
      0, stream, random_seed, num_picks, num_rows, slice_rows, in_ptr,
      in_cols, data, out_ptr, out_rows, out_cols, out_idxs);
    // cudaEventRecord(stop);
  } else {  // without replacement
    // const dim3 block(BLOCK_SIZE);
    // const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);
    // CUDA_KERNEL_CALL(
    //     (_CSRRowWiseSampleUniformKernel<IdType, TILE_SIZE>), grid, block, 0,
    //     stream, random_seed, num_picks, num_rows, slice_rows, in_ptr, in_cols,
    //     data, out_ptr, out_rows, out_cols, out_idxs);
    //
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);
    CUDA_KERNEL_CALL(
      (_CSRRowWiseSampleUniformKernel1<IdType, TILE_SIZE>), grid, block, num_picks,
      stream, random_seed, num_picks, num_rows, slice_rows, in_ptr, in_cols,
      data, out_ptr, out_rows, out_cols, out_idxs, d_part_array);

  }
  device->FreeWorkspace(ctx, out_ptr);
  // cudaDeviceSynchronize();
  // wait for copying `new_len` to finish
  CUDA_CALL(cudaEventSynchronize(copyEvent));
  CUDA_CALL(cudaEventDestroy(copyEvent));
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  sampling_time += milliseconds/1000;
  printf("cuda sapmling time %.6f\n", sampling_time);
  // cudaFree(d_part_array);

  const IdType new_len = static_cast<const IdType*>(new_len_tensor->data)[0];
  picked_row = picked_row.CreateView({new_len}, picked_row->dtype);
  picked_col = picked_col.CreateView({new_len}, picked_col->dtype);
  picked_idx = picked_idx.CreateView({new_len}, picked_idx->dtype);

  // std::cout
  // std::cout<<"new_len: "<<new_len<<"\n";
  // std::cout<<"picked_row: "<<picked_row<<"\n";
  // std::cout<<"picked_col: "<<picked_col<<"\n";
  // std::cout<<"picked_idx: "<<picked_idx<<"\n";

  return COOMatrix(
    mat.num_rows, mat.num_cols, picked_row, picked_col, picked_idx);
}

template <DGLDeviceType XPU, typename IdType>
COOMatrix _CSRRowWiseSamplingUniform2(
  CSRMatrix mat, IdArray rows, const int64_t num_picks, const bool replace) {
  const auto& ctx = rows->ctx;
  auto device = runtime::DeviceAPI::Get(ctx);
  cudaStream_t stream = runtime::getCurrentCUDAStream();

  // size_t size = part_array->shape[0];
  // int64_t* parts_array = static_cast<int64_t*>(part_array->data);

  // printf("data from _CSRRowWiseSamplingUniform2 function line 448");
  // if(flag == 0)
  // {
  // printf("flag activated");
  // flag = 1;
  // allocate gpu memory
  // IdType* d_part_array = static_cast<IdType*>(device->AllocWorkspace(ctx, (size) * sizeof(IdType)));
  // cudaMalloc(&d_part_array, size * sizeof(int64_t));

  // cudaMemcpy(d_part_array, parts_array, size * sizeof(int64_t), cudaMemcpyHostToDevice);
  // printf("Cudamemcpy called");
  // flag = 1;
  // }
  const int64_t num_rows = rows->shape[0];
  const IdType* const slice_rows = static_cast<const IdType*>(rows->data);

  IdArray picked_row =
    NewIdArray(num_rows * num_picks, ctx, sizeof(IdType) * 8);
  IdArray picked_col =
    NewIdArray(num_rows * num_picks, ctx, sizeof(IdType) * 8);
  IdArray picked_idx =
    NewIdArray(num_rows * num_picks, ctx, sizeof(IdType) * 8);
  IdType* const out_rows = static_cast<IdType*>(picked_row->data);
  IdType* const out_cols = static_cast<IdType*>(picked_col->data);
  IdType* const out_idxs = static_cast<IdType*>(picked_idx->data);

  const IdType* in_ptr = static_cast<IdType*>(GetDevicePointer(mat.indptr));
  const IdType* in_cols = static_cast<IdType*>(GetDevicePointer(mat.indices));
  const IdType* data = CSRHasData(mat)
    ? static_cast<IdType*>(GetDevicePointer(mat.data))
    : nullptr;


  // size_t size = parts_array->shape[0];
  // int64_t* part_array = static_cast<int64_t*>(parts_array->data);
  // 
  // printf("data from rowwise_sampling.cu line 478\n");
  // for(int i=0; i< size; i++)
  // {
  //     std::cout << part_array[i] << " ";
  // }
  // // int64_t* d_part_array;
  // cudaMalloc(&d_part_array, size * sizeof(int64_t));

  // cudaMemcpy(cuda_array, data_ptr, size * sizeof(int64_t), cudaMemcpyHostToDevice);


  // compute degree
  IdType* out_deg = static_cast<IdType*>(device->AllocWorkspace(ctx, (num_rows + 1) * sizeof(IdType)));
  if (replace) {
    const dim3 block(512);
    const dim3 grid((num_rows + block.x - 1) / block.x);
    CUDA_KERNEL_CALL(
      _CSRRowWiseSampleDegreeReplaceKernel, grid, block, 0, stream, num_picks,
      num_rows, slice_rows, in_ptr, out_deg);
  } else {
    const dim3 block(512);
    const dim3 grid((num_rows + block.x - 1) / block.x);
    CUDA_KERNEL_CALL(
      _CSRRowWiseSampleDegreeKernel, grid, block, 0, stream, num_picks,
      num_rows, slice_rows, in_ptr, out_deg);
  }

  // fill out_ptr
  IdType* out_ptr = static_cast<IdType*>(
    device->AllocWorkspace(ctx, (num_rows + 1) * sizeof(IdType)));
  size_t prefix_temp_size = 0;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
    nullptr, prefix_temp_size, out_deg, out_ptr, num_rows + 1, stream));
  void* prefix_temp = device->AllocWorkspace(ctx, prefix_temp_size);
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
    prefix_temp, prefix_temp_size, out_deg, out_ptr, num_rows + 1, stream));
  device->FreeWorkspace(ctx, prefix_temp);
  device->FreeWorkspace(ctx, out_deg);

  cudaEvent_t copyEvent;
  CUDA_CALL(cudaEventCreate(&copyEvent));

  NDArray new_len_tensor;
  if (TensorDispatcher::Global()->IsAvailable()) {
    new_len_tensor = NDArray::PinnedEmpty(
      {1}, DGLDataTypeTraits<IdType>::dtype, DGLContext{kDGLCPU, 0});
  } else {
    // use pageable memory, it will unecessarily block but be functional
    new_len_tensor = NDArray::Empty(
      {1}, DGLDataTypeTraits<IdType>::dtype, DGLContext{kDGLCPU, 0});
  }

  // copy using the internal current stream
  CUDA_CALL(cudaMemcpyAsync(
    new_len_tensor->data, out_ptr + num_rows, sizeof(IdType),
    cudaMemcpyDeviceToHost, stream));
  CUDA_CALL(cudaEventRecord(copyEvent, stream));

  const uint64_t random_seed = RandomEngine::ThreadLocal()->RandInt(1000000000);

  // select edges
  // the number of rows each thread block will cover
  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  constexpr int TILE_SIZE = BLOCK_SIZE / BLOCK_SIZE;
  // constexpr int TILE_SIZE = 128 / BLOCK_SIZE;
  cudaEventRecord(start);
  if (replace) {  // with replacement
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);

    // cudaEventRecord(start);
    CUDA_KERNEL_CALL(
      (_CSRRowWiseSampleUniformReplaceKernel<IdType, TILE_SIZE>), grid, block,
      0, stream, random_seed, num_picks, num_rows, slice_rows, in_ptr,
      in_cols, data, out_ptr, out_rows, out_cols, out_idxs);
    // cudaEventRecord(stop);
  } else {  // without replacement
    // const dim3 block(BLOCK_SIZE);
    // const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);
    // CUDA_KERNEL_CALL(
    //     (_CSRRowWiseSampleUniformKernel<IdType, TILE_SIZE>), grid, block, 0,
    //     stream, random_seed, num_picks, num_rows, slice_rows, in_ptr, in_cols,
    //     data, out_ptr, out_rows, out_cols, out_idxs);
    //
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);
    CUDA_KERNEL_CALL(
      (_CSRRowWiseSampleUniformKernel1<IdType, TILE_SIZE>), grid, block, num_picks,
      stream, random_seed, num_picks, num_rows, slice_rows, in_ptr, in_cols,
      data, out_ptr, out_rows, out_cols, out_idxs, d_part_array);

  }
  device->FreeWorkspace(ctx, out_ptr);
  // cudaDeviceSynchronize();
  // wait for copying `new_len` to finish
  CUDA_CALL(cudaEventSynchronize(copyEvent));
  CUDA_CALL(cudaEventDestroy(copyEvent));
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  sampling_time += milliseconds/1000;
  printf("\ncuda sapmling time %.6f\t \n", sampling_time);
  // cudaFree(d_part_array);

  const IdType new_len = static_cast<const IdType*>(new_len_tensor->data)[0];
  picked_row = picked_row.CreateView({new_len}, picked_row->dtype);
  picked_col = picked_col.CreateView({new_len}, picked_col->dtype);
  picked_idx = picked_idx.CreateView({new_len}, picked_idx->dtype);

  // std::cout<<"U2 new_len: "<<new_len<<"\n";
  // std::cout<<"U2 picked_row: "<<picked_row<<"\n";
  // std::cout<<"U2 picked_col: "<<picked_col<<"\n";
  // std::cout<<"U2 picked_idx: "<<picked_idx<<"\n";


  return COOMatrix(
    mat.num_rows, mat.num_cols, picked_row, picked_col, picked_idx);
}


template <DGLDeviceType XPU, typename IdType>
COOMatrix CSRRowWiseSamplingUniform(
  CSRMatrix mat, IdArray rows, const int64_t num_picks, const bool replace) {

  // size_t size = parts_array->shape[0];
  // int64_t* part_array = static_cast<int64_t*>(parts_array->data);

  // printf("data from rowwise_sampling.cu line 265\n");
  // for(int i=0; i< size; i++)
  // {
  // std::cout << part_array[i] << " ";
  // }

  if (num_picks == -1) {
    // Basically this is UnitGraph::InEdges().
    COOMatrix coo = CSRToCOO(CSRSliceRows(mat, rows), false);
    IdArray sliced_rows = IndexSelect(rows, coo.row);
    return COOMatrix(
      mat.num_rows, mat.num_cols, sliced_rows, coo.col, coo.data);
  } else {
    return _CSRRowWiseSamplingUniform<XPU, IdType>(
      mat, rows, num_picks, replace);
  }
}

template <DGLDeviceType XPU, typename IdType>
COOMatrix CSRRowWiseSamplingUniform1(
  CSRMatrix mat, IdArray rows, const int64_t num_picks, const NDArray& parts_array, const bool replace) {


  // size_t size = parts_array->shape[0];
  // IdArray* part_array = static_cast<IdArray*>(parts_array->data);
  //
  // int64_t* d_part_array;
  // cudaMalloc(&d_part_array, size * sizeof(int64_t));
  //
  // cudaMemcpy(d_part_array, part_array, size * sizeof(int64_t), cudaMemcpyHostToDevice);

  // size_t size = parts_array->shape[0];
  // int64_t* part_array = static_cast<int64_t*>(parts_array->data);

  // printf("data from rowwise_sampling.cu line 265\n");
  // for(int i=0; i< size; i++)
  // {
  // std::cout << "data :" << part_array[i] << " ";
  // }
  // dgl::IdArray part_array = dgl::aten::AsIdArray(parts_array);
  // dgl::IdArray part_array = dgl::IdArray::FromDLPack(parts_array.ToDLPack());

  if (num_picks == -1) {
    // Basically this is UnitGraph::InEdges().
    COOMatrix coo = CSRToCOO(CSRSliceRows(mat, rows), false);
    IdArray sliced_rows = IndexSelect(rows, coo.row);
    return COOMatrix(
      mat.num_rows, mat.num_cols, sliced_rows, coo.col, coo.data);
  } else {
    return _CSRRowWiseSamplingUniform1<XPU, IdType>(
      mat, rows, num_picks, parts_array, replace);
  }
}

template <DGLDeviceType XPU, typename IdType>
COOMatrix CSRRowWiseSamplingUniform2(
  CSRMatrix mat, IdArray rows, const int64_t num_picks, const bool replace) {
  if (num_picks == -1) {
    // Basically this is UnitGraph::InEdges().
    COOMatrix coo = CSRToCOO(CSRSliceRows(mat, rows), false);
    IdArray sliced_rows = IndexSelect(rows, coo.row);
    return COOMatrix(
      mat.num_rows, mat.num_cols, sliced_rows, coo.col, coo.data);
  } else {
    return _CSRRowWiseSamplingUniform2<XPU, IdType>(
      mat, rows, num_picks, replace);
  }
}


template COOMatrix CSRRowWiseSamplingUniform<kDGLCUDA, int32_t>(
  CSRMatrix, IdArray, int64_t, bool);
template COOMatrix CSRRowWiseSamplingUniform<kDGLCUDA, int64_t>(
  CSRMatrix, IdArray, int64_t, bool);
template COOMatrix CSRRowWiseSamplingUniform1<kDGLCUDA, int32_t>(
  CSRMatrix, IdArray, int64_t, const NDArray& , bool);
template COOMatrix CSRRowWiseSamplingUniform1<kDGLCUDA, int64_t>(
  CSRMatrix, IdArray, int64_t, const NDArray& , bool);
template COOMatrix CSRRowWiseSamplingUniform2<kDGLCUDA, int32_t>(
  CSRMatrix, IdArray, int64_t, bool);
template COOMatrix CSRRowWiseSamplingUniform2<kDGLCUDA, int64_t>(
  CSRMatrix, IdArray, int64_t, bool);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
