/**
 *  Copyright (c) 2020 by Contributors
 * @file array/cuda/spmm.cu
 * @brief SPMM C APIs and definitions.
 */
#include <dgl/array.h>

#include "../../runtime/cuda/cuda_common.h"
#include "./functor.cuh"
#include "./ge_spmm.cuh"
#include "./spmm.cuh"

namespace dgl {

using namespace cuda;

namespace aten {

/**
 * @brief CUDA implementation of g-SpMM on Csr format.
 * @note use cusparse if the reduce operator is `sum` and there is
 *       no broadcast, use dgl's kernel in other cases.
 */


template <int XPU, typename IdType, typename DType>
void CustomCsr(const std::string& op, const std::string& reduce,
             const BcastOff& bcast,
             const CSRMatrix& csr,
             NDArray ufeat,
             NDArray efeat,
             NDArray out,
             std::vector<NDArray> out_aux) {
  //bool is_scalar_efeat = efeat.NumElements() == csr.indices->shape[0];
  //bool use_efeat = op != "copy_lhs";

  if (reduce == "sum") {
    //bool more_nnz = (csr.indices->shape[0] > csr.num_rows * csr.num_cols);
    if (op == "copy_lhs") {
      // cusparse
      int64_t x_length = 1;
      for (int i = 1; i < ufeat->ndim; ++i)
        x_length *= ufeat->shape[i];
    /*  SWITCH_BITS(bits, DType, {
        printf("before call 0");
	cuda::CustomCsr<DType, IdType>(
            csr,
            static_cast<DType*>(ufeat->data),
            nullptr,
            static_cast<DType*>(out->data),
            x_length);
        printf("after call 0");
      });*/
      // SWITCH_BITS(bits, DType, {
        SWITCH_OP(op, Op, {
                        printf("before call 2");
          cuda::CustomCsr<IdType, DType >(bcast,
                csr, static_cast<DType*>(ufeat->data),
            nullptr,
            static_cast<DType*>(out->data),
            x_length);
        });
      // });
    }
  }
}



template <int XPU, typename IdType, typename DType>
void SpMMCsr(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {
  bool is_scalar_efeat = efeat.NumElements() == csr.indices->shape[0];
  bool use_efeat = op != "copy_lhs";

  if (reduce == "sum") {
    bool more_nnz = (csr.indices->shape[0] > csr.num_rows * csr.num_cols);
    if (op == "copy_lhs" && cusparse_available<DType, IdType>(more_nnz)) {
      // cusparse
      int64_t x_length = 1;
      // printf("CusparseCsrmm2 called op=sum\n");
      for (int i = 1; i < ufeat->ndim; ++i) x_length *= ufeat->shape[i];
      
      // SWITCH_OP(op, Op {
      cuda::CustomCsr<IdType, DType>(
        bcast, csr, static_cast<DType*>(ufeat->data),
        nullptr, 
        static_cast<DType*>(out->data),
        x_length
      );
      // });

      // CusparseCsrmm2<DType, IdType>(
      //     ufeat->ctx, csr, static_cast<DType*>(ufeat->data), nullptr,
      //     static_cast<DType*>(out->data), x_length);
    } else if (
        op == "mul" && is_scalar_efeat &&
        cusparse_available<DType, IdType>(more_nnz)) {
      // cusparse
      int64_t x_length = 1;
      for (int i = 1; i < ufeat->ndim; ++i) x_length *= ufeat->shape[i];
      if (!IsNullArray(csr.data)) {
        efeat = IndexSelect(efeat, csr.data);
      }
      // printf("CusparseCsrmm2 called op= mul\n");
      CusparseCsrmm2<DType, IdType>(
          ufeat->ctx, csr, static_cast<DType*>(ufeat->data),
          static_cast<DType*>(efeat->data), static_cast<DType*>(out->data),
          x_length);
    } else {  // general kernel
      SWITCH_OP(op, Op, {
        cuda::SpMMCsr<IdType, DType, Op, cuda::reduce::Sum<IdType, DType> >(
            bcast, csr, ufeat, efeat, out, NullArray(), NullArray());
      });
    }
  } else if (reduce == "max") {
    SWITCH_OP(op, Op, {
      cuda::SpMMCsr<IdType, DType, Op, cuda::reduce::Max<IdType, DType> >(
          bcast, csr, ufeat, efeat, out, out_aux[0], out_aux[1]);
    });
  } else if (reduce == "min") {
    SWITCH_OP(op, Op, {
      cuda::SpMMCsr<IdType, DType, Op, cuda::reduce::Min<IdType, DType> >(
          bcast, csr, ufeat, efeat, out, out_aux[0], out_aux[1]);
    });
  } else {
    LOG(FATAL) << "Not implemented";
  }
}

/**
 * @brief CUDA implementation of g-SpMM on Coo format.
 */
template <int XPU, typename IdType, typename DType>
void SpMMCoo(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {
  if (reduce == "sum") {
    SWITCH_OP(op, Op, {
      cuda::SpMMCoo<IdType, DType, Op, cuda::reduce::Sum<IdType, DType, true> >(
          bcast, coo, ufeat, efeat, out, NullArray(), NullArray());
    });
  } else if (reduce == "max") {
    SWITCH_OP(op, Op, {
      cuda::SpMMCoo<IdType, DType, Op, cuda::reduce::Max<IdType, DType, true> >(
          bcast, coo, ufeat, efeat, out, out_aux[0], out_aux[1]);
    });
  } else if (reduce == "min") {
    SWITCH_OP(op, Op, {
      cuda::SpMMCoo<IdType, DType, Op, cuda::reduce::Min<IdType, DType, true> >(
          bcast, coo, ufeat, efeat, out, out_aux[0], out_aux[1]);
    });
  } else {
    LOG(FATAL) << "Not implemented";
  }
}

// template<typename Idx, typename Dtype>
// void CoustomCsr(const BcastOff& bcast,
//                 const CSRMatrix& csr,
//                 const Dtype* B_data,
//                 int x_length){
//     const int m = csr.num_rows;
//     const int n = x_length;
//     int width = 16;
//   la_spmm<Idx, Dtype>(m, n, static_cast<Idx*>(csr.indptr -> data), static_cast<Idx*>(csr.indices-data), B_data, C_data, width);
// }
//
//
// template<typename Idx, typename Dtype>
// void ls_spmm(const int m, const int k, const Idx* A_indptr, const Idx* A_indices, const Dtype* B_data, Dtype* C_data, int width)
// {
//   static float summation=0;
//   float elapsed_time;
//   cudaEvent_t start, stop;
//   cudaEventCreate(&start);
//   cudaEventCreate(&stop);
//
//   int bx, by;
//   bx = (m/16) + (m%16 != o);
//   int len;
//   if(k < 64)
//   {
//     len = k;
//   }
//   else {
//     len = 64;
//   }
//   by = (k/64) + (k%64 != 0);
//   
//   cudaEventRecord(start);
//   dkernel<Idx, Dtype><<<dim3(bx,16,1), dim3(len,by,1), len*by*sizeof(int)>>>(m ,k, A_indptr, A_indices, B_data, C_data, len);
//   cudaDeviceSynchronize();
//   cudaEventRecord(stop);
//   cudaEventSynchronize(stop);
//   cudaEventElapsedTime(&elapsed_time, start, stop);
//   summation+=elapsed_time;
//   printf("sum = %f\n", summation);
//
// }

template void CustomCsr<kDGLCUDA, int32_t, __half>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);

template void CustomCsr<kDGLCUDA, int64_t, __half>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);

template void CustomCsr<kDGLCUDA, int32_t, float>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void CustomCsr<kDGLCUDA, int64_t, float>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);

template void CustomCsr<kDGLCUDA, int32_t, double>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void CustomCsr<kDGLCUDA, int64_t, double>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
//
// template void CoustomCsr<kDGLCUDA, int32_t, __half>(
//     const std::string& op, const std::string& reduce, const BcastOff& bcast,
//     const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
//     std::vector<NDArray> out_aux);
//
// template void CoustomCsr<kDGLCUDA, int64_t, __half>(
//     const std::string& op, const std::string& reduce, const BcastOff& bcast,
//     const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
//     std::vector<NDArray> out_aux);


template void SpMMCsr<kDGLCUDA, int32_t, __half>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
template void SpMMCsr<kDGLCUDA, int64_t, __half>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
#if BF16_ENABLED
template void SpMMCsr<kDGLCUDA, int32_t, __nv_bfloat16>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
template void SpMMCsr<kDGLCUDA, int64_t, __nv_bfloat16>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);

template void CustomCsr<kDGLCUDA, int32_t, __nv_bfloat16>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
template void CustomCsr<kDGLCUDA, int64_t, __nv_bfloat16>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);


#endif  // BF16_ENABLED
template void SpMMCsr<kDGLCUDA, int32_t, float>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
template void SpMMCsr<kDGLCUDA, int64_t, float>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
template void SpMMCsr<kDGLCUDA, int32_t, double>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
template void SpMMCsr<kDGLCUDA, int64_t, double>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);

template void SpMMCoo<kDGLCUDA, int32_t, __half>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
template void SpMMCoo<kDGLCUDA, int64_t, __half>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
#if BF16_ENABLED
template void SpMMCoo<kDGLCUDA, int32_t, __nv_bfloat16>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
template void SpMMCoo<kDGLCUDA, int64_t, __nv_bfloat16>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
#endif  // BF16_ENABLED
template void SpMMCoo<kDGLCUDA, int32_t, float>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
template void SpMMCoo<kDGLCUDA, int64_t, float>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
template void SpMMCoo<kDGLCUDA, int32_t, double>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
template void SpMMCoo<kDGLCUDA, int64_t, double>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);

}  // namespace aten
}  // namespace dgl
