/**
 *  Copyright (c) 2020 by Contributors
 * @file array/cuda/spmm.cuh
 * @brief SPMM CUDA kernel function header.
 */
#ifndef DGL_ARRAY_CUDA_SPMM_CUH_
#define DGL_ARRAY_CUDA_SPMM_CUH_

#include <dgl/bcast.h>

#include <algorithm>
#include <limits>

#include "../../runtime/cuda/cuda_common.h"
#include "./utils.h"
#include "atomic.cuh"
#include "bf16.cuh"
#include "fp16.cuh"
#include "macro.cuh"
#include "global_array.h"

#define Shared_mem_size 512
namespace dgl {

using namespace cuda;

namespace aten {

/**
 * @brief Determine whether cusparse SpMM function is applicable.
 */
template <typename DType, typename IdType>
inline bool cusparse_available(bool more_nnz_than_matrix_size) {
#if CUDART_VERSION < 11000
  if (std::is_same<IdType, int>::value &&
      (std::is_same<DType, float>::value || std::is_same<DType, double>::value))
    return true;
  return false;
#else
  if (std::is_same<DType, __half>::value ||
      std::is_same<DType, __nv_bfloat16>::value)
    return false;  // cusparse's SpMM on fp16 is slow, temporally disabled.
  // If the CSR matrix has more NNZ than matrix size, we should not use
  // cuSPARSE 11.1.
  return !more_nnz_than_matrix_size;
#endif
}

namespace {

/** @brief Call cuBLAS geam API for transpose operation for float and double. */
template <typename DType>
cublasStatus_t Xgeam(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, const DType* alpha, const DType* A, int lda,
    const DType* beta, const DType* B, int ldb, DType* C, int ldc) {
  LOG(FATAL) << "Not supported dtype";
  return CUBLAS_STATUS_EXECUTION_FAILED;
}

template <>
cublasStatus_t Xgeam<__half>(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, const __half* alpha, const __half* A, int lda,
    const __half* beta, const __half* B, int ldb, __half* C, int ldc) {
  // TODO(ndickson): There is no cublasHgeam, so a different
  // implementation would be required.
  LOG(FATAL) << "Xgeam does not support dtype half (FP16)";
  return CUBLAS_STATUS_EXECUTION_FAILED;
}

#if BF16_ENABLED
template <>
cublasStatus_t Xgeam<__nv_bfloat16>(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, const __nv_bfloat16* alpha, const __nv_bfloat16* A, int lda,
    const __nv_bfloat16* beta, const __nv_bfloat16* B, int ldb,
    __nv_bfloat16* C, int ldc) {
  // TODO(ndickson): There is no cublasHgeam, so a different
  // implementation would be required.
  LOG(FATAL) << "Xgeam does not support dtype bfloat16 (BF16)";
  return CUBLAS_STATUS_EXECUTION_FAILED;
}
#endif  // BF16_ENABLED

template <>
cublasStatus_t Xgeam<float>(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, const float* alpha, const float* A, int lda,
    const float* beta, const float* B, int ldb, float* C, int ldc) {
  return cublasSgeam(
      handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

template <>
cublasStatus_t Xgeam<double>(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, const double* alpha, const double* A, int lda,
    const double* beta, const double* B, int ldb, double* C, int ldc) {
  return cublasDgeam(
      handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

/**
 * @brief Transpose operator kernel implementation.
 * @note not efficient but it's not a bottleneck, used for float16 dtype.
 */
template <typename DType>
__global__ void _TransposeKernel(
    const DType* __restrict__ in, DType* __restrict__ out, int n, int m) {
  int i = blockIdx.x;
  for (int j = threadIdx.x; j < m; j += blockDim.x)
    out[i * m + j] = in[j * n + i];
}

/**
 * @brief Tranpose the input matrix.
 * @param row number of rows of input matrix.
 * @param col number of columns of input matrix.
 */
template <typename DType>
void _Transpose(const DType* in, DType* out, int row, int col) {
  DType alpha = 1., beta = 0.;
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  if (!thr_entry->cublas_handle)
    CUBLAS_CALL(cublasCreate(&(thr_entry->cublas_handle)));
  CUBLAS_CALL(cublasSetStream(thr_entry->cublas_handle, stream));
  CUBLAS_CALL(Xgeam<DType>(
      thr_entry->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, row, col, &alpha, in,
      col, &beta, nullptr, row, out, row));
}

/**
 * @brief Tranpose the input matrix for data type half.
 * @note cuBLAS has no geam API for half data type, fallback to our kernel.
 */
template <>
void _Transpose<__half>(const __half* in, __half* out, int row, int col) {
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  int nt = FindNumThreads(row);
  int nb = col;
  CUDA_KERNEL_CALL(_TransposeKernel, nb, nt, 0, stream, in, out, col, row);
}

#if BF16_ENABLED
/**
 * @brief Tranpose the input matrix for data type half.
 * @note cuBLAS has no geam API for bf16 data type, fallback to our kernel.
 */
template <>
void _Transpose<__nv_bfloat16>(
    const __nv_bfloat16* in, __nv_bfloat16* out, int row, int col) {
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  int nt = FindNumThreads(row);
  int nb = col;
  CUDA_KERNEL_CALL(_TransposeKernel, nb, nt, 0, stream, in, out, col, row);
}
#endif  // BF16_ENABLED

#if CUDART_VERSION < 11000
template <typename DType>
cusparseStatus_t Xcsrmm2(
    cusparseHandle_t handle, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int n, int k, int nnz,
    const DType* alpha, const cusparseMatDescr_t descrA, const DType* csrValA,
    const int* csrRowPtrA, const int* csrColIndA, const DType* B, int ldb,
    const DType* beta, DType* C, int ldc) {
  LOG(INFO) << "Not supported dtype";
  return CUSPARSE_STATUS_EXECUTION_FAILED;
}

template <>
cusparseStatus_t Xcsrmm2<float>(
    cusparseHandle_t handle, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int n, int k, int nnz,
    const float* alpha, const cusparseMatDescr_t descrA, const float* csrValA,
    const int* csrRowPtrA, const int* csrColIndA, const float* B, int ldb,
    const float* beta, float* C, int ldc) {
  return cusparseScsrmm2(
      handle, transA, transB, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA,
      csrColIndA, B, ldb, beta, C, ldc);
}

template <>
cusparseStatus_t Xcsrmm2<double>(
    cusparseHandle_t handle, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int n, int k, int nnz,
    const double* alpha, const cusparseMatDescr_t descrA, const double* csrValA,
    const int* csrRowPtrA, const int* csrColIndA, const double* B, int ldb,
    const double* beta, double* C, int ldc) {
  return cusparseDcsrmm2(
      handle, transA, transB, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA,
      csrColIndA, B, ldb, beta, C, ldc);
}
#endif

/** Cusparse implementation of SpMM on Csr format. */
template <typename DType, typename IdType>
void CusparseCsrmm2(
    const DGLContext& ctx, const CSRMatrix& csr, const DType* B_data,
    const DType* A_data, DType* C_data, int x_length) {
  // We use csrmm2 to perform following operation:
  // C = A x B, where A is a sparse matrix in csr format, B is the dense matrix
  // for node feature tensor. However, since cusparse only supports
  // column-major, while our tensor is stored in row-major, the actual
  // computation is: C = trans(A x trans(B)). Currently, we use cublasXgeam to
  // implement transposition and allocate intermediate workspace memory for
  // this.
  const int m = csr.num_rows;
  const int n = x_length;
  const int k = csr.num_cols;
  const int nnz = csr.indices->shape[0];
  const DType alpha = 1.0;
  const DType beta = 0.0;
  static float spmm_time = 0;
  // device
  auto device = runtime::DeviceAPI::Get(ctx);
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  // allocate cusparse handle if needed
  if (!thr_entry->cusparse_handle) {
    CUSPARSE_CALL(cusparseCreate(&(thr_entry->cusparse_handle)));
  }
  CUSPARSE_CALL(cusparseSetStream(thr_entry->cusparse_handle, stream));
  // all one data array
  DType* valptr = nullptr;
  if (!A_data) {
    valptr =
        static_cast<DType*>(device->AllocWorkspace(ctx, nnz * sizeof(DType)));
    _Fill(valptr, nnz, static_cast<DType>(1.));
  }
#if CUDART_VERSION >= 11000
  cusparseSpMatDescr_t matA;
  cusparseDnMatDescr_t matB, matC;
  constexpr auto dtype = cuda_dtype<DType>::value;
  constexpr auto idtype = cusparse_idtype<IdType>::value;
  CUSPARSE_CALL(cusparseCreateCsr(
      &matA, m, k, nnz, static_cast<IdType*>(csr.indptr->data),
      static_cast<IdType*>(csr.indices->data),
      const_cast<DType*>(valptr ? valptr : A_data), idtype, idtype,
      CUSPARSE_INDEX_BASE_ZERO, dtype));
  CUSPARSE_CALL(cusparseCreateDnMat(
      &matB, k, n, n, const_cast<DType*>(B_data), dtype, CUSPARSE_ORDER_ROW));
  CUSPARSE_CALL(
      cusparseCreateDnMat(&matC, m, n, n, C_data, dtype, CUSPARSE_ORDER_ROW));

  auto transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  auto transB = CUSPARSE_OPERATION_NON_TRANSPOSE;
  size_t workspace_size;
  CUSPARSE_CALL(cusparseSpMM_bufferSize(
      thr_entry->cusparse_handle, transA, transB, &alpha, matA, matB, &beta,
      matC, dtype, CUSPARSE_SPMM_CSR_ALG2, &workspace_size));
  void* workspace = device->AllocWorkspace(ctx, workspace_size);
  // printf("cusparse called\n");
  float elapsed_time_cusprse;
  cudaEvent_t start_cusparse, stop_cusparse;
  cudaEventCreate(&start_cusparse);
  cudaEventCreate(&stop_cusparse);
  cudaEventRecord(start_cusparse);

  CUSPARSE_CALL(cusparseSpMM(
      thr_entry->cusparse_handle, transA, transB, &alpha, matA, matB, &beta,
      matC, dtype, CUSPARSE_SPMM_CSR_ALG2, workspace));
  device->FreeWorkspace(ctx, workspace);
  cudaEventRecord(stop_cusparse);
  cudaEventSynchronize(stop_cusparse);
  cudaEventElapsedTime(&elapsed_time_cusprse, start_cusparse, stop_cusparse);
  spmm_time+=elapsed_time_cusprse/1000;
  printf("cusparse spmm time %.6f \n",spmm_time);

  CUSPARSE_CALL(cusparseDestroySpMat(matA));
  CUSPARSE_CALL(cusparseDestroyDnMat(matB));
  CUSPARSE_CALL(cusparseDestroyDnMat(matC));
#else
  // allocate matrix for temporary transposed output
  DType* trans_out =
      static_cast<DType*>(device->AllocWorkspace(ctx, m * n * sizeof(DType)));

  cusparseMatDescr_t descr;
  CUSPARSE_CALL(cusparseCreateMatDescr(&descr));
  CUSPARSE_CALL(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
  CUSPARSE_CALL(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));
  CUSPARSE_CALL(Xcsrmm2<DType>(
      thr_entry->cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_TRANSPOSE, m, n, k, nnz, &alpha, descr,
      (valptr) ? valptr : A_data, static_cast<int32_t*>(csr.indptr->data),
      static_cast<int32_t*>(csr.indices->data), B_data, n, &beta, trans_out,
      m));
  CUSPARSE_CALL(cusparseDestroyMatDescr(descr));
  // transpose the output matrix
  _Transpose(trans_out, C_data, n, m);
  device->FreeWorkspace(ctx, trans_out);
#endif
  if (valptr) device->FreeWorkspace(ctx, valptr);
}

/** Cusparse implementation of SpMM on Csr format. */
template <typename DType, typename IdType>
void CusparseCsrmm2Hetero(
    const DGLContext& ctx, const CSRMatrix& csr, const DType* B_data,
    const DType* A_data, DType* C_data, int64_t x_length,
    cudaStream_t strm_id) {
  // We use csrmm2 to perform following operation:
  // C = A x B, where A is a sparse matrix in csr format, B is the dense matrix
  // for node feature tensor. However, since cusparse only supports
  // column-major, while our tensor is stored in row-major, the actual
  // computation is: C = trans(A x trans(B)). Currently, we use cublasXgeam to
  // implement transposition and allocate intermediate workspace memory for
  // this.
  int int_maxlimit = std::numeric_limits<int>::max();
  CHECK_GE(int_maxlimit, (csr.num_rows));
  CHECK_GE(int_maxlimit, csr.num_cols);
  CHECK_GE(int_maxlimit, csr.indices->shape[0]);
  const int m = csr.num_rows;
  const int n = x_length;
  const int k = csr.num_cols;
  const int nnz = csr.indices->shape[0];
  const DType alpha = 1.0;
  const DType beta = 1.0;
  // device
  auto device = runtime::DeviceAPI::Get(ctx);
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  // allocate cusparse handle if needed
  if (!thr_entry->cusparse_handle) {
    CUSPARSE_CALL(cusparseCreate(&(thr_entry->cusparse_handle)));
  }
  CUSPARSE_CALL(cusparseSetStream(thr_entry->cusparse_handle, strm_id));
  // all one data array
  DType* valptr = nullptr;
  if (!A_data) {
    valptr =
        static_cast<DType*>(device->AllocWorkspace(ctx, nnz * sizeof(DType)));
    _Fill(valptr, nnz, static_cast<DType>(1.));
  }
#if CUDART_VERSION >= 11000
  cusparseSpMatDescr_t matA;
  cusparseDnMatDescr_t matB, matC;
  constexpr auto dtype = cuda_dtype<DType>::value;
  constexpr auto idtype = cusparse_idtype<IdType>::value;
  CUSPARSE_CALL(cusparseCreateCsr(
      &matA, m, k, nnz, static_cast<IdType*>(csr.indptr->data),
      static_cast<IdType*>(csr.indices->data),
      const_cast<DType*>(valptr ? valptr : A_data), idtype, idtype,
      CUSPARSE_INDEX_BASE_ZERO, dtype));
  CUSPARSE_CALL(cusparseCreateDnMat(
      &matB, k, n, n, const_cast<DType*>(B_data), dtype, CUSPARSE_ORDER_ROW));
  CUSPARSE_CALL(
      cusparseCreateDnMat(&matC, m, n, n, C_data, dtype, CUSPARSE_ORDER_ROW));

  auto transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  auto transB = CUSPARSE_OPERATION_NON_TRANSPOSE;
  size_t workspace_size;
  CUSPARSE_CALL(cusparseSpMM_bufferSize(
      thr_entry->cusparse_handle, transA, transB, &alpha, matA, matB, &beta,
      matC, dtype, CUSPARSE_SPMM_CSR_ALG2, &workspace_size));
  void* workspace = device->AllocWorkspace(ctx, workspace_size);
  CUSPARSE_CALL(cusparseSpMM(
      thr_entry->cusparse_handle, transA, transB, &alpha, matA, matB, &beta,
      matC, dtype, CUSPARSE_SPMM_CSR_ALG2, workspace));
  device->FreeWorkspace(ctx, workspace);

  CUSPARSE_CALL(cusparseDestroySpMat(matA));
  CUSPARSE_CALL(cusparseDestroyDnMat(matB));
  CUSPARSE_CALL(cusparseDestroyDnMat(matC));
#else
  cusparseMatDescr_t descr;
  CUSPARSE_CALL(cusparseCreateMatDescr(&descr));
  CUSPARSE_CALL(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
  CUSPARSE_CALL(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));
  CHECK_EQ(sizeof(IdType), sizeof(int32_t));
  CUSPARSE_CALL(Xcsrmm2<DType>(
      thr_entry->cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_TRANSPOSE, m, n, k, nnz, &alpha, descr,
      (valptr) ? valptr : A_data, static_cast<int32_t*>(csr.indptr->data),
      static_cast<int32_t*>(csr.indices->data), B_data, n, &beta, C_data, m));
  CUSPARSE_CALL(cusparseDestroyMatDescr(descr));
#endif
  if (valptr) device->FreeWorkspace(ctx, valptr);
}

}  // namespace

#define SWITCH_OP(op, Op, ...)                                  \
  do {                                                          \
    if ((op) == "add") {                                        \
      typedef cuda::binary::Add<DType> Op;                      \
      { __VA_ARGS__ }                                           \
    } else if ((op) == "sub") {                                 \
      typedef cuda::binary::Sub<DType> Op;                      \
      { __VA_ARGS__ }                                           \
    } else if ((op) == "mul") {                                 \
      typedef cuda::binary::Mul<DType> Op;                      \
      { __VA_ARGS__ }                                           \
    } else if ((op) == "div") {                                 \
      typedef cuda::binary::Div<DType> Op;                      \
      { __VA_ARGS__ }                                           \
    } else if ((op) == "copy_lhs") {                            \
      typedef cuda::binary::CopyLhs<DType> Op;                  \
      { __VA_ARGS__ }                                           \
    } else if ((op) == "copy_rhs") {                            \
      typedef cuda::binary::CopyRhs<DType> Op;                  \
      { __VA_ARGS__ }                                           \
    } else {                                                    \
      LOG(FATAL) << "Unsupported SpMM binary operator: " << op; \
    }                                                           \
  } while (0)

namespace cuda {

/**
 * @brief CUDA kernel of g-SpMM on Coo format.
 * @note it uses edge parallel strategy, different threadblocks (on y-axis)
 *       is responsible for the computation on different edges. Threadblocks
 *       on the x-axis are responsible for the computation on different
 * positions in feature dimension. To avoid possible data hazards, it uses
 * atomic operators for reduction.
 */
template <
    typename Idx, typename DType, typename BinaryOp, typename ReduceOp,
    bool UseBcast = false, bool UseIdx = false>
__global__ void SpMMCooKernel(
    const DType* __restrict__ ufeat, const DType* __restrict__ efeat,
    DType* __restrict__ out, Idx* __restrict__ arg_u, Idx* __restrict__ arg_e,
    const Idx* __restrict__ row, const Idx* __restrict__ col,
    const Idx* __restrict__ edge_map, int64_t N, int64_t M, int64_t E,
    const int64_t* __restrict__ ubcast_off,
    const int64_t* __restrict__ ebcast_off, int64_t ufeat_len,
    int64_t efeat_len, int64_t out_len) {
  // SPMM with COO.
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  while (ty < E) {
    const Idx src = _ldg(row + ty);
    const Idx dst = _ldg(col + ty);
    const Idx eid = UseIdx ? _ldg(edge_map + ty) : ty;
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride_x = blockDim.x * gridDim.x;
    const DType* uoff = BinaryOp::use_lhs ? (ufeat + src * ufeat_len) : nullptr;
    const DType* eoff = BinaryOp::use_rhs ? (efeat + eid * efeat_len) : nullptr;
    DType* outoff = out + dst * out_len;
    while (tx < out_len) {
      const int64_t lhs_add = UseBcast ? ubcast_off[tx] : tx;
      const int64_t rhs_add = UseBcast ? ebcast_off[tx] : tx;
      DType val = BinaryOp::Call(uoff + lhs_add, eoff + rhs_add);
      Idx* arguoff = nullptr;  // arguoff is not used in SpMMCoo.
      Idx* argeoff = nullptr;  // argeoff is not used in SpMMCoo.
      ReduceOp::Call(outoff + tx, arguoff, argeoff, val, src, eid);
      tx += stride_x;
    }
    ty += stride_y;
  }
}

/**
 * @brief CUDA kernel to compute argu and arge in g-SpMM on Coo format.
 * @note it uses edge parallel strategy, different threadblocks (on y-axis)
 *       is responsible for the computation on different edges. Threadblocks
 *       on the x-axis are responsible for the computation on different
 * positions in feature dimension.
 */
template <
    typename Idx, typename DType, typename BinaryOp, typename ReduceOp,
    bool UseBcast = false, bool UseIdx = false>
__global__ void ArgSpMMCooKernel(
    const DType* __restrict__ ufeat, const DType* __restrict__ efeat,
    DType* __restrict__ out, Idx* __restrict__ arg_u, Idx* __restrict__ arg_e,
    const Idx* __restrict__ row, const Idx* __restrict__ col,
    const Idx* __restrict__ edge_map, int64_t N, int64_t M, int64_t E,
    const int64_t* __restrict__ ubcast_off,
    const int64_t* __restrict__ ebcast_off, int64_t ufeat_len,
    int64_t efeat_len, int64_t out_len) {
  // SPMM with COO arg max/min.
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  while (ty < E) {
    const Idx src = _ldg(row + ty);
    const Idx dst = _ldg(col + ty);
    const Idx eid = UseIdx ? _ldg(edge_map + ty) : ty;
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride_x = blockDim.x * gridDim.x;
    const DType* uoff = BinaryOp::use_lhs ? (ufeat + src * ufeat_len) : nullptr;
    const DType* eoff = BinaryOp::use_rhs ? (efeat + eid * efeat_len) : nullptr;
    const DType* outoff = out + dst * out_len;
    Idx* arguoff = BinaryOp::use_lhs ? (arg_u + dst * out_len) : nullptr;
    Idx* argeoff = BinaryOp::use_rhs ? (arg_e + dst * out_len) : nullptr;
    while (tx < out_len) {
      int64_t lhs_add = UseBcast ? ubcast_off[tx] : tx;
      int64_t rhs_add = UseBcast ? ebcast_off[tx] : tx;
      DType val = BinaryOp::Call(uoff + lhs_add, eoff + rhs_add);
      ReduceOp::CallArg(tx, arguoff, argeoff, val, outoff[tx], src, eid);
      tx += stride_x;
    }
    ty += stride_y;
  }
}

/**
 * @brief CUDA kernel of g-SpMM on Csr format.
 * @note it uses node parallel strategy, different threadblocks (on y-axis)
 *       is responsible for the computation on different destination nodes.
 *       Threadblocks on the x-axis are responsible for the computation on
 *       different positions in feature dimension.
 */



__device__ __forceinline__ float sum_reduce(float acc, float x) {
  return acc + x;
}

__device__ __forceinline__ float sum_init() {
  return 0;
}

template <typename Idx, typename DType>
__global__ void topoCacheCoarsenSPMMKernel(
  int m, int k, const Idx* A_indptr, const Idx* A_indices, const DType* B, DType* C
) {
  extern __shared__ int sh[];
  int sm_offset = (threadIdx.y<<5);
  int thread_idx = sm_offset+threadIdx.x;

  int rid = blockDim.y*blockIdx.x+threadIdx.y;
  if (rid<m) {
//	if(rid==0 && threadIdx.x==0)
//		printf("hello");

    int cid = (blockIdx.y<<6)+threadIdx.x;
    int lb = A_indptr[rid];
    int hb = A_indptr[rid+1];
    int ptr = lb+threadIdx.x;
    int offset;
    float acc1 = sum_init();
    float acc2 = sum_init();
    if (blockIdx.y != gridDim.y-1) {
      for (int jj=lb; jj<hb; jj+=32) {
        if (ptr<hb) {
          sh[thread_idx] = A_indices[ptr]*k;
          // sh[thread_idx] = __ldg(A_indices+ptr)*k;
        }
        __syncwarp();
        ptr += 32;
        for (int kk=0; kk<32&&jj+kk<hb; kk++) {
          offset = sh[(sm_offset+kk)] + cid;
          acc1 = sum_reduce(acc1, B[offset]);
          acc2 = sum_reduce(acc2, B[(offset+32)]);
          // acc1 = sum_reduce(acc1, __ldg(B+offset));
          // acc2 = sum_reduce(acc2, __ldg(B+offset+32));
        }
        __syncwarp();
      }
      offset = rid*k+cid;
      C[offset] = acc1;
      C[offset+32] = acc2;
    }
    else { // threadIdx.y==blockDim.y-1
      int nout = (k-cid+31)/32;
      for (int jj=lb; jj<hb; jj+=32) {
        if (ptr<hb) {
          sh[thread_idx] = A_indices[ptr]*k;
          // sh[thread_idx] = __ldg(A_indices+ptr)*k;
        }
        __syncwarp();
        ptr += 32;
        for (int kk=0; kk<32&&jj+kk<hb; kk++) {
          offset = sh[(sm_offset+kk)] + cid;
          if (nout>0) {
          acc1 = sum_reduce(acc1, B[offset]);}
          // acc1 = sum_reduce(acc1, __ldg(B+offset)); }
          if (nout>1) {
          acc2 = sum_reduce(acc2, B[(offset+32)]);}
          // acc2 = sum_reduce(acc2, __ldg(B+offset+32));}
        }
        __syncwarp();
      }
      offset = rid*k+cid;
      if (nout>0) {
      C[offset] = acc1;}
      if (nout>1) {
      C[offset+32] = acc2;}
    }
  }
} 
template <typename Idx, typename DType>
__global__ void topoCacheSPMMKernel(
  int m, int k, const Idx* A_indptr, const Idx* A_indices, const DType* B, DType* C 
) {
  extern __shared__ int sh[];
  int sm_offset = (threadIdx.y<<5);
  int thread_idx = sm_offset + threadIdx.x;
  
  int cid = (blockIdx.y<<5)+threadIdx.x;
  int rid = blockDim.y*blockIdx.x+threadIdx.y;
     //if(rid==0 && threadIdx.x==0)
       //         printf("hello");

  if (rid<m) {
    int lb = A_indptr[rid];
    int hb = A_indptr[(rid+1)];
    int offset;
    int ptr = lb+threadIdx.x;
    float acc1 = sum_init();
    if (blockIdx.y != gridDim.y-1) {
      for (int jj=lb; jj<hb; jj+=32) {
        if (ptr<hb) {
          sh[thread_idx] = A_indices[ptr]*k;
          // sh[thread_idx] = __ldg(A_indices+ptr)*k;
        }
        __syncwarp();
        ptr += 32;
        for (int kk=0; kk<32&&jj+kk<hb; kk++) {
          offset = sh[sm_offset+kk]+cid;
          acc1 = sum_reduce(acc1, B[offset]);
          // acc1 = sum_reduce(acc1, __ldg(B+offset));
        }
        __syncwarp();
      }
      offset = rid*k+cid;
      C[offset] = acc1;
     // if(rid==0)
       //       printf("%f\t %d\n",C[offset],offset);

    }
    else { // threadIdx.y==blockDim.y-1
      int nout = (k-cid+31)/32;
      for (int jj=lb; jj<hb; jj+=32) {
        if (ptr<hb) {
          sh[thread_idx] = A_indices[ptr]*k;
          // sh[thread_idx] = __ldg(A_indices+ptr)*k;
        }
        __syncwarp();
        ptr += 32;
        for (int kk=0; kk<32&&jj+kk<hb; kk++) {
          offset = sh[(sm_offset+kk)] + cid;
          if (nout>0) {
          acc1 = sum_reduce(acc1, B[offset]);}
          // acc1 = sum_reduce(acc1, __ldg(B+offset)); }
        }
        __syncwarp();
      }
      offset = rid*k+cid;
      if (nout>0) {
      C[offset] = acc1;
     // if(rid==0)
//	      printf("%f\t %d\n",C[offset],offset);
      }
    }
  }
}
template <typename Idx, typename DType>
__global__ void topoSimpleSPMMKernel(
  int m, int k, const Idx* A_indptr, const Idx* A_indices, const DType* B, DType* C 
) {
  int rid = blockDim.y*blockIdx.x+threadIdx.y;
   //if(rid==0 && threadIdx.x==0)
     //           printf("hello");

  if (rid<m) {
    int lb = A_indptr[rid];
    int hb = A_indptr[(rid+1)];
    float acc1 = sum_init();
    int offset;
    for (int ptr=lb; ptr<hb; ptr++) {
      // offset = __ldg(A_indices+ptr)*k+threadIdx.x;
      // acc1 = sum_reduce(acc1, __ldg(B+offset));
      offset = A_indices[ptr]*k+threadIdx.x;
      acc1 = sum_reduce(acc1, B[offset]);
    }
    C[(rid*k+threadIdx.x)] = acc1;
    //if(rid==0)
      //        printf("%f\t %d\n",C[(rid*k+threadIdx.x)],(rid*k+threadIdx.x));

  }
}

template <typename Idx, typename DType>
void la_spmm(const int m, const int n, Idx* A_indptr, const Idx* A_indices, const DType* B_data, DType* C_data, int M, int W_SIZE)
{
    static float summation=0;
    float elapsed_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
	if (n<32) {
    const int row_per_block = 128/n;
    const int n_block = (m+row_per_block-1)/row_per_block;
    topoSimpleSPMMKernel<Idx,DType><<< dim3(n_block,1,1),dim3(n, row_per_block, 1), 0>>>(m,n,A_indptr,A_indices,B_data,C_data);
  }
  if (n<64 && n>32) {
    const int tile_k = (n+31)/32;
    const int n_block = (m+3)/4;
    topoCacheSPMMKernel<Idx,DType><<< dim3(n_block,tile_k,1), dim3(32,4,1), 128*sizeof(int)>>>(m,n,A_indptr,A_indices,B_data,C_data);
  }
  else {
    const int tile_k = (n+63)/64;
    const int n_block = (m+8-1)/8;
    topoCacheCoarsenSPMMKernel<Idx,DType><<< dim3(n_block,tile_k,1), dim3(32,8,1), 8*32*sizeof(int)>>>(m,n,A_indptr,A_indices,B_data,C_data);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time, start, stop);
  summation+=elapsed_time/1000;
  printf("ge-spmm time %.6f\t \n",summation);
  // printf("sum = %f\n",summation);

}


//----------------------------------reorderd_spmm kernel--------------------------------------------------------------
template <typename IdType, typename DType>
__global__ void dkernel_reorderd(int m, int k, const IdType* A_indptr, const IdType* A_indices, const DType* B, DType* C, IdType *reorderd_arr)
// __global__ void dkernel_ksn(int m, int k, int* A_indptr, int* A_indices, int* B, int* C)
{
  // int rid = reorderd_arr[blockIdx.x];
  int rid = blockIdx.x;
  __shared__ int Neb[Shared_mem_size];    //initilize the shared memory
  // extern __shared__ int result[];         //initilize external/ dynamic shared memory
  // __shared__ int start;
  // __shared__ int end;
  // __shared__ int pos;

  if(rid < m)
  {
    // int tid = threadIdx.x+blockIdx.x*blockDim.x;
    int tid = threadIdx.x;
    // if(threadIdx.x == 0)
    // {
    // pos = 0;
    // start = A_indptr[rid];
    // end = A_indptr[rid+1];
    // }
    // __syncthreads();
    int start = A_indptr[rid];      
    int end = A_indptr[rid+1];
    // int deg = end - start;
    // printf("bid: %d, tid: %d, start: %d, end: %d, deg: %d, start_org: %d, end_org: %d\n",blockIdx.x, threadIdx.x, start, end, deg);
    int neb_size = end-start;
    float sum = 0;
    // int sum1 = 0;
    if(neb_size < Shared_mem_size)
    {
      for(int ii=tid; ii<neb_size; ii+=blockDim.x)          //copy all neb in shared memory
      {
        Neb[ii] = A_indices[ii+start];
        // printf("tid: %d, bid: %d, neb: %d\n", tid, blockIdx.x, Neb[ii]);
      }
      __syncthreads();
      // __syncwarp();
      int tile = ceil((float)(k)/blockDim.x);             //Feature matrix divided initilizeo tiles.
      // printf("rid: %d, tid: %d, start: %d, end: %d, sum: %d, tile: %d, \n",rid, tid, start, end, sum, tile);
      for(int tt=0; tt<tile; tt++)
      {
        sum = 0.0;
        // for(int zz=start; zz<end; zz+=blockDim.x)
        for(int zz=0; zz<neb_size; zz++)
        {
          int offset = Neb[zz]*k+tid;
          offset += (tt*blockDim.x);                //find the location in danse matrix for multiplication.
          // printf("if -> rid: %d, tid: %d, strat: %d, end: %d, sum: %d, offset: %d, tile: %d, Neb: %d\n",rid, tid, start, end, sum, offset, tile, Neb[zz]);
          // sum = sum + B[offset];
          sum = sum_reduce(sum, B[offset]);
          // sum1 += B[offset + blockDim.x];
          // sum += offset;
          // }
          // __syncwarp();
          // __syncthreads();
        }
        int out = rid*k+tid;
        // printf("rid: %d, out: %d, tid: %d\n", rid, (blockDim.x*tt+out), tid);
        // C[blockDim.x * tt + out] = sum;
        // result[blockDim.x * tt + tid] = sum;
        C[blockDim.x * tt + out] = sum_reduce(C[blockDim.x * tt + out], sum);
        // C[(blockDim.x * tt + out) + blockDim.x] = C[(blockDim.x * tt + out) + blockDim.x] +sum;
      }
      // __syncthreads();
      // for(int ii=tid; ii<k; ii+=blockDim.x)
      // {
      //   int out = rid*k+ii;
      //   // C[blockDim.x * tt + out] = result[ii]; 
      //   C[out] = result[ii]; 
      // }
    }
    else
  {
      int sharetile = ceil((float)(neb_size)/Shared_mem_size);
      int remining_size = neb_size;
      // printf("else  ->  bid: %d, rid: %d\n", blockIdx.x, rid);
      for(int ii=0; ii<sharetile; ii++)
      {
        if (remining_size > Shared_mem_size)
        {
          for(int xx=tid; xx<Shared_mem_size; xx+=blockDim.x)
          {
            int index_ptr = ii*Shared_mem_size+xx;
            // printf("rid: %d, tid: %d, start : %d, end: %d, ii: %d, xx: %d, sharetile: %d, index_ptr: %d\n", rid, tid, start, end, ii, xx, sharetile, index_ptr);
            if(index_ptr < end)
            {
              // atomicAdd(&pos, 1);
              Neb[xx] = A_indices[index_ptr+start];
              // Neb[xx] = A_indices;
              // printf("rid: %d, tid: %d, start : %d, end: %d, ii: %d, xx: %d, sharetile: %d, index_ptr: %d, Neb_%d: %d\n", rid, tid, start, end, ii, xx, sharetile, index_ptr, xx, Neb[xx]);
            }
          }
          __syncthreads();
          // __syncwarp();
          int tile = ceil((float)(k)/blockDim.x);
          for(int tt=0; tt<tile; tt++)
          {
            sum = 0;
            // for(int zz=start; zz<end; zz+=blockDim.x)
            for(int zz=0; zz<Shared_mem_size; zz++)
            {
              int offset = Neb[zz]*k+tid;
              offset += (tt*blockDim.x);
              // printf("else -> rid: %d, tid: %d, strat: %d, end: %d, sum: %d, offset: %d, tile: %d, sharetile: %d Neb0: %d, Neb1: %d ii: %d, tt: %d, zz: %d, Neb_zz: %d\n",rid, tid, start, end, sum, offset, tile, sharetile, Neb[0], Neb[1], ii, tt, zz, Neb[zz]);
              // sum += B[offset];
              sum = sum_reduce(sum, B[offset]);
              // sum1 += B[offset + blockDim.x];
              // sum += offset;

              __syncwarp();
              // }
            }
            int out = rid*k+tid;
            C[blockDim.x * tt + out] = sum_reduce(C[blockDim.x * tt + out], sum);
            // result[blockDim.x * tt + tid] = result[blockDim.x * tt + tid] +sum;
          }
          remining_size = remining_size - Shared_mem_size;
        }
        else
      {
          for(int xx=tid; xx<remining_size; xx+=blockDim.x)
          {
            int index_ptr = ii*Shared_mem_size+xx;
            // printf("rid: %d, tid: %d, start : %d, end: %d, ii: %d, xx: %d, sharetile: %d, index_ptr: %d\n", rid, tid, start, end, ii, xx, sharetile, index_ptr);
            if(index_ptr < end)
            {
              Neb[xx] = A_indices[index_ptr+start];
              // printf("rid: %d, tid: %d, start : %d, end: %d, ii: %d, xx: %d, sharetile: %d, index_ptr: %d, Neb_%d: %d\n", rid, tid, start, end, ii, xx, sharetile, index_ptr, xx, Neb[xx]);
            }
          }
          __syncthreads();
          // __syncwarp();
          int tile = ceil((float)(k)/blockDim.x);
          for(int tt=0; tt<tile; tt++)
          {
            sum = 0;
            // for(int zz=start; zz<end; zz+=blockDim.x)
            for(int zz=0; zz<remining_size; zz++)
            {
              // sum = 0;
              // for(int kk=0; (zz+kk)<end ; kk++)
              // {
              int offset = Neb[zz]*k+tid;
              // int offset = Neb[kk]*k+tid;
              offset += (tt*blockDim.x);
              // printf("else -> rid: %d, tid: %d, strat: %d, end: %d, sum: %d, offset: %d, tile: %d, sharetile: %d Neb0: %d, Neb1: %d ii: %d, tt: %d, zz: %d, Neb_zz: %d\n",rid, tid, start, end, sum, offset, tile, sharetile, Neb[0], Neb[1], ii, tt, zz, Neb[zz]);
              // sum += B[offset];
              sum = sum_reduce(sum, B[offset]);
              // sum1 += B[offset + blockDim.x];
              // sum += offset;
              // }
              __syncwarp();
            }
            int out = rid*k+tid;
            C[blockDim.x * tt + out] = sum_reduce(C[blockDim.x * tt + out], sum);
            // result[blockDim.x * tt + tid] = result[blockDim.x * tt + tid] +sum;
          }
          // __syncthreads();
          // for(int tt=0; tt<tile; tt++)
          // {
          // for(int ii=tid; ii<k; ii+=blockDim.x)
          // {
          //   int out = rid*k+ii;
          //   // C[blockDim.x * tt + out] = result[ii]; 
          //   C[out] = result[ii]; 
          // }
        }
      }
    }
    // } //this is for the neighbour chek
  }
}


template <typename IdType, typename DType>
void reorderd_kernel_call(const int m, const int n, IdType* A_indptr, const IdType* A_indices, const DType* B_data, DType* C_data, int M, int W_SIZE, IdType* d_part_array)
{
    static float summation=0;
    float elapsed_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
	// if (n<32) {
 //    const int row_per_block = 128/n;
 //    const int n_block = (m+row_per_block-1)/row_per_block;
 //    topoSimpleSPMMKernel<Idx,DType><<< dim3(n_block,1,1),dim3(n, row_per_block, 1), 0>>>(m,n,A_indptr,A_indices,B_data,C_data);
 //  }
 //  if (n<64 && n>32) {
 //    const int tile_k = (n+31)/32;
 //    const int n_block = (m+3)/4;
 //    topoCacheSPMMKernel<Idx,DType><<< dim3(n_block,tile_k,1), dim3(32,4,1), 128*sizeof(int)>>>(m,n,A_indptr,A_indices,B_data,C_data);
 //  }
 //  else {
 //    const int tile_k = (n+63)/64;
 //    const int n_block = (m+8-1)/8;
 //    topoCacheCoarsenSPMMKernel<Idx,DType><<< dim3(n_block,tile_k,1), dim3(32,8,1), 8*32*sizeof(int)>>>(m,n,A_indptr,A_indices,B_data,C_data);
 //  }
  // printf("Indptr: "); 
  // for (int i=0; i<m; i++)
  // {
  //   std::cout<<A_indptr[i]<<" ";
  // }
  // printf("\n");
  if(n < 256)
  {
    // dkernel_ksn<<<M,N>>>(M,N,d_indptr,d_indices,d_b,d_c,len);
    dkernel_reorderd<IdType,DType><<<m, n >>>(m, n, A_indptr, A_indices, B_data, C_data, d_part_array);
  }
  else {
    // dkernel_ksn<<<M, BLOCKSIZE, N*sizeof(int)>>>(M, N, d_indptr, d_indices, d_b, d_c, min, max, processed_arr);
    dkernel_reorderd<IdType,DType><<<m, 256 >>>(m, n, A_indptr, A_indices, B_data, C_data, d_part_array);
  }
  cudaDeviceSynchronize();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time, start, stop);
  summation+=elapsed_time/1000;
  printf("re_orderd_spmm time %.6f\t \n",summation);
  // printf("sum = %f\n",summation);

}




template <
    typename Idx, typename DType, typename BinaryOp, typename ReduceOp,
    bool UseBcast = false, bool UseIdx = false>
__global__ void SpMMCsrKernel(
    const DType* __restrict__ ufeat, const DType* __restrict__ efeat,
    DType* __restrict__ out, Idx* __restrict__ arg_u, Idx* __restrict__ arg_e,
    const Idx* __restrict__ indptr, const Idx* __restrict__ indices,
    const Idx* __restrict__ edge_map, int64_t num_rows, int64_t num_cols,
    const int64_t* __restrict__ ubcast_off,
    const int64_t* __restrict__ ebcast_off, int64_t ufeat_len,
    int64_t efeat_len, int64_t out_len) {
  // SPMM with CSR.
  int ty = blockIdx.x * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.x;
  const int stride_x = blockDim.x * gridDim.y;
  while (ty < num_rows) {
    int tx = blockIdx.y * blockDim.x + threadIdx.x;
    while (tx < out_len) {
      typename accum_dtype<DType>::type local_accum = ReduceOp::zero();
      Idx local_argu = 0, local_arge = 0;
      const int lhs_add = UseBcast ? ubcast_off[tx] : tx;
      const int rhs_add = UseBcast ? ebcast_off[tx] : tx;
      for (Idx i = indptr[ty]; i < indptr[ty + 1]; ++i) {
        const Idx eid = UseIdx ? _ldg(edge_map + i) : i;
        const Idx cid = _ldg(indices + i);
        const DType* uoff =
            BinaryOp::use_lhs ? (ufeat + cid * ufeat_len) : nullptr;
        const DType* eoff =
            BinaryOp::use_rhs ? (efeat + eid * efeat_len) : nullptr;
        DType out = BinaryOp::Call(uoff + lhs_add, eoff + rhs_add);
        ReduceOp::Call(&local_accum, &local_argu, &local_arge, out, cid, eid);
      }
      // The use of += is to compute cross-type reducing on heterogeneous graph
      // when reduce op is `sum`.
      //     C = SpMM(SpA, B) + C
      // Separate kernel `SpMMCmpCsrHeteroKernel` is used for max- and
      // min-reducer. It does not affect the output on homogeneous graph as
      // `out` is initialized to zero.
      out[ty * out_len + tx] += static_cast<DType>(local_accum);
      if (ReduceOp::require_arg && BinaryOp::use_lhs)
        arg_u[ty * out_len + tx] = local_argu;
      if (ReduceOp::require_arg && BinaryOp::use_rhs)
        arg_e[ty * out_len + tx] = local_arge;
      tx += stride_x;
    }
    ty += stride_y;
  }
}

/**
 * @brief CUDA kernel of SpMM-Min/Max on Csr format.
 * @note it uses node parallel strategy, different threadblocks (on y-axis)
 *       is responsible for the computation on different destination nodes.
 *       Threadblocks on the x-axis are responsible for the computation on
 *       different positions in feature dimension.
 */
template <
    typename Idx, typename DType, typename BinaryOp, typename ReduceOp,
    bool UseBcast = false, bool UseIdx = false>
__global__ void SpMMCmpCsrHeteroKernel(
    const DType* __restrict__ ufeat, const DType* __restrict__ efeat,
    DType* __restrict__ out, Idx* __restrict__ arg_u, Idx* __restrict__ arg_e,
    Idx* __restrict__ arg_u_ntype, Idx* __restrict__ arg_e_etype,
    const Idx* __restrict__ indptr, const Idx* __restrict__ indices,
    const Idx* __restrict__ edge_map, int64_t num_rows, int64_t num_cols,
    const int64_t* __restrict__ ubcast_off,
    const int64_t* __restrict__ ebcast_off, int64_t ufeat_len,
    int64_t efeat_len, int64_t out_len, const int src_type, const int etype) {
  // SPMM with CSR.
  int ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  const int stride_x = blockDim.x * gridDim.x;
  while (ty < num_rows) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    while (tx < out_len) {
      using accum_type = typename accum_dtype<DType>::type;
      accum_type local_accum = static_cast<accum_type>(
          out[ty * out_len + tx]);  // ReduceOp::zero();
      Idx local_argu = 0, local_arge = 0;
      const int lhs_add = UseBcast ? ubcast_off[tx] : tx;
      const int rhs_add = UseBcast ? ebcast_off[tx] : tx;
      for (Idx i = indptr[ty]; i < indptr[ty + 1]; ++i) {
        const Idx eid = UseIdx ? _ldg(edge_map + i) : i;
        const Idx cid = _ldg(indices + i);
        const DType* uoff =
            BinaryOp::use_lhs ? (ufeat + cid * ufeat_len) : nullptr;
        const DType* eoff =
            BinaryOp::use_rhs ? (efeat + eid * efeat_len) : nullptr;
        DType tmp_out = BinaryOp::Call(uoff + lhs_add, eoff + rhs_add);
        ReduceOp::Call(
            &local_accum, &local_argu, &local_arge, tmp_out, cid, eid);
      }
      // Update output only when max/min values are different that original
      // output
      DType new_out = static_cast<DType>(local_accum);
      if (out[ty * out_len + tx] != new_out) {
        out[ty * out_len + tx] = new_out;
        if (ReduceOp::require_arg && BinaryOp::use_lhs) {
          arg_u[ty * out_len + tx] = local_argu;
          arg_u_ntype[ty * out_len + tx] = src_type;
        }
        if (ReduceOp::require_arg && BinaryOp::use_rhs) {
          arg_e[ty * out_len + tx] = local_arge;
          arg_e_etype[ty * out_len + tx] = etype;
        }
      }
      tx += stride_x;
    }
    ty += stride_y;
  }
}

/**
 * @brief CUDA implementation of g-SpMM on Coo format.
 * @param bcast Broadcast information.
 * @param coo The Coo matrix.
 * @param ufeat The feature on source nodes.
 * @param efeat The feature on edges.
 * @param out The result feature on destination nodes.
 * @param argu Arg-Min/Max on source nodes, which refers the source node indices
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max
 * reducer.
 * @param arge Arg-Min/Max on edges. which refers the source node indices
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max
 * reducer.
 */
template <typename Idx, typename DType, typename BinaryOp, typename ReduceOp>
void SpMMCoo(
    const BcastOff& bcast, const COOMatrix& coo, NDArray ufeat, NDArray efeat,
    NDArray out, NDArray argu, NDArray arge) {
  /**
   * TODO(Xin): Disable half precision for SpMMCoo due to the round-off error.
   * We should use fp32 for the accumulation but it's hard to modify the 
   * current implementation.
   */
#if BF16_ENABLED
  if (std::is_same<DType, __half>::value ||
      std::is_same<DType, __nv_bfloat16>::value)
#else
  if (std::is_same<DType, __half>::value)
#endif  // BF16_ENABLED
    LOG(FATAL) << "SpMMCoo doesn't support half precision fow now. "
               << "Please use SpMMCsr instead by allowing the graph "
               << "materialize CSR/CSC formats.";
  const Idx *row = coo.row.Ptr<Idx>(), *col = coo.col.Ptr<Idx>(),
            *edge_map = coo.data.Ptr<Idx>();
  const DType *ufeat_data = ufeat.Ptr<DType>(),
              *efeat_data = efeat.Ptr<DType>();
  DType* out_data = out.Ptr<DType>();
  Idx *argu_data = argu.Ptr<Idx>(), *arge_data = arge.Ptr<Idx>();
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  const int64_t N = coo.num_rows, M = coo.num_cols, E = coo.row->shape[0];

  int64_t *ubcast_off = nullptr, *ebcast_off = nullptr;
  int64_t len = bcast.out_len, lhs_len = bcast.lhs_len, rhs_len = bcast.rhs_len;

  int64_t out_size = out.NumElements();
  const int nt = FindNumThreads(out_size);
  const int nb = (out_size + nt - 1) / nt;
  CUDA_KERNEL_CALL(
      _FillKernel, nb, nt, 0, stream, out_data, out_size, ReduceOp::zero());

  const int ntx = FindNumThreads(len);
  const int nty = CUDA_MAX_NUM_THREADS / ntx;
  const int nbx = (len + ntx - 1) / ntx;
  const int nby = FindNumBlocks<'y'>((E + nty - 1) / nty);
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  const bool use_idx = !IsNullArray(coo.data);

  BCAST_IDX_CTX_SWITCH(bcast, use_idx, ufeat->ctx, ubcast_off, ebcast_off, {
    CUDA_KERNEL_CALL(
        (SpMMCooKernel<Idx, DType, BinaryOp, ReduceOp, UseBcast, UseIdx>),
        nblks, nthrs, 0, stream, ufeat_data, efeat_data, out_data, argu_data,
        arge_data, row, col, edge_map, N, M, E, ubcast_off, ebcast_off, lhs_len,
        rhs_len, len);
    if (ReduceOp::require_arg) {
      CUDA_KERNEL_CALL(
          (ArgSpMMCooKernel<Idx, DType, BinaryOp, ReduceOp, UseBcast, UseIdx>),
          nblks, nthrs, 0, stream, ufeat_data, efeat_data, out_data, argu_data,
          arge_data, row, col, edge_map, N, M, E, ubcast_off, ebcast_off,
          lhs_len, rhs_len, len);
    }
  });
}

/**
 * @brief CUDA implementation of g-SpMM on Csr format.
 * @param bcast Broadcast information.
 * @param csr The Csr matrix.
 * @param ufeat The feature on source nodes.
 * @param efeat The feature on edges.
 * @param out The result feature on destination nodes.
 * @param argu Arg-Min/Max on source nodes, which refers the source node indices
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max
 * reducer.
 * @param arge Arg-Min/Max on edges. which refers the source node indices
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max
 * reducer.
 */

template <typename Idx, typename DType>
void CustomCsr(const BcastOff& bcast,
    const CSRMatrix& csr,
    const DType* B_data, const DType* A_data,
    DType* C_data,
    int x_length){
	 const int m = csr.num_rows;
         const int n = x_length;
	 int W_SIZE=1024;
         int T_MBlock;
         T_MBlock=m/W_SIZE;
         if(m%W_SIZE)
            T_MBlock+=1;

        T_MBlock *= W_SIZE;
        la_spmm<Idx,DType>(m,n,static_cast<Idx*>(csr.indptr->data),static_cast<Idx*>(csr.indices->data),B_data,C_data,T_MBlock,W_SIZE);
	//la_spmm<int64_t,float>(m,n,static_cast<Idx*>(csr.indptr->data),static_cast<Idx*>(csr.indices->data),B_data,C_data,T_MBlock,W_SIZE);
	//la_spmm<int32_t,double>(m,n,static_cast<Idx*>(csr.indptr->data),static_cast<Idx*>(csr.indices->data),B_data,C_data,T_MBlock,W_SIZE);
	//la_spmm<int64_t,double>(m,n,static_cast<Idx*>(csr.indptr->data),static_cast<Idx*>(csr.indices->data),B_data,C_data,T_MBlock,W_SIZE);

}

template <typename IdType, typename DType>
void reorderd_Csr(const BcastOff& bcast,
                  const CSRMatrix& csr,
                  const DType* B_data, const DType* A_data,
                  DType* C_data,
                  int x_length,
                  IdType* d_part_array)
{
  const int m = csr.num_rows;
  const int n = x_length;
  // printf("num_vertex: %d\n",m);
  // printf("feat_size: %d\n",n);
  int W_SIZE=1024;
  int T_MBlock;
  T_MBlock=m/W_SIZE;
  if(m%W_SIZE)
    T_MBlock+=1;

  T_MBlock *= W_SIZE;
  reorderd_kernel_call<IdType,DType>(m,n,static_cast<IdType*>(csr.indptr->data),static_cast<IdType*>(csr.indices->data),B_data,C_data,T_MBlock,W_SIZE, d_part_array);
}


template <typename Idx, typename DType, typename BinaryOp, typename ReduceOp>
void SpMMCsr(
    const BcastOff& bcast, const CSRMatrix& csr, NDArray ufeat, NDArray efeat,
    NDArray out, NDArray argu, NDArray arge) {
  const Idx* indptr = csr.indptr.Ptr<Idx>();
  const Idx* indices = csr.indices.Ptr<Idx>();
  const Idx* edge_map = csr.data.Ptr<Idx>();
  const DType* ufeat_data = ufeat.Ptr<DType>();
  const DType* efeat_data = efeat.Ptr<DType>();
  DType* out_data = out.Ptr<DType>();
  Idx* argu_data = argu.Ptr<Idx>();
  Idx* arge_data = arge.Ptr<Idx>();

  cudaStream_t stream = runtime::getCurrentCUDAStream();

  int64_t *ubcast_off = nullptr, *ebcast_off = nullptr;
  int64_t len = bcast.out_len, lhs_len = bcast.lhs_len, rhs_len = bcast.rhs_len;
  const int ntx = FindNumThreads(len);
  const int nty = CUDA_MAX_NUM_THREADS / ntx;
  const int nby = (len + ntx - 1) / ntx;
  const int nbx = FindNumBlocks<'x'>((csr.num_rows + nty - 1) / nty);
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  const bool use_idx = !IsNullArray(csr.data);

  BCAST_IDX_CTX_SWITCH(
      bcast, use_idx, ufeat->ctx, ubcast_off, ebcast_off,
      {CUDA_KERNEL_CALL(
          (SpMMCsrKernel<Idx, DType, BinaryOp, ReduceOp, UseBcast, UseIdx>),
          nblks, nthrs, 0, stream, ufeat_data, efeat_data, out_data, argu_data,
          arge_data, indptr, indices, edge_map, csr.num_rows, csr.num_cols,
          ubcast_off, ebcast_off, lhs_len, rhs_len, len)});
}

/**
 * @brief CUDA kernel of SpMM-Min/Max on Csr format on heterogeneous graph.
 * @param bcast Broadcast information.
 * @param csr The Csr matrix.
 * @param ufeat The feature on source nodes.
 * @param efeat The feature on edges.
 * @param out The result feature on destination nodes.
 * @param argu Arg-Min/Max on source nodes, which refers the source node indices
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max
 * reducer.
 * @param arge Arg-Min/Max on edges. which refers the source node indices
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max
 * reducer.
 * @param argu_ntype Node type of the arg-Min/Max on source nodes, which refers
 * the source node types correspond to the minimum/maximum values of reduction
 * result on destination nodes. It's useful in computing gradients of Min/Max
 * reducer.
 * @param arge_etype Edge-type of the arg-Min/Max on edges. which refers the
 * source node indices correspond to the minimum/maximum values of reduction
 * result on destination nodes. It's useful in computing gradients of Min/Max
 * reducer.
 * @param src_type Node type of the source nodes of an etype
 * @param etype Edge type
 */
template <typename Idx, typename DType, typename BinaryOp, typename ReduceOp>
void SpMMCmpCsrHetero(
    const BcastOff& bcast, const CSRMatrix& csr, NDArray ufeat, NDArray efeat,
    NDArray out, NDArray argu, NDArray arge, NDArray argu_ntype,
    NDArray arge_etype, const int src_type, const int etype) {
  const Idx* indptr = csr.indptr.Ptr<Idx>();
  const Idx* indices = csr.indices.Ptr<Idx>();
  const Idx* edge_map = csr.data.Ptr<Idx>();
  const DType* ufeat_data = ufeat.Ptr<DType>();
  const DType* efeat_data = efeat.Ptr<DType>();
  DType* out_data = out.Ptr<DType>();
  Idx* argu_data = argu.Ptr<Idx>();
  Idx* arge_data = arge.Ptr<Idx>();

  cudaStream_t stream = runtime::getCurrentCUDAStream();

  int64_t *ubcast_off = nullptr, *ebcast_off = nullptr;
  int64_t len = bcast.out_len, lhs_len = bcast.lhs_len, rhs_len = bcast.rhs_len;
  const int ntx = FindNumThreads(len);
  const int nty = CUDA_MAX_NUM_THREADS / ntx;
  const int nbx = (len + ntx - 1) / ntx;
  const int nby = FindNumBlocks<'y'>((csr.num_rows + nty - 1) / nty);
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  const bool use_idx = !IsNullArray(csr.data);

  BCAST_IDX_CTX_SWITCH(
      bcast, use_idx, ufeat->ctx, ubcast_off, ebcast_off,
      {CUDA_KERNEL_CALL(
          (SpMMCmpCsrHeteroKernel<
              Idx, DType, BinaryOp, ReduceOp, UseBcast, UseIdx>),
          nblks, nthrs, 0, stream, ufeat_data, efeat_data, out_data, argu_data,
          arge_data, static_cast<Idx*>(argu_ntype->data),
          static_cast<Idx*>(arge_etype->data), indptr, indices, edge_map,
          csr.num_rows, csr.num_cols, ubcast_off, ebcast_off, lhs_len, rhs_len,
          len, src_type, etype)});
}

}  // namespace cuda
}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CUDA_SPMM_CUH_
