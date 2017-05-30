#pragma once

#include <cusp/multiply.h>
#include <cusp/system/cuda/detail/multiply/coo_flat_k.h>
#include <cusp/system/cuda/detail/multiply/csr_scalar.h>

#include "bytes_per_spmv.h"
#include "utility.h"
#include "../timer.h"

#include <string>
#include <iostream>
#include <stdio.h>

// lld: whether only run one test kernel
#define TEST_MODE
#ifndef TEST_MODE_SINGLE_ITER
  #define TEST_MODE_SINGLE_ITER 2
#endif
#define GPU_TEST_MODE // lld: this must be open when running experiments
#define ENLARGE_INPUT
#ifndef INPUT_TIME
  #define INPUT_TIME 100
#endif
#define DEBUG_PRINT printf("here: %d\n", __LINE__); fflush(stdout);
// lld: whether unified memory is used
//#define CUDA_UM_ALLOC
//#define CUDA_UM_PREFETCH
//#define CUDA_UM_HOST_PREFETCH
//#define CUDA_UM_DUPLICATE
//#define CUDA_UM_PREFERRED_GPU
//#define CUDA_UM_PREFERRED_CPU
//#define CUDA_HOST_ALLOC
//#define CUDA_HOST_ACCESSEDBY
#if defined(CUDA_UM_ALLOC) || defined(CUDA_HOST_ALLOC)
#error "This is only for hybrid allocation."
#endif
//#define CUDA_HYBRID_ALLOC
//#define CUDA_TRUE_HYBRID_ALLOC
#ifndef CUDA_DEVICE_ALLOC_SIZE
  #define CUDA_DEVICE_ALLOC_SIZE ( (size_t) (14 * 1024 * 1024) / (sizeof(IndexType) + sizeof(ValueType)) * 1024 )
#endif

const char * BENCHMARK_OUTPUT_FILE_NAME = "benchmark_output.log";
int global_device_id;

template <typename HostMatrix, typename TestMatrix, typename TestKernel>
float check_spmv(HostMatrix& host_matrix, TestMatrix& test_matrix, TestKernel test_kernel)
{
    typedef typename TestMatrix::index_type   IndexType; // ASSUME same as HostMatrix::index_type
    typedef typename TestMatrix::value_type   ValueType; // ASSUME same as HostMatrix::value_type
    typedef typename TestMatrix::memory_space MemorySpace;

    const IndexType M = host_matrix.num_rows;
    const IndexType N = host_matrix.num_cols;

    // create host input (x) and output (y) vectors
    cusp::array1d<ValueType,cusp::host_memory> host_x(N);
    cusp::array1d<ValueType,cusp::host_memory> host_y(M);
    //for(IndexType i = 0; i < N; i++) host_x[i] = (rand() % 21) - 10;
    for(IndexType i = 0; i < N; i++) host_x[i] = (int(i % 21) - 10);
    for(IndexType i = 0; i < M; i++) host_y[i] = 0;

    // create test input (x) and output (y) vectors
    cusp::array1d<ValueType, MemorySpace> test_x(host_x.begin(), host_x.end());
    cusp::array1d<ValueType, MemorySpace> test_y(host_y.begin(), host_y.end());

    // compute SpMV on host and device
    cusp::multiply(host_matrix, host_x, host_y);
    test_kernel(test_matrix, test_x, test_y);

    // compare results
    cusp::array1d<ValueType,cusp::host_memory> test_y_copy(test_y.begin(), test_y.end());
    double error = l2_error(M, thrust::raw_pointer_cast(&test_y_copy[0]), thrust::raw_pointer_cast(&host_y[0]));

    return error;
}

template <typename HostMatrix, typename TestMatrix, typename TestKernel>
float check_block_spmv(HostMatrix& host_matrix, TestMatrix& test_matrix, TestKernel test_kernel, size_t num_cols)
{
    typedef typename TestMatrix::index_type   IndexType; // ASSUME same as HostMatrix::index_type
    typedef typename TestMatrix::value_type   ValueType; // ASSUME same as HostMatrix::value_type
    typedef typename TestMatrix::memory_space MemorySpace;

    const IndexType M = host_matrix.num_rows;
    const IndexType N = host_matrix.num_cols;

    // create host input (x) and output (y) vectors
    cusp::array2d<ValueType,cusp::host_memory> host_x(N, num_cols);
    cusp::array2d<ValueType,cusp::host_memory> host_y(M, num_cols, 0);

    // initialize host_x to random array
    cusp::copy(cusp::random_array<ValueType>(host_x.values.size()), host_x.values);

    //// create test input (x) and output (y) vectors
    //cusp::array2d<ValueType, MemorySpace> test_x(host_x);
    //cusp::array2d<ValueType, MemorySpace> test_y(host_y);

    // lld: manuallly allocate memory
    cusp::array2d<ValueType, MemorySpace> test_x;
    cusp::array2d<ValueType, MemorySpace> test_y;
    test_x.num_cols = num_cols;
    test_y.num_cols = num_cols;
    test_x.num_rows = N;
    test_y.num_rows = M;
    cudaMallocManaged((void **)&test_x.values.my_begin, N*num_cols*sizeof(ValueType));
    cudaMallocManaged((void **)&test_y.values.my_begin, M*num_cols*sizeof(ValueType));
    for(unsigned long long i = 0; i < N*num_cols; i++)
      test_x.values.my_begin[i] = host_x.values.begin()[i];
    for(unsigned long long i = 0; i < M*num_cols; i++)
      test_y.values.my_begin[i] = (ValueType)0;

    // compute SpMV on host and device
    cusp::multiply(host_matrix, host_x, host_y);
    test_kernel(test_matrix, test_x, test_y);

    // compare results
    //cusp::array2d<ValueType,cusp::host_memory> test_y_copy(test_y);
    cusp::array2d<ValueType,cusp::host_memory> test_y_copy(host_y); // lld

    // lld: copy results back
    cudaDeviceSynchronize();
    for(unsigned long long i = 0; i < M*num_cols; i++)
      test_y_copy.values.begin()[i] = test_y.values.my_begin[i];

    ValueType error = 0;
    for(size_t i = 0; i < num_cols; i++)
        error = std::max(error, l2_error(test_y_copy.column(i), host_y.column(i)));

    return error;
}

template <typename TestMatrix, typename TestKernel>
float time_spmv(TestMatrix& test_matrix, TestKernel test_spmv, double seconds = 3.0, size_t min_iterations = 100, size_t max_iterations = 500)
{
    typedef typename TestMatrix::index_type   IndexType; // ASSUME same as HostMatrix::index_type
    typedef typename TestMatrix::value_type   ValueType; // ASSUME same as HostMatrix::value_type
    typedef typename TestMatrix::memory_space MemorySpace;

    const IndexType M = test_matrix.num_rows;
    const IndexType N = test_matrix.num_cols;

    // create test input (x) and output (y) vectors
    cusp::array1d<ValueType, MemorySpace> test_x(N);
    cusp::array1d<ValueType, MemorySpace> test_y(M);

    // warmup
    timer time_one_iteration;
    test_spmv(test_matrix, test_x, test_y);
    cudaThreadSynchronize();
    double estimated_time = time_one_iteration.seconds_elapsed();

    // determine # of seconds dynamically
    size_t num_iterations;
    if (estimated_time == 0)
        num_iterations = max_iterations;
    else
        num_iterations = std::min(max_iterations, std::max(min_iterations, (size_t) (seconds / estimated_time)) );

    // time several SpMV iterations
    timer t;
    for(size_t i = 0; i < num_iterations; i++)
        test_spmv(test_matrix, test_x, test_y);
    cudaThreadSynchronize();

    float sec_per_iteration = t.seconds_elapsed() / num_iterations;

    return sec_per_iteration;
}

template <typename TestMatrix, typename TestKernel>
float time_spmv_block(TestMatrix& test_matrix, size_t num_cols, TestKernel test_spmv, double seconds = 3.0, size_t min_iterations = 100, size_t max_iterations = 500)
{
    typedef typename TestMatrix::index_type   IndexType; // ASSUME same as HostMatrix::index_type
    typedef typename TestMatrix::value_type   ValueType; // ASSUME same as HostMatrix::value_type
    typedef typename TestMatrix::memory_space MemorySpace;

    const IndexType M = test_matrix.num_rows;
    const IndexType N = test_matrix.num_cols;

    // lld: my timer
    timer my;

    //// create test input (x) and output (y) vectors
    //cusp::array2d<ValueType, MemorySpace, cusp::row_major> test_x(N, num_cols);
    //cusp::array2d<ValueType, MemorySpace, cusp::row_major> test_y(M, num_cols);

    // lld: manuallly allocate memory
    cusp::array2d<ValueType, MemorySpace, cusp::row_major> test_x;
    cusp::array2d<ValueType, MemorySpace, cusp::row_major> test_y;
    test_x.num_cols = num_cols;
    test_y.num_cols = num_cols;
    test_x.num_rows = N;
    test_y.num_rows = M;
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&test_x.values.my_begin, N*num_cols*sizeof(ValueType)));
    //CUDA_SAFE_CALL_NO_SYNC(cudaMallocManaged((void **)&test_x.values.my_begin, N*num_cols*sizeof(ValueType)));
    //CUDA_SAFE_CALL_NO_SYNC(cudaMallocHost((void **)&test_x.values.my_begin, N*num_cols*sizeof(ValueType)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&test_y.values.my_begin, M*num_cols*sizeof(ValueType)));
    //CUDA_SAFE_CALL_NO_SYNC(cudaMallocManaged((void **)&test_y.values.my_begin, M*num_cols*sizeof(ValueType)));
    //CUDA_SAFE_CALL_NO_SYNC(cudaMallocHost((void **)&test_y.values.my_begin, M*num_cols*sizeof(ValueType)));

    // warmup
    timer time_one_iteration;
    test_spmv(test_matrix, test_x, test_y);
    cudaThreadSynchronize();
    double estimated_time = time_one_iteration.seconds_elapsed();
    printf("\t%-20s: %8.4f ms\n", "warmup", estimated_time * 1e3);

    // determine # of seconds dynamically
    size_t num_iterations;
    // lld: iteration number
#ifdef TEST_MODE_SINGLE_ITER
    num_iterations = TEST_MODE_SINGLE_ITER;
#else
    if (estimated_time == 0)
        num_iterations = max_iterations;
    else
        num_iterations = std::min(max_iterations, std::max(min_iterations, (size_t) (seconds / estimated_time)) );
#endif

    // time several SpMV iterations
    timer t;
    for(size_t i = 0; i < num_iterations; i++)
      test_spmv(test_matrix, test_x, test_y);
    cudaThreadSynchronize();

    float sec_per_iteration = t.seconds_elapsed() / num_iterations;

    // lld: more accurate total execution time
    double my_time = my.seconds_elapsed() * 1e3;
    printf("\t%-20s: %8.4f ms\n", "runtime", my_time);

    // lld: free vector memory
    cudaFree(test_x.values.my_begin);
    cudaFree(test_y.values.my_begin);

    return sec_per_iteration;
}

template <typename HostMatrix, typename TestMatrixOnHost, typename TestMatrixOnDevice, typename TestKernel>
void test_spmv(std::string         kernel_name,
               HostMatrix&         host_matrix,
               TestMatrixOnHost&   test_matrix_on_host,
               TestMatrixOnDevice& test_matrix_on_device,
               TestKernel          test_spmv)
{
    float error = check_spmv(host_matrix, test_matrix_on_device, test_spmv);
    float time  = time_spmv(              test_matrix_on_device, test_spmv);
    float gbyte = bytes_per_spmv(test_matrix_on_host);

    float GFLOPs = (time == 0) ? 0 : (2 * host_matrix.num_entries / time) / 1e9;
    float GBYTEs = (time == 0) ? 0 : (gbyte / time)                       / 1e9;

    printf("\t%-20s: %8.4f ms ( %5.2f GFLOP/s %5.1f GB/s) [L2 error %f]\n", kernel_name.c_str(), 1e3 * time, GFLOPs, GBYTEs, error);

    //record results to file
    FILE * fid = fopen(BENCHMARK_OUTPUT_FILE_NAME, "a");
    fprintf(fid, "kernel=%s gflops=%f gbytes=%f msec=%f\n", kernel_name.c_str(), GFLOPs, GBYTEs, 1e3 * time);
    fclose(fid);
}

template <typename HostMatrix, typename TestMatrixOnHost, typename TestMatrixOnDevice, typename TestKernel>
void test_spmv_block(std::string         kernel_name,
                     size_t              num_cols,
                     HostMatrix&         host_matrix,
                     TestMatrixOnHost&   test_matrix_on_host,
                     TestMatrixOnDevice& test_matrix_on_device,
                     TestKernel          test_spmv)
{
    std::ostringstream block_string;
    block_string << "(" << num_cols << ")";
    kernel_name += block_string.str();

#ifdef GPU_TEST_MODE
    float time  = time_spmv_block(test_matrix_on_device, num_cols, test_spmv);
    printf("\t%-20s: %8.4f ms ( %5.2f GFLOP/s %5.1f GB/s) [L2 error %f]\n", kernel_name.c_str(), 1e3 * time, 0.0, 0.0, 0.0);
#else
    float error = check_block_spmv(host_matrix, test_matrix_on_device, test_spmv, num_cols);
    float time  = time_spmv_block(test_matrix_on_device, num_cols, test_spmv);
    float gbyte = bytes_per_spmv_block(test_matrix_on_host, num_cols);

    float GFLOPs = (time == 0) ? 0 : (num_cols * 2 * host_matrix.num_entries / time) / 1e9;
    float GBYTEs = (time == 0) ? 0 : (gbyte / time)                       / 1e9;

    printf("\t%-20s: %8.4f ms ( %5.2f GFLOP/s %5.1f GB/s) [L2 error %f]\n", kernel_name.c_str(), 1e3 * time, GFLOPs, GBYTEs, error);

    //record results to file
    FILE * fid = fopen(BENCHMARK_OUTPUT_FILE_NAME, "a");
    fprintf(fid, "kernel=%s gflops=%f gbytes=%f msec=%f\n", kernel_name.c_str(), GFLOPs, GBYTEs, 1e3 * time);
    fclose(fid);
#endif
}

/////////////////////////////////////////////////////
// These methods test specific formats and kernels //
/////////////////////////////////////////////////////

template <typename HostMatrix>
void test_coo(HostMatrix& host_matrix)
{
    typedef typename HostMatrix::index_type IndexType;
    typedef typename HostMatrix::value_type ValueType;

    // convert HostMatrix to TestMatrix on host
    cusp::coo_matrix<IndexType, ValueType, cusp::host_memory> test_matrix_on_host(host_matrix);

    // transfer TestMatrix to device
    typedef typename cusp::coo_matrix<IndexType, ValueType, cusp::device_memory> DeviceMatrix;
    typedef typename cusp::array1d<ValueType,cusp::device_memory>                DeviceArray;
    DeviceMatrix test_matrix_on_device(test_matrix_on_host);

    test_spmv("coo",     host_matrix, test_matrix_on_host, test_matrix_on_device, cusp::multiply<DeviceMatrix,DeviceArray,DeviceArray>);
}

template <typename HostMatrix>
void test_csr(HostMatrix& host_matrix)
{
    typedef typename HostMatrix::index_type IndexType;
    typedef typename HostMatrix::value_type ValueType;

    // convert HostMatrix to TestMatrix on host
#ifdef TEST_MODE
  #define test_matrix_on_host host_matrix
#else
    cusp::csr_matrix<IndexType, ValueType, cusp::host_memory> test_matrix_on_host(host_matrix);
#endif

    // transfer csr_matrix to device
    typedef typename cusp::csr_matrix<IndexType, ValueType, cusp::device_memory> DeviceMatrix;
    typedef typename cusp::array1d<ValueType,cusp::device_memory>                DeviceArray;
    typedef typename cusp::array2d<ValueType,cusp::device_memory>                DeviceArray2d;
    //DeviceMatrix test_matrix_on_device(test_matrix_on_host);
    cudaEvent_t start;
    cudaEvent_t end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEvent_t total_start;
    cudaEvent_t total_end;
    cudaEventCreate(&total_start);
    cudaEventCreate(&total_end);
    float elapsed_time;
    DeviceMatrix test_matrix_on_device;
#ifdef ENLARGE_INPUT
    test_matrix_on_device.num_rows = test_matrix_on_host.num_rows * INPUT_TIME;
    test_matrix_on_device.num_cols = test_matrix_on_host.num_cols;
    test_matrix_on_device.num_entries = test_matrix_on_host.num_entries * INPUT_TIME;
#else
    test_matrix_on_device.num_rows = test_matrix_on_host.num_rows;
    test_matrix_on_device.num_cols = test_matrix_on_host.num_cols;
    test_matrix_on_device.num_entries = test_matrix_on_host.num_entries;
#endif
#ifdef CUDA_TRUE_HYBRID_ALLOC
    if(test_matrix_on_device.num_entries <= CUDA_DEVICE_ALLOC_SIZE) {
        printf("\tMatrix is too small for truly hybrid allocation. (%lu <= %lu)\n", test_matrix_on_device.num_entries, CUDA_DEVICE_ALLOC_SIZE);
        exit(0);
    }
#endif
    printf("\tHere we go.\n");
    cudaEventRecord(total_start,0);

#if defined (CUDA_HYBRID_ALLOC)
    cudaEventRecord(start,0);
    //CUDA_SAFE_CALL_NO_SYNC(cudaMallocManaged((void **)&test_matrix_on_device.row_offsets.my_begin, (test_matrix_on_device.num_rows+1)*sizeof(IndexType)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&test_matrix_on_device.row_offsets.my_begin, (test_matrix_on_device.num_rows+1)*sizeof(IndexType)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMallocManaged((void **)&test_matrix_on_device.column_indices.my_begin, test_matrix_on_device.num_entries*sizeof(IndexType)));
    //CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&test_matrix_on_device.column_indices.my_begin, test_matrix_on_device.num_entries*sizeof(IndexType)));
    //CUDA_SAFE_CALL_NO_SYNC(cudaMallocManaged((void **)&test_matrix_on_device.values.my_begin, test_matrix_on_device.num_entries*sizeof(ValueType)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&test_matrix_on_device.values.my_begin, test_matrix_on_device.num_entries*sizeof(ValueType)));
#if defined (CUDA_UM_HOST_PREFETCH)
    //cudaMemPrefetchAsync(test_matrix_on_device.row_offsets.my_begin, (test_matrix_on_device.num_rows+1)*sizeof(IndexType), cudaCpuDeviceId);
    cudaMemPrefetchAsync(test_matrix_on_device.column_indices.my_begin, test_matrix_on_device.num_entries*sizeof(IndexType), cudaCpuDeviceId);
    //cudaMemPrefetchAsync(test_matrix_on_device.values.my_begin, test_matrix_on_device.num_entries*sizeof(ValueType), cudaCpuDeviceId);
#endif
    // lld: reorganize host matrix to be linear
    IndexType * host_row_offsets = new IndexType[test_matrix_on_device.num_rows+1];
    //IndexType * host_column_indices = new IndexType[test_matrix_on_device.num_entries];
    ValueType * host_values = new ValueType[test_matrix_on_device.num_entries];
#ifdef ENLARGE_INPUT
    size_t originalsize = test_matrix_on_host.row_offsets.begin()[test_matrix_on_host.num_rows];
    for(unsigned long long i = 0; i < test_matrix_on_device.num_rows+1; i++)
        host_row_offsets[i] = test_matrix_on_host.row_offsets.begin()[i % test_matrix_on_host.num_rows] + originalsize * (i / test_matrix_on_host.num_rows);
    //for(unsigned long long i = 0; i < test_matrix_on_device.num_entries; i++)
    //    host_column_indices[i] = test_matrix_on_host.column_indices.begin()[i % test_matrix_on_host.num_entries];
    for(unsigned long long i = 0; i < test_matrix_on_device.num_entries; i++)
        host_values[i] = test_matrix_on_host.values.begin()[i % test_matrix_on_host.num_entries];
#else
    for(unsigned long long i = 0; i < test_matrix_on_device.num_rows+1; i++)
        host_row_offsets[i] = test_matrix_on_host.row_offsets.begin()[i];
    //for(unsigned long long i = 0; i < test_matrix_on_device.num_entries; i++)
    //    host_column_indices[i] = test_matrix_on_host.column_indices.begin()[i];
    for(unsigned long long i = 0; i < test_matrix_on_device.num_entries; i++)
        host_values[i] = test_matrix_on_host.values.begin()[i];
#endif
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(test_matrix_on_device.row_offsets.my_begin, (void *)host_row_offsets, (test_matrix_on_device.num_rows+1)*sizeof(IndexType), cudaMemcpyHostToDevice));
    //CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(test_matrix_on_device.column_indices.my_begin, (void *)host_column_indices, test_matrix_on_device.num_entries*sizeof(IndexType), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(test_matrix_on_device.values.my_begin, (void *)host_values, test_matrix_on_device.num_entries*sizeof(ValueType), cudaMemcpyHostToDevice));
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, start, end);
    printf("\t%-20s: %8.4f ms\n", "cudaMallocHybrid", elapsed_time);

    cudaEventRecord(start,0);
#ifdef ENLARGE_INPUT
    assert(test_matrix_on_host.row_offsets.begin()[0] == 0);
    //for(unsigned long long i = 0; i < test_matrix_on_device.num_rows+1; i++)
    //    test_matrix_on_device.row_offsets.my_begin[i] = test_matrix_on_host.row_offsets.begin()[i % test_matrix_on_host.num_rows] + originalsize * (i / test_matrix_on_host.num_rows);
    for(unsigned long long i = 0; i < test_matrix_on_device.num_entries; i++)
        test_matrix_on_device.column_indices.my_begin[i] = test_matrix_on_host.column_indices.begin()[i % test_matrix_on_host.num_entries];
    //for(unsigned long long i = 0; i < test_matrix_on_device.num_entries; i++)
    //    test_matrix_on_device.values.my_begin[i] = test_matrix_on_host.values.begin()[i % test_matrix_on_host.num_entries];
#else
    //for(unsigned long long i = 0; i < test_matrix_on_host.num_rows+1; i++)
    //    test_matrix_on_device.row_offsets.my_begin[i] = test_matrix_on_host.row_offsets.begin()[i];
    for(unsigned long long i = 0; i < test_matrix_on_host.num_entries; i++)
        test_matrix_on_device.column_indices.my_begin[i] = test_matrix_on_host.column_indices.begin()[i];
    //for(unsigned long long i = 0; i < test_matrix_on_host.num_entries; i++)
    //    test_matrix_on_device.values.my_begin[i] = test_matrix_on_host.values.begin()[i];
#endif
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, start, end);
    printf("\t%-20s: %8.4f ms\n", "initData", elapsed_time);

#if defined (CUDA_UM_DUPLICATE)
    //cudaMemAdvise(test_matrix_on_device.row_offsets.my_begin, (test_matrix_on_device.num_rows+1)*sizeof(IndexType), cudaMemAdviseSetReadMostly, global_device_id);
    cudaMemAdvise(test_matrix_on_device.column_indices.my_begin, test_matrix_on_device.num_entries*sizeof(IndexType), cudaMemAdviseSetReadMostly, global_device_id);
    //cudaMemAdvise(test_matrix_on_device.values.my_begin, test_matrix_on_device.num_entries*sizeof(ValueType), cudaMemAdviseSetReadMostly, global_device_id);
#elif defined (CUDA_UM_PREFERRED_GPU)
    //cudaMemAdvise(test_matrix_on_device.row_offsets.my_begin, (test_matrix_on_device.num_rows+1)*sizeof(IndexType), cudaMemAdviseSetPreferredLocation, global_device_id);
    cudaMemAdvise(test_matrix_on_device.column_indices.my_begin, test_matrix_on_device.num_entries*sizeof(IndexType), cudaMemAdviseSetPreferredLocation, global_device_id);
    //cudaMemAdvise(test_matrix_on_device.values.my_begin, test_matrix_on_device.num_entries*sizeof(ValueType), cudaMemAdviseSetPreferredLocation, global_device_id);
#elif defined (CUDA_UM_PREFERRED_CPU)
    //cudaMemAdvise(test_matrix_on_device.row_offsets.my_begin, (test_matrix_on_device.num_rows+1)*sizeof(IndexType), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemAdvise(test_matrix_on_device.column_indices.my_begin, test_matrix_on_device.num_entries*sizeof(IndexType), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    //cudaMemAdvise(test_matrix_on_device.values.my_begin, test_matrix_on_device.num_entries*sizeof(ValueType), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    //cudaMemAdvise(test_matrix_on_device.row_offsets.my_begin, (test_matrix_on_device.num_rows+1)*sizeof(IndexType), cudaMemAdviseSetAccessedBy, global_device_id);
    cudaMemAdvise(test_matrix_on_device.column_indices.my_begin, test_matrix_on_device.num_entries*sizeof(IndexType), cudaMemAdviseSetAccessedBy, global_device_id);
    //cudaMemAdvise(test_matrix_on_device.values.my_begin, test_matrix_on_device.num_entries*sizeof(ValueType), cudaMemAdviseSetAccessedBy, global_device_id);
#endif

#ifdef CUDA_UM_PREFETCH
    //cudaMemPrefetchAsync(test_matrix_on_device.row_offsets.my_begin, (test_matrix_on_device.num_rows+1)*sizeof(IndexType), global_device_id);
    cudaMemPrefetchAsync(test_matrix_on_device.column_indices.my_begin, test_matrix_on_device.num_entries*sizeof(IndexType), global_device_id);
    //cudaMemPrefetchAsync(test_matrix_on_device.values.my_begin, test_matrix_on_device.num_entries*sizeof(ValueType), global_device_id);
#endif
#elif defined (CUDA_TRUE_HYBRID_ALLOC)
    cudaEventRecord(start,0);
    size_t device_alloc_size = CUDA_DEVICE_ALLOC_SIZE;
    size_t managed_alloc_size = test_matrix_on_device.num_entries - CUDA_DEVICE_ALLOC_SIZE;
    test_matrix_on_device.column_indices.my_device_size = device_alloc_size;
    test_matrix_on_device.values.my_device_size = device_alloc_size;
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&test_matrix_on_device.row_offsets.my_begin, (test_matrix_on_device.num_rows+1)*sizeof(IndexType)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&test_matrix_on_device.column_indices.my_begin, device_alloc_size*sizeof(IndexType)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMallocManaged((void **)&test_matrix_on_device.column_indices.my_second_begin, managed_alloc_size*sizeof(IndexType)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&test_matrix_on_device.values.my_begin, device_alloc_size*sizeof(ValueType)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMallocManaged((void **)&test_matrix_on_device.values.my_second_begin, managed_alloc_size*sizeof(ValueType)));
#if defined (CUDA_UM_HOST_PREFETCH)
    cudaMemPrefetchAsync(test_matrix_on_device.column_indices.my_second_begin, managed_alloc_size*sizeof(IndexType), cudaCpuDeviceId);
    cudaMemPrefetchAsync(test_matrix_on_device.values.my_second_begin, managed_alloc_size*sizeof(ValueType), cudaCpuDeviceId);
#endif
    // lld: reorganize host matrix to be linear
    IndexType * host_row_offsets = new IndexType[test_matrix_on_device.num_rows+1];
    IndexType * host_column_indices = new IndexType[device_alloc_size];
    ValueType * host_values = new ValueType[device_alloc_size];
#ifdef ENLARGE_INPUT
    size_t originalsize = test_matrix_on_host.row_offsets.begin()[test_matrix_on_host.num_rows];
    for(unsigned long long i = 0; i < test_matrix_on_device.num_rows+1; i++)
        host_row_offsets[i] = test_matrix_on_host.row_offsets.begin()[i % test_matrix_on_host.num_rows] + originalsize * (i / test_matrix_on_host.num_rows);
    for(unsigned long long i = 0; i < device_alloc_size; i++)
        host_column_indices[i] = test_matrix_on_host.column_indices.begin()[i % test_matrix_on_host.num_entries];
    for(unsigned long long i = 0; i < device_alloc_size; i++)
        host_values[i] = test_matrix_on_host.values.begin()[i % test_matrix_on_host.num_entries];
#else
    for(unsigned long long i = 0; i < test_matrix_on_device.num_rows+1; i++)
        host_row_offsets[i] = test_matrix_on_host.row_offsets.begin()[i];
    for(unsigned long long i = 0; i < device_alloc_size; i++)
        host_column_indices[i] = test_matrix_on_host.column_indices.begin()[i];
    for(unsigned long long i = 0; i < device_alloc_size; i++)
        host_values[i] = test_matrix_on_host.values.begin()[i];
#endif
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(test_matrix_on_device.row_offsets.my_begin, (void *)host_row_offsets, (test_matrix_on_device.num_rows+1)*sizeof(IndexType), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(test_matrix_on_device.column_indices.my_begin, (void *)host_column_indices, device_alloc_size*sizeof(IndexType), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(test_matrix_on_device.values.my_begin, (void *)host_values, device_alloc_size*sizeof(ValueType), cudaMemcpyHostToDevice));
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, start, end);
    printf("\t%-20s: %8.4f ms\n", "cudaMallocTrueHybrid", elapsed_time);

    cudaEventRecord(start,0);
#ifdef ENLARGE_INPUT
    assert(test_matrix_on_host.row_offsets.begin()[0] == 0);
    for(unsigned long long i = device_alloc_size; i < test_matrix_on_device.num_entries; i++)
        test_matrix_on_device.column_indices.my_second_begin[i - device_alloc_size] = test_matrix_on_host.column_indices.begin()[i % test_matrix_on_host.num_entries];
    for(unsigned long long i = device_alloc_size; i < test_matrix_on_device.num_entries; i++)
        test_matrix_on_device.values.my_second_begin[i - device_alloc_size] = test_matrix_on_host.values.begin()[i % test_matrix_on_host.num_entries];
#else
    for(unsigned long long i = device_alloc_size; i < test_matrix_on_host.num_entries; i++)
        test_matrix_on_device.column_indices.my_second_begin[i - device_alloc_size] = test_matrix_on_host.column_indices.begin()[i];
    for(unsigned long long i = device_alloc_size; i < test_matrix_on_host.num_entries; i++)
        test_matrix_on_device.values.my_second_begin[i - device_alloc_size] = test_matrix_on_host.values.begin()[i];
#endif
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, start, end);
    printf("\t%-20s: %8.4f ms\n", "initData", elapsed_time);

#if defined (CUDA_UM_DUPLICATE)
    cudaMemAdvise(test_matrix_on_device.column_indices.my_second_begin, managed_alloc_size*sizeof(IndexType), cudaMemAdviseSetReadMostly, global_device_id);
    cudaMemAdvise(test_matrix_on_device.values.my_second_begin, managed_alloc_size*sizeof(ValueType), cudaMemAdviseSetReadMostly, global_device_id);
#elif defined (CUDA_UM_PREFERRED_GPU)
    cudaMemAdvise(test_matrix_on_device.column_indices.my_second_begin, managed_alloc_size*sizeof(IndexType), cudaMemAdviseSetPreferredLocation, global_device_id);
    cudaMemAdvise(test_matrix_on_device.values.my_second_begin, managed_alloc_size*sizeof(ValueType), cudaMemAdviseSetPreferredLocation, global_device_id);
#elif defined (CUDA_UM_PREFERRED_CPU)
    cudaMemAdvise(test_matrix_on_device.column_indices.my_second_begin, managed_alloc_size*sizeof(IndexType), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemAdvise(test_matrix_on_device.values.my_second_begin, managed_alloc_size*sizeof(ValueType), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemAdvise(test_matrix_on_device.column_indices.my_second_begin, managed_alloc_size*sizeof(IndexType), cudaMemAdviseSetAccessedBy, global_device_id);
    cudaMemAdvise(test_matrix_on_device.values.my_second_begin, managed_alloc_size*sizeof(ValueType), cudaMemAdviseSetAccessedBy, global_device_id);
#endif

#ifdef CUDA_UM_PREFETCH
    cudaMemPrefetchAsync(test_matrix_on_device.column_indices.my_second_begin, managed_alloc_size*sizeof(IndexType), global_device_id);
    cudaMemPrefetchAsync(test_matrix_on_device.values.my_second_begin, managed_alloc_size*sizeof(ValueType), global_device_id);
#endif
#elif defined (CUDA_HOST_ALLOC)
    cudaEventRecord(start,0);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, start, end);
    printf("\t%-20s: %8.4f ms\n", "cudaMallocHost", elapsed_time);

    cudaEventRecord(start,0);
#ifdef ENLARGE_INPUT
    size_t originalsize = test_matrix_on_host.row_offsets.begin()[test_matrix_on_host.num_rows];
    assert(test_matrix_on_host.row_offsets.begin()[0] == 0);
    for(unsigned long long i = 0; i < test_matrix_on_device.num_rows+1; i++)
        test_matrix_on_device.row_offsets.my_begin[i] = test_matrix_on_host.row_offsets.begin()[i % test_matrix_on_host.num_rows] + originalsize * (i / test_matrix_on_host.num_rows);
    for(unsigned long long i = 0; i < test_matrix_on_device.num_entries; i++)
        test_matrix_on_device.column_indices.my_begin[i] = test_matrix_on_host.column_indices.begin()[i % test_matrix_on_host.num_entries];
    for(unsigned long long i = 0; i < test_matrix_on_device.num_entries; i++)
        test_matrix_on_device.values.my_begin[i] = test_matrix_on_host.values.begin()[i % test_matrix_on_host.num_entries];
#else
    for(unsigned long long i = 0; i < test_matrix_on_host.num_rows+1; i++)
        test_matrix_on_device.row_offsets.my_begin[i] = test_matrix_on_host.row_offsets.begin()[i];
    for(unsigned long long i = 0; i < test_matrix_on_host.num_entries; i++)
        test_matrix_on_device.column_indices.my_begin[i] = test_matrix_on_host.column_indices.begin()[i];
    for(unsigned long long i = 0; i < test_matrix_on_host.num_entries; i++)
        test_matrix_on_device.values.my_begin[i] = test_matrix_on_host.values.begin()[i];
#endif
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, start, end);
    printf("\t%-20s: %8.4f ms\n", "initHostData", elapsed_time);

#ifdef CUDA_HOST_ACCESSEDBY
    cudaMemAdvise(test_matrix_on_device.row_offsets.my_begin, (test_matrix_on_device.num_rows+1)*sizeof(IndexType), cudaMemAdviseSetAccessedBy, global_device_id);
    cudaMemAdvise(test_matrix_on_device.column_indices.my_begin, test_matrix_on_device.num_entries*sizeof(IndexType), cudaMemAdviseSetAccessedBy, global_device_id);
    cudaMemAdvise(test_matrix_on_device.values.my_begin, test_matrix_on_device.num_entries*sizeof(ValueType), cudaMemAdviseSetReadMostly, global_device_id);
#endif
#else
    cudaEventRecord(start,0);
    // lld: reorganize host matrix to be linear
#ifdef ENLARGE_INPUT
    IndexType * host_row_offsets = new IndexType[test_matrix_on_device.num_rows+1];
    size_t originalsize = test_matrix_on_host.row_offsets.begin()[test_matrix_on_host.num_rows];
    for(unsigned long long i = 0; i < test_matrix_on_device.num_rows+1; i++)
        host_row_offsets[i] = test_matrix_on_host.row_offsets.begin()[i % test_matrix_on_host.num_rows] + originalsize * (i / test_matrix_on_host.num_rows);
    IndexType * host_column_indices = new IndexType[test_matrix_on_device.num_entries];
    for(unsigned long long i = 0; i < test_matrix_on_device.num_entries; i++)
        host_column_indices[i] = test_matrix_on_host.column_indices.begin()[i % test_matrix_on_host.num_entries];
    ValueType * host_values = new ValueType[test_matrix_on_device.num_entries];
    for(unsigned long long i = 0; i < test_matrix_on_device.num_entries; i++)
        host_values[i] = test_matrix_on_host.values.begin()[i % test_matrix_on_host.num_entries];
#else
    IndexType * host_row_offsets = new IndexType[test_matrix_on_host.num_rows+1];
    for(unsigned long long i = 0; i < test_matrix_on_host.num_rows+1; i++)
        host_row_offsets[i] = test_matrix_on_host.row_offsets.begin()[i];
    IndexType * host_column_indices = new IndexType[test_matrix_on_host.num_entries];
    for(unsigned long long i = 0; i < test_matrix_on_host.num_entries; i++)
        host_column_indices[i] = test_matrix_on_host.column_indices.begin()[i];
    ValueType * host_values = new ValueType[test_matrix_on_host.num_entries];
    for(unsigned long long i = 0; i < test_matrix_on_host.num_entries; i++)
        host_values[i] = test_matrix_on_host.values.begin()[i];
#endif
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, start, end);
    printf("\t%-20s: %8.4f ms\n", "initHostData", elapsed_time);

    cudaEventRecord(start,0);
    // lld: manually allocate GPU memory and copy data
    cudaMalloc((void **)&test_matrix_on_device.row_offsets.my_begin, (test_matrix_on_device.num_rows+1)*sizeof(IndexType));
    cudaMalloc((void **)&test_matrix_on_device.column_indices.my_begin, test_matrix_on_device.num_entries*sizeof(IndexType));
    cudaMalloc((void **)&test_matrix_on_device.values.my_begin, test_matrix_on_device.num_entries*sizeof(ValueType));

    //IndexType idxfirst = test_matrix_on_host.row_offsets.begin()[0];
    //cudaMemcpy(test_matrix_on_device.row_offsets.my_begin, (void *)&idxfirst, (test_matrix_on_device.num_rows+1)*sizeof(IndexType), cudaMemcpyHostToDevice);
    //idxfirst = test_matrix_on_host.column_indices.begin()[0];
    //cudaMemcpy(test_matrix_on_device.column_indices.my_begin, (void *)&idxfirst, test_matrix_on_device.num_entries*sizeof(IndexType), cudaMemcpyHostToDevice);
    //ValueType valfirst = test_matrix_on_host.values.begin()[0];
    //cudaMemcpy(test_matrix_on_device.values.my_begin, (void *)&valfirst, test_matrix_on_device.num_entries*sizeof(ValueType), cudaMemcpyHostToDevice);
    cudaMemcpy(test_matrix_on_device.row_offsets.my_begin, (void *)host_row_offsets, (test_matrix_on_device.num_rows+1)*sizeof(IndexType), cudaMemcpyHostToDevice);
    cudaMemcpy(test_matrix_on_device.column_indices.my_begin, (void *)host_column_indices, test_matrix_on_device.num_entries*sizeof(IndexType), cudaMemcpyHostToDevice);
    cudaMemcpy(test_matrix_on_device.values.my_begin, (void *)host_values, test_matrix_on_device.num_entries*sizeof(ValueType), cudaMemcpyHostToDevice);
    free(host_row_offsets);
    free(host_column_indices);
    free(host_values);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, start, end);
    printf("\t%-20s: %8.4f ms\n", "cudaMalloc", elapsed_time);
#endif

#ifdef TEST_MODE
    test_spmv_block("csr_block",  2, host_matrix, test_matrix_on_host, test_matrix_on_device, cusp::multiply<DeviceMatrix,DeviceArray2d,DeviceArray2d>);
#undef test_matrix_on_host
#else
    // lld: TODO: kernel needs to be modified to support manual memory allocation
    //test_spmv("csr_vector", host_matrix, test_matrix_on_host, test_matrix_on_device, cusp::multiply<DeviceMatrix,DeviceArray,DeviceArray>);
    //test_spmv("csr_scalar", host_matrix, test_matrix_on_host, test_matrix_on_device, cusp::system::cuda::detail::spmv_csr_scalar<DeviceMatrix,DeviceArray,DeviceArray>);

    for(size_t num_cols = 2; num_cols < 64; num_cols *= 2)
      test_spmv_block("csr_block",  num_cols, host_matrix, test_matrix_on_host, test_matrix_on_device, cusp::multiply<DeviceMatrix,DeviceArray2d,DeviceArray2d>);
#endif

    // lld: free memory
    cudaFree(test_matrix_on_device.row_offsets.my_begin);
    cudaFree(test_matrix_on_device.column_indices.my_begin);
    cudaFree(test_matrix_on_device.values.my_begin);
    cudaEventRecord(total_end, 0);
    cudaEventSynchronize(total_end);
    cudaEventElapsedTime(&elapsed_time, total_start, total_end);
    printf("\t%-20s: %8.4f ms\n", "total csr", elapsed_time);
    float data_size = ((test_matrix_on_device.num_rows+1)*sizeof(IndexType)
                       + test_matrix_on_device.num_entries*sizeof(IndexType)
                       + test_matrix_on_device.num_entries*sizeof(ValueType)
                       + test_matrix_on_device.num_rows*sizeof(ValueType)*2
                       + test_matrix_on_device.num_cols*sizeof(ValueType)*2) / (1024.0 * 1024.0);
    printf("\t%-20s: %8.2f MB\n", "data size", data_size);

    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaEventDestroy(total_start);
    cudaEventDestroy(total_end);
}

template <typename HostMatrix>
void test_dia(HostMatrix& host_matrix)
{
    typedef typename HostMatrix::index_type IndexType;
    typedef typename HostMatrix::value_type ValueType;

    // convert HostMatrix to TestMatrix on host
    cusp::dia_matrix<IndexType, ValueType, cusp::host_memory> test_matrix_on_host;

    try
    {
        test_matrix_on_host = host_matrix;
    }
    catch (cusp::format_conversion_exception)
    {
        std::cout << "\tRefusing to convert to DIA format" << std::endl;
        return;
    }

    // transfer TestMatrix to device
    typedef typename cusp::dia_matrix<IndexType, ValueType, cusp::device_memory> DeviceMatrix;
    typedef typename cusp::array1d<ValueType, cusp::device_memory>               DeviceArray;
    DeviceMatrix test_matrix_on_device(test_matrix_on_host);

    test_spmv("dia",     host_matrix, test_matrix_on_host, test_matrix_on_device, cusp::multiply<DeviceMatrix,DeviceArray,DeviceArray>);
}

template <typename HostMatrix>
void test_ell(HostMatrix& host_matrix)
{
    typedef typename HostMatrix::index_type IndexType;
    typedef typename HostMatrix::value_type ValueType;

    // convert HostMatrix to TestMatrix on host
    cusp::ell_matrix<IndexType, ValueType, cusp::host_memory> test_matrix_on_host;

    try
    {
        test_matrix_on_host = host_matrix;
    }
    catch (cusp::format_conversion_exception)
    {
        std::cout << "\tRefusing to convert to ELL format" << std::endl;
        return;
    }

    // transfer TestMatrix to device
    typedef typename cusp::ell_matrix<IndexType, ValueType, cusp::device_memory> DeviceMatrix;
    typedef typename cusp::array1d<ValueType, cusp::device_memory>               DeviceArray;
    DeviceMatrix test_matrix_on_device(test_matrix_on_host);

    test_spmv("ell",     host_matrix, test_matrix_on_host, test_matrix_on_device, cusp::multiply<DeviceMatrix,DeviceArray,DeviceArray>);
}

template <typename HostMatrix>
void test_hyb(HostMatrix& host_matrix)
{
    typedef typename HostMatrix::index_type IndexType;
    typedef typename HostMatrix::value_type ValueType;

    // convert HostMatrix to TestMatrix on host
    cusp::hyb_matrix<IndexType, ValueType, cusp::host_memory> test_matrix_on_host(host_matrix);

    // transfer TestMatrix to device
    typedef typename cusp::hyb_matrix<IndexType, ValueType, cusp::device_memory> DeviceMatrix;
    typedef typename cusp::array1d<ValueType, cusp::device_memory>               DeviceArray;
    DeviceMatrix test_matrix_on_device(test_matrix_on_host);

    test_spmv("hyb",     host_matrix, test_matrix_on_host, test_matrix_on_device, cusp::multiply<DeviceMatrix,DeviceArray,DeviceArray>);
}

