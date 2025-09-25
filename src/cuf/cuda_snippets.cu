#include "cuda_settings.h"

template <typename T>
void print_matrix(const int &m, const int &n, const T *A, const int &lda);

template <>
void print_matrix(const int &m, const int &n, const float *A, const int &lda)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%0.2f ", A[j * lda + i]);
        }
        printf("\n");
    }
}

template <>
void print_matrix(const int &m, const int &n, const double *A, const int &lda)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%0.2f ", A[j * lda + i]);
        }
        printf("\n");
    }
}

template <>
void print_matrix(const int &m, const int &n, const cuComplex *A, const int &lda)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%0.2f + %0.2fj ", A[j * lda + i].x, A[j * lda + i].y);
        }
        printf("\n");
    }
}

template <>
void print_matrix(const int &m, const int &n, const cuDoubleComplex *A, const int &lda)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%12.6f + %12.6fj ", A[j * lda + i].x, A[j * lda + i].y);
        }
        printf("\n");
    }
}

extern "C"
{
    /*
     *cuda matrix operations
     */

    cublasOperation_t char_to_cublas_trans(char trans)
    {
        cublasOperation_t cuTrans;
        switch (trans)
        {
        case 'n':
        case 'N':
            cuTrans = CUBLAS_OP_N;
            break;
        case 't':
        case 'T':
            cuTrans = CUBLAS_OP_T;
            break;
        case 'c':
        case 'C':
            cuTrans = CUBLAS_OP_C;
            break;
        default:
            exit(-1);
        }
        return cuTrans;
    }
    void run_ddot_(int *_n, double *x, double *y, double *z)
    {
        int n = *_n;
        double *dx, *dy, *dz;
        cublasHandle_t blasHandle;
        CUDA_CHECK(cudaMalloc((void **)&dx, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&dy, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&dz, sizeof(double)));

        // cublas set vector api also works
        CUDA_CHECK(cudaMemcpy(dx, x, n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dy, y, n * sizeof(double), cudaMemcpyHostToDevice));

        cublasCreate(&blasHandle);
        cublasDdot(blasHandle, n, dx, 1, dy, 1, dz);
        // cublas get vector api also works
        CUDA_CHECK(cudaMemcpy(z, dz, sizeof(double), cudaMemcpyDeviceToHost));
        cudaFree(dx);
        cudaFree(dy);
        cudaFree(dz);
        cublasDestroy(blasHandle);
    }

    void run_cublas_dgemv_(char *_trans, int *_m, int *_n, double *_alpha, double *A, int *_lda, double *x, int *_incx, double *_beta, double *y, int *_incy)
    {
        char trans = *_trans;
        int m = *_m, n = *_n, lda = *_lda, incx = *_incx, incy = *_incy;
        double alpha = *_alpha, beta = *_beta;
        double *dA, *dX, *dY;
        cublasHandle_t blasHandle;
        cudaStream_t stream = NULL;
        CUDA_CHECK(cudaMallocAsync((void **)&dA, m * n * sizeof(double), stream));
        CUDA_CHECK(cudaMallocAsync((void **)&dX, n * sizeof(double), stream));
        CUDA_CHECK(cudaMallocAsync((void **)&dY, m * sizeof(double), stream));

        CUBLAS_CHECK(cublasCreate(&blasHandle));
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUBLAS_CHECK(cublasSetStream(blasHandle, stream));

        CUBLAS_CHECK(cublasSetMatrixAsync(m, n, sizeof(double), A, m, dA, m, stream));
        CUBLAS_CHECK(cublasSetVectorAsync(n, sizeof(double), x, 1, dX, 1, stream));
        CUBLAS_CHECK(cublasDgemv(blasHandle, char_to_cublas_trans(trans), m, n, &alpha, dA, lda, dX, incx, &beta, dY, incy));
        CUBLAS_CHECK(cublasGetVectorAsync(m, sizeof(double), dY, 1, y, 1, stream));

        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFreeAsync(dA, stream));
        CUDA_CHECK(cudaFreeAsync(dX, stream));
        CUDA_CHECK(cudaFreeAsync(dY, stream));
        CUBLAS_CHECK(cublasDestroy(blasHandle));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }

    void run_cublas_zgemv_(char *_trans, int *_m, int *_n, cuDoubleComplex *alpha, cuDoubleComplex *A, int *_lda, cuDoubleComplex *x, int *_incx, cuDoubleComplex *beta, cuDoubleComplex *y, int *_incy)
    {
        char trans = *_trans;
        int m = *_m, n = *_n, lda = *_lda, incx = *_incx, incy = *_incy;
        cuDoubleComplex *dA = nullptr, *dX = nullptr, *dY = nullptr;
        cublasHandle_t blasHandle;
        cudaStream_t stream = NULL;
        CUDA_CHECK(cudaMallocAsync((void **)&dA, m * n * sizeof(cuDoubleComplex), stream));
        CUDA_CHECK(cudaMallocAsync((void **)&dX, n * sizeof(cuDoubleComplex), stream));
        CUDA_CHECK(cudaMallocAsync((void **)&dY, m * sizeof(cuDoubleComplex), stream));

        CUBLAS_CHECK(cublasCreate(&blasHandle));
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUBLAS_CHECK(cublasSetStream(blasHandle, stream));
        CUBLAS_CHECK(cublasSetMatrixAsync(m, n, sizeof(cuDoubleComplex), A, m, dA, m, stream));
        CUBLAS_CHECK(cublasSetVectorAsync(n, sizeof(cuDoubleComplex), x, 1, dX, 1, stream));
        CUBLAS_CHECK(cublasZgemv(blasHandle, char_to_cublas_trans(trans), m, n, (cuDoubleComplex *)alpha, dA, lda, dX, incx, (cuDoubleComplex *)beta, dY, incy));
        CUBLAS_CHECK(cublasGetVectorAsync(m, sizeof(cuDoubleComplex), dY, 1, y, 1, stream));

        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFreeAsync(dA, stream));
        CUDA_CHECK(cudaFreeAsync(dX, stream));
        CUDA_CHECK(cudaFreeAsync(dY, stream));
        CUBLAS_CHECK(cublasDestroy(blasHandle));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }

    void run_cublas_dgemm_(char *_transa, char *_transb, int *_m, int *_n, int *_k, double *_alpha, double *A, int *_lda, double *B, int *_ldb, double *_beta, double *C, int *_ldc)
    {
        char transa = *_transa, transb = *_transb;
        int m = *_m, n = *_n, k = *_k, lda = *_lda, ldb = *_ldb, ldc = *_ldc;
        double *dA = nullptr, *dB = nullptr, *dC = nullptr, alpha = *_alpha, beta = *_beta;
        cublasHandle_t blasHandle;
        cudaStream_t stream = NULL;
        CUDA_CHECK(cudaMallocAsync((void **)&dA, m * k * sizeof(double), stream));
        CUDA_CHECK(cudaMallocAsync((void **)&dB, k * n * sizeof(double), stream));
        CUDA_CHECK(cudaMallocAsync((void **)&dC, m * n * sizeof(double), stream));

        CUBLAS_CHECK(cublasCreate(&blasHandle));
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUBLAS_CHECK(cublasSetStream(blasHandle, stream));
        CUBLAS_CHECK(cublasSetMatrixAsync(m, k, sizeof(double), A, m, dA, m, stream));
        CUBLAS_CHECK(cublasSetMatrixAsync(k, n, sizeof(double), B, k, dB, k, stream));
        CUBLAS_CHECK(cublasDgemm(blasHandle, char_to_cublas_trans(transa), char_to_cublas_trans(transb), m, n, k, &alpha, dA, lda, dB, ldb, &beta, dC, ldc));
        CUBLAS_CHECK(cublasGetMatrixAsync(m, n, sizeof(double), dC, m, C, m, stream));

        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFreeAsync(dA, stream));
        CUDA_CHECK(cudaFreeAsync(dB, stream));
        CUDA_CHECK(cudaFreeAsync(dC, stream));
        cublasDestroy(blasHandle);
        CUDA_CHECK(cudaStreamDestroy(stream));
    }

    void run_cublas_zgemm_(char *_transa, char *_transb, int *_m, int *_n, int *_k, cuDoubleComplex *alpha, cuDoubleComplex *A, int *_lda, cuDoubleComplex *B, int *_ldb, cuDoubleComplex *beta, cuDoubleComplex *C, int *_ldc)
    {
        char transa = *_transa, transb = *_transb;
        int m = *_m, n = *_n, k = *_k, lda = *_lda, ldb = *_ldb, ldc = *_ldc;
        cuDoubleComplex *dA = nullptr, *dB = nullptr, *dC = nullptr;
        cublasHandle_t blasHandle;
        cudaStream_t stream = NULL;

        CUDA_CHECK(cudaMallocAsync((void **)&dA, m * k * sizeof(cuDoubleComplex), stream));
        CUDA_CHECK(cudaMallocAsync((void **)&dB, k * n * sizeof(cuDoubleComplex), stream));
        CUDA_CHECK(cudaMallocAsync((void **)&dC, m * n * sizeof(cuDoubleComplex), stream));

        CUBLAS_CHECK(cublasCreate(&blasHandle));
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUBLAS_CHECK(cublasSetStream(blasHandle, stream));
        CUBLAS_CHECK(cublasSetMatrixAsync(m, k, sizeof(cuDoubleComplex), A, m, dA, m, stream));
        CUBLAS_CHECK(cublasSetMatrixAsync(k, n, sizeof(cuDoubleComplex), B, k, dB, k, stream));
        CUBLAS_CHECK(cublasZgemm(blasHandle, char_to_cublas_trans(transa), char_to_cublas_trans(transb), m, n, k, (cuDoubleComplex *)alpha, dA, lda, dB, ldb, (cuDoubleComplex *)beta, dC, ldc));
        CUBLAS_CHECK(cublasGetMatrixAsync(m, n, sizeof(cuDoubleComplex), dC, m, C, m, stream));

        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaFreeAsync(dA, stream));
        CUDA_CHECK(cudaFreeAsync(dB, stream));
        CUDA_CHECK(cudaFreeAsync(dC, stream));

        cublasDestroy(blasHandle);
        CUDA_CHECK(cudaStreamDestroy(stream));
    }

    void run_cublas_zgemm_strided_batched_(char *_transa, char *_transb, int *_m, int *_n, int *_k, cuDoubleComplex *alpha, cuDoubleComplex *A, int *_lda, long long int *_stridea, cuDoubleComplex *B, int *_ldb, long long int *_strideb, cuDoubleComplex *beta, cuDoubleComplex *C, int *_ldc, long long int *_stridec, int *_batch_count)
    {
        char transa = *_transa, transb = *_transb;
        int m = *_m, n = *_n, k = *_k, lda = *_lda, ldb = *_ldb, ldc = *_ldc, stridea = *_stridea, strideb = *_strideb, stridec = *_stridec, batchCount = *_batch_count;
        cuDoubleComplex *dA = nullptr, *dB = nullptr, *dC = nullptr;
        cublasHandle_t blasHandle;
        cudaStream_t stream = NULL;

        CUDA_CHECK(cudaMallocAsync((void **)&dA, batchCount * m * k * sizeof(cuDoubleComplex), stream));
        CUDA_CHECK(cudaMallocAsync((void **)&dB, batchCount * k * n * sizeof(cuDoubleComplex), stream));
        CUDA_CHECK(cudaMallocAsync((void **)&dC, batchCount * m * n * sizeof(cuDoubleComplex), stream));

        CUBLAS_CHECK(cublasCreate(&blasHandle));
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUBLAS_CHECK(cublasSetStream(blasHandle, stream));

        CUDA_CHECK(cudaMemcpyAsync(dA, A, batchCount * m * k * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(dB, B, batchCount * k * n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream));

        CUBLAS_CHECK(cublasZgemmStridedBatched(blasHandle, char_to_cublas_trans(transa), char_to_cublas_trans(transb), m, n, k, (cuDoubleComplex *)alpha, dA, lda, stridea, dB, ldb, strideb, (cuDoubleComplex *)beta, dC, ldc, stridec, batchCount));

        CUDA_CHECK(cudaMemcpyAsync(C, dC, batchCount * m * n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, stream));

        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaFreeAsync(dA, stream));
        CUDA_CHECK(cudaFreeAsync(dB, stream));
        CUDA_CHECK(cudaFreeAsync(dC, stream));

        cublasDestroy(blasHandle);
        CUDA_CHECK(cudaStreamDestroy(stream));
    }

    void run_cublas_zgemm_strided_batched_no_ab_(char *_transa, char *_transb, int *_m, int *_n, int *_k, cuDoubleComplex *A, int *_lda, long long int *_stridea, cuDoubleComplex *B, int *_ldb, long long int *_strideb, cuDoubleComplex *C, int *_ldc, long long int *_stridec, int *_batch_count)
    {
        char transa = *_transa, transb = *_transb;
        int m = *_m, n = *_n, k = *_k, lda = *_lda, ldb = *_ldb, ldc = *_ldc, stridea = *_stridea, strideb = *_strideb, stridec = *_stridec, batchCount = *_batch_count;
        cuDoubleComplex *dA = nullptr, *dB = nullptr, *dC = nullptr;
        cublasHandle_t blasHandle;
        cudaStream_t stream = NULL;

        CUDA_CHECK(cudaMalloc((void **)&dA, batchCount * m * k * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc((void **)&dB, batchCount * k * n * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc((void **)&dC, batchCount * m * n * sizeof(cuDoubleComplex)));

        CUBLAS_CHECK(cublasCreate(&blasHandle));
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUBLAS_CHECK(cublasSetStream(blasHandle, stream));

        CUDA_CHECK(cudaMemcpyAsync(dA, A, batchCount * m * k * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(dB, B, batchCount * k * n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream));

        CUBLAS_CHECK(cublasZgemmStridedBatched(blasHandle, char_to_cublas_trans(transa), char_to_cublas_trans(transb), m, n, k, (cuDoubleComplex *)&cone_, dA, lda, stridea, dB, ldb, strideb, (cuDoubleComplex *)&czero_, dC, ldc, stridec, batchCount));

        CUDA_CHECK(cudaMemcpyAsync(C, dC, batchCount * m * n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, stream));

        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));

        cublasDestroy(blasHandle);
        CUDA_CHECK(cudaStreamDestroy(stream));
    }

    /*
     *cuSolver
     */

    cublasFillMode_t char_to_cublas_fillmode(char _uplo)
    {
        // Fortran follows columns while C/C++ follow rows
        cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
        switch (_uplo)
        {
        case 'u':
        case 'U':
            uplo = CUBLAS_FILL_MODE_UPPER;
            break;
        case 'l':
        case 'L':
            uplo = CUBLAS_FILL_MODE_LOWER;
        default:
            break;
        }
        return uplo;
    }
    void run_cusolver_zheevj_(int *_m, int *_n, char *_uplo, cuDoubleComplex *A, cuDoubleComplex *V, double *W)
    {
        const int m = *_m, n = *_n, lda = (m>n)?m:n;

        cusolverDnHandle_t cusolverH = NULL;
        cudaStream_t stream = NULL;
        syevjInfo_t syevj_params = NULL;
        cuDoubleComplex *d_A = nullptr, *d_work = nullptr;
        double *d_W = nullptr;
        int *devInfo = nullptr, lwork = 0, info_gpu = 0;

        /* configuration of syevj  */
        const double tol = 1.e-7;
        const int max_sweeps = 15;
        const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvectors.
        const cublasFillMode_t uplo = char_to_cublas_fillmode(*_uplo);

        /* step 1: create cusolver handle, bind a stream */
        CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));

        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));

        /* step 2: configuration of syevj */
        CUSOLVER_CHECK(cusolverDnCreateSyevjInfo(&syevj_params));

        /* default value of tolerance is machine zero */
        CUSOLVER_CHECK(cusolverDnXsyevjSetTolerance(syevj_params, tol));

        /* default value of max. sweeps is 100 */
        CUSOLVER_CHECK(cusolverDnXsyevjSetMaxSweeps(syevj_params, max_sweeps));

        /* step 3: copy A to device */
        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void **>(&d_A), sizeof(cuDoubleComplex) * lda * m, stream));
        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void **>(&d_W), sizeof(double) * m, stream));
        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void **>(&devInfo), sizeof(int), stream));

        CUDA_CHECK(
            cudaMemcpyAsync(d_A, A, sizeof(cuDoubleComplex) * lda * m, cudaMemcpyHostToDevice, stream));

        /* step 4: query working space of syevj */
        CUSOLVER_CHECK(
            cusolverDnZheevj_bufferSize(cusolverH, jobz, uplo, m, d_A, lda, d_W, &lwork, syevj_params));

        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void **>(&d_work), sizeof(cuDoubleComplex) * lwork, stream));

        /* step 5: compute eigen-pair   */
        CUSOLVER_CHECK(cusolverDnZheevj(cusolverH, jobz, uplo, m, d_A, lda, d_W, d_work, lwork, devInfo, syevj_params));

        CUDA_CHECK(cudaMemcpyAsync(V, d_A, sizeof(cuDoubleComplex) * lda * m, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(W, d_W, sizeof(double) * m, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost, stream));

        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (0 > info_gpu)
        {
            printf("%d-th parameter is wrong \n", -info_gpu);
            exit(1);
        }
#ifdef DEBUG
        if (0 == info_gpu)
            printf("syevj converges \n");
        else
            printf("WARNING: info = %d : syevj does not converge \n", info_gpu);

        printf("Eigenvalue = (matlab base-1), ascending order\n");
        for (int i = 0; i < m; i++)
        {
            printf("W[%d] = %E\n", i + 1, W[i]);
        }
#endif

        /* free resources */
        CUDA_CHECK(cudaFreeAsync(d_A, stream));
        CUDA_CHECK(cudaFreeAsync(d_W, stream));
        CUDA_CHECK(cudaFreeAsync(devInfo, stream));
        CUDA_CHECK(cudaFreeAsync(d_work, stream));

        CUSOLVER_CHECK(cusolverDnDestroySyevjInfo(syevj_params));

        CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));

        CUDA_CHECK(cudaStreamDestroy(stream));
    }
}