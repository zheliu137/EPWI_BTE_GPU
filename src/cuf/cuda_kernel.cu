#include <math.h>
#include "cuda_settings.h"
#include "cuda_timer.h"

extern "C"
{   
    static GPU_Timer timer_GPUr_;
    static CPU_Timer timer_CPUr_;
    cublasOperation_t char_to_cublas_trans(char trans);
    void init_cuda_dev_(int *_mpime, int *_gpu_id)
    {
        int mpime = *_mpime;
        CUDA_CHECK(cudaSetDevice(*_gpu_id>=0?*_gpu_id:mpime % GPUNUM));
        timer_GPUr_ = GPU_Timer();
        timer_CPUr_ = CPU_Timer();
#ifdef DEBUG
        printf("Use GPU DEVICE : %d by process %d \n", mpime % GPUNUM, mpime);
#endif
    }
    
    void reset_cuda_dev(int *_mpime)
    {
        CUDA_CHECK(cudaDeviceReset());
#ifdef DEBUG
        printf("Reset GPU DEVICE : %d by process %d\n", *_mpime % GPUNUM, *_mpime);
#endif
    }

    void reset_cuda_dev_wrapper_(int *_mpime)
    {
        reset_cuda_dev(_mpime);
    }

    void ephwan2bloch_cuda_(int *_nbnd, int *_nmodes, cuDoubleComplex *cufkq, cuDoubleComplex *cufkk, cuDoubleComplex *epmatf)
    {
        int nbnd = *_nbnd, nmodes = *_nmodes;
        cuDoubleComplex cone = {1.0, 0.0}, czero = {0.0, 0.0};
        cuDoubleComplex *d_cufkq = nullptr, *d_cufkk = nullptr, *d_eptmp = nullptr, *d_epmatf = nullptr;
        cublasHandle_t blasHandle;
        cudaStream_t stream = NULL;

        CUBLAS_CHECK(cublasCreate(&blasHandle));
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUBLAS_CHECK(cublasSetStream(blasHandle, stream));

        CUDA_CHECK(cudaMallocAsync((void **)&d_cufkk, nbnd * nbnd * sizeof(cuDoubleComplex), stream));
        CUDA_CHECK(cudaMallocAsync((void **)&d_cufkq, nbnd * nbnd * sizeof(cuDoubleComplex), stream));
        CUDA_CHECK(cudaMallocAsync((void **)&d_eptmp, nbnd * nbnd * nmodes * sizeof(cuDoubleComplex), stream));
        CUDA_CHECK(cudaMallocAsync((void **)&d_epmatf, nbnd * nbnd * nmodes * sizeof(cuDoubleComplex), stream));

        CUBLAS_CHECK(cublasSetMatrixAsync(nbnd, nbnd, sizeof(cuDoubleComplex), cufkq, nbnd, d_cufkq, nbnd, stream));
        CUBLAS_CHECK(cublasSetMatrixAsync(nbnd, nbnd, sizeof(cuDoubleComplex), cufkk, nbnd, d_cufkk, nbnd, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_epmatf, epmatf, nbnd * nbnd * nmodes * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream));

        CUBLAS_CHECK(cublasZgemmStridedBatched(blasHandle, char_to_cublas_trans('n'), char_to_cublas_trans('n'), nbnd, nbnd, nbnd, &cone, d_cufkq, nbnd, 0, d_epmatf, nbnd, nbnd * nbnd, &czero, d_eptmp, nbnd, nbnd * nbnd, nmodes));

        CUBLAS_CHECK(cublasZgemmStridedBatched(blasHandle, char_to_cublas_trans('n'), char_to_cublas_trans('c'), nbnd, nbnd, nbnd, &cone, d_eptmp, nbnd, nbnd * nbnd, d_cufkk, nbnd, 0, &czero, d_epmatf, nbnd, nbnd * nbnd, nmodes));

        CUDA_CHECK(cudaMemcpyAsync(epmatf, d_epmatf, nbnd * nbnd * nmodes * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, stream));

        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaFreeAsync(d_cufkk, stream));
        CUDA_CHECK(cudaFreeAsync(d_cufkq, stream));
        CUDA_CHECK(cudaFreeAsync(d_eptmp, stream));
        CUDA_CHECK(cudaFreeAsync(d_epmatf, stream));

        cublasDestroy(blasHandle);
        CUDA_CHECK(cudaStreamDestroy(stream));
    }

    __global__ void init_champ_matrix(int n, cuDoubleComplex *champ, cuDoubleComplex *chf)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        int idy = blockDim.y * blockIdx.y + threadIdx.y;
        if (idx < n && idy < n)
        {
            champ[idy * n + idx] =
                cuCmul(
                    cuCadd(
                        chf[idy * n + idx], cuConj(chf[idx * n + idy])),
                    {0.5, 0.0});
        }
    }
    
    int kplusq(int k, int q, int *NKgrid, int *NQgrid)
    {
        int __kplusq;
        int veck[3], vecq[3], veckq[3];

        veck[0] = int((k - 1) / (NKgrid[1] * NKgrid[2]));
        veck[1] = int((k - 1) % (NKgrid[1] * NKgrid[2]) / NKgrid[2]);
        veck[2] = (k - 1) % NKgrid[2];

        vecq[0] = int((q - 1) / (NQgrid[1] * NQgrid[2]));
        vecq[1] = int((q - 1) % (NQgrid[1] * NQgrid[2]) / NQgrid[2]);
        vecq[2] = (q - 1) % NQgrid[2];

        vecq[0] = vecq[0] * (NKgrid[0] / NQgrid[0]);
        vecq[1] = vecq[1] * (NKgrid[1] / NQgrid[1]);
        vecq[2] = vecq[2] * (NKgrid[2] / NQgrid[2]);

        for (int i = 0; i < 3; i++)
        {
            veckq[i] = (veck[i] + vecq[i]) % NKgrid[i];
        }

        __kplusq = veckq[0] * NKgrid[1] * NKgrid[2] + veckq[1] * NKgrid[2] + veckq[2] + 1;

        return __kplusq;
    }
}