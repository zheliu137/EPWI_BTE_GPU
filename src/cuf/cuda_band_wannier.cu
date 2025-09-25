// #include "cuda_elphwann_namespace.h"
#include "cuda_settings.h"
#include "cuda_elphwann_namespace.h"
#include "cuda_timer.h"
#include <algorithm>
#include <bits/types/FILE.h>
#include <cmath>
#include <cstdio>
#include <ios>
#include <iostream>
#include <thrust/host_vector.h>
// #include <cstddef>
// #include <cuda/std/complex>
// #include <cmath>

using namespace cuda::std;
using namespace device_funcs;
using namespace polar_funcs;
// using thrust::max_element;
// using thrust::min_element;
// using namespace cuda_elphwann_wannier;

#define IDX2F(i, j, ld) ((((j - 1)) * (ld)) + ((i - 1)))

namespace
{
extern "C"
{

    /*
     * global variables
     */

    static const double radps2ev = 6.582119263352476E-004,
                        ryd2ev = 13.6056981,
                        ryd2nmev = 0.719982285311279,
                        ryd2nm = 0.052917721092,
                        Pi = M_PI,
                        Kb = 8.6173324e-5; // ev/K
    static const double done = 1.0, dzero = 0.0;

    static int nrr_k_,
        Nsymm_,
        Nlist_,
        nqc_[3],
        nrx_[3],
        *NKgrid_,
        NPTK_K_,
        *eqidx_,
        nbndsub_,
        nbands_,
        nbnd_irr_,
        nmodes_,
        batchsize_;

    bool lmetal_;
    bool lpolar_;

    static int nat_,
               *ityp_ = nullptr;
    static double *at_ = nullptr,
                  *celldm1_ = nullptr,
                  *bg_ = nullptr,
                  Volume_,
                  *orthcar_ = nullptr,
                  *amass_ = nullptr,
                  *tau_ = nullptr,
                  *el_energy_ = nullptr;

    static int *d_ndegen_k_ = nullptr,
               *d_NKgrid_ = nullptr,
               *d_eqidx_ = nullptr,
               *d_ityp_ = nullptr;

    static double *d_irvec_r_ = nullptr,
                  *d_irvec_cart_ = nullptr,
                  *d_at_ = nullptr,
                  *d_bg_ = nullptr,
                  *d_orthcar_ = nullptr,
                  *d_celldm1_ = nullptr,
                  *d_amass_ = nullptr,
                  *d_tau_ = nullptr,
                  *d_xxq_cart_ = nullptr,
                  *d_epsil_ = nullptr,
                  *d_zstar_ = nullptr,
                  *d_rdotk_ = nullptr,
                  *d_xkk_ = nullptr,
                  *d_w_ = nullptr,
                  *d_vmef_ = nullptr; // (nmodes, nk, 3)
    
    const double gmax_ = 14.0;


    static cuDoubleComplex *d_cfac_ = nullptr,
                           *d_cfac_v_ = nullptr,
                           *d_chv_ = nullptr,
                           *d_chv_tmp_ = nullptr,
                           *d_chw_ = nullptr,
                           *d_chf_ = nullptr,
                           *d_champ_ = nullptr;

    static cublasHandle_t blasHandle_ = NULL;
    cusolverDnHandle_t solverHandle_ = NULL;
    syevjInfo_t syevj_params_ = NULL;
    static int *devInfo_ = nullptr;

    static const double solver_tol_ = 1.e-12;
    static const int solver_max_sweeps_ = 150;
    const cusolverEigMode_t jobz_ = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvectors.
    const cublasFillMode_t uplo_ = CUBLAS_FILL_MODE_UPPER;

    static GPU_Timer timer_GPUr_;
    static CPU_Timer timer_CPUr_;
}
} // namespace

extern "C" {

    __global__ static void calc_vec_xkk(int *List, int *NKgrid, int *vec, double *xkk, int Nlist);
    __global__ static void get_cfac_from_rdot(ComplexD *cfac, double *rdot, int *ndegen, int nrr, int batchsize);
    __global__ static void vel_matrix_0(int nrr, int batchsize, double celldm1, double *irvec_cart, ComplexD *cfac, ComplexD *cfac_v);
    __global__ static void vel_collect_ph(cuDoubleComplex *vc, double *vmef, int nbnd, int Nlist, double *w);
    __global__ static void vel_collect(cuDoubleComplex *vc, double *vmef, int nbnd, int Nlist);
    __global__ static void init_champ_matrix(int n, ComplexD *champ, ComplexD *chf, int Nlist);
    __global__ static void d_irvec_cart_init(double *irvec_cart, double *ivrec_cryst, double *at, int nrr);

    void cuda_band_wannier_init_(int *_nbndsub, int *_nbands, int *_nbnd_irr, int *_nrr_k, int *_Nlist, int *List, int *_NKgrid, double *irvec_r, int *ndegen_k, cuDoubleComplex *chw,
                                 int *_batchsize, double *_at, double *_celldm1, double *_bg, int *_eqidx, double *_orthcar, int *_Nsymm)
    {
        int *d_vec = nullptr;
        nrr_k_ = *_nrr_k;
        NKgrid_ = _NKgrid;
        nbndsub_ = *_nbndsub;
        nbands_ = *_nbands;
        nbnd_irr_ = *_nbnd_irr;
        Nlist_ = *_Nlist;
        batchsize_ = *_batchsize;
        at_ = _at;
        celldm1_ = _celldm1; // in the unit of alat
        bg_ = _bg;           // in the unit of nm
        eqidx_ = _eqidx;
        orthcar_ = _orthcar;
        Nsymm_ = *_Nsymm;

        NPTK_K_ = _NKgrid[0] * _NKgrid[1] * _NKgrid[2];
        timer_GPUr_.start_clock("band_wannier_init");

        CUBLAS_CHECK(cublasCreate(&blasHandle_));
        CUSOLVER_CHECK(cusolverDnCreate(&solverHandle_));
        CUSOLVER_CHECK(cusolverDnCreateSyevjInfo(&syevj_params_));
        CUSOLVER_CHECK(cusolverDnXsyevjSetTolerance(syevj_params_, solver_tol_));
        CUSOLVER_CHECK(cusolverDnXsyevjSetMaxSweeps(syevj_params_, solver_max_sweeps_));

        CUDA_CHECK(cudaMalloc((void **)&d_vec, 3 * Nlist_ * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&d_xkk_, 3 * Nlist_ * sizeof(double)));

        CUDA_CHECK(cudaMalloc((void **)&d_at_, 9 * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_at_, at_, 9 * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc((void **)&d_celldm1_, sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_celldm1_, celldm1_, sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc((void **)&d_bg_, 9 * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_bg_, bg_, 9 * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc((void **)&d_NKgrid_, 3 * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_NKgrid_, _NKgrid, 3 * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc((void **)&d_eqidx_, NPTK_K_ * 2 * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_eqidx_, eqidx_, NPTK_K_ * 2 * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc((void **)&d_orthcar_, 9 * Nsymm_ * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_orthcar_, orthcar_, 9 * Nsymm_ * sizeof(double), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc((void **)&d_ndegen_k_, nrr_k_ * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_ndegen_k_, ndegen_k, nrr_k_ * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc((void **)&d_irvec_r_, 3 * nrr_k_ * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_irvec_r_, irvec_r, 3 * nrr_k_ * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc((void **)&d_irvec_cart_, 3 * nrr_k_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_rdotk_, nrr_k_ * batchsize_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_cfac_, nrr_k_ * batchsize_ * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc((void **)&d_cfac_v_, 3 * nrr_k_ * batchsize_ * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc((void **)&d_chv_tmp_, nbndsub_ * nbndsub_ * batchsize_ * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc((void **)&d_chv_, nbndsub_ * nbndsub_ * batchsize_ * 3 * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc((void **)&d_vmef_, nbndsub_ * Nlist_ * 3 * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_chw_, nbndsub_ * nbndsub_ * nrr_k_ * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMemcpy(d_chw_, chw, nbndsub_ * nbndsub_ * nrr_k_ * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc((void **)&d_chf_, nbndsub_ * nbndsub_ * batchsize_ * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc((void **)&d_champ_, nbndsub_ * nbndsub_ * batchsize_ * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc((void **)&d_w_, nbndsub_ * Nlist_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&devInfo_, batchsize_ * sizeof(int)));

        int *d_List = nullptr;
        CUDA_CHECK(cudaMalloc((void **)&d_List, Nlist_ * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_List, List, Nlist_ * sizeof(int), cudaMemcpyHostToDevice));

        calc_vec_xkk<<<(Nlist_ + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0>>>(d_List, d_NKgrid_, d_vec, d_xkk_, Nlist_);

        d_irvec_cart_init<<<(nrr_k_ + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_irvec_cart_, d_irvec_r_, d_at_, nrr_k_);

        CUDA_CHECK(cudaFree(d_vec));
        // CUDA_CHECK(cudaFree(d_NKgrid_));
        CUDA_CHECK(cudaFree(d_List));
        timer_GPUr_.stop_clock("band_wannier_init");
    }

    void cuda_band_wannier_(double *w, double *el_velocity)
    {
        int nbnd = nbndsub_,
            nrr = nrr_k_,
            lwork = 0;
        el_energy_ = w;
        int batchSize = batchsize_;
        cuDoubleComplex cone = {1.0, 0.0}, czero = {0.0, 0.0};
        cuDoubleComplex *d_work = nullptr;
        double tpi = M_PI * 2.0;

        for (int ibatch = 0; ibatch * batchSize < Nlist_; ibatch = ibatch + 1)
        {
            int offset = ibatch * batchSize;
            int Size_thisbatch = (ibatch + 1) * batchSize > Nlist_ ? Nlist_ - ibatch * batchSize : batchSize;

            timer_GPUr_.start_clock("band_wannier");
            
            CUBLAS_CHECK(cublasDgemm(blasHandle_, char_to_cublas_trans('t'), char_to_cublas_trans('n'), nrr_k_, Size_thisbatch, 3, &tpi, d_irvec_r_, 3, d_xkk_ + offset * 3, 3, &dzero, d_rdotk_, nrr_k_));

            dim3 block(BLOCK_SIZE, BLOCK_SIZE); // dim3 variable holds 3 dimensions
            dim3 grid((nrr + block.x - 1) / block.x, (Size_thisbatch + block.y - 1) / block.y);
            get_cfac_from_rdot<<<grid, block>>>((ComplexD *)d_cfac_, d_rdotk_, d_ndegen_k_, nrr, Size_thisbatch);
            CUBLAS_CHECK(cublasZgemm(blasHandle_, char_to_cublas_trans('n'), char_to_cublas_trans('n'), nbnd * nbnd, Size_thisbatch, nrr, &cone, d_chw_, nbnd * nbnd, d_cfac_, nrr, &czero, d_chf_, nbnd * nbnd));

            dim3 _block(BLOCK_SIZE, BLOCK_SIZE); // dim3 variable holds 3 dimensions
            dim3 _grid((nbnd * nbnd + _block.x - 1) / _block.x, (Size_thisbatch + _block.y - 1) / _block.y);
            init_champ_matrix<<<_grid, _block>>>(nbnd, (ComplexD *)d_champ_, (ComplexD *)d_chf_, Size_thisbatch);

            CUSOLVER_CHECK(cusolverDnZheevjBatched_bufferSize(solverHandle_, jobz_, uplo_, nbnd, d_champ_, nbnd, d_w_ + offset * nbnd, &lwork, syevj_params_, Size_thisbatch));

            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(cuDoubleComplex) * lwork));

            CUSOLVER_CHECK(cusolverDnZheevjBatched(solverHandle_, jobz_, uplo_, nbnd, d_champ_, nbnd, d_w_ + offset * nbnd, d_work, lwork, devInfo_, syevj_params_, Size_thisbatch));

            CUDA_CHECK(cudaFree(d_work));

            CUDA_CHECK(cudaDeviceSynchronize());

            /** 
             * @debug: cusolverDnZheevjBatched
            */
            //  int *__devinfo = (int*) malloc(sizeof(int));
            //  cudaMemcpy(__devinfo, devInfo_, sizeof(int), cudaMemcpyDeviceToHost);
            //  printf("\nCuSolver DevInfo is : %d \n", *__devinfo);
            //  free(__devinfo);

            // check_eigenvec<<<1, 1>>>(d_champ_,nbnd);
            CUDA_CHECK(cudaMemcpy(w + nbnd * offset, d_w_ + offset * nbnd, sizeof(double) * nbnd * Size_thisbatch, cudaMemcpyDeviceToHost));

            _grid = {(nrr + block.x - 1) / block.x, (Size_thisbatch + block.y - 1) / block.y};
            vel_matrix_0<<<_grid, _block>>>(nrr, Size_thisbatch, *celldm1_, d_irvec_cart_, (ComplexD *)d_cfac_, (ComplexD *)d_cfac_v_);
            int m = nbndsub_ * nbndsub_;
            CUBLAS_CHECK(cublasZgemm(blasHandle_, char_to_cublas_trans('n'), char_to_cublas_trans('n'), m, 3 * Size_thisbatch, nrr_k_, &cone, d_chw_, m, d_cfac_v_, nrr_k_, &czero, d_chv_, m));
            // _block = BLOCK_SIZE;
            // _grid  = ( Size_thisbatch + block.x - 1 ) / block.x;
            // vel_collect<<<_grid, _block>>>( d_chv_, d_vmef_, nbnd, Size_thisbatch);
            for (int ipol = 0; ipol < 3; ipol++)
            {
                // CUBLAS_CHECK(cublasZgemmStridedBatched(blasHandle_, char_to_cublas_trans('n'), char_to_cublas_trans('n'), nbndsub_, nbndsub_, nbndsub_, &cone, d_champ_, nbndsub_, m, d_chv_+ipol*m*Nlist_, nbndsub_, m, &czero,d_chv_tmp_,nbndsub_,m,Nlist_));
                CUBLAS_CHECK(cublasZgemmStridedBatched(blasHandle_, char_to_cublas_trans('n'), char_to_cublas_trans('n'), nbnd, nbnd, nbnd, &cone, d_chv_ + ipol * m * Size_thisbatch, nbnd, m, d_champ_, nbnd, m, &czero, d_chv_tmp_, nbnd, m, Size_thisbatch));
                // CUBLAS_CHECK(cublasZgemmStridedBatched(blasHandle_, char_to_cublas_trans('n'), char_to_cublas_trans('c'), nbndsub_, nbndsub_, nbndsub_, &cone, d_chv_tmp_, nbndsub_, m, d_champ_, nbndsub_, m, &czero,d_chv_+ipol*m*Nlist_,nbndsub_,m,Nlist_));
                CUBLAS_CHECK(cublasZgemmStridedBatched(blasHandle_, char_to_cublas_trans('c'), char_to_cublas_trans('n'), nbnd, nbnd, nbnd, &cone, d_champ_, nbnd, m, d_chv_tmp_, nbnd, m, &czero, d_chv_ + ipol * m * Size_thisbatch, nbnd, m, Size_thisbatch));
            }

            _block = BLOCK_SIZE;
            _grid = (Size_thisbatch + block.x - 1) / block.x;
            vel_collect<<<_grid, _block>>>(d_chv_, d_vmef_ + offset * nbnd * 3, nbnd, Size_thisbatch);
            CUDA_CHECK(cudaMemcpy(el_velocity + offset * nbnd * 3, d_vmef_ + offset * nbnd * 3, nbnd * Size_thisbatch * 3 * sizeof(double), cudaMemcpyDeviceToHost));
            
            timer_GPUr_.stop_clock("band_wannier");
        }
    }

    void cuda_band_wannier_destroy_()
    {
        timer_GPUr_.print_clock("band_wannier_init");
        timer_GPUr_.print_clock("band_wannier");
        timer_GPUr_.print_clock("band_dos");
        timer_GPUr_.start_clock("band_wannier_destroy");
        CUDA_CHECK(cudaFree(d_NKgrid_));
        CUDA_CHECK(cudaFree(d_xkk_));
        CUDA_CHECK(cudaFree(d_at_));
        CUDA_CHECK(cudaFree(d_celldm1_));
        CUDA_CHECK(cudaFree(d_ndegen_k_));
        CUDA_CHECK(cudaFree(d_irvec_r_));
        CUDA_CHECK(cudaFree(d_irvec_cart_));
        CUDA_CHECK(cudaFree(d_rdotk_));
        CUDA_CHECK(cudaFree(d_cfac_));
        CUDA_CHECK(cudaFree(d_cfac_v_));
        CUDA_CHECK(cudaFree(d_chv_tmp_));
        CUDA_CHECK(cudaFree(d_chv_));
        CUDA_CHECK(cudaFree(d_vmef_));
        CUDA_CHECK(cudaFree(d_chw_));
        CUDA_CHECK(cudaFree(d_chf_));
        CUDA_CHECK(cudaFree(d_champ_));
        CUDA_CHECK(cudaFree(d_w_));
        CUDA_CHECK(cudaFree(devInfo_));

        CUSOLVER_CHECK(cusolverDnDestroySyevjInfo(syevj_params_));
        CUSOLVER_CHECK(cusolverDnDestroy(solverHandle_));
        CUBLAS_CHECK(cublasDestroy(blasHandle_));
        timer_GPUr_.stop_clock("band_wannier_destroy");
        timer_GPUr_.print_clock("band_wannier_destroy");
    }
    extern __global__ void fc_massfac(cuDoubleComplex *rdw, double *mass, int *ityp, int nmodes, int nrr);

    void cuda_ph_wannier_init_(int *_nmodes, int *_nrr_q, int *_Nlist, int *List, int *NKgrid, double *irvec_r, int *ndegen_k, cuDoubleComplex *chw,
                               int *_batchsize, double *_at, double *_bg, double *_celldm1, double *_Volume, int *_nat, double *_amass, int *_ityp, double *_tau,
                               bool *_lpolar, double *_zstar, double *_epsil, int *_nq1, int *_nq2, int *_nq3)
    {
        int *d_vec = nullptr;
        nrr_k_ = *_nrr_q;
        nmodes_ = *_nmodes;
        Nlist_ = *_Nlist;
        batchsize_ = *_batchsize;
        at_ = _at;
        bg_ = _bg;
        celldm1_ = _celldm1;
        Volume_ = *_Volume;
        nat_ = *_nat;
        amass_ = _amass;
        ityp_ = _ityp;
        tau_ = _tau;
        lpolar_ = *_lpolar;

        nqc_[0] = *_nq1;
        nqc_[1] = *_nq2;
        nqc_[2] = *_nq3;

        if(lpolar_){
            double geg = gmax_*4.0;
            for (int i=0;i<3;i++){
                if(nqc_[i]==1){
                    nrx_[i]=0;
                }
                else{
                    nrx_[i] = round(sqrt(geg) / sqrt(bg_[0+i*3]*bg_[0+i*3] + bg_[1+i*3]*bg_[1+i*3]+ bg_[2+i*3]*bg_[2+i*3])) + 1;
                }
            }
        }

        // printf("celldm1 = %f\n",*celldm1_);

        timer_GPUr_.start_clock("ph_wannier_init");

        CUBLAS_CHECK(cublasCreate(&blasHandle_));
        CUSOLVER_CHECK(cusolverDnCreate(&solverHandle_));
        CUSOLVER_CHECK(cusolverDnCreateSyevjInfo(&syevj_params_));
        CUSOLVER_CHECK(cusolverDnXsyevjSetTolerance(syevj_params_, solver_tol_));
        CUSOLVER_CHECK(cusolverDnXsyevjSetMaxSweeps(syevj_params_, solver_max_sweeps_));

        CUDA_CHECK(cudaMalloc((void **)&d_vec, 3 * Nlist_ * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&d_NKgrid_, 3 * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&d_xkk_, 3 * Nlist_ * sizeof(double)));

        CUDA_CHECK(cudaMalloc((void **)&d_at_, 9 * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_bg_, 9 * sizeof(double)));

        CUDA_CHECK(cudaMalloc((void **)&d_celldm1_, sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_amass_, nat_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_ityp_, nat_ * sizeof(int)));

        CUDA_CHECK(cudaMalloc((void **)&d_tau_, nat_ * 3 * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_xxq_cart_, Nlist_ * 3 * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_epsil_, 9 * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_zstar_, 9 * nat_ * sizeof(double)));

        CUDA_CHECK(cudaMalloc((void **)&d_ndegen_k_, nrr_k_ * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&d_irvec_r_, 3 * nrr_k_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_irvec_cart_, 3 * nrr_k_ * sizeof(double)));
        // CUDA_CHECK(cudaMalloc((void **)&d_rdotk_, nrr_k_ * batchsize_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_cfac_, nrr_k_ * batchsize_ * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc((void **)&d_cfac_v_, 3 * nrr_k_ * batchsize_ * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc((void **)&d_chv_tmp_, nmodes_ * nmodes_ * batchsize_ * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc((void **)&d_chv_, nmodes_ * nmodes_ * batchsize_ * 3 * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc((void **)&d_vmef_, nmodes_ * batchsize_ * 3 * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_chw_, nmodes_ * nmodes_ * nrr_k_ * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc((void **)&d_chf_, nmodes_ * nmodes_ * batchsize_ * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc((void **)&d_champ_, nmodes_ * nmodes_ * batchsize_ * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc((void **)&d_w_, nmodes_ * batchsize_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&devInfo_, batchsize_ * sizeof(int)));

        CUDA_CHECK(cudaMemcpy(d_at_, at_, 9 * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bg_, bg_, 9 * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_celldm1_, celldm1_, sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_amass_, amass_, nat_ * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ityp_, ityp_, nat_ * sizeof(int), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(d_tau_, tau_, nat_ * 3 * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_epsil_, _epsil, 9 * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_zstar_, _zstar, nat_ * 9 * sizeof(double), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(d_NKgrid_, NKgrid, 3 * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ndegen_k_, ndegen_k, nrr_k_ * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_irvec_r_, irvec_r, 3 * nrr_k_ * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_chw_, chw, nmodes_ * nmodes_ * nrr_k_ * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

        int *d_List = nullptr;
        CUDA_CHECK(cudaMalloc((void **)&d_List, Nlist_ * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_List, List, Nlist_ * sizeof(int), cudaMemcpyHostToDevice));

        calc_vec_xkk<<<(Nlist_ + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_List, d_NKgrid_, d_vec, d_xkk_, Nlist_);

        d_irvec_cart_init<<<(nrr_k_ + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_irvec_cart_, d_irvec_r_, d_at_, nrr_k_);

        // dim3 block_ = {BLOCK_SIZE, BLOCK_SIZE};
        // dim3 grid_ = {(nmodes_ * nmodes_ + block_.x - 1) / block_.x, (nrr_k_ + block_.y - 1) / block_.y};
        // fc_massfac<<<grid_, block_>>>(d_chw_, d_amass_, d_ityp_, nmodes_, nrr_k_);

        CUDA_CHECK(cudaFree(d_vec));
        CUDA_CHECK(cudaFree(d_NKgrid_));
        CUDA_CHECK(cudaFree(d_List));
        timer_GPUr_.stop_clock("ph_wannier_init");
    }

    __global__ void sqrt_wq2(double *w, int batchsize, int nmodes)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < batchsize)
        {
            for (int im = 0; im < nmodes; im++)
            {
                w[idx * nmodes + im] = w[idx * nmodes + im] > 0.0 ? sqrt(w[idx * nmodes + im]) : -sqrt(-w[idx * nmodes + im]);
            }
        }
    }

    // static __global__ void output_d_eptmp(char name, int offset, cuDoubleComplex *chf)
    // {
    //     for (int i = 0; i < 60; i++)
    //     {
    //         printf("%c d_eptmp : %e\n", name, cabs_(chf[i * offset]));
    //         // printf("d_eptmp : %e %e\n", chf[i*offset].x,chf[i*offset].y);
    //     }
    // }

    void cuda_ph_wannier_(double *w, double *ph_velocity)
    {
        int nbnd = nmodes_,
            nrr = nrr_k_,
            lwork = 0;
        int batchSize = batchsize_;
        cuDoubleComplex cone = {1.0, 0.0}, czero = {0.0, 0.0};
        cuDoubleComplex *d_work = nullptr;
        // double tpi = M_PI * 2.0;

        // extern __global__ void hermitianize_matrix_batched(int n, int batchSize, cuDoubleComplex *A);

        // extern __global__ void cfac_batched(int nrr, double *irvec, int *ndegen, double *d_xkk, ComplexD *cfac, int batchsize);

        // extern __global__ void iktoxxk_batch(int batchsize, double *xxk, int *ngrid, int ik);

        timer_GPUr_.start_clock("ph_wannier");
        for (int ibatch = 0; ibatch * batchSize < Nlist_; ibatch = ibatch + 1)
        {

            int offset = ibatch * batchSize;
            int Size_thisbatch = (ibatch + 1) * batchSize > Nlist_ ? Nlist_ - ibatch * batchSize : batchSize;

            // CUBLAS_CHECK(cublasDgemm(blasHandle_, char_to_cublas_trans('t'), char_to_cublas_trans('n'), nrr_k_, Size_thisbatch, 3, &tpi, d_irvec_r_, 3, d_xkk_, 3, &dzero, d_rdotk_, nrr_k_));

            dim3 block(BLOCK_SIZE, BLOCK_SIZE); // dim3 variable holds 3 dimensions
            dim3 grid((nrr + block.x - 1) / block.x, (batchSize + block.y - 1) / block.y);
            cfac_batched<<<grid, block>>>(nrr, d_irvec_r_, d_ndegen_k_, d_xkk_ + offset * 3, (ComplexD *)d_cfac_, Size_thisbatch);

            CUBLAS_CHECK(cublasZgemm(blasHandle_, char_to_cublas_trans('n'), char_to_cublas_trans('n'), nbnd * nbnd, Size_thisbatch, nrr, &cone, d_chw_, nbnd * nbnd, d_cfac_, nrr, &czero, d_champ_, nbnd * nbnd));

            if(lpolar_) {
                int nat=nat_;
                double e2 = 2.0; // e^2 in rydberg unit
                double omega = Volume_/ryd2nm/ryd2nm/ryd2nm;
                double fac0 = e2  * 4.0 * Pi /omega; // in the unit of rydberg 
                double  *tau = d_tau_,
                        *bg = d_bg_,
                        *xqc = d_xxq_cart_,
                        *epsil = d_epsil_,
                        *zeu = d_zstar_,
                        *d_xxq = d_xkk_,
                        gmax = gmax_;

                cuDoubleComplex *dyn=d_champ_;

                block = {BLOCK_SIZE2}; // dim3 variable holds 3 dimensions
                grid= {(batchSize + block.x - 1) / block.x};

                cryst_to_cart_global<<<grid, block>>>(d_bg_, d_xxq, xqc,Size_thisbatch);

                dynmat_polar<<<grid, block>>>(Size_thisbatch, nrx_[0], nrx_[1],nrx_[2], nbnd, nat, bg,
                                                tau, xqc, (ComplexD *)dyn, epsil, zeu, gmax, fac0);

            }


            // hermitianize and devided by mass factor 
            double *amass = d_amass_;
            int *ityp = d_ityp_;

            block={BLOCK_SIZE, BLOCK_SIZE}; // dim3 variable holds 3 dimensions
            grid={(nbnd * (nbnd + 1) / 2 + block.x - 1) / block.x, (Size_thisbatch + block.y - 1) / block.y};
            // hermitianize_matrix_batched<<<grid, block>>>(nbnd, Size_thisbatch, d_champ_);
            dynmat_prep<<<grid, block>>>((ComplexD *)d_champ_, amass, ityp, nbnd, batchSize);

            CUSOLVER_CHECK(cusolverDnZheevjBatched_bufferSize(solverHandle_, jobz_, uplo_, nbnd, d_champ_, nbnd, d_w_, &lwork, syevj_params_, Size_thisbatch));

            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(cuDoubleComplex) * lwork));
            CUSOLVER_CHECK(cusolverDnZheevjBatched(solverHandle_, jobz_, uplo_, nbnd, d_champ_, nbnd, d_w_, d_work, lwork, devInfo_, syevj_params_, Size_thisbatch));
            CUDA_CHECK(cudaFree(d_work));

            sqrt_wq2<<<(Size_thisbatch + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_w_, Size_thisbatch, nmodes_);
            CUDA_CHECK(cudaMemcpy(w + offset * nbnd, d_w_, sizeof(double) * nbnd * Size_thisbatch, cudaMemcpyDeviceToHost));

            grid = {(nrr + block.x - 1) / block.x, (Size_thisbatch + block.y - 1) / block.y};
            vel_matrix_0<<<grid, block>>>(nrr, Size_thisbatch, *celldm1_, d_irvec_cart_, (ComplexD *)d_cfac_, (ComplexD *)d_cfac_v_);
            int m = nmodes_ * nmodes_;
            CUBLAS_CHECK(cublasZgemm(blasHandle_, char_to_cublas_trans('n'), char_to_cublas_trans('n'), m, 3 * Size_thisbatch, nrr_k_, &cone, d_chw_, m, d_cfac_v_, nrr_k_, &czero, d_chv_, m));
            for (int ipol = 0; ipol < 3; ipol++)
            {
                // CUBLAS_CHECK(cublasZgemmStridedBatched(blasHandle_, char_to_cublas_trans('n'), char_to_cublas_trans('n'), nmodes_, nmodes_, nmodes_, &cone, d_champ_, nmodes_, m, d_chv_+ipol*m*Nlist_, nmodes_, m, &czero,d_chv_tmp_,nmodes_,m,Nlist_));
                CUBLAS_CHECK(cublasZgemmStridedBatched(blasHandle_, char_to_cublas_trans('n'), char_to_cublas_trans('n'), nmodes_, nmodes_, nmodes_, &cone, d_chv_ + ipol * m * Size_thisbatch, nmodes_, m, d_champ_, nmodes_, m, &czero, d_chv_tmp_, nmodes_, m, Size_thisbatch));
                // CUBLAS_CHECK(cublasZgemmStridedBatched(blasHandle_, char_to_cublas_trans('n'), char_to_cublas_trans('c'), nmodes_, nmodes_, nmodes_, &cone, d_chv_tmp_, nmodes_, m, d_champ_, nmodes_, m, &czero,d_chv_+ipol*m*Nlist_,nmodes_,m,Nlist_));
                CUBLAS_CHECK(cublasZgemmStridedBatched(blasHandle_, char_to_cublas_trans('c'), char_to_cublas_trans('n'), nmodes_, nmodes_, nmodes_, &cone, d_champ_, nmodes_, m, d_chv_tmp_, nmodes_, m, &czero, d_chv_ + ipol * m * Size_thisbatch, nmodes_, m, Size_thisbatch));
            }

            block = BLOCK_SIZE;
            grid = (Size_thisbatch + block.x - 1) / block.x;
            vel_collect_ph<<<grid, block>>>(d_chv_, d_vmef_, nbnd, Size_thisbatch, d_w_);
            CUDA_CHECK(cudaMemcpy(ph_velocity + offset * nbnd * 3, d_vmef_, nbnd * Size_thisbatch * 3 * sizeof(double), cudaMemcpyDeviceToHost));
        }

        timer_GPUr_.stop_clock("ph_wannier");
    }

    void cuda_ph_wannier_destroy_()
    {
        timer_GPUr_.print_clock("ph_wannier_init");
        timer_GPUr_.print_clock("ph_wannier");

        timer_GPUr_.start_clock("ph_wannier_destroy");

        CUDA_CHECK(cudaFree(d_xkk_));
        CUDA_CHECK(cudaFree(d_at_));
        CUDA_CHECK(cudaFree(d_bg_));
        CUDA_CHECK(cudaFree(d_celldm1_));
        CUDA_CHECK(cudaFree(d_amass_));
        CUDA_CHECK(cudaFree(d_ityp_));
        CUDA_CHECK(cudaFree(d_tau_));
        CUDA_CHECK(cudaFree(d_epsil_));
        CUDA_CHECK(cudaFree(d_zstar_));
        CUDA_CHECK(cudaFree(d_xxq_cart_));
        CUDA_CHECK(cudaFree(d_ndegen_k_));
        CUDA_CHECK(cudaFree(d_irvec_r_));
        CUDA_CHECK(cudaFree(d_irvec_cart_));
        // CUDA_CHECK(cudaFree(d_rdotk_));
        CUDA_CHECK(cudaFree(d_cfac_));
        CUDA_CHECK(cudaFree(d_cfac_v_));
        CUDA_CHECK(cudaFree(d_chv_tmp_));
        CUDA_CHECK(cudaFree(d_chv_));
        CUDA_CHECK(cudaFree(d_vmef_));
        CUDA_CHECK(cudaFree(d_chw_));
        CUDA_CHECK(cudaFree(d_chf_));
        CUDA_CHECK(cudaFree(d_champ_));
        CUDA_CHECK(cudaFree(d_w_));
        CUDA_CHECK(cudaFree(devInfo_));

        CUSOLVER_CHECK(cusolverDnDestroySyevjInfo(syevj_params_));
        CUSOLVER_CHECK(cusolverDnDestroy(solverHandle_));
        CUBLAS_CHECK(cublasDestroy(blasHandle_));
        timer_GPUr_.stop_clock("ph_wannier_destroy");

        timer_GPUr_.print_clock("ph_wannier_destroy");
    }

    __global__ static void DOS_of_el(double *dos, double *el_energy, double *el_vel, double *G, int *Eqindex, double *orth, int ismear_ecp, double degauss, double scalebroad, int nk, int nkfull, int nbnd, int ntick, int ntick_cut, double *etick, int *Ngrid)
    {
        int ik = blockDim.x * blockIdx.x + threadIdx.x;
        // printf("Ngrid = %d %d %d\n", Ngrid[0],Ngrid[1],Ngrid[2]);

        if (ik < nkfull)
        {
            double estep = (etick[ntick - 1] - etick[0]) / (ntick - 1);
            for (int ie = -ntick_cut; ie < ntick_cut + 1; ie++)
            {
                double sigma;
                double weight = 0.0;
                for (int ib = 0; ib < nbnd; ib++)
                {
                    int s_idx = Eqindex[ik] - 1;
                    double ek = el_energy[s_idx * nbnd + ib];
                    int tick0 = round((ek - etick[0]) / estep);
                    tick0 = tick0 > ntick - ntick_cut - 2 ? ntick - ntick_cut - 2 : tick0;
                    tick0 = tick0 < ntick_cut ? ntick_cut : tick0;
                    // tick0 = 0;
                    double e = etick[ie + tick0];
                    ek = ek - e;
                    double v[3];
                    if (ismear_ecp == 0)
                    {
                        int orth_dix = Eqindex[ik + nkfull] - 1;
                        matmul(v, &el_vel[3 * s_idx * nbnd + 3 * ib], &orth[orth_dix * 9]);
                        sigma = sigma_ags(v, G, scalebroad, Ngrid);
                    }
                    else
                    {
                        sigma = degauss;
                    }
                    if (abs(ek) < sigma * 3.5)
                    {
                        weight = exp(-pow(ek, 2) / pow(sigma, 2)) / sigma * M_2_SQRTPI * 0.5 / double(nkfull);
                        atomicAdd(&dos[ie + tick0], weight);
                    }
                }
            }
        }
    }

    __global__ void recons_vel(double *vel_new, double *vel_old, int nbnd,int Nlist, double ryd2nmev){
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        int idy = blockDim.y * blockIdx.y + threadIdx.y;
        if ( idx < Nlist && idy < 3*nbnd){
            int ib = idy/3; // 
            int ipol = idy%3; //
            vel_new[ipol + ib * 3 + idx * 3 * nbnd] = vel_old[ib + ipol * nbnd + idx * nbnd * 3 ]*ryd2nmev;
        }
    }

    __global__ void trans_ener(double *ener_new, double *ener_old, int nbnd, int Nlist, double ryd2ev)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        int idy = blockDim.y * blockIdx.y + threadIdx.y;
        if (idx < Nlist && idy < nbnd)
        {
            ener_new[idy + idx * nbnd] = ener_old[idy + idx * nbnd] * ryd2ev;
        }
    }

    __global__ void sumk_gpu(double *ne, double e, double *el_energy, double invTemp, int *Eqindex, int nkfull, int nbnd, double spin)
    {
        int ik = blockDim.x * blockIdx.x + threadIdx.x;
        if (ik < nkfull)
        {
            int s_idx = Eqindex[ik] - 1;
            for (int ib = 0; ib < nbnd; ib++)
            {
                double ek = el_energy[s_idx * nbnd + ib] - e;
                double weight;
                weight = spin * 1.0 / (exp(ek * invTemp) + 1.0) / double(nkfull);
                atomicAdd(ne, weight);
            }
        }
    }

    __global__ static void calc_DosFermi(double *DosFermi, double *V2Fermi, double *el_energy, double *el_vel, double *G, int *Eqindex, double *orth, int ismear_ecp, double degauss, double scalebroad, int nk, int nkfull, int nbnd, double ef, int *Ngrid)
    {
        int ik = blockDim.x * blockIdx.x + threadIdx.x;

        if (ik < nkfull)
        {
            double sigma;
            for (int ib = 0; ib < nbnd; ib++)
            {
                int s_idx = Eqindex[ik] - 1;
                double etmp = el_energy[s_idx * nbnd + ib];
                double ek = etmp - ef;
                double v[3];
                if (ismear_ecp == 0)
                {
                    int orth_dix = Eqindex[ik + nkfull] - 1;
                    matmul(v, &el_vel[3 * s_idx * nbnd + 3 * ib], &orth[orth_dix * 9]);
                    sigma = sigma_ags(v, G, scalebroad, Ngrid);
                }
                else
                {
                    sigma = degauss;
                }
                if (abs(ek) < sigma * 3.5)
                {
                    double tmp_v = exp(-pow(ek, 2) / pow(sigma, 2)) / sigma * M_2_SQRTPI * 0.5 / double(nkfull);
                    
                    atomicAdd(DosFermi, tmp_v);

                    for (int i = 0; i < 3; ++i)
                    {
                        atomicAdd(&V2Fermi[i], tmp_v * pow(v[i], 2));
                    }
                }
            }
        }
    }

    static double find_ef(double *d_el_energy, double ne0, double Temp, double emin, double emax, int *d_Eqindex, int nkfull, int nbnd)
    {
        double invTemp;
        double diff_e = 1.0;
        double eps = 1e-10;
        invTemp = 1.0 / (Temp);
        double *d_ne;
        double ne;
        // double ne_prev = 0.0;
        double e_up, e_lw;
        double spin_deg = 2.0;
        e_up = emax;
        e_lw = emin;
        double e_test = (emax + emin) / 2.0;
        while (abs(diff_e) > eps)
        {
            dim3 block = BLOCK_SIZE;
            dim3 grid = (nkfull - 1 + block.x) / block.x;
            cudaMalloc((void **)&d_ne, sizeof(double));
            cudaMemset(d_ne, 0, sizeof(double));
            sumk_gpu<<<grid, block>>>(d_ne, e_test, d_el_energy, invTemp, d_Eqindex, nkfull, nbnd, spin_deg);
            CUDA_CHECK(cudaMemcpy(&ne, d_ne, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaDeviceSynchronize());
            diff_e = ne - ne0;
            e_up = (diff_e < 0 ? e_up : e_test);
            e_lw = (diff_e < 0 ? e_test : e_lw);
            e_test = (e_up + e_lw) / 2.0;
        }
        return e_test;
    }

    void cuda_band_dos_(double *_ef, double *_delta_ef, double *ne, double *Temp, int *_ismear_ecp, double *_degauss, double *_scalebroad, bool *_lmetal, double *_DosFermi, double *_V2Fermi)
    {
        int nkfull = NPTK_K_;
        int nk = Nlist_;
        int nbnd = nbndsub_;
        // int nbnd_eff = nbands_;
        // int nbnd_irr = nbnd_irr_;
        int ismear = *_ismear_ecp;
        double degauss = *_degauss;
        int scalebroad = *_scalebroad;
        bool lmetal = *_lmetal;
        static int ntick = 5001;
        double spin_deg = 2.0;

        // static int ntick = 1;

        double *d_energy_ = d_w_;
        double *d_ener;
        double *d_vel_old = d_vmef_;
        double *d_vel;
        double emin;
        double emax;
        // double ener_in[nbands_ * nk];
        // double vel_in[3 * nbands_ * nk];
        double *eticks;
        double *d_eticks;
        double *ef = _ef;
        double delta_ef = *_delta_ef;
        double *dos;
        double *d_dos;
        double *DosFermi = _DosFermi;
        double *d_DosFermi;
        double *V2Fermi = _V2Fermi;
        double *d_V2Fermi;

        timer_GPUr_.start_clock("band_dos");

        // for (int ib=0; ib<nbnd_eff; ib ++){
        //     for(int ik = 0; ik<nk; ik++){
        //         ener_in[ib+ik*nbnd_eff]=el_energy_[ib+nbnd_irr+ik*nbnd];
        //         // printf("ener_in = %d %d %f\n", ik,ib,ener_in[ib+ik*nbnd_eff]*ryd2ev);
        //     }
        // }

        // double *d_ener_in;
        // CUDA_CHECK(cudaMalloc((void **) &d_ener_in,nbnd_eff*nk*sizeof(double)));
        // CUDA_CHECK(cudaMemcpy(d_ener_in,ener_in,nbnd_eff*nk*sizeof(double),cudaMemcpyHostToDevice));

        printf("Reconstructing velocities...\n");
        dim3 block = {BLOCK_SIZE, BLOCK_SIZE};
        dim3 grid = {(nk - 1 + block.x) / block.x, (3 * nbnd - 1 + block.y) / block.y};
        CUDA_CHECK(cudaMalloc((void **)&d_vel, 3 * nbnd * nk * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_ener, nbnd * nk * sizeof(double)));

        // dim3 grid = {(nk - 1 + block.x)/block.x, ( 3*nbnd_eff - 1 + block.y)/ block.y};
        // CUDA_CHECK(cudaMalloc((void **) &d_vel,3*nbnd_eff*nk*sizeof(double)));
        // CUDA_CHECK(cudaMalloc((void **) &d_ener,nbnd_eff*nk*sizeof(double)));

        /*
        d_vel_old(nbnd,nk_irr,3) in the unit of ryd to d_vel(3,nbnd,nk_irr) in the unit of nm*eV
        */

        recons_vel<<<grid, block>>>(d_vel, d_vel_old, nbnd, nk, ryd2nmev);

        grid = {(nk - 1 + block.x) / block.x, (nbnd - 1 + block.y) / block.y};
        trans_ener<<<grid, block>>>(d_ener, d_energy_, nbnd, nk, ryd2ev);

        emax = *thrust::max_element(el_energy_, el_energy_ + Nlist_ * nbnd) * ryd2ev + 0.1;
        emin = *thrust::min_element(el_energy_, el_energy_ + Nlist_ * nbnd) * ryd2ev - 0.1;

        eticks = (double *)malloc(ntick * sizeof(double));
        for (int i = 0; i < ntick; i++)
        {
            eticks[i] = emin + double(i) / (ntick - 1) * (emax - emin); // in the unit of ev
        }

        CUDA_CHECK(cudaMalloc((void **)&d_eticks, ntick * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_eticks, eticks, ntick * sizeof(double), cudaMemcpyHostToDevice));

        dos = (double *)malloc(ntick * sizeof(double));
        CUDA_CHECK(cudaMalloc((void **)&d_dos, ntick * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_dos, 0, ntick * sizeof(double)));

        block = BLOCK_SIZE;
        grid = {(nkfull - 1 + block.x) / block.x};

        int ntick_cut = 50;

        DOS_of_el<<<grid, block>>>(d_dos, d_ener, d_vel, d_bg_, d_eqidx_, d_orthcar_, ismear, degauss, scalebroad, nk, nkfull, nbnd, ntick, ntick_cut, d_eticks, d_NKgrid_);
        // DOS_of_el<<<grid,block>>>(d_dos, d_ener, d_vel, d_bg_, d_eqidx_, d_orthcar_, ismear, degauss, scalebroad, nk, nkfull, nbnd_eff, ntick, ntick_cut, d_eticks, d_NKgrid_);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(dos, d_dos, ntick * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());

        for (int i = 0; i < ntick; i++)
        {
            dos[i] *= spin_deg;
        }

        /*
         * write DOS file
         */

        FILE *dosfile;

        dosfile = fopen("BTE.dos", "w");

        for (int i = 0; i < ntick; i++)
        {
            fprintf(dosfile, "%20.10f %20.10f\n", eticks[i], dos[i]);
        }

        fclose(dosfile);

        /*
         * write cummunitive DOS file
         */

        dosfile = fopen("BTE.cumulative.dos", "w");

        double realaux = 0.0;
        for (int i = 0; i < ntick; i++)
        {
            realaux += dos[i] / (ntick - 1) * (emax - emin);
            fprintf(dosfile, "%20.10f %20.10f\n", eticks[i], realaux);
        }

        fclose(dosfile);

        /*
         * calculate Ef for metal
         */

        if (lmetal)
        {
            if (*ef < -1.0e-10)
            {
                *ef = find_ef(d_ener, *ne, *Temp * Kb, emin, emax, d_eqidx_, nkfull, nbnd);
                printf("Info: Calculated Fermi energy = %15.7f eV\n", *ef);
            }
            else
            {
                printf("Info: Input Fermi energy = %15.7f eV\n", *ef);
            }
            *ef = *ef + delta_ef;
            printf("Info: Shifted energy = %15.7f eV\n", *ef);
        }

        /*
         * TODO: calculate DOS_Ef and AV2_Ef for metal
         */
        if (lmetal)
        {
            CUDA_CHECK(cudaMalloc((void **)&d_DosFermi, sizeof(double)));
            CUDA_CHECK(cudaMalloc((void **)&d_V2Fermi, 3 * sizeof(double)));
            CUDA_CHECK(cudaMemset(d_DosFermi, 0, sizeof(double)));
            CUDA_CHECK(cudaMemset(d_V2Fermi, 0, 3 * sizeof(double)));

            calc_DosFermi<<<grid, block>>>(d_DosFermi, d_V2Fermi, d_ener, d_vel, d_bg_, d_eqidx_, d_orthcar_, ismear, degauss, scalebroad, nk, nkfull, nbnd, *ef, d_NKgrid_);

            CUDA_CHECK(cudaMemcpy(DosFermi, d_DosFermi, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(V2Fermi, d_V2Fermi, 3 * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaDeviceSynchronize());

            for (int i = 0; i < 3; ++i)
            {
                V2Fermi[i] /= *DosFermi;
            }

            printf("Info: Density of DosFermi = %15.7f eV\n", *DosFermi);
            printf("Info: Velocity of V2Fermi %10.5f \t %10.5f \t %10.5f\n", V2Fermi[0], V2Fermi[1], V2Fermi[2]);

            CUDA_CHECK(cudaFree(d_DosFermi));
            CUDA_CHECK(cudaFree(d_V2Fermi));
        }

        CUDA_CHECK(cudaFree(d_vel));
        CUDA_CHECK(cudaFree(d_ener));
        CUDA_CHECK(cudaFree(d_eticks));
        CUDA_CHECK(cudaFree(d_dos));

        timer_GPUr_.stop_clock("band_dos");
    }
}

__global__ static void calc_vec_xkk(int *List, int *NKgrid, int *vec, double *xkk, int Nlist)
{
    int ik = blockDim.x * blockIdx.x + threadIdx.x;

    if (ik < Nlist)
    {
        int ikk = List[ik] - 1;
        vec[0 + ik * 3] = ikk / (NKgrid[1] * NKgrid[2]);
        vec[1 + ik * 3] = ikk % (NKgrid[1] * NKgrid[2]) / NKgrid[2];
        vec[2 + ik * 3] = ikk % NKgrid[2];

        for (int i = 0; i < 3; ++i)
        {
            xkk[i + ik * 3] = double(vec[i + ik * 3]) / double(NKgrid[i]);
        }
    }
}

__global__ static void get_cfac_from_rdot(ComplexD *cfac, double *rdot, int *ndegen, int nrr, int batchsize)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    if (idx < nrr && idy < batchsize)
    {
        cfac[idy * nrr + idx] = exp(ComplexD(0.0, 1.0) * rdot[idy * nrr + idx]) / (double)ndegen[idx];
        // if(idy == 1){
        // printf("cfac : %d,%f,%f\n", idy * nrr + idx, rdot[idy * nrr + idx], cfac[idy * nrr + idx].real());
        // }
    }
}

__global__ static void vel_matrix_0(int nrr, int Nlist, double celldm1, double *irvec_cart, ComplexD *cfac, ComplexD *cfac_v)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    ComplexD ci(0.0, 1.0);

    if (idx < nrr && idy < Nlist)
    {
        for (int ipol = 0; ipol < 3; ipol++)
        {
            cfac_v[idx + idy * nrr + ipol * nrr * Nlist] = ci * celldm1 * cfac[idx + idy * nrr] * irvec_cart[ipol + idx * 3];
            // printf("%f\n",cfac_v[idx + idy*nrr + ipol*nrr*Nlist].real());
        }
    }
}

    __global__ static void vel_collect_ph(cuDoubleComplex *vc, double *vmef, int nbnd, int Nlist, double *w)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < Nlist){
            for (int ipol=0; ipol<3; ipol++)
            {
                for (int ibnd=0; ibnd<nbnd; ibnd++)
                {
                    // vmef[ibnd+ idx*nbnd+ipol*nbnd*Nlist]=vc[ibnd+ibnd*nbnd+nbnd*nbnd*idx+ipol*nbnd*nbnd*Nlist].x/(1.e-10+2.0*w[ibnd+ idx*nbnd]);
                    vmef[ibnd+ ipol*nbnd +  idx*nbnd*3]=vc[ibnd+ibnd*nbnd+nbnd*nbnd*idx+ipol*nbnd*nbnd*Nlist].x/(1.e-10+2.0*w[ibnd+ idx*nbnd]);
                }
            }
        }
    }
    __global__ static void vel_collect(cuDoubleComplex *vc, double *vmef, int nbnd, int Nlist)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;

        if (idx < Nlist)
        {
            for (int ipol=0; ipol<3; ipol++)
            {
                for (int ibnd=0; ibnd<nbnd; ibnd++)
                {
                    // vmef[ibnd+ idx*nbnd+ipol*nbnd*Nlist]=vc[ibnd+ibnd*nbnd+nbnd*nbnd*idx+ipol*nbnd*nbnd*Nlist].x;
                    vmef[ibnd+ ipol*nbnd + idx*nbnd*3]=vc[ibnd+ibnd*nbnd+nbnd*nbnd*idx+ipol*nbnd*nbnd*Nlist].x;
                }
            }
        }
    }

__global__ static void init_champ_matrix(int n, ComplexD *champ, ComplexD *chf, int Nlist)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    if (idx < n * n && idy < Nlist)
    {
        int ibnd = idx / n;
        int jbnd = idx % n;
        champ[ibnd * n + jbnd + idy * n * n] = (chf[ibnd * n + jbnd + idy * n * n] + conj(chf[jbnd * n + ibnd + idy * n * n])) * ComplexD(0.5, 0.0);
        // printf("%d champ[%d,%d] : %f %f\n",idy,ibnd,jbnd,champ[ibnd * n + jbnd + idy*n*n].real(),champ[ibnd * n + jbnd + idy*n*n].imag());
    }
}

__global__ static void d_irvec_cart_init(double *irvec_cart, double *ivrec_cryst, double *at, int nrr)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < nrr)
    {
        for (int ipol = 0; ipol < 3; ipol++)
        {
            irvec_cart[ipol + idx * 3] = 0.0;
            for (int jpol = 0; jpol < 3; jpol++)
            {
                // irvec_cart[ipol+idx*3] += ivrec_cryst[jpol+idx*3] * at[jpol+ipol*3];
                irvec_cart[ipol + idx * 3] += ivrec_cryst[jpol + idx * 3] * at[ipol + jpol * 3];
            }
        }
    }
}
