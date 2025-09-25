// #include <unistd.h>
#include "cuda_elphwann_namespace.h"
#include "cuda_settings.h"
#include "cuda_timer.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <thrust/copy.h>
#include <complex>

using thrust::device;
using thrust::reduce;

using namespace cuda::std;
using namespace cuda_elphwann_wannier;
using namespace device_funcs;
using namespace polar_funcs;

// using namespace std;

extern "C"
{
    void gpu_cuda_init_(int *_nbands,
                        int *_nbndsub,
                        int *_nrr_k,
                        int *_nrr_q,
                        int *_nmodes,
                        double *_celldm1,
                        double *_bg,
                        double *_rlattvec,
                        double *_Volume,
                        int *_nat,
                        int *_ntypx,
                        double *_amass,
                        int *_ityp,
                        double *_spin_degen,
                        double *_tau,
                        int *_Nstates,
                        int *_NStateInterest,
                        int *_StateInterest,
                        double *_irvec_r,
                        int *_irvec,
                        int *_ndegen_k,
                        int *_ndegen_q,
                        cuDoubleComplex *epmatwp,
                        cuDoubleComplex *chw,
                        cuDoubleComplex *rdw,
                        bool *_lpolar,
                        double *_epsil,
                        double *_zstar,
                        int *_nq1,
                        int *_nq2,
                        int *_nq3,
                        int *_NKgrid,
                        int *_NQgrid,
                        int *_Nsymm,
                        int *_Nlist_K,
                        int *_Nlist_Q,
                        int *_NPTK_K,
                        int *_NPTK_Q,
                        int *_List,
                        int *_Eqindex_K,
                        int *_Eqindex_Q,
                        double *_Te,
                        int *_ismear_ecp,
                        double *_scalebroad,
                        double *_degauss,
                        double *_delta_mult,
                        double *_ph_cut,
                        double *_el_energy,
                        double *_ph_energy,
                        double *_el_velocity,
                        double *_ph_velocity,
                        double *_Orthcar,
                        bool *_convergence,
                        double *_iter_tolerance,
                        int *_maxiter,
                        int *_batch_size,
                        int *_mpime)
    {
        nbands_ = *_nbands;
        nbndsub_ = *_nbndsub;
        nrr_k_ = *_nrr_k;
        nrr_q_ = *_nrr_q;
        nmodes_ = *_nmodes;
        celldm1_ = *_celldm1;
        bg_ = _bg;
        Volume_ = *_Volume;
        nat_ = *_nat;
        ntypx_ = *_ntypx;
        amass_ = _amass;
        ityp_ = _ityp;
        spin_degen_ = *_spin_degen;
        tau_ = _tau; //
        Nstates_ = *_Nstates;
        NStateInterest_ = *_NStateInterest;
        StateInterest_ = _StateInterest;
        irvec_r_ = _irvec_r;
        irvec_ = _irvec;
        ndegen_k_ = _ndegen_k;
        ndegen_q_ = _ndegen_q;

        lpolar_ = *_lpolar;
        epsil_ = _epsil;
        zstar_ = _zstar;

        nqc_ = (int *) malloc(3*sizeof(int));
        nqc_[0] = *_nq1;
        nqc_[1] = *_nq2;
        nqc_[2] = *_nq3;

        if (lpolar_) {
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

        NKgrid_ = _NKgrid;
        NQgrid_ = _NQgrid;
        Nsymm_ = *_Nsymm;
        Nlist_K_ = *_Nlist_K;
        Nlist_Q_ = *_Nlist_Q;
        NPTK_K_ = *_NPTK_K;
        NPTK_Q_ = *_NPTK_Q;
        List_ = _List;
        Eqindex_K_ = _Eqindex_K;
        Eqindex_Q_ = _Eqindex_Q;
        Te_ = *_Te;
        ismear_ecp_ = *_ismear_ecp;
        scalebroad_ = *_scalebroad;
        degauss_ = *_degauss;
        delta_mult_ = *_delta_mult;
        ph_cut_ = *_ph_cut;
        el_energy_ = _el_energy;
        ph_energy_ = _ph_energy;
        el_velocity_ = (double *)malloc(3 * nbands_ * Nlist_K_ * sizeof(double));
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < Nlist_K_; j++)
            {
                for (int k = 0; k < nbands_; k++)
                {
                    el_velocity_[i + k * 3 + j * 3 * nbands_] = _el_velocity[j + k * Nlist_K_ + i * Nlist_K_ * nbands_];
                }
            }
        }
        ph_velocity_ = (double *)malloc(3 * nmodes_ * Nlist_Q_ * sizeof(double));
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < Nlist_Q_; j++)
            {
                for (int k = 0; k < nmodes_; k++)
                {
                    ph_velocity_[i + k * 3 + j * 3 * nmodes_] = _ph_velocity[j + k * Nlist_Q_ + i * Nlist_Q_ * nmodes_];
                }
            }
        }

        Orthcar_ = _Orthcar;
        convergence_ = *_convergence;
        iter_tolerance_ = *_iter_tolerance;
        maxiter_ = *_maxiter;

        batch_size_ = *_batch_size;
        batch_size_k_ = batch_size_*10;
        batch_size_q_ = batch_size_*100;
        batch_size_iter_ = batch_size_*10000;
        mpime_ = *_mpime;

        int m = nbndsub_ * nbndsub_ * nrr_q_ * nmodes_,
            n = nrr_k_;

        /* allocate global variables */
        CUDA_CHECK(cudaMalloc((void **)&d_amass_,  ntypx_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_ityp_,  nat_ * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&d_tau_,  nat_ * 3 * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_rlattvec_, 3 * 3 * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_bg_, 3 * 3 * sizeof(double)));

        CUDA_CHECK(cudaMalloc((void **)&d_epsil_, 3 * 3 * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_zstar_, 3 * 3 * nat_ * sizeof(double)));

        CUDA_CHECK(cudaMalloc((void **)&d_irvec_r_, 3 * nrr_k_ * sizeof(double)));

        CUDA_CHECK(cudaMalloc((void **)&d_irvec_, 3 * nrr_k_ * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&d_ndegen_k_, nrr_k_ * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&d_ndegen_q_, nrr_q_ * sizeof(int)));

        CUDA_CHECK(cudaMalloc((void **)&d_Orthcar_, 9 * Nsymm_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_Orthcar_ptr_, NPTK_Q_ * sizeof(double *)));
        CUDA_CHECK(cudaMalloc((void **)&d_epmatwp_, m * n * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc((void **)&d_chw_, nbndsub_ * nbndsub_ * nrr_k_ * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc((void **)&d_rdw_, nmodes_ * nmodes_ * nrr_q_ * sizeof(cuDoubleComplex)));

        int l_eig = max(nbndsub_ * batch_size_k_, nmodes_*batch_size_);
        CUDA_CHECK(cudaMalloc((void **)&d_eig_, sizeof(double) * l_eig));
        CUDA_CHECK(cudaMalloc((void **)&d_wfqq_, nmodes_ * batch_size_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_etkk_,  nbndsub_ * batch_size_k_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_w2_, nmodes_ * batch_size_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_etkq_, nbndsub_ * batch_size_ * sizeof(double)));

        CUDA_CHECK(cudaMalloc((void **)&d_cufkk_,  nbndsub_ * nbndsub_ * batch_size_k_ * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc((void **)&d_cuf_, nmodes_ * nmodes_ * batch_size_ * sizeof(cuDoubleComplex) ));
        CUDA_CHECK(cudaMalloc((void **)&d_cufkq_,  nbndsub_ * nbndsub_ * batch_size_ * sizeof(cuDoubleComplex)));

        CUDA_CHECK(cudaMalloc((void **)&d_epmatf_, batch_size_ * nbndsub_ * nbndsub_ * nmodes_ * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc((void **)&d_epmatfl_, batch_size_ * nbndsub_ * nbndsub_ * nmodes_ * sizeof(cuDoubleComplex)));
        // CUDA_CHECK(cudaMalloc((void **)&d_umn_, batch_size_ * nbndsub_ * nbndsub_ * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc((void **)&d_umn_, batch_size_ * nbands_ * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc((void **)&d_epmatf_dout, batch_size_ * nbands_  * nmodes_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_epwef_, nmodes_ * nrr_q_ * nbndsub_ * sizeof(cuDoubleComplex)));

        CUDA_CHECK(cudaMalloc((void **)&d_el_energy_, nbands_ * Nlist_K_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_ph_energy_, nmodes_ * Nlist_Q_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_el_velocity_, 3 * nbands_ * Nlist_K_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_ph_velocity_, 3 * nmodes_ * Nlist_Q_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_el_velocity_tmp_, 3 * nbands_ * NPTK_K_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_ph_velocity_tmp_, 3 * nmodes_ * NPTK_Q_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_el_velocity_ptr_, NPTK_K_ * sizeof(double *)));
        CUDA_CHECK(cudaMalloc((void **)&d_el_velocity_tmp_ptr_, NPTK_K_ * sizeof(double *)));
        CUDA_CHECK(cudaMalloc((void **)&d_ph_velocity_ptr_,  NPTK_Q_ * sizeof(double *)));
        CUDA_CHECK(cudaMalloc((void **)&d_ph_velocity_tmp_ptr_,  NPTK_Q_ * sizeof(double *)));

        CUDA_CHECK(cudaMalloc((void **)&d_xkk_, 3 * batch_size_k_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_xxq_, 3 * batch_size_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_xxq_cart_, 3 * batch_size_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_xkq_, 3 * batch_size_ * sizeof(double)));
        int l_eptmp = max(batch_size_  * nbndsub_ * nmodes_, nmodes_ * nbndsub_ * nrr_q_* nbndsub_);
        CUDA_CHECK(cudaMalloc((void **)&d_eptmp_, l_eptmp* sizeof(cuDoubleComplex)));

        /* temporary variables */
        int l_cfac = max(nrr_k_ * batch_size_k_, nrr_q_ * batch_size_);
        CUDA_CHECK(cudaMalloc((void **)&d_cfac_, sizeof(cuDoubleComplex) * l_cfac ));

        CUDA_CHECK(cudaMalloc((void **)&solverDevInfo_, batch_size_k_*sizeof(int)));

        CUDA_CHECK(cudaMemcpy(d_rlattvec_, _rlattvec, 3 * 3 * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bg_, bg_, 3 * 3 * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_irvec_r_, irvec_r_, 3 * nrr_k_ * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_irvec_, irvec_, 3 * nrr_k_ * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_epsil_, epsil_, 3 * 3 * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_zstar_, zstar_, 3 * 3 * nat_ * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ndegen_k_, ndegen_k_, nrr_k_ * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ndegen_q_, ndegen_q_, nrr_q_ * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Orthcar_, Orthcar_, 9 * Nsymm_ * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_epmatwp_, epmatwp, m * n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_chw_, chw, nbndsub_ * nbndsub_ * nrr_k_ * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rdw_, rdw, nmodes_ * nmodes_ * nrr_q_ * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_amass_, amass_, ntypx_ * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_tau_, tau_, nat_* 3 * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ityp_, ityp_, nat_ * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_el_energy_, el_energy_, nbands_ * Nlist_K_ * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ph_energy_, ph_energy_, nmodes_ * Nlist_Q_ * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_el_velocity_, el_velocity_, 3 * nbands_ * Nlist_K_ * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ph_velocity_, ph_velocity_, 3 * nmodes_ * Nlist_Q_ * sizeof(double), cudaMemcpyHostToDevice));

        // check valid variables
        CUDA_CHECK(cudaMalloc((void **)&d_kpq_, NPTK_Q_ * sizeof(int)));

        CUDA_CHECK(cudaMalloc((void **)&d_NQgrid_, 3 * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&d_NKgrid_, 3 * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_NQgrid_, NQgrid_, 3 * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_NKgrid_, NKgrid_, 3 * sizeof(int), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc((void **)&d_Eqindex_K_, NPTK_K_ * 2 * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_Eqindex_K_, Eqindex_K_, NPTK_K_ * 2 * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc((void **)&d_Eqindex_Q_, NPTK_Q_ * 2 * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_Eqindex_Q_, Eqindex_Q_, NPTK_Q_ * 2 * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc((void **)&d_valid_i_, NPTK_Q_ * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&d_valid_plus_i_, NPTK_Q_ * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&d_PhononInterest_, NPTK_Q_ * sizeof(int)));

        // CUDA_CHECK(cudaMalloc((void **)&d_e1_, 1 * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_e2_, nmodes_ *NPTK_Q_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_e3_, NPTK_Q_ * nbands_ * sizeof(double)));

        CUDA_CHECK(cudaMalloc((void **)&d_StateInterest_, Nstates_ * 2 * sizeof(int) ));
        CUDA_CHECK(cudaMalloc((void **)&d_List_K_, Nlist_K_ * sizeof(int) ));
        CUDA_CHECK(cudaMemcpy(d_StateInterest_, StateInterest_, Nstates_ * 2 * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_List_K_, List_, Nlist_K_ * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc((void **)&d_v2_, nmodes_ * 3 * NPTK_Q_ * sizeof(double) ));

        CUBLAS_CHECK(cublasCreate(&blasHandle_));
        CUSOLVER_CHECK(cusolverDnCreate(&solverHandle_));

        CUSOLVER_CHECK(cusolverDnCreateSyevjInfo(&syevj_params_));
        CUSOLVER_CHECK(cusolverDnXsyevjSetTolerance(syevj_params_, solver_tol_));
        CUSOLVER_CHECK(cusolverDnXsyevjSetMaxSweeps(syevj_params_, solver_max_sweeps_));

        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void nprocesses_cuda_deal_()
    {   

        CUDA_CHECK(cudaFree(d_Orthcar_ptr_));

        // CUDA_CHECK(cudaFree(d_ph_energy_));
        // CUDA_CHECK(cudaFree(d_ph_velocity_));
        CUDA_CHECK(cudaFree(d_ph_velocity_tmp_));
        CUDA_CHECK(cudaFree(d_ph_velocity_ptr_));
        CUDA_CHECK(cudaFree(d_ph_velocity_tmp_ptr_));
        CUDA_CHECK(cudaFree(d_el_velocity_tmp_));
        CUDA_CHECK(cudaFree(d_el_velocity_ptr_));
        CUDA_CHECK(cudaFree(d_el_velocity_tmp_ptr_));

        /* check valid vars */

        CUDA_CHECK(cudaFree(d_valid_plus_i_));
        CUDA_CHECK(cudaFree(d_PhononInterest_));
        // CUDA_CHECK(cudaFree(d_e1_));
        CUDA_CHECK(cudaFree(d_e2_));
        CUDA_CHECK(cudaFree(d_e3_));
        CUDA_CHECK(cudaFree(d_kpq_));
        CUDA_CHECK(cudaFree(d_v2_));

        /* lib vars */
        
        CUBLAS_CHECK(cublasDestroy(blasHandle_));
        CUSOLVER_CHECK(cusolverDnDestroySyevjInfo(syevj_params_));
        CUSOLVER_CHECK(cusolverDnDestroy(solverHandle_));

        printf("Pre-NPRO\n");        
        timer_GPUr_.print_clock("np_omega_init");
        timer_GPUr_.print_clock("np_vel_init");
        timer_GPUr_.print_clock("np_get_ener");
        timer_GPUr_.print_clock("np_vel_rot");
        timer_GPUr_.print_clock("np_counting");
        // timer_GPUr_.print_clock("np_memcpy");
        // timer_CPUr_.print_clock("np_summ");
        timer_GPUr_.print_clock("np_summ");
        printf("\n");

        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void cuda_elphwann_restart_( int *_nbnd_irrel,
                                     double *_ChemPot,
                                     double *_el_velocity,
                                     double *_ph_velocity,
                                     double *_Gamma,
                                     int *_indscatt
                                     )
    {
        
        ChemPot_ = *_ChemPot;
        nbnd_irrel_ = *_nbnd_irrel;
        // gpu_id_ = *_gpu_id;
        double *el_velocity_tmp = (double *)malloc(3 * nbands_ * Nlist_K_ * sizeof(double));
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < Nlist_K_; j++)
            {
                for (int k = 0; k < nbands_; k++)
                {
                    el_velocity_tmp[i + j * 3 + k * 3 * Nlist_K_] = el_velocity_[i + k * 3 + j * 3 * nbands_];
                }
            }
        }

        free(el_velocity_);
        el_velocity_ = el_velocity_tmp;

        double *ph_velocity_tmp = (double *)malloc(3 * nmodes_ * Nlist_Q_ * sizeof(double));
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < Nlist_Q_; j++)
            {
                for (int k = 0; k < nmodes_; k++)
                {
                    ph_velocity_tmp[i + j * 3 + k * 3 * Nlist_Q_] = ph_velocity_[i + k * 3 + j * 3 * nmodes_];
                }
            }
        }

        free(ph_velocity_);
        ph_velocity_ = ph_velocity_tmp;
        if(convergence_){
            Gamma_ = _Gamma;
            indscatt_ = _indscatt;
        }

        int l_d_chf=max(nmodes_*nmodes_*batch_size_, nbndsub_ * nbndsub_ * batch_size_k_);
        
        CUDA_CHECK(cudaMalloc((void **)&d_valid_list_,  NPTK_Q_ * sizeof(int) ));
        // CUDA_CHECK(cudaMalloc((void **)&d_chf_, l_d_chf*sizeof(cuDoubleComplex)));

        CUDA_CHECK(cudaMalloc((void **)&d_Orthcar_ptr_, batch_size_q_ * sizeof(double *)));
        CUDA_CHECK(cudaMalloc((void **)&d_Orthcar_ptr_q_, batch_size_q_ * sizeof(double *)));
        CUDA_CHECK(cudaMalloc((void **)&d_e1_,  sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_e2_, batch_size_q_ * nmodes_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_e3_, batch_size_q_ * nbands_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_el_velocity_tmp_, 3 * nbands_ * batch_size_q_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_el_velocity_ptr_, batch_size_q_ * sizeof(double *)));
        CUDA_CHECK(cudaMalloc((void **)&d_el_velocity_tmp_ptr_, batch_size_q_ * sizeof(double *)));
        CUDA_CHECK(cudaMalloc((void **)&d_ph_velocity_tmp_, 3 * nmodes_ * batch_size_q_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_ph_velocity_ptr_, batch_size_q_ * sizeof(double *)));
        CUDA_CHECK(cudaMalloc((void **)&d_ph_velocity_tmp_ptr_, batch_size_q_ * sizeof(double *)));
        CUDA_CHECK(cudaMalloc((void **)&d_kpq_, batch_size_q_ * sizeof(int)));
        // CUDA_CHECK(cudaMalloc((void **)&d_ph_velocity_, 3 * nmodes_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_v1_, 3 * sizeof(double) ));
        CUDA_CHECK(cudaMalloc((void **)&d_v2_, nmodes_ * 3 * sizeof(double) ));

        CUDA_CHECK(cudaMalloc((void **)&d_rate_rta_, batch_size_ * sizeof(double) ));
        CUDA_CHECK(cudaMalloc((void **)&d_rate_mrta_, batch_size_ * sizeof(double) ));
        CUDA_CHECK(cudaMalloc((void **)&d_el_cond_, 9 * sizeof(double) ));
        CUDA_CHECK(cudaMalloc((void **)&d_th_cond_, 9 * sizeof(double) ));
        CUDA_CHECK(cudaMalloc((void **)&d_niterm_, 9 * sizeof(double) ));

        CUDA_CHECK(cudaMalloc((void **)&d_gamma_tmp_, batch_size_ * nbands_ * sizeof(double) ));
        CUDA_CHECK(cudaMalloc((void **)&d_gamma_, batch_size_ * nbands_* sizeof(double) ));
        CUDA_CHECK(cudaMalloc((void **)&d_idxgm_tmp_, batch_size_ * nbands_ * sizeof(int) ));
        CUDA_CHECK(cudaMalloc((void **)&d_idxgm_, batch_size_ * nbands_ * sizeof(int) ));

        CUBLAS_CHECK(cublasCreate(&blasHandle_));
        CUSOLVER_CHECK(cusolverDnCreate(&solverHandle_));

        CUSOLVER_CHECK(cusolverDnCreateSyevjInfo(&syevj_params_));
        CUSOLVER_CHECK(cusolverDnXsyevjSetTolerance(syevj_params_, solver_tol_));
        CUSOLVER_CHECK(cusolverDnXsyevjSetMaxSweeps(syevj_params_, solver_max_sweeps_));


        const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvectors.
        const cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
        int lwork_kk;
        int lwork_kq;
        int lwork_q;
        // int lwork_;
        CUSOLVER_CHECK(cusolverDnZheevjBatched_bufferSize(solverHandle_, jobz, uplo, nbndsub_, d_cufkq_, nbndsub_, d_etkq_, &lwork_kq, syevj_params_, batch_size_));
        CUSOLVER_CHECK(cusolverDnZheevjBatched_bufferSize(solverHandle_, jobz, uplo, nmodes_, d_cuf_, nmodes_, d_w2_, &lwork_q, syevj_params_, batch_size_));
        CUSOLVER_CHECK(cusolverDnZheevjBatched_bufferSize(solverHandle_, jobz, uplo, nbndsub_, d_cufkk_, nbndsub_, d_etkk_, &lwork_kk, syevj_params_, batch_size_k_));
        lwork_ = max( max( lwork_kk, lwork_kq), lwork_q );
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work_), sizeof(cuDoubleComplex) * lwork_));

        // block_ = {BLOCK_SIZE,BLOCK_SIZE};
        // grid_ = {( nmodes_ * nmodes_ + block_.x - 1) / block_.x,( nrr_q_ + block_.y - 1) / block_.y};
        // fc_massfac<<<grid_,block_>>>(d_rdw_, d_amass_, d_ityp_,nmodes_,nrr_q_);

        if (lpolar_) {
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

        // System Info
        size_t free_mem, total_mem;
        cudaMemGetInfo( &free_mem, &total_mem);
        printf("Info: Total GPU memory is %lu MBs\n", total_mem/1024/1024);
        // printf("Free memory is %lu MBs\n", free_mem/1024/1024);
        
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    void cuda_elphwann_wannier_destroy_()
    {
        CUDA_CHECK(cudaDeviceSynchronize());

        printf("Eph-interp-e\n");        
        timer_GPUr_.print_clock("Ham1_prep");
        timer_GPUr_.print_clock("Ham1_diag");
        timer_GPUr_.print_clock("GPU_ephe_cfac");
        timer_GPUr_.print_clock("GPU_ephe_trans");
        timer_GPUr_.print_clock("GPU_ephe_core");
        printf("\n");
        timer_GPUr_.print_clock("Valid-Q");
        timer_GPUr_.print_clock("Valid-Q-ener");
        timer_GPUr_.print_clock("Valid-Q-vel");
        timer_GPUr_.print_clock("Valid-Q-check");
        timer_GPUr_.print_clock("Valid-Q-count");
        
        printf("Eph-interp-p\n");
        timer_GPUr_.print_clock("GPU_ham2_prep");
        timer_GPUr_.print_clock("GPU_ham2_prep_cfac");
        timer_GPUr_.print_clock("GPU_ham2_prep_trans");
        timer_GPUr_.print_clock("GPU_ham2_prep_herm");
        timer_GPUr_.print_clock("GPU_ham2_diag");
        timer_GPUr_.print_clock("GPU_ephq_trans");
        timer_GPUr_.print_clock("GPU_ukq_trans");
        timer_GPUr_.print_clock("GPU_dyn_prep");
        timer_GPUr_.print_clock("GPU_dyn_diag");
        timer_GPUr_.print_clock("GPU_uq_trans");

        if (lpolar_) {
            printf("Eph-interp-p\n");
            timer_GPUr_.print_clock("Polar-Umn");
            timer_GPUr_.print_clock("Polar-Trans_Cart");
            timer_GPUr_.print_clock("Polar-Eph_LR");
            timer_GPUr_.print_clock("Polar-Eph_Sum");
        }

        CUDA_CHECK(cudaFree(d_xkk_));
        CUDA_CHECK(cudaFree(d_xxq_));
        CUDA_CHECK(cudaFree(d_xxq_cart_));
        CUDA_CHECK(cudaFree(d_xkq_));

        CUDA_CHECK(cudaFree(d_eptmp_));

        CUDA_CHECK(cudaFree(d_wfqq_));
        CUDA_CHECK(cudaFree(d_eig_));
        CUDA_CHECK(cudaFree(d_amass_));
        CUDA_CHECK(cudaFree(d_tau_));
        CUDA_CHECK(cudaFree(d_irvec_r_));
        CUDA_CHECK(cudaFree(d_irvec_));
        CUDA_CHECK(cudaFree(d_ndegen_k_));
        CUDA_CHECK(cudaFree(d_ndegen_q_));

        // CUDA_CHECK(cudaFree(d_Orthcar_));
        CUDA_CHECK(cudaFree(d_Orthcar_ptr_));
        CUDA_CHECK(cudaFree(d_epmatwp_));
        CUDA_CHECK(cudaFree(d_chw_));
        CUDA_CHECK(cudaFree(d_rdw_));

        CUDA_CHECK(cudaFree(d_etkk_));
        CUDA_CHECK(cudaFree(d_w2_));
        CUDA_CHECK(cudaFree(d_etkq_));

        CUDA_CHECK(cudaFree(d_cufkk_));
        CUDA_CHECK(cudaFree(d_cuf_));
        CUDA_CHECK(cudaFree(d_cufkq_));

        CUDA_CHECK(cudaFree(d_epmatf_));
        CUDA_CHECK(cudaFree(d_epmatfl_));
        CUDA_CHECK(cudaFree(d_umn_));
        CUDA_CHECK(cudaFree(d_epwef_));

        // CUDA_CHECK(cudaFree(d_el_energy_));
        CUDA_CHECK(cudaFree(d_ph_energy_));
        // CUDA_CHECK(cudaFree(d_el_velocity_));
        CUDA_CHECK(cudaFree(d_ph_velocity_));
        CUDA_CHECK(cudaFree(d_el_velocity_tmp_));
        CUDA_CHECK(cudaFree(d_ph_velocity_tmp_));
        CUDA_CHECK(cudaFree(d_el_velocity_ptr_));
        CUDA_CHECK(cudaFree(d_el_velocity_tmp_ptr_));
        CUDA_CHECK(cudaFree(d_ph_velocity_ptr_));
        CUDA_CHECK(cudaFree(d_ph_velocity_tmp_ptr_));

        // temporary variables 
        CUDA_CHECK(cudaFree(d_cfac_));

        CUDA_CHECK(cudaFree(solverDevInfo_));

        CUDA_CHECK(cudaFree(d_kpq_));
        CUDA_CHECK(cudaFree(d_NQgrid_));
        CUDA_CHECK(cudaFree(d_NKgrid_));
        // CUDA_CHECK(cudaFree(d_Eqindex_K_));
        CUDA_CHECK(cudaFree(d_Eqindex_Q_));
        CUDA_CHECK(cudaFree(d_valid_i_));

        CUDA_CHECK(cudaFree(d_e1_));
        CUDA_CHECK(cudaFree(d_e2_));
        CUDA_CHECK(cudaFree(d_e3_));

        // CUDA_CHECK(cudaFree(d_StateInterest_ ));
        CUDA_CHECK(cudaFree(d_List_K_ ));
        CUDA_CHECK(cudaFree(d_v2_ ));
        CUDA_CHECK(cudaFree(d_v1_));

        CUDA_CHECK(cudaFree(d_rate_rta_));
        CUDA_CHECK(cudaFree(d_rate_mrta_));
        
        CUDA_CHECK(cudaDeviceSynchronize());
        CUBLAS_CHECK(cublasDestroy(blasHandle_));
        CUSOLVER_CHECK(cusolverDnDestroySyevjInfo(syevj_params_));
        CUSOLVER_CHECK(cusolverDnDestroy(solverHandle_));
        CUDA_CHECK(cudaFree(d_work_));
    }

    void nprocesses_cuda_(int *N_plus, long int *naccum, int *PhononInterest)
    {
        int nstates = Nstates_;
        int nbnd = nbands_;
        int nmodes = nmodes_;
        int ismear = ismear_ecp_;
        double degauss = degauss_;

        double *d_el_energy = d_el_energy_;
        double *d_el_velocity = d_el_velocity_;
        double **d_el_velocity_ptr = d_el_velocity_ptr_;
        double *d_el_velocity_tmp = d_el_velocity_tmp_;
        double **d_el_velocity_tmp_ptr = d_el_velocity_tmp_ptr_;
        double *d_rlattvec = d_rlattvec_;
        double *d_Orthcar = d_Orthcar_;
        double **d_Orthcar_ptr = d_Orthcar_ptr_;

        double *d_ph_velocity = d_ph_velocity_;

        int *d_valid_plus_i = d_valid_plus_i_;
        int *d_PhononInterest = d_PhononInterest_;

        int *d_kpq = d_kpq_;

        double *d_e2 = d_e2_,
               *d_e3 = d_e3_;

        double e1;

        int *List_K = List_;

        double *d_v2 = d_ph_velocity_tmp_;
        block_ = BLOCK_SIZE*BLOCK_SIZE;
        grid_ = {(nmodes*NPTK_Q_ + block_.x - 1) / block_.x, 1};

        // printf("phonon energy...\n");

        CUDA_CHECK(cudaMemset(d_PhononInterest, 0, NPTK_Q_ * sizeof(int)));

        timer_GPUr_.start_clock("np_omega_init");
        omega_init_npro<<<grid_, block_>>> ( NPTK_Q_,  d_Eqindex_Q_,  d_e2, radps2ev,  d_ph_energy_,  nmodes,  Nlist_Q_);
        timer_GPUr_.stop_clock("np_omega_init");

        // printf("phonon velocities...\n");
        timer_GPUr_.start_clock("np_vel_init");
        if ( ismear == 0 ) {
            ph_vel_init_npro<<<grid_, block_>>>(NPTK_Q_, d_ph_velocity, d_ph_velocity_ptr_, d_ph_velocity_tmp_, d_ph_velocity_tmp_ptr_,
                                    d_Orthcar, d_Orthcar_ptr, d_Eqindex_Q_, d_e2, ph_energy_, nmodes, Nlist_Q_);

            CUBLAS_CHECK(cublasDgemmBatched(blasHandle_, char_to_cublas_trans('n'), char_to_cublas_trans('n'), 3, nmodes, 3, &one_,
                                        d_Orthcar_ptr, 3, d_ph_velocity_ptr_, 3, &zero_, d_ph_velocity_tmp_ptr_, 3, NPTK_Q_));
        } 
        timer_GPUr_.stop_clock("np_vel_init");

        for (int istate=0;istate<nstates;istate++)
        {

            int k = List_K[StateInterest_[istate*2+1]-1];
            int ibnd = StateInterest_[istate*2];

            e1 = el_energy_[ IDX2F(Eqindex_K_[IDX2F(k, 1, NPTK_K_)], ibnd, Nlist_K_) ];

            block_ = { BLOCK_SIZE*BLOCK_SIZE };
            grid_ = {(nbnd * NPTK_Q_ + block_.x - 1) / block_.x};
            timer_GPUr_.start_clock("np_get_ener");
            get_kpq_ene_npro<<< grid_, block_ >>>(NPTK_Q_, k-1, d_Eqindex_K_, d_kpq, d_NKgrid_, d_NQgrid_,
                                       d_e3, d_el_energy, nbnd, Nlist_K_);
            timer_GPUr_.stop_clock("np_get_ener");

            timer_GPUr_.start_clock("np_vel_rot");
            if ( ismear == 0 ) {
                vel_rotation_ptr_npro<<< grid_, block_ >>>(d_kpq, d_Orthcar_ptr, d_el_velocity_ptr, d_el_velocity_tmp_ptr,
                                           d_Orthcar, d_el_velocity, d_el_velocity_tmp, d_Eqindex_K_, NPTK_K_, nbnd, NPTK_Q_);

                CUBLAS_CHECK(cublasDgemmBatched(blasHandle_, char_to_cublas_trans('n'), char_to_cublas_trans('n'), 3, nbnd, 3, &one_,
                                            d_Orthcar_ptr, 3, d_el_velocity_ptr, 3, &zero_, d_el_velocity_tmp_ptr, 3, NPTK_Q_));
            }
            timer_GPUr_.stop_clock("np_vel_rot");

            block_ = { BLOCK_SIZE*BLOCK_SIZE };
            grid_ = { ( NPTK_Q_ + block_.x - 1 ) / block_.x};

            timer_GPUr_.start_clock("np_counting");
            count_npro<<<grid_, block_>>>( d_v2, d_el_velocity_tmp, e1, d_e2, d_e3, d_rlattvec,
                                  d_NQgrid_, nmodes, nbnd, radps2ev, scalebroad_, ismear, degauss, delta_mult_, NPTK_Q_, d_valid_plus_i, ph_cut_);
            timer_GPUr_.stop_clock("np_counting");

            timer_GPUr_.start_clock("np_summ");
            N_plus[istate] = reduce(device, d_valid_plus_i,  d_valid_plus_i+NPTK_Q_ ); // last is not included
            // printf(" is: %d, np: %d \n", istate, N_plus[istate]);

            block_ = BLOCK_SIZE*BLOCK_SIZE ;
            grid_ =  ( NPTK_Q_ + block_.x - 1 ) / block_.x ;

	        countvalidph_npro<<< grid_, block_ >>>(d_valid_plus_i,d_PhononInterest,NPTK_Q_);
            timer_GPUr_.stop_clock("np_summ");

            naccum[istate+1] = naccum[istate] + N_plus[istate];
            // timer_CPUr_.stop_clock("np_summ");
        }

        // memcpy
        CUDA_CHECK(cudaMemcpy(PhononInterest, d_PhononInterest, NPTK_Q_ * sizeof(int), cudaMemcpyDeviceToHost));

        npro_ = naccum[nstates];
        // for ( int i=0; i<NPTK_Q_; i++ )
        // {
        //     if(PhononInterest_[i] != PhononInterest[i]){
        //         printf(" ERROR : npro counting ph %d %d %d \n", i, PhononInterest_[i], PhononInterest[i]);
        //         exit(-1);
        //     }
        // }
    }

    void cuda_hamwan2bloch_( int *_is, int *_batch_id )
    {
        int batchSize_full = batch_size_k_, 
            batch_id = *_batch_id,
            is_offset = *_is,
            ns_done = batch_id  * batchSize_full,
            nbnd = nbndsub_,
            nrr = nrr_k_,
            batchSize;
        int *d_StateInterest = d_StateInterest_;
        int *d_List = d_List_K_;
        double* d_irvec = d_irvec_r_;


        if (ns_done + batchSize_full > Nstates_)
        {
            batchSize = Nstates_ - ns_done;
        }
        else {
            batchSize = batchSize_full;
        }

        // cuDoubleComplex *d_chf = d_chf_;
        cuDoubleComplex *d_cfac =  d_cfac_;

        /*
         *  zgemv cublas routine
         */
        cuDoubleComplex *d_chw = d_chw_;
        cublasHandle_t blasHandle = blasHandle_;

        timer_GPUr_.start_clock("Ham1_prep");

        block_ = BLOCK_SIZE*BLOCK_SIZE; // dim3 variable holds 3 dimensions
        grid_ =(batchSize + block_.x - 1) / block_.x;

        istoxxk_batch<<< grid_, block_ >>>( batchSize, d_StateInterest, d_List, d_xkk_, d_NKgrid_, is_offset);

        block_ = {BLOCK_SIZE,BLOCK_SIZE};
        grid_ = {(nrr + block_.x - 1) / block_.x, (batchSize + block_.y - 1) / block_.y};
        cfac_batched<<<grid_, block_>>>(nrr, d_irvec, d_ndegen_k_, d_xkk_, (ComplexD *)d_cfac, batchSize);

        /* hermitianize */
        cublasOperation_t transa=char_to_cublas_trans('n');
        cublasOperation_t transb=char_to_cublas_trans('n');

        cuDoubleComplex *d_champ = d_cufkk_;

        CUBLAS_CHECK(cublasZgemm(blasHandle, transa, transb, nbnd * nbnd, batchSize, nrr, &conee_, d_chw, nbnd * nbnd, d_cfac, nrr, &czeroo_, d_champ, nbnd * nbnd));

        block_={BLOCK_SIZE, BLOCK_SIZE, 1}; // dim3 variable holds 3 dimensions
        grid_={(nbnd + block_.x - 1) / block_.x,(nbnd + block_.y - 1) / block_.y,(batchSize + block_.z - 1) / block_.z};
        hermitianize_matrix_batched<<<grid_, block_>>>(nbnd, batchSize, d_champ);
        // output_u<<<1,1>>>('0', nbnd, d_champ);
        timer_GPUr_.stop_clock("Ham1_prep");

        /*
         *   zhpevx cusolver routine
         */
        timer_GPUr_.start_clock("Ham1_diag");
        
        cusolverDnHandle_t solverHandle = solverHandle_;
        syevjInfo_t syevj_params = syevj_params_;
        cuDoubleComplex *d_work = d_work_;
        int *devInfo = solverDevInfo_, lwork = 0;
        double *d_ek=d_eig_;


        const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvectors.
        const cublasFillMode_t uplo = char_to_cublas_fillmode('u');

        CUSOLVER_CHECK(cusolverDnZheevjBatched_bufferSize(solverHandle, jobz, uplo, nbnd, d_champ, nbnd, d_ek, &lwork, syevj_params, batchSize));

        CUSOLVER_CHECK(cusolverDnZheevjBatched(solverHandle, jobz, uplo, nbnd, d_champ, nbnd, d_ek, d_work, lwork, devInfo, syevj_params, batchSize));

        timer_GPUr_.stop_clock("Ham1_diag");

    }

    void cuda_ephwan2bloche_(int *_is)
    {
        // double *xk=_xk;
        is_ = *_is;
        int is=*_is;        
        int nbnd = nbndsub_, 
            nrr_k = nrr_k_, 
            nmodes = nmodes_, 
            nrr_q = nrr_q_;

        // int *d_StateInterest=d_StateInterest_;

        cuDoubleComplex *d_epmatwp = d_epmatwp_,
                        *d_eptmp = d_eptmp_;

        cuDoubleComplex cone = {1.0, 0.0}, czero = {0.0, 0.0};
        cuDoubleComplex *d_cfac = d_cfac_;
        cuDoubleComplex *d_cufkk = d_cufkk_;
        // cuDoubleComplex **d_cufkk_ptr = d_cufkk_ptr_, **d_cufkq_ptr = d_cufkq_ptr_, **d_eptmp_ptr = d_eptmp_ptr_, **d_epmatf_ptr = d_epmatf_ptr_;

        double *d_irvec = d_irvec_r_;
        int *d_ndegen = d_ndegen_k_;
        double *d_xkk = d_xkk_;

        cublasHandle_t blasHandle = blasHandle_;

        // double* d_e1 = d_e1_;

        int* StateInterest = StateInterest_;
        // double* el_energy = el_energy_;


        int ibnd = StateInterest[ 2*is ] - 1;
        // int k = StateInterest[ 2*is + 1 ] - 1;
        // int Nlist_K = Nlist_K_;

        int batchSize = batch_size_k_;

        int is_inbatch = is - is/batchSize*batchSize;

        timer_GPUr_.start_clock("GPU_ephe_cfac");

        block_ = BLOCK_SIZE;
        grid_ = (nrr_k + BLOCK_SIZE - 1) / BLOCK_SIZE;

        cfac_batched<<<grid_, block_>>>(nrr_k, d_irvec, d_ndegen, d_xkk+is_inbatch*3, (ComplexD *)d_cfac, 1);
        CUDA_CHECK(cudaGetLastError());

        timer_GPUr_.stop_clock("GPU_ephe_cfac");

        /***
         *** epmatwp[nbnd(k+q) * nmodes * nrr_q * nbnd(k), nrr_k] * cfac[nrr_k]
         *** = eptmp[nbnd(k+q) * nmodes * nrr_q * nbnd(k)]
         ***/
        
        int m = nbnd * nmodes * nrr_q * nbnd;
        int n = nrr_k;
        int l;

        timer_GPUr_.start_clock("GPU_ephe_trans");

        CUBLAS_CHECK(cublasZgemv(blasHandle, char_to_cublas_trans('n'), m, n, &cone, d_epmatwp, m, d_cfac, 1, &czero, d_eptmp, 1));

        // output_d_eptmp<<<1,1>>>('-',1,d_epmatwp);
        // output_real_image<<<1,1>>>('c',d_cfac);
#ifdef DEBUG
        output_d_eptmp<<<1,1>>>('0',1,d_eptmp);
#endif
        timer_GPUr_.stop_clock("GPU_ephe_trans");

        /*
         *   eptmp(nmodes, nrr_q, nb, nb) * cufkk(nb, 1) = epmatwef(nmodes, nrr_q, nb, 1) 
         */

        cuDoubleComplex *d_epwef = d_epwef_;
        m = nbnd * nmodes * nrr_q;
        l = nbnd;
        n = 1;

        timer_GPUr_.start_clock("GPU_ephe_core");

        cuDoubleComplex *d_cuf_k;
        d_cuf_k = d_cufkk+is_inbatch*nbnd*nbnd+(nbnd_irrel_+ibnd)*nbnd;
        CUBLAS_CHECK(cublasZgemm(blasHandle, char_to_cublas_trans('n'), char_to_cublas_trans('n'), m, n, l, &cone, d_eptmp, m, d_cuf_k, l, &czero, d_epwef, m));
        CUDA_CHECK(cudaGetLastError());

        // output_u<<<1,1>>>('k', nbnd, d_cufkk);
#ifdef DEBUG
        output_d_eptmp<<<1,1>>>('1',1,d_epwef);
#endif

        timer_GPUr_.stop_clock("GPU_ephe_core");

    }

    void check_valid_q_(int *valid_list, int *_is, int *_nvalid) 
    {
        int batchSize_full = batch_size_q_, 
            batchSize;
        int nbnd = nbands_;
        int nmodes = nmodes_;
        int is = *_is;
        int ismear = ismear_ecp_;
        double degauss = degauss_;

        int* StateInterest = StateInterest_;

        double *el_energy = el_energy_;
        double *el_velocity = el_velocity_;
        double *d_el_energy = d_el_energy_;
        double *d_ph_energy = d_ph_energy_;
        double *d_el_velocity = d_el_velocity_;
        double *d_ph_velocity = d_ph_velocity_;
        double **d_el_velocity_ptr = d_el_velocity_ptr_;
        double **d_ph_velocity_ptr = d_ph_velocity_ptr_;
        double *d_el_velocity_tmp = d_el_velocity_tmp_;
        double *d_ph_velocity_tmp=d_ph_velocity_tmp_;
        double **d_el_velocity_tmp_ptr = d_el_velocity_tmp_ptr_;
        double **d_ph_velocity_tmp_ptr = d_ph_velocity_tmp_ptr_;
        double *d_rlattvec = d_rlattvec_;
        double *d_Orthcar = d_Orthcar_,
               **d_Orthcar_ptr = d_Orthcar_ptr_,
               **d_Orthcar_ptr_q = d_Orthcar_ptr_q_;

        int* d_Eqindex_K=d_Eqindex_K_;
        int* d_Eqindex_Q=d_Eqindex_Q_;
        int NPTK_K=NPTK_K_;
        int NPTK_Q=NPTK_Q_;


        int *d_valid_i = d_valid_i_;
        int *d_kpq = d_kpq_;

        double *d_e1 = d_e1_,
               *d_e2 = d_e2_,
               *d_e3 = d_e3_;

        *_nvalid = 0;

        int q_offset;

        int irr_k = StateInterest[2*is+1]-1;
        int k = List_[irr_k]-1; 

        int ibnd = StateInterest[2*is]-1;
        int Nlist_K = Nlist_K_;
        double *e1 = &el_energy[ irr_k + ibnd*Nlist_K ];
        double *v1 = &el_velocity[ irr_k*3 + ibnd*Nlist_K*3 ];
      //  printf("%d valid state : %d %d, band : %d ......\n", is,irr_k,k,ibnd);

        timer_GPUr_.start_clock("Valid-Q");

        CUDA_CHECK(cudaMemcpy(d_e1, e1, sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v1_, v1, 3*sizeof(double), cudaMemcpyHostToDevice));

        // printf("%d valid ......\n", k);

        int reduce_offset=0;

        for (int ibatch = 0; ibatch < (NPTK_Q_-1)/batchSize_full+1; ibatch++)
        {
            
            if ( ( ibatch + 1 ) * batchSize_full > NPTK_Q_)
            {
                batchSize = NPTK_Q_ - ibatch * batchSize_full;
            }
            else {
                batchSize = batchSize_full;
            }

            q_offset = ibatch *  batchSize_full;
            
            timer_GPUr_.start_clock("Valid-Q-ener");
            block_ = { BLOCK_SIZE*BLOCK_SIZE };
            grid_ = { (batchSize + block_.x - 1) / block_.x };
            get_kpq_ene<<< grid_, block_ >>>(batchSize, d_Eqindex_K_, d_Eqindex_Q_, k, q_offset, d_kpq, d_NKgrid_, d_NQgrid_,
                                           d_e2, d_e3, d_el_energy, d_ph_energy, radps2ev, nbnd, nmodes, Nlist_K_, Nlist_Q_);
            timer_GPUr_.stop_clock("Valid-Q-ener");                        
            CUDA_CHECK(cudaGetLastError());

            timer_GPUr_.start_clock("Valid-Q-vel");

            if( ismear == 0 ){

                // el and ph velocities rorate matrices pointer

                set_vel_rotation_ptr<<< grid_, block_ >>>( d_kpq, q_offset, d_Orthcar_ptr, d_el_velocity_ptr, d_el_velocity_tmp_ptr, d_Orthcar, d_el_velocity, d_el_velocity_tmp, 
                                                                             d_Orthcar_ptr_q, d_ph_velocity_ptr, d_ph_velocity_tmp_ptr, d_ph_velocity, d_ph_velocity_tmp, 
                                                                             d_Eqindex_K,  d_Eqindex_Q,  NPTK_K,  NPTK_Q,  nbnd,  nmodes,  batchSize);

                CUBLAS_CHECK(cublasDgemmBatched(blasHandle_, char_to_cublas_trans('n'), char_to_cublas_trans('n'), 3, nbnd, 3, &one_, d_Orthcar_ptr, 3, d_el_velocity_ptr, 3, &zero_, d_el_velocity_tmp_ptr, 3, batchSize));
                CUBLAS_CHECK(cublasDgemmBatched(blasHandle_, char_to_cublas_trans('n'), char_to_cublas_trans('n'), 3, nmodes, 3, &one_, d_Orthcar_ptr_q, 3, d_ph_velocity_ptr, 3, &zero_, d_ph_velocity_tmp_ptr, 3, batchSize));

            }
            timer_GPUr_.stop_clock("Valid-Q-vel");
            CUDA_CHECK(cudaGetLastError());

            block_ = { BLOCK_SIZE, BLOCK_SIZE };
            grid_ = { ( batchSize + block_.x - 1 ) / block_.x, ( nmodes*nbnd + block_.y - 1 ) / block_.y };
            
            timer_GPUr_.start_clock("Valid-Q-check");

            CUDA_CHECK(cudaMemset(d_valid_i, 0, batchSize*sizeof(int)));

            if (ismear == 0) {
                check_validk_ags<<<grid_, block_>>>( d_ph_velocity_tmp, d_el_velocity_tmp, d_e1, d_e2, d_e3, d_rlattvec, 
                                        d_NQgrid_, nmodes, nbnd, radps2ev, scalebroad_, delta_mult_, ph_cut_, d_valid_i, q_offset, batchSize);
            } else {
                check_validk_cgs<<<grid_, block_>>>( d_e1, d_e2, d_e3, d_NQgrid_, nmodes, nbnd, degauss, delta_mult_, ph_cut_, d_valid_i, q_offset, batchSize);
            }

            block_ = BLOCK_SIZE*BLOCK_SIZE;
            grid_ = ( batchSize + block_.x - 1 ) / block_.x;

            count_validk<<<grid_, block_>>>(d_valid_i, q_offset, batchSize);

            CUDA_CHECK(cudaGetLastError());

            timer_GPUr_.stop_clock("Valid-Q-check");

            timer_GPUr_.start_clock("Valid-Q-count");

            int nvalid_thisbatch =
                thrust::remove_copy_if(device, d_valid_i, d_valid_i + batchSize,
                                        d_valid_list_ + reduce_offset,
                                        is_negative()) -
                (d_valid_list_ + reduce_offset);

            reduce_offset += nvalid_thisbatch;
            timer_GPUr_.stop_clock("Valid-Q-count");
        }

        *_nvalid = reduce_offset;
        nvalid_ = *_nvalid;
        CUDA_CHECK(cudaMemcpy(valid_list, d_valid_list_, *_nvalid * sizeof(int) , cudaMemcpyDeviceToHost));

        // printf("valid list[0] = %d\n", valid_list[0]);
        // free(valid);
        // int* d_valid_list = d_valid_list_;
        // printf("number of valid is %d \n", nvalid_);
        // CUDA_CHECK(cudaMemcpy(d_valid_list, valid_list, nvalid_ * sizeof(int) , cudaMemcpyHostToDevice));
        timer_GPUr_.stop_clock("Valid-Q");
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    /***************************************************************************
     * Inner Nstates loop
     **************************************************************************/

    void cuda_hamw2b_kq_(int *_batch_id, int *_k)
    {
        // double *xxq;
        int batchSize_full = batch_size_, 
            batch_id = *_batch_id,
            k = *_k,
            // istate = batch_id * batchSize_full,
            nbnd = nbndsub_,
            nmodes = nmodes_,
            nrr_k = nrr_k_,
            nrr_q = nrr_q_,
            nvalid = nvalid_,
            batchSize;
        double* d_xxq=d_xxq_;
        double* d_xkq=d_xkq_;

        int *d_valid_list = d_valid_list_;

        int q_offset = batch_id * batchSize_full;

        batchSize = q_offset + batchSize_full > nvalid?(nvalid - q_offset):batchSize_full;

        block_ =  BLOCK_SIZE;
        grid_ =  ( batchSize + block_.x - 1 ) / block_.x;

        iqtoxkxq_batch<<< grid_, block_ >>> ( batchSize, d_NKgrid_, d_NQgrid_, k, q_offset, d_valid_list, d_xxq, d_xkq);

        cuDoubleComplex *d_chw = d_chw_,
                        *d_cfac = d_cfac_;
        cublasHandle_t blasHandle = blasHandle_;

        double* d_irvec = d_irvec_r_;
        int* d_ndegen_k = d_ndegen_k_;


        timer_GPUr_.start_clock("GPU_ham2_prep");

        timer_GPUr_.start_clock("GPU_ham2_prep_cfac");
        block_ = {BLOCK_SIZE, BLOCK_SIZE};
        grid_ = { (nrr_k + block_.x - 1) / block_.y, (batchSize + block_.y - 1)/block_.y};
        cfac_batched<<< grid_, block_ >>>(nrr_k, d_irvec, d_ndegen_k, d_xkq, (ComplexD *)d_cfac, batchSize);
        timer_GPUr_.stop_clock("GPU_ham2_prep_cfac");

        // output_tmp<<<1,1>>>(batchSize, nrr_k, d_cfac);

        cuDoubleComplex cone = {1.0, 0.0}, czero = {0.0, 0.0};

        /*
         ***  transform to bloch 
         */

        cusolverDnHandle_t solverHandle = solverHandle_;
        syevjInfo_t syevj_params = syevj_params_;
        cuDoubleComplex *d_champ = nullptr, *d_work = d_work_;
        d_champ = d_cufkq_;
        double *d_ekq_tmp = d_etkq_;

        timer_GPUr_.start_clock("GPU_ham2_prep_trans");
        CUBLAS_CHECK(cublasZgemm(blasHandle, char_to_cublas_trans('n'), char_to_cublas_trans('n'), nbnd * nbnd, batchSize, nrr_k, &cone, d_chw, nbnd * nbnd, d_cfac, nrr_k, &czero, d_champ, nbnd * nbnd));
        timer_GPUr_.stop_clock("GPU_ham2_prep_trans");

        /*
         ***   diagonalization
         */

        /* hermitianize matrix */
        block_ = {BLOCK_SIZE, BLOCK_SIZE}; // dim3 variable holds 3 dimensions
        grid_ = {(nbnd*(nbnd+1) + block_.x - 1) / block_.x, (batchSize + block_.y - 1) / block_.y};

        timer_GPUr_.start_clock("GPU_ham2_prep_herm");
        hermitianize_matrix_batched<<<grid_, block_>>>(nbnd, batchSize, d_champ);
        timer_GPUr_.stop_clock("GPU_ham2_prep_herm");

        CUDA_CHECK(cudaDeviceSynchronize());

        timer_GPUr_.stop_clock("GPU_ham2_prep");

        // output_u<<<1,1>>>('1', nbnd, d_champ);
        // output_u<<<1,1>>>('2', nbnd, d_champ+nbnd*nbnd);

        const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvectors.
        const cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;

        int *devInfo = solverDevInfo_, lwork = 0;

        timer_GPUr_.start_clock("GPU_ham2_diag");
        CUSOLVER_CHECK(cusolverDnZheevjBatched_bufferSize(solverHandle, jobz, uplo, nbnd * nbnd, d_champ, nbnd, d_ekq_tmp, &lwork, syevj_params, batchSize));
        // double * test= new double[batchSize  * nmodes];
        // CUDA_CHECK(cudaMemcpy(test, d_epmatf_dout, sizeof(double), cudaMemcpyDeviceToHost));

        CUSOLVER_CHECK(cusolverDnZheevjBatched(solverHandle, jobz, uplo, nbnd, d_champ, nbnd, d_ekq_tmp, d_work, lwork, devInfo, syevj_params, batchSize));
        // output_u<<<1,1>>>('K', nbnd, d_champ);
        // output_real<<<1,1>>>(d_ekq_tmp);

        timer_GPUr_.stop_clock("GPU_ham2_diag");    

        /*
         ***  Description:
         ***  G(k,q) = \sum_{Rp} U_ph*U^\dagger_{k+q}*G(k,Rp)*exp(iq\cdot Rp)
         ***  transformed into
         ***  G(k,q) = U_ph*U^\dagger_{k+q}*\sum_{Rp} G(k,Rp)*exp(iq\cdot Rp)
         ***         = U_ph*U^\dagger_{k+q}*\tilde G(k,q)
         ***  where 
         ***  \tilde G(k,q) = \sum_{Rp} G(k,Rp)*exp(iq\cdot Rp)
         ***
         ***  Data Structure :
         ***  G(k,Rp) : d_epwef_(nbndsub, nmodes, nrr_q)
         ***  exp(iq\cdot Rp) : cfac_(nrr_q,batchsize)
         ***  \tilde G(k,q) : d_epmatf_(nbnd, nmodes, batchsize)
         ***  \dagger_{k+q}*\tilde G(k,q) : d_eptmp_(nbnd_eff, nmodes, batchsize)
         ***  G(k,q) : d_epmatf_(nbnd_eff, nmodes, batchsize), 1st offset = nbnd_eff
         ***  
         */

        // phase factor exp(iq\cdot Rp) transfer

        timer_GPUr_.start_clock("GPU_ephq_trans");

        block_ = {BLOCK_SIZE, BLOCK_SIZE};
        grid_ = {(nrr_q + block_.x - 1) / block_.x, (batchSize + block_.y - 1) / block_.y};

        cfac_batched<<<grid_,block_>>>(nrr_q, d_irvec, d_ndegen_q_, d_xxq, (ComplexD *)d_cfac, batchSize);

        cuDoubleComplex *d_epwef = d_epwef_,
                        *d_eptmp = d_eptmp_,
                        *d_epmatf = d_epmatf_;

        CUBLAS_CHECK(cublasZgemm(blasHandle, char_to_cublas_trans('n'), char_to_cublas_trans('n'), nbnd * nmodes, batchSize, nrr_q, &cone, d_epwef, nbnd * nmodes, d_cfac, nrr_q, &czero, d_epmatf, nbnd * nmodes));
#ifdef DEBUG
        output_d_eptmp<<<1,1>>>('2', 1, d_epmatf+nbnd * nmodes);
#endif
        // CUDA_CHECK(cudaDeviceSynchronize());
        timer_GPUr_.stop_clock("GPU_ephq_trans");

        int nbnd_eff = nbands_,
            nbnd_irrel = nbnd_irrel_;
        
        timer_GPUr_.start_clock("GPU_ukq_trans");
        CUBLAS_CHECK(cublasZgemmStridedBatched(blasHandle, char_to_cublas_trans('c'), char_to_cublas_trans('n'), nbnd_eff, nmodes, nbnd, &cone,d_champ + nbnd_irrel*nbnd, nbnd, nbnd*nbnd, d_epmatf, nbnd, nbnd * nmodes,  &czero, d_eptmp, nbnd_eff, nbnd_eff * nmodes, batchSize));
#ifdef DEBUG
        output_d_eptmp<<<1,1>>>('3',1,d_eptmp+nbnd_eff * nmodes);
#endif
        // CUDA_CHECK(cudaDeviceSynchronize());
        timer_GPUr_.stop_clock("GPU_ukq_trans");
    }

    __global__ void cuda_elphwann_wq_kernel(int nmodes, double *w, double *wfqq, int batchsize)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        int idy = blockDim.y * blockIdx.y + threadIdx.y;
        if (idx < nmodes && idy < batchsize)
        {
            wfqq[idx+idy*nmodes] = w[idx+idy*nmodes] > 1e-12 ? sqrt(abs(w[idx+idy*nmodes])) : -sqrt(abs(w[idx+idy*nmodes]));
            // printf(" wq and wq^2 :%d %d %g %g\n",idx,idy, wfqq[idx+idy*nmodes], w[idx+idy*nmodes] );
        }
    }

    __global__ void cuda_elphwann_renor_kernel(int nmodes, cuDoubleComplex *uf, double *amass, int *ityp, int batchsize)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        int idy = blockDim.y * blockIdx.y + threadIdx.y;

        if (idx < nmodes * nmodes && idy < batchsize)
        {

            /*
            *** int na = (idx%nmodes - 1) / 3; // fortran is na + 1
            *** int it = ityp[na] - 1;  // fortran is ityp + 1
            *** double di = sqrt(amass[it]);
            */

            double di = sqrt(amass[ityp[(idx%nmodes) / 3] - 1]);

            uf[idy * nmodes * nmodes + idx].y = uf[ idy * nmodes * nmodes + idx ].y / di; 
            uf[idy * nmodes * nmodes + idx].x = uf[ idy * nmodes * nmodes + idx ].x / di; 
            // printf("uf : %d %d %d %g %g\n", idy, idx%nmodes, idx/nmodes,
            //      uf[idy * nmodes * nmodes + idx].x, uf[idy * nmodes * nmodes + idx].y);
        }
    }

    __global__ void reduce_epmat(cuDoubleComplex *d_epmatf, cuDoubleComplex *d_eptmp, int batch_size, int nbnd_eff, int nbndsub, int nmodes)
    {
        int ibatch = blockIdx.x * blockDim.x + threadIdx.x;
        int idy = blockIdx.y * blockDim.y + threadIdx.y;
        int imode = idy/nbnd_eff;
        int ibnd = idy%nbnd_eff;
        if(ibatch<batch_size && idy<nmodes*nbnd_eff){
            d_epmatf[ibnd + ibatch * nbnd_eff + imode * batch_size * nbnd_eff] = 
                d_eptmp[ibnd + ibatch * nbndsub * nbndsub  + imode * batch_size * nbndsub*nbndsub];
        }
    }

    void cuda_dynwan2bloch_batched_(int *_batch_id, double *_eps)
    // void cuda_dynwan2bloch_batched_(int *_batch_id, double *_eps, ComplexD *_epmatf)
    // void cuda_dynwan2bloch_batched_(int *_batch_id, double *_eps, ComplexD *_uf, ComplexD *_epmatf, double *_wq)
    {
        // default set lifc=false
        int batchSize_full = batch_size_;
        int nmodes = nmodes_;
        int batchSize;
        int is = batchSize_full * *_batch_id;
        cuDoubleComplex *d_champ = d_cuf_;
        cuDoubleComplex *d_cfac =  d_cfac_;
        cuDoubleComplex *d_rdw =  d_rdw_;
        cuDoubleComplex *d_work = d_work_;
        double *d_W = d_eig_;
        int nrr_q = nrr_q_;

        cuDoubleComplex const cone = {1.0, 0.0}, czero = {0.0, 0.0};

        int nvalid = nvalid_;

        if (is + batchSize_full > nvalid)
        {
            batchSize = nvalid - is;
        }
        else{
            batchSize = batchSize_full;
        }

        timer_GPUr_.start_clock("GPU_dyn_prep");
        cublasHandle_t blasHandle=blasHandle_;

        CUBLAS_CHECK(cublasZgemm(blasHandle, char_to_cublas_trans('n'), char_to_cublas_trans('n'), nmodes * nmodes, batchSize, nrr_q, &conee_, d_rdw, nmodes * nmodes, d_cfac, nrr_q, &czeroo_, d_champ, nmodes * nmodes));

        if (lpolar_) {
            block_ = {BLOCK_SIZE2}; // dim3 variable holds 3 dimensions
            grid_= {(batchSize + block_.x - 1) / block_.x};
            int nat=nat_;
            double *d_zstar = d_zstar_;
            double e2 = 2.0; // e^2 in rydberg unit
            double omega = Volume_/ryd2nm/ryd2nm/ryd2nm;
            double fac0 = e2  * 4.0 * Pi /omega; // in the unit of rydberg 
            double  *tau = d_tau_,
                    *bg = d_bg_,
                    *xqc = d_xxq_cart_,
                    *epsil = d_epsil_,
                    *zeu = d_zstar,
                    gmax = gmax_;

            cuDoubleComplex *dyn=d_champ;

            cryst_to_cart_global<<<grid_, block_>>>(d_bg_, d_xxq_, d_xxq_cart_,batchSize);

            dynmat_polar<<<grid_, block_>>>(batchSize, nrx_[0], nrx_[1],nrx_[2], nmodes, nat, bg,
                                            tau, xqc, (ComplexD*)dyn, epsil, zeu, gmax, fac0);
        }

        // printf(" hermitianizing \n");
        block_={BLOCK_SIZE, BLOCK_SIZE}; // dim3 variable holds 3 dimensions
        grid_={(nmodes*(nmodes+1) + block_.x - 1) / block_.x,(batchSize + block_.y - 1) / block_.y};
        // hermitianize_matrix_batched<<<grid_, block_>>>(nmodes, batchSize, d_champ);
        double *amass = d_amass_;
        int *ityp = d_ityp_;
        dynmat_prep<<<grid_, block_>>>((ComplexD *)d_champ, amass, ityp, nmodes, batchSize);
        timer_GPUr_.stop_clock("GPU_dyn_prep");

        const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvectors.
        const cublasFillMode_t uplo = char_to_cublas_fillmode('u');
        
        // printf(" Diaging... \n");
        int lwork;
        timer_GPUr_.start_clock("GPU_dyn_diag");
        CUSOLVER_CHECK(
            cusolverDnZheevjBatched_bufferSize(solverHandle_, jobz, uplo, nmodes, d_champ, nmodes, d_W, &lwork, syevj_params_, batchSize));

        CUSOLVER_CHECK(cusolverDnZheevjBatched(solverHandle_, jobz, uplo, nmodes, d_champ, nmodes, d_W, d_work, lwork, solverDevInfo_, syevj_params_, batchSize));

        // output_u<<<1,1>>>('p', nmodes, d_champ+nmodes*nmodes);

        double *d_wfqq = d_wfqq_, *d_amass = d_amass_;
        int *d_ityp = d_ityp_;

        // printf(" decorating and cutting... \n");
        block_ = {BLOCK_SIZE, BLOCK_SIZE}; // dim3 variable holds 3 dimensions
        grid_ = {(nmodes + block_.x - 1) / block_.x,(batchSize + block_.y - 1) / block_.y};
        cuda_elphwann_wq_kernel<<< grid_, block_>>>(nmodes, d_W, d_wfqq, batchSize);

        block_={BLOCK_SIZE,BLOCK_SIZE}; // dim3 variable holds 3 dimensions
        grid_={(nmodes*nmodes + block_.x - 1) / block_.x, (batchSize + block_.y - 1) / block_.y};

        cuda_elphwann_renor_kernel<<<grid_, block_>>>(nmodes, d_champ, d_amass, d_ityp, batchSize);
        timer_GPUr_.stop_clock("GPU_dyn_diag");

        // output_u<<<1,1>>>('p', nmodes, d_champ+9*nmodes*nmodes);
        timer_GPUr_.start_clock("GPU_uq_trans");

        int nbnd_eff = nbands_;
        // int nbnd = nbndsub_;
        cuDoubleComplex* d_eptmp = d_eptmp_;
        cuDoubleComplex* d_epmatf = d_epmatf_;
        CUBLAS_CHECK(cublasZgemmStridedBatched(blasHandle, char_to_cublas_trans('n'), char_to_cublas_trans('n'), nbnd_eff, nmodes, nmodes, &cone, d_eptmp, nbnd_eff, nbnd_eff * nmodes, d_champ, nmodes, nmodes*nmodes, &czero, d_epmatf, nbnd_eff, nbnd_eff * nmodes, batchSize));
#ifdef DEBUG
        output_d_eptmp<<<1,1>>>('4',1,d_epmatf);
#endif
        CUDA_CHECK(cudaDeviceSynchronize());

        // reduce_epmat<<< grid_, block_ >>>( d_epmatf, d_eptmp, batchSize, nbnd_eff, nbnd, nmodes );

        // double eps = 1e-10;
        // double eps = ph_cut_;

        // block_ = {BLOCK_SIZE, BLOCK_SIZE}; // dim3 variable holds 3 dimensions
        // grid_= {(batchSize + block_.x - 1) / block_.x, (nmodes*nbnd + block_.y - 1) / block_.y};
        // trans_epmat<<< grid_, block_ >>>(d_epmatf_dout, d_epmatf, d_wfqq, eps, batchSize, nbnd_eff, nmodes, ryd2ev);
        // // trans_epmat_2<<< grid_, block_ >>>(d_epmatf_dout, d_epmatf, d_ph_energy_, d_Eqindex_Q_, Nlist_Q_, d_valid_list_,  radps2ev/ryd2ev,  eps,  batchSize, nbnd_eff,  nmodes);

        // output_d_out<<<1,1>>>(1,d_epmatf_dout+nbnd_eff*nmodes);
        // CUDA_CHECK(cudaMemcpy(_epmatf, d_epmatf_dout, nbnd_eff * nmodes * batchSize * sizeof(double), cudaMemcpyDeviceToHost));
        // CUDA_CHECK(cudaMemcpy(_epmatf, d_epmatf, nbnd_eff * nmodes * batchSize * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        timer_GPUr_.stop_clock("GPU_uq_trans");
    
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    /***************************************************************************
     * Frohlich Polar term
     **************************************************************************/

    __global__ void g2_longrange_3d(int batchsize, double eps, int nrx1, int nrx2, int nrx3, int nmodes, int nbnd, int nat, double alat, double *bg,
                                        double *tau, double *xqc, ComplexD *uq, ComplexD *epmat, ComplexD *umn, 
                                        double *epsil, double *zeu, ComplexD fac0, double ryd2ev){
        
        int idx = blockDim.x * blockIdx.x + threadIdx.x;

        double g1,g2,g3;

        double qeq;
        double tpi = 2.0*M_PI;

        ComplexD czero={0.0,0.0};
        ComplexD fac1, fac, phase;
        double zaq;
        // double xxq_cart[3];
        double gmax = 14.0;

        if( idx<batchsize ){
            for (int im = 0; im<nmodes*nbnd; im++){
                epmat[idx*nmodes*nbnd+im] = czero;
            }
        }

        if( idx<batchsize && (abs(xqc[idx*3+0])>eps||abs(xqc[idx*3+1])>eps||abs(xqc[idx*3+2])>eps))
        {
            for (int r1 = -nrx1; r1<=nrx1; r1++ ){
                for (int r2 = -nrx2; r2<=nrx2; r2++ ){
                // for (int r2 = -0; r2<=0; r2++ ){
                    // for (int r3 = -0; r3<=0; r3++ ){
                    for (int r3 = -nrx3; r3<=nrx3; r3++ ){
                        // printf("old kernel index : %d %d %d %d \n", idx, r1, r2, r3);
                        g1 = bg[0]*r1 + bg[3]*r2 + bg[6] * r3 + xqc[idx*3];
                        g2 = bg[1]*r1 + bg[4]*r2 + bg[7] * r3 + xqc[idx*3+1];
                        g3 = bg[2]*r1 + bg[5]*r2 + bg[8] * r3 + xqc[idx*3+2];

                        qeq = (g1*(epsil[0]*g1+epsil[3]*g2+epsil[6]*g3 )+      
                            g2*(epsil[1]*g1+epsil[4]*g2+epsil[7]*g3 )+      
                            g3*(epsil[2]*g1+epsil[5]*g2+epsil[8]*g3 )); //*twopi/alat

                        // printf("old kernel qeq : %d %d %d %d %g \n", idx, r1, r2, r3, qeq);
                        if (qeq > 1.0e-14 && qeq/4.0 < gmax ) {
                            qeq = qeq*tpi/alat;
                            fac1 = {exp(-qeq/4.0)/qeq,0.0};
                            for (int iat = 0; iat<nat; iat++ ){
                                double arg = - tpi * ( g1*tau[0+iat*3] + g2*tau[1+iat*3] + g3*tau[2+iat*3]);
                                phase = {cos(arg),sin(arg)};
                                fac = fac1*phase;
                                for (int idir = 0; idir<3; idir++ ){
                                    zaq = g1*zeu[iat*9+3*idir]+g2*zeu[iat*9+3*idir+1]+g3*zeu[iat*9+3*idir+2];
                                    for (int im = 0; im<nmodes; im++){
                                        for (int ib = 0; ib<nbnd; ib++){
                                            epmat[idx*nmodes*nbnd+im*nbnd+ib] =  epmat[idx*nmodes*nbnd + im*nbnd + ib] + 
                                            zaq * fac * uq[ idx*nmodes*nmodes+im*nmodes + iat*3 + idir] * umn[idx*nbnd + ib];
                                            // printf("old term %d %d %d, %d %d %d %d %d, %g * %g * %g * %g += %g\n",
                                            //         idx, im, ib,r1,r2,r3,iat,idir,
                                            //         zaq,
                                            //         abs(fac),
                                            //         abs(uq[ idx*nmodes*nmodes+im*nmodes + iat*3 + idir]),
                                            //         abs(umn[idx*nbnd + ib]),
                                            //         abs(epmat[idx*nmodes*nbnd+im*nbnd+ib])
                                            // );
                                        }
                                    }                    
                                }
                            }
                        }
                    }
                }
            }

            for (int im = 0; im<nmodes; im++)
            {
                // for (int ib = 0; ib<nbnd*nbnd; ib++)
                for (int ib = 0; ib<nbnd; ib++)
                {
                    // epmat[idx*nmodes*nbnd*nbnd+im*nbnd*nbnd+ib] =  epmat[idx*nmodes*nbnd*nbnd + im*nbnd*nbnd + ib] * fac0 * ryd2ev;
                    epmat[idx*nmodes*nbnd+im*nbnd+ib] =  epmat[idx*nmodes*nbnd + im*nbnd + ib] * fac0;
                    // printf("kernel epmat : %d %d %d %g\n",idx, im, ib, abs(epmat[idx*nmodes*nbnd*nbnd+im*nbnd*nbnd+ib])/ryd2ev);
                    // printf("old kernel epmat : %d %d %d, %g\n", idx, im, ib, abs(epmat[idx*nmodes*nbnd+im*nbnd+ib]) );
                }
            }     
        }
    }

    /***************************************************************************
     * Frohlich Polar term
     **************************************************************************/

    __global__ void g2_longrange_3d_v2(int batchsize, double eps, int nrx1, int nrx2, int nrx3, int nmodes, int nbnd, int nat, double alat, double *bg,
                                        double *tau, double *xqc, ComplexD *uq, ComplexD *epmat, ComplexD *umn, 
                                        double *epsil, double *zeu, ComplexD fac0, double ryd2ev)
    {
        
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        int idy = blockDim.y * blockIdx.y + threadIdx.y;

        double g1,g2,g3;
        double qeq;
        double tpi = 2.0*M_PI;

        ComplexD czero={0.0,0.0};
        ComplexD fac1, fac, phase;
        double zaq;
        double gmax = 14.0;

        int im = idy/nbnd;
        int ib = idy%nbnd;
        // int im = idy%nmodes;
        // int ib = idy/nmodes;

        if( idx<batchsize && idy <nmodes*nbnd )
        {

            epmat[idx*nmodes*nbnd+idy] = czero;

            if ((abs(xqc[idx*3+0])>eps||abs(xqc[idx*3+1])>eps||abs(xqc[idx*3+2])>eps)){
                for (int r1 = -nrx1; r1<=nrx1; r1++ ){
                    for (int r2 = -nrx2; r2<=nrx2; r2++ ){
                        for (int r3 = -nrx3; r3<=nrx3; r3++ ){
                            g1 = bg[0]*r1 + bg[3]*r2 + bg[6] * r3 + xqc[idx*3];
                            g2 = bg[1]*r1 + bg[4]*r2 + bg[7] * r3 + xqc[idx*3+1];
                            g3 = bg[2]*r1 + bg[5]*r2 + bg[8] * r3 + xqc[idx*3+2];

                            qeq = (g1*(epsil[0]*g1+epsil[3]*g2+epsil[6]*g3 )+      
                                g2*(epsil[1]*g1+epsil[4]*g2+epsil[7]*g3 )+      
                                g3*(epsil[2]*g1+epsil[5]*g2+epsil[8]*g3 )); //*twopi/alat

                            // printf("new kernel qeq : %d %d %d %d %g \n", idx, r1, r2, r3, qeq);
                            if (qeq > 1.0e-14 && qeq/4.0 < gmax ) {
                                qeq = qeq*tpi/alat;
                                fac1 = {exp(-qeq/4.0)/qeq,0.0};
                                for (int iat = 0; iat<nat; iat++ ){
                                    double arg = - tpi * ( g1*tau[0+iat*3] + g2*tau[1+iat*3] + g3*tau[2+iat*3]);
                                    phase = {cos(arg),sin(arg)};
                                    fac = fac1*phase;
                                    for (int idir = 0; idir<3; idir++ ){
                                        zaq = g1*zeu[iat*9+3*idir]+g2*zeu[iat*9+3*idir+1]+g3*zeu[iat*9+3*idir+2];
                                        epmat[idx*nmodes*nbnd+idy] += 
                                            zaq * fac * uq[ idx*nmodes*nmodes+im*nmodes + iat*3 + idir] * umn[idx*nbnd + ib];
                                        // printf("new term %d %d %d, %d %d %d %d %d, %g * %g * %g * %g += %g\n",
                                        //     idx, im, ib, r1, r2, r3, iat, idir,
                                        //     zaq,
                                        //     abs(fac),
                                        //     abs(uq[ idx*nmodes*nmodes+im*nmodes + iat*3 + idir]),
                                        //     abs(umn[idx*nbnd + ib]),
                                        //     abs(epmat[idx*nmodes*nbnd+idy])
                                        // );
                                    }                    
                                }
                            }
                        }
                    }
                }
            }

            // epmat[idx*nmodes*nbnd+idy] =  epmat[idx*nmodes*nbnd + idy] * fac0;
            epmat[idx*nmodes*nbnd+idy] *= fac0;
            // printf("new kernel epmat : %d %d %d, %g\n", idx, im, ib, abs(epmat[idx*nmodes*nbnd+idy]) );
        }
    }

    __global__ void g2_longrange_sum(int batchsize, int nmodes, int nbnd_eff, int nbnd_irr, int ibnd, ComplexD *epmat, ComplexD *epmatl){

        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        int idy = blockDim.y * blockIdx.y + threadIdx.y;

        // int im = idy/nbnd;
        // int ib = idy%nbnd;

        if( idx<batchsize && idy<nbnd_eff*nmodes ){
            // for (int im = 0; im<nmodes; im++)
            // {
            //     for (int ib = 0; ib<nbnd_eff; ib++){
                    // int ibfull=ib + nbnd_irr + ibnd*nbnd;
                    // printf("kernel polar sum: %d %d %d ( %g %g ) ( %g %g )  \n", idx,im,ib,
                    // epmat[idx*nmodes*nbnd_eff+im*nbnd_eff+ib].real(), epmat[idx*nmodes*nbnd_eff+im*nbnd_eff+ib].imag(),
                    // epmatl[idx*nmodes*nbnd_eff+im*nbnd_eff+ib].real(), epmatl[idx*nmodes*nbnd_eff+im*nbnd_eff+ib].imag());

                    // epmat[idx*nmodes*nbnd_eff+im*nbnd_eff+ib] = epmat[idx*nmodes*nbnd_eff+im*nbnd_eff+ib] + epmatl[idx*nmodes*nbnd*nbnd+im*nbnd*nbnd+ibfull];
                    // epmat[idx*nmodes*nbnd_eff+im*nbnd_eff+ib] = epmat[idx*nmodes*nbnd_eff+im*nbnd_eff+ib] + epmatl[idx*nmodes*nbnd_eff+im*nbnd_eff+ib];
                    epmat[idx*nmodes*nbnd_eff+idy] += epmatl[idx*nmodes*nbnd_eff+idy];
                // }
            // }
        }
    }

    // void cuda_polar_(int *_batch_id, double *_eps, ComplexD *epmatl, ComplexD *epmat,
    //                  cuDoubleComplex *cufkk,cuDoubleComplex *cufkq, 
    //                  cuDoubleComplex *_uf, cuDoubleComplex *umn)
    void cuda_polar_(int *_batch_id, double *_eps)
    {

        cuDoubleComplex *d_cuf = d_cuf_;

        int nbnd = nbndsub_,
            nbnd_eff = nbands_,
            nbnd_irrel = nbnd_irrel_,
            nmodes = nmodes_;

        int batchSize_full = batch_size_, 
            batch_id = *_batch_id,
            nvalid = nvalid_,
            batchSize;

        int is = is_;

        int q_offset = batch_id * batchSize_full;

        batchSize = q_offset + batchSize_full > nvalid?(nvalid - q_offset):batchSize_full;

        cublasHandle_t blasHandle=blasHandle_;
        
        /*
        u_mn(k,q) = U(k)*U(k+q)^\dagger
        */
        
        cuDoubleComplex *d_cuf_k;
        int batchSize_k = batch_size_k_;
        int is_inbatch = is - is/batchSize_k*batchSize_k ;
        // d_cuf_k = d_cufkk+is_inbatch*nbnd*nbnd+(nbnd_irrel_+ibnd)*nbnd;
        cuDoubleComplex *d_cufkk = d_cufkk_;
        int ibnd_full = nbnd_irrel + StateInterest_[2*is]-1;
        d_cuf_k = d_cufkk+is_inbatch*nbnd*nbnd+ibnd_full*nbnd;
        cuDoubleComplex *d_cufkq = d_cufkq_+nbnd*nbnd_irrel;
        cuDoubleComplex *d_umn = d_umn_;

        timer_GPUr_.start_clock("Polar-Umn");
        // CUBLAS_CHECK(cublasZgemmStridedBatched(blasHandle, CUBLAS_OP_C, CUBLAS_OP_N, nbnd, nbnd, nbnd, &conee_, 
        //                                        d_cufkq, nbnd, nbnd * nbnd, d_cuf_k, nbnd, 0, &czeroo_, 
        //                                        d_umn, nbnd, nbnd * nbnd, batchSize));

        CUBLAS_CHECK(cublasZgemmStridedBatched(blasHandle, CUBLAS_OP_C, CUBLAS_OP_N, nbnd_eff, 1, nbnd, &conee_, 
                                               d_cufkq, nbnd, nbnd * nbnd, d_cuf_k, nbnd, 0, &czeroo_, 
                                               d_umn, nbnd_eff, nbnd_eff, batchSize));
        timer_GPUr_.stop_clock("Polar-Umn");
        CUDA_CHECK(cudaGetLastError());

        // CUDA_CHECK(cudaMemcpy( cufkk, d_cufkk+is_inbatch*nbnd*nbnd, nbnd*nbnd*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        // CUDA_CHECK(cudaMemcpy( cufkq, d_cufkq_, nbnd*nbnd*batchSize*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        // CUDA_CHECK(cudaMemcpy( _uf, d_cuf, nmodes*nmodes*batchSize*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        // CUDA_CHECK(cudaMemcpy( umn, d_umn, nbnd_eff*batchSize*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        /*
            G_l(k,q) = prefac * \sum prefac_kappa q.Z * uq * phase * qeq^{-1} *u_mn(k,q)
            
            $$ g_{mn\nu}^{\mathcal L}({\bf k},{\bf q) = i\frac{4\pi e^2}{\Omega} \sum_{\kappa}
            \left(\frac{\hbar}{2 {M_\kappa \omega_{{\bf q}\nu}}}\right)^{\!\!\frac{1}{2}}
                \sum_{{\bf G}\ne -{\bf q}} e^{-({\bf q}+{\bf G})^2/4\alpha}
                \frac{ ({\bf q}+{\bf G})\cdot{\bf Z}^*_\kappa \cdot {\bf e}_{\kappa\nu}({\bf q}) } 
                {({\bf q}+{\bf G})\cdot\bm\epsilon^\infty\!\cdot({\bf q}+{\bf G})}\,
                \left[ U_{{\bf k}+{\bf q}}\:U_{{\bf k}}^{\dagger} \right]_{mn} $$
        */

        int nat = nat_;
        double eps = *_eps;
        double *d_bg = d_bg_;
        double *d_tau = d_tau_;
        double *d_xq  = d_xxq_;
        double *d_xqc  = d_xxq_cart_;
        cuDoubleComplex *d_epmat = d_epmatf_; // in the unit of rydberg
        cuDoubleComplex *d_epmatl = d_epmatfl_; // in the unit of rydberg
        double *d_epsil = d_epsil_;
        double *d_zstar = d_zstar_;
        double e2 = 2.0; // e^2 in rydberg unit
        double omega = Volume_/ryd2nm/ryd2nm/ryd2nm,
               alat = celldm1_;

        ComplexD fac0 = ci_ * e2  * 4.0 * Pi /omega; // in the unit of rydberg 

        block_ = {BLOCK_SIZE2}; // dim3 variable holds 3 dimensions
        grid_= {(batchSize + block_.x - 1) / block_.x};
        /* d_xqc = d_xq * d_bg; */
        timer_GPUr_.start_clock("Polar-Trans_Cart");
        cryst_to_cart_global<<<grid_, block_>>>( d_bg, d_xq, d_xqc, batchSize);
        timer_GPUr_.stop_clock("Polar-Trans_Cart");
        // fflush(stdout);

        timer_GPUr_.start_clock("Polar-Eph_LR");
        // g2_longrange_3d<<<grid_, block_>>>( batchSize, eps, nrx_[0], nrx_[1],nrx_[2], 
        //                                                       nmodes, nbnd_eff, nat, alat, d_bg, d_tau, d_xqc, 
        //                                                       (ComplexD *)d_cuf, (ComplexD *)d_epmatl, (ComplexD *)d_umn,
        //                                                 d_epsil,  d_zstar, fac0, ryd2ev);
        // CUDA_CHECK(cudaGetLastError());
        // CUDA_CHECK(cudaDeviceSynchronize());

        block_={BLOCK_SIZE, BLOCK_SIZE/2};
        grid_= {(batchSize + block_.x - 1) / block_.x, (nbnd_eff*nmodes + block_.y - 1) / block_.y};
        g2_longrange_3d_v2<<<grid_, block_>>>( batchSize, eps, nrx_[0], nrx_[1],nrx_[2], 
                                                              nmodes, nbnd_eff, nat, alat, d_bg, d_tau, d_xqc, 
                                                              (ComplexD *)d_cuf, (ComplexD *)d_epmatl, (ComplexD *)d_umn,
                                                        d_epsil,  d_zstar, fac0, ryd2ev);
        timer_GPUr_.stop_clock("Polar-Eph_LR");

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        // exit(0);
        // CUDA_CHECK(cudaMemcpy( epmatl, d_epmatl, batchSize*nmodes*nbnd_eff*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

        // output_epmatl(batchSize, nmodes, nbnd, epmatl);

        timer_GPUr_.start_clock("Polar-Eph_Sum");
        // int ibnd_full = StateInterest_[2*is]-1 + nbnd_irrel;
        block_ = {BLOCK_SIZE,BLOCK_SIZE}; // dim3 variable holds 3 dimensions
        grid_= {(batchSize + block_.x - 1) / block_.x, (nbnd_eff*nmodes + block_.y - 1) / block_.y};
        g2_longrange_sum<<<grid_, block_>>>( batchSize, nmodes, nbnd_eff, nbnd_irrel, ibnd_full, (ComplexD *)d_epmat, (ComplexD *)d_epmatl);
        timer_GPUr_.stop_clock("Polar-Eph_Sum");
                                    
        // CUDA_CHECK(cudaMemcpy( epmat, d_epmat, batchSize*nmodes*nbnd_eff*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        // block_ = {BLOCK_SIZE*BLOCK_SIZE}; // dim3 variable holds 3 dimensions
        // grid_= {(batchSize + block_.x - 1) / block_.x};

        // output_d_epmatl<<<1, 1>>>(batchSize, nmodes, nbnd, (ComplexD *)d_epmatl);
        
    }

    /***************************************************************************
     * Scattering rates and Gamma
     **************************************************************************/

    // void cuda_g2_(int *_batch_id, cuDoubleComplex *_g2){
    void cuda_g2_(int *_batch_id){
        int nbnd_eff = nbands_,
            nbnd     = nbndsub_,
            nmodes   = nmodes_,
            batchSize_full = batch_size_;

        cuDoubleComplex *d_epmatf = d_epmatf_;

        int is = batchSize_full * *_batch_id;

        double *d_wfqq = d_wfqq_;

        int nvalid = nvalid_;

        int batchSize;

        if (is + batchSize_full > nvalid)
        {
            batchSize = nvalid - is;
        }
        else{
            batchSize = batchSize_full;
        }

        double eps = ph_cut_;

        block_ = {BLOCK_SIZE, BLOCK_SIZE}; // dim3 variable holds 3 dimensions
        grid_= {(batchSize + block_.x - 1) / block_.x, (nmodes*nbnd + block_.y - 1) / block_.y};
        trans_epmat<<< grid_, block_ >>>(d_epmatf_dout, d_epmatf, d_wfqq, eps, batchSize, nbnd_eff, nmodes, ryd2ev);
    }


    __global__ void cuda_indpro_glb( double *e1, double *v1, double *ph_ener, double *el_ener, int k, int *q, double *el_vel, double *ph_vel, double *g2,
                                    int nbnd, int nmodes, int Nlist_K, int Nlist_Q, int NPTK_K, int NPTK_Q, int batch_q,
                                    int *Ngrid_K, int *Ngrid_Q, int *Eqidx_K, int *Eqidx_Q, double *orth,
                                    int ismear_ecp, double degauss, double delta_mult,double ph_cut, double SB, double *G,
                                    double one_kbt, double chempot, double radps2ev, double ryd2ev,
                                    double *rate_rta,   double *rate_mrta, 
                                    bool iter, int *indofgm, double *gm, double gm_eps)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
 
        // double tpi = 2.0*M_PI;
        double sigma;
        double v[3];
        // double v1_m2;
        double v2[3];
        double v3[3];
        double e2, e3;
        double weight;
        // double e, wg;
        int idx_gm;
        bool l_valid;

        if(idx < batch_q){
            rate_rta[idx] = 0.0;
            rate_mrta[idx] = 0.0;
            // printf(" idx : %d, k+q : %d , q : %d  \n", idx, kpq[idx], q[idx] );
            int kpq = kplusq_kernel( k, q[idx], Ngrid_K, Ngrid_Q);
            int idx_kq = Eqidx_K[kpq] - 1;
            // int idx_qq = Eqidx_Q[q[idx]] - 1;
            int idx_qq = Eqidx_Q[q[idx]] - 1;
            // printf(" idx : %d, idx_kq : %d , idx_qq : %d ,ph_ener : %g \n", idx, idx_kq, idx_qq,ph_ener[idx_qq]*radps2ev);
            // printf(" idx : %d, idx_kq : %d , idx_qq : %d \n", idx, kpq, q[idx]);
            // printf(" ???? : %d  \n", idx );
            for(int ib = 0; ib < nbnd; ib++){
                if (iter){
                    l_valid=false;
                    idx_gm = idx*nbnd + ib;
                    indofgm[idx_gm] = kpq*nbnd + ib;
                }
                for(int im = 0; im < nmodes; im++){
                    int idx_g2 = ib + im * nbnd + idx * nbnd * nmodes;
                    e2 = ph_ener[idx_qq+im*Nlist_Q]*radps2ev;
                    e3 = el_ener[idx_kq+ib*Nlist_K];

                    // if( e2 < 1.e-10 ) e2 = 1.e-10;
                    // if( e2 > 1.e-10 ){
                    if( e2 > ph_cut ){
                        double f2 = Bose_dist(e2, one_kbt);
                        // double f2 = 1.0/(exp(e2*one_kbt)-1.0);
                        double f3 = Fermi_dist(e3-chempot, one_kbt);
                        int orth_idx_kq = Eqidx_K[kpq+NPTK_K] - 1;
                        matmul( v3, &el_vel[3*idx_kq*nbnd+3*ib], &orth[orth_idx_kq*9] );
                        if (ismear_ecp == 0) 
                        {
                            int orth_idx_qq = Eqidx_Q[q[idx]+NPTK_Q] - 1;
                            matmul( v2, &ph_vel[3*idx_qq*nmodes+3*im], &orth[orth_idx_qq*9] );
                            for (int ip=0;ip<3;ip++){
                                v[ip] = v2[ip] * radps2ev - v3[ip];
                            }
                            sigma = sigma_ags( v, G, SB, Ngrid_Q );
                        }
                        else 
                        {
                            sigma = degauss;
                        }
                        // v1_m2=ddot3_device(v1, v1);
                        double vkqfactor;
                        if ( ddot3_device(v1, v1) < 1.e-10) 
                        {
                            vkqfactor = 1.0;
                        } 
                        else{
                            vkqfactor=1.0 - ddot3_device(v1,v3)/ddot3_device(v1, v1);
                        }
                        // printf("tau term : %d %d %d, %g %g %g \n", idx, ib ,im, Delta(*e1+e2-e3,sigma,NPTK_Q), abs(g2[idx_g2]) , (f2 + f3));
                        if(abs(*e1+e2-e3)<delta_mult*sigma){
                            weight = 2.0*M_PI * Delta(*e1+e2-e3,sigma,NPTK_Q) * g2[idx_g2] * (f2 + f3) / radps2ev;
                            if (iter) {
                                l_valid=true;
                                gm[idx_gm] = gm[idx_gm] + weight;
                            }
                            rate_rta[idx] += weight ;
                            weight = weight * vkqfactor;
                            rate_mrta[idx] += weight ;
                        }
                        if(abs(*e1-e2-e3)<delta_mult*sigma){
                            weight = 2.0*M_PI * Delta(*e1-e2-e3,sigma,NPTK_Q) * g2[idx_g2] * (1.e0 + f2 - f3) / radps2ev;
                            if (iter) {
                                l_valid=true;
                                gm[idx_gm] = gm[idx_gm] + weight;
                            }
                            rate_rta[idx] += weight ;
                            weight = weight * vkqfactor;
                            rate_mrta[idx] += weight ;
                        }
                    }
                }
                if (iter){
                    // printf(" process %d %d \n",idx,ib);
                    // if ( !l_valid || abs(gm[idx_gm]) < gm_eps ) {
                    if ( !l_valid ) {
                    // if ( !l_valid ) {
                        indofgm[idx_gm] = -1;
                        // indofgm[idx_gm] = 0;
                        // gm[idx_gm] = 0.0;
                    }
                }
            }
            // printf(" all : ~~ %d, %20.10g \n", q[idx], rate_rta[idx] );
        }
    }

    void cuda_indpro_(int *_batch_id, double *rate_rta, double *rate_mrta, long int *gm_offset){
        int batchSize_full = batch_size_;
        int nmodes = nmodes_;
        int nbnd = nbands_;
        int batchSize;
        double one_kbt;
        int is = batchSize_full * *_batch_id;

        // int nrr_q = nrr_q_;
        // double* d_irvec=d_irvec_r_;
        double *d_rate_rta=d_rate_rta_;
        double *d_rate_mrta=d_rate_mrta_;

        int nvalid = nvalid_;

        if (is + batchSize_full > nvalid)
        {
            batchSize = nvalid - is;
        }
        else
        {
            batchSize = batchSize_full;
        }

        int offset = is;

        one_kbt = 1.0/Te_/Kb;

        block_ = 256; 
        // block_ = 1;
        // block_ = BLOCK_SIZE*BLOCK_SIZE;
        grid_ = (batchSize+block_.x-1)/block_.x;

        // printf(" is : %d, batchSize_full : %d, nvalid : %d \n", is, batchSize_full, nvalid);
        // printf(" nbnd : %d, nmodes : %d, batch : %d \n", nbnd, nmodes, batchSize);

        int k = List_[StateInterest_[2*is_+1]-1]-1; 

        // printf(" ik(full index) = %d \n", k + 1 );

        bool iter = convergence_;
        double *d_gm_tmp=d_gamma_tmp_;
        double *h_gm_all=Gamma_;
        double *d_gm_all=d_gamma_;
        int *d_idxgm_tmp=d_idxgm_tmp_;
        int *h_idxgm_all=indscatt_;
        int *d_idxgm_all=d_idxgm_;
        int ngm=batchSize*nbnd;

        CUDA_CHECK(cudaMemset(d_gm_tmp, 0, ngm*sizeof(double)));
        cuda_indpro_glb<<< grid_, block_ >>>( d_e1_, d_v1_, d_ph_energy_, d_el_energy_, k, d_valid_list_+offset, d_el_velocity_, d_ph_velocity_, d_epmatf_dout,
                                            nbnd, nmodes, Nlist_K_, Nlist_Q_, NPTK_K_, NPTK_Q_, batchSize,
                                            d_NKgrid_, d_NQgrid_, d_Eqindex_K_, d_Eqindex_Q_, d_Orthcar_,
                                            ismear_ecp_, degauss_,delta_mult_, ph_cut_, scalebroad_, d_rlattvec_,
                                            one_kbt, ChemPot_, radps2ev, ryd2ev, d_rate_rta, d_rate_mrta,
                                            iter, d_idxgm_tmp, d_gm_tmp, gm_eps);
        
        if(iter){
            int nvalid_gm =  thrust::remove_copy_if(device, d_gm_tmp, d_gm_tmp + ngm,
                                                d_gm_all, is_zero_d0()) 
                                                - (d_gm_all);

            int nvalid_idx =  thrust::remove_copy_if(device, d_idxgm_tmp, d_idxgm_tmp + ngm,
                                                d_idxgm_all, is_mone_i()) 
                                                - (d_idxgm_all);

            CUDA_CHECK(cudaDeviceSynchronize());
            if(nvalid_gm!=nvalid_idx){
                printf("ERROR, number of valid gamma not equal to number of valid gamma index in No.%d state: %d %d \n", is_, nvalid_gm, nvalid_idx);
                exit(-1);
            } 

            CUDA_CHECK(cudaMemcpy(h_gm_all+*gm_offset, d_gm_all, nvalid_gm * sizeof(double) , cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_idxgm_all+*gm_offset, d_idxgm_all, nvalid_idx * sizeof(int) , cudaMemcpyDeviceToHost));

            *gm_offset +=nvalid_gm;
            
            // printf(" gamma offset : %ld \n", *gm_offset);

            // CUDA_CHECK(cudaDeviceSynchronize());
        }
    
        CUDA_CHECK(cudaDeviceSynchronize());

        *rate_rta += reduce(device, d_rate_rta, d_rate_rta+batchSize);
        *rate_mrta += reduce(device, d_rate_mrta, d_rate_mrta+batchSize);
    }

    /***************************************************************************
     * Iteration and transport coefficients
     **************************************************************************/

    __global__ void iteration_glb( int nstates, int nbnd, long npro, long offset, long *naccum, int *indofgamma, double *gamma, double *F, double *delta_F, 
                                   int *state, double *orth, int *eqidx, int NPTK, int Nirk ){
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        double diff[3];
        double F_m[3];
        if(idx < npro){
            int is = find_box(idx+offset, nstates, naccum);
            int m = indofgamma[idx]%nbnd;
            int kpq = indofgamma[idx]/nbnd;
            matmul(F_m, &F[( (eqidx[kpq]-1) + m*Nirk)*3], &orth[(eqidx[kpq+NPTK]-1)*9]);
            diff[0] = gamma[idx] * F_m[0];
            diff[1] = gamma[idx] * F_m[1];
            diff[2] = gamma[idx] * F_m[2];
            int n=state[is*2]-1;
            int k=state[is*2+1]-1;
            int idx_Fn = (k+n*Nirk)*3;

            atomicAdd(&delta_F[idx_Fn],diff[0]);
            atomicAdd(&delta_F[idx_Fn+1],diff[1]);
            atomicAdd(&delta_F[idx_Fn+2],diff[2]);
        }        
    }

    __global__ void init_F0(int nstates,   int nbnd, double *F, double *delta_F, 
                               double *tau,   double *vel,
                                  int *state, int Nirk){                     
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if(idx<nstates){
            int n=state[idx*2]-1;
            int k=state[idx*2+1]-1;
            int idx_Fn = (k+n*Nirk)*3;
            int idx_vel = (n+k*nbnd)*3;
            for(int idir=0; idir<3; idir++){
                // Fold[idx_Fn+idir] = Fnew[idx_Fn+idir];
                F[idx_Fn+idir] = tau[idx] * vel[idx_vel+idir];
            }
            // printf(" Fn_gpu : %d %d %d, %g %g %g %g %g\n", k, n, idx_Fn, tau[idx], vel[idx_vel], F[idx_Fn], F[idx_Fn+1], F[idx_Fn+2] );
        }                        
    }

    __global__ void refresh_Fn(int nstates,   int    nbnd, double *F, double *delta_F, 
                               double *tau,   double *vel,
                                  int *state, int    Nirk){                     
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if(idx<nstates){
            int n=state[idx*2]-1;
            int k=state[idx*2+1]-1;
            int idx_Fn = (k+n*Nirk)*3;
            int idx_vel = (n+k*nbnd)*3;
            for(int idir=0; idir<3; idir++){
                // Fold[idx_Fn+idir] = Fnew[idx_Fn+idir];
                F[idx_Fn+idir] = tau[idx] *  (delta_F[idx_Fn+idir] + vel[idx_vel+idir]) ;
            }
            // printf(" dFn_gpu : %d %d %d, %g %g %g %g %g\n", k, n, idx_Fn, tau[idx], vel[idx_vel], delta_F[idx_Fn], delta_F[idx_Fn+1], delta_F[idx_Fn+2] );
        }                        
    }

    __global__ void elcond(int nbnd, double *F, double *energy,
                               double *tau, double *vel, double *orth, int *eqidx,
                               int *state, int Nirk, int NPTK,
                               double one_kbt, double chempot,
                               double *el_cond, double *th_cond, double *Ninterm){
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        double v[3];
        double F_n[3];
        double tmp[9];

        if(idx<NPTK*nbnd)
        {
            int k = idx/nbnd; // fully index
            int n = idx%nbnd; // band
            // int idx_Fn = (k+n*Nirk)*3;
            // double e = energy[eqidx[k]-1+n*Nirk] - chempot;
            double e = energy[eqidx[k]-1+n*Nirk];
            matmul(v, &vel[(n + (eqidx[k]-1)*nbnd)*3], &orth[(eqidx[k+NPTK]-1)*9]);
            matmul(F_n, &F[(eqidx[k]-1 + n*Nirk)*3], &orth[(eqidx[k+NPTK]-1)*9]);
            double f = Fermi_dist(e-chempot, one_kbt); //f
            double f_1mf = f*(1.0-f); //f(1-f)
            for(int idir=0;idir<3;idir++)
            {
                for(int idir1=0;idir1<3;idir1++)
                {
                    tmp[idir+idir1*3] = v[idir]*F_n[idir1];
                    atomicAdd(&el_cond[idir+idir1*3],tmp[idir+idir1*3]*f_1mf);
                    atomicAdd(&th_cond[idir+idir1*3],tmp[idir+idir1*3]*f_1mf*(e - chempot)*(e - chempot));
                    atomicAdd(&Ninterm[idir+idir1*3],tmp[idir+idir1*3]*f_1mf*(e - chempot));
                }
            }
            // printf("elcond_part_gpu : %d %d %d %d  %d %e %d %e %e\n", k, n, eqidx[k]-1+n*Nirk, eqidx[k]-1,Nirk, e, eqidx[k+NPTK]-1, v[0],vel[(n + (eqidx[k]-1)*nbnd)*3]);
            // printf("elcond_part_gpu : %d %d %e %e %e %e %e %e %e %e\n", k, n, e, v[0], F_n[0],F_n[1],F_n[2],F[(eqidx[k]-1 + n*Nirk)*3],F[(eqidx[k]-1 + n*Nirk)*3+1],F[(eqidx[k]-1 + n*Nirk)*3+2]);
        }
    }

    void cuda_iter_init_(){
        CUDA_CHECK(cudaMalloc((void **)&d_rate_rta_iter_,  Nstates_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_Fn_,  3 * Nlist_K_ * nbands_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_delta_Fn_,  3 * Nlist_K_ * nbands_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_naccum_,  Nstates_ * sizeof(long)));
        CUDA_CHECK(cudaMalloc((void **)&d_Gamma_iter_,  batch_size_iter_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_IndGamma_iter_,  batch_size_iter_ * sizeof(double)));
    }

    void cuda_iter_(double *_rate_rta, long *_naccum, double *_Elcond){

        double *d_vel=d_el_velocity_;
        double *d_ener=d_el_energy_;
        int *d_eqidx = d_Eqindex_K_;

        double *Gamma = Gamma_;
        int *ind_Gamma = indscatt_;
        double *d_Gamma = d_Gamma_iter_;
        int *d_ind_Gamma = d_IndGamma_iter_;
        int batchsize_iter = batch_size_iter_;
        // int batchsize_iter = 100;
        int NPTK=NPTK_K_;
        int nir=Nlist_K_;
        int nbnd=nbands_;

        double spin_degen=spin_degen_;

        double one_kbt = 1.0/Te_/Kb;

        double *d_Fn = d_Fn_; // (3,irrk,n)
        double *d_delta_Fn = d_delta_Fn_; // (3,irrk,n)
        double *d_tau = d_rate_rta_iter_;
        long *d_naccum = d_naccum_;
        long npro=npro_;
        int nstates=Nstates_;
        bool converged=false;
        double criteria = iter_tolerance_;

        double  *d_el_cond = d_el_cond_, 
                *d_th_cond = d_th_cond_, 
                *d_niterm  = d_niterm_;
        double Vol = Volume_;
        double echarge = 1.60217657e-19;

        double el_cond_new[9]={};
        double el_cond_old[9]={0.0};
        double el_cond_sm[9];
        std::fill_n(el_cond_new, 9, 1.0);

        CUDA_CHECK(cudaMemcpy(d_tau, _rate_rta, Nstates_*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_naccum, _naccum+1, Nstates_*sizeof(long), cudaMemcpyHostToDevice));

        int niter = 0,
            max_iter = maxiter_;

        printf(" maxiter = %d \n", max_iter);
        printf(" nstates = %d \n", nstates);
        printf(" batchsize_iter = %d \n", batchsize_iter);
        printf(" %ld processeses are devided into %ld batch(es) \n", npro, (npro-1)/batchsize_iter+1);

        block_ = BLOCK_SIZE*BLOCK_SIZE;
        grid_ = (nstates+block_.x-1)/block_.x;
        CUDA_CHECK(cudaMemset(d_Fn, 0, 3 * nir * nbnd * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_delta_Fn, 0, 3 * nir * nbnd * sizeof(double)));
        init_F0<<<grid_,block_>>>(nstates, nbnd, d_Fn, d_delta_Fn, d_tau, d_vel, d_StateInterest_, nir);
        CUDA_CHECK(cudaDeviceSynchronize());

        grid_ = (NPTK*nbnd+block_.x-1)/block_.x;
        CUDA_CHECK(cudaMemset(d_el_cond, 0, 9 * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_th_cond, 0, 9 * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_niterm, 0, 9 * sizeof(double)));
        elcond<<<grid_,block_>>>(nbnd,d_Fn,d_ener,d_tau,d_vel,d_Orthcar_,d_eqidx,d_StateInterest_, nir, NPTK, 
        one_kbt, ChemPot_, d_el_cond, d_th_cond, d_niterm);

        CUDA_CHECK(cudaMemcpy(&el_cond_new, d_el_cond, sizeof(double)*9, cudaMemcpyDeviceToHost));
        for (int i=0;i<9;i++){
            el_cond_sm[i]  = spin_degen * el_cond_new[i]  * echarge*1.e21/(radps2ev*radps2ev)/(Kb*Te_*Vol*NPTK);
        }
        printf("  cond(S/m)   %g           %g           %g        \n", el_cond_sm[0], el_cond_sm[1], el_cond_sm[2]);
        printf("              %g           %g           %g        \n", el_cond_sm[3], el_cond_sm[4], el_cond_sm[5]);
        printf("              %g           %g           %g        \n", el_cond_sm[6], el_cond_sm[7], el_cond_sm[8]);
 
        // output_cond<<<1,1>>>(d_el_cond);

        // CUDA_CHECK(cudaDeviceSynchronize());
        
        // max_iter = 10;

        while(!converged && niter < max_iter)        
        {
            niter +=1;
            for( int idir=0; idir<9; idir++) el_cond_old[idir] = el_cond_new[idir];
            CUDA_CHECK(cudaMemset(d_delta_Fn, 0, 3 * nir * nbnd * sizeof(double)));
            for( int ibatch=0; batchsize_iter*ibatch < npro; ibatch ++ ){
                long offset = batchsize_iter*ibatch;
                long ncount = offset+batchsize_iter>npro?npro-offset:batchsize_iter;
                // printf("  ibatch = %d, offset = %ld\n", ibatch, offset);

                timer_GPUr_.start_clock("iter-Cpyin");
                CUDA_CHECK(cudaMemcpy(d_Gamma, Gamma+offset, ncount*sizeof(double), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_ind_Gamma, ind_Gamma+offset, ncount*sizeof(int), cudaMemcpyHostToDevice));
                timer_GPUr_.stop_clock("iter-Cpyin");

                timer_GPUr_.start_clock("iter-Core");
                block_ = BLOCK_SIZE*BLOCK_SIZE;
                grid_ = (ncount+block_.x-1)/block_.x;

                iteration_glb<<<grid_,block_>>>(nstates, nbnd, ncount, offset, d_naccum, d_ind_Gamma, d_Gamma,
                                                d_Fn, d_delta_Fn, d_StateInterest_, d_Orthcar_, d_eqidx, NPTK, nir);
                timer_GPUr_.stop_clock("iter-Core");
                CUDA_CHECK(cudaDeviceSynchronize());
            }
            // compare and outloop

            // printf("  iteration %d done. \n", niter);
            grid_ = (nstates+block_.x-1)/block_.x;
            refresh_Fn<<<grid_,block_>>>(nstates, nbnd, d_Fn, d_delta_Fn, d_tau, d_vel, d_StateInterest_, nir);


            grid_ = (NPTK*nbnd+block_.x-1)/block_.x;

            CUDA_CHECK(cudaMemset(d_el_cond, 0, 9 * sizeof(double)));
            CUDA_CHECK(cudaMemset(d_th_cond, 0, 9 * sizeof(double)));
            CUDA_CHECK(cudaMemset(d_niterm, 0, 9 * sizeof(double)));
            elcond<<<grid_,block_>>>(nbnd,d_Fn,d_ener,d_tau,d_vel,d_Orthcar_,d_eqidx,d_StateInterest_, nir, NPTK, 
            one_kbt, ChemPot_, d_el_cond, d_th_cond, d_niterm);

            CUDA_CHECK(cudaMemcpy(&el_cond_new, d_el_cond, sizeof(double)*9, cudaMemcpyDeviceToHost));

            // output_cond<<<1,1>>>(d_el_cond);

            CUDA_CHECK(cudaDeviceSynchronize());

            for( int idir=0; idir<9; idir++) el_cond_new[idir] = abs(el_cond_new[idir])>1e-10?el_cond_new[idir]:1e-10;

            // printf("   new:       %f          old:       %f          \n", el_cond_new[0], el_cond_old[0]);
            double diff = abs(((el_cond_new[0]-el_cond_old[0])/el_cond_old[0]+
                                    (el_cond_new[4]-el_cond_old[4])/el_cond_old[4]+
                                    (el_cond_new[8]-el_cond_old[8])/el_cond_old[8])/3.0);
            converged = diff<criteria;
            for (int i=0;i<9;i++){
                el_cond_sm[i]  = spin_degen * el_cond_new[i]  * echarge*1.e21/(radps2ev*radps2ev)/(Kb*Te_*Vol*NPTK);
            }
            printf("  iteration   %d,  relative difference  %f          \n", niter, diff);
            printf("  cond(S/m)   %g           %g           %g        \n", el_cond_sm[0], el_cond_sm[1], el_cond_sm[2]);
            printf("              %g           %g           %g        \n", el_cond_sm[3], el_cond_sm[4], el_cond_sm[5]);
            printf("              %g           %g           %g        \n", el_cond_sm[6], el_cond_sm[7], el_cond_sm[8]);
        }
        
        // copy to host

        double Elcond[9];

        CUDA_CHECK(cudaMemcpy(&Elcond, d_el_cond, sizeof(double)*9, cudaMemcpyDeviceToHost));

        // convert to common units
        // sigma=e**2/(Kb*Te*Vol*NPTK)sum_k [f_k(1-f_k)*v_k*F_k]
        // velocity: nm*ev/hbar=nm*THZ/radps2ev, F_n: nm*ev/hbar/THz,Vol:nm**3,Elcond: Siemens/m = C^2/(J*m*s)

        for (int i=0;i<9;i++){
            _Elcond[i]  = spin_degen * Elcond[i]  * echarge*1.e21/(radps2ev*radps2ev)/(Kb*Te_*Vol*NPTK);
        }
    }

    void cuda_iter_destroy_(){
        
        CUDA_CHECK(cudaFree(d_Orthcar_));

        CUDA_CHECK(cudaFree(d_el_energy_));
        CUDA_CHECK(cudaFree(d_el_velocity_));

        CUDA_CHECK(cudaFree(d_Eqindex_K_));
        CUDA_CHECK(cudaFree(d_StateInterest_ ));

        CUDA_CHECK(cudaFree(d_el_cond_));
        CUDA_CHECK(cudaFree(d_th_cond_));
        CUDA_CHECK(cudaFree(d_niterm_));

        CUDA_CHECK(cudaFree(d_rate_rta_iter_));
        CUDA_CHECK(cudaFree(d_Fn_));
        CUDA_CHECK(cudaFree(d_delta_Fn_));
        CUDA_CHECK(cudaFree(d_naccum_));
        CUDA_CHECK(cudaFree(d_Gamma_iter_));
        CUDA_CHECK(cudaFree(d_IndGamma_iter_));
    }

} // extern "C"
