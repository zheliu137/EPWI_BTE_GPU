#pragma once
#include "cuda_settings.h"
#include "cuda_timer.h"
// #include <cuda/std/complex>
// #include <cmath>

// using namespace cuda::std;

// namespace 
namespace cuda_elphwann_wannier
// namespace 
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
    static const double solver_tol_ = 1.e-12;
    static const int solver_max_sweeps_ = 150;

    static const ComplexD ci_(0.0, 1.0);
    static const cuDoubleComplex conee_ = {1.0, 0.0}, czeroo_ = {0.0, 0.0};
    // static const ComplexD cone_ = {1.0, 0.0}, czero_ = {0.0, 0.0};

    static const double one_ = 1.0,
                        zero_ = 0.0;

    static cudaEvent_t elphwann_wannier_event_start_ = NULL,
                       elphwann_wannier_event_end_ = NULL, 
                       f_dynwan_2_ephwan_event_start = NULL,
                       f_dynwan_2_ephwan_event_end_;

    static int gpu_id_,
        nmodes_,
        nbands_,
        nbnd_irrel_,
        nbndsub_,
        nrr_k_,
        nrr_q_,
        nat_,
        ntypx_,
        Nstates_,
        NStateInterest_,
        *StateInterest_,
        *d_StateInterest_, // (2,Nstates) ibnd and k in irr index(get full index from List_K)
        Nsymm_,
        Nlist_K_,
        Nlist_Q_,
        NPTK_K_,
        NPTK_Q_,
        batch_size_,
        batch_size_k_,
        batch_size_q_,
        batch_size_iter_,
        mpime_,
        nvalid_;
        
    static double *amass_ = nullptr,
                  *d_amass_ = nullptr,
                  celldm1_,
                  omega_,
                  *tau_ = nullptr,    // in Cart coord
                  *d_tau_ = nullptr,
                  *bg_ = nullptr,
                  *d_bg_ = nullptr,
                  *irvec_r_ = nullptr,
                  *d_irvec_r_ = nullptr,
                  *rlattvec_ = nullptr,   // in the unit of 1/nm
                  *d_rlattvec_ = nullptr,
                  Volume_,            // in the unit of nm^3
                  spin_degen_;

    static int is_;

    static int *d_kpq_  = nullptr,
               *NKgrid_ = nullptr,
               *d_NKgrid_ = nullptr,
               *NQgrid_ = nullptr,
               *nqc_ = nullptr,
               *d_NQgrid_ = nullptr,
               *ityp_ = nullptr,
               *d_ityp_ = nullptr,
               *irvec_ = nullptr,
               *ndegen_k_ = nullptr,
               *ndegen_q_ = nullptr,
               *d_irvec_ = nullptr,
               *d_ndegen_k_ = nullptr,
               *d_ndegen_q_ = nullptr,
               *List_ = nullptr,
               *d_List_K_ = nullptr,
               *Eqindex_K_ = nullptr,
               *d_Eqindex_K_ = nullptr,
               *Eqindex_Q_ = nullptr,
               *d_Eqindex_Q_ = nullptr,
               *valid_list_ = nullptr,
               *indscatt_ = nullptr,
               *d_idxgm_ = nullptr,
               *d_idxgm_tmp_ = nullptr,
               *d_IndGamma_iter_ = nullptr,
               *d_valid_i_ = nullptr,
               *d_valid_plus_i_ = nullptr,
               *d_PhononInterest_ = nullptr;

    // polar variables
    static int nrx_[3];
    static const double gmax_=14.0;

    static long *d_naccum_ = nullptr,
                npro_;
    
    static int ismear_ecp_; // smearing method in energy conservation principles

    static double Te_,
        ChemPot_,
        scalebroad_,
        degauss_,
        delta_mult_,
        ph_cut_;

    static double *d_xkk_ = nullptr,
                  *d_xkq_ = nullptr,
                  *d_xxq_ = nullptr,
                  *d_xxq_cart_ = nullptr,
                  *d_eig_ = nullptr,
                  *d_wfqq_ = nullptr,
                  *d_etkk_ = nullptr,
                  *d_w2_ = nullptr,
                  *d_etkq_ = nullptr,
                  *Orthcar_ = nullptr,
                  *d_Orthcar_ = nullptr,
                  **d_Orthcar_ptr_ = nullptr,
                  **d_Orthcar_ptr_q_ = nullptr,
                  *el_energy_ = nullptr,      // (Nlist_K,nbnd), in the unit of eV
                  *d_el_energy_ = nullptr,    // (Nlist_K,nbnd), in the unit of eV
                  *ph_energy_ = nullptr,      // (Nlist_Q,nmodes), in the unit of radps
                  *d_ph_energy_ = nullptr,    // (Nlist_Q,nmodes), in the unit of radps
                  *el_velocity_ = nullptr,    // 3,nbnd,Nlist_K, in the unit of nm*eV
                  *d_v1_ = nullptr,
                  *d_v2_ = nullptr,
                  *d_el_velocity_ = nullptr,  // 3,nbnd,Nlist_K
                  *d_el_velocity_tmp_ = nullptr,
                  **d_el_velocity_ptr_ = nullptr,
                  **d_el_velocity_tmp_ptr_ = nullptr,
                  *ph_velocity_ = nullptr,    // 3,nmodes,Nlist_Q
                  *d_elk_velocity_ = nullptr,
                  *d_ph_velocity_ = nullptr,  // 3,nmodes,Nlist_Q, in the unit of nm*radps
                  *d_ph_velocity_tmp_ = nullptr,
                  **d_ph_velocity_ptr_ = nullptr,
                  **d_ph_velocity_tmp_ptr_ = nullptr,
                  *Gamma_ = nullptr,
                  *d_gamma_ = nullptr,
                  *d_gamma_tmp_ = nullptr,
                  *d_Gamma_iter_ = nullptr,
                  *d_Orthcar_q_ = nullptr,
                  *d_Orthcar_k_ = nullptr,
                  *d_e1_ = nullptr,
                  *d_e2_ = nullptr,
                  *d_e3_ = nullptr;

    static double *d_rate_rta_,
                  *d_rate_rta_iter_,
                  *d_rate_mrta_,
                  *d_Fn_, // (3,irrk,n)
                  *d_delta_Fn_, // (3,irrk,n)
                  *d_el_cond_,
                  *d_th_cond_,
                  *d_niterm_;

    static bool convergence_;
    static bool lpolar_;

    static double *epsil_,
                  *d_epsil_,
                  *zstar_,
                  *d_zstar_;

    static double iter_tolerance_;

    static int maxiter_;

    static cuDoubleComplex *d_cuf_ = nullptr,
                           *d_chf_ = nullptr,
                           *d_work_ = nullptr,
                           *d_work_k_ = nullptr,
                           *d_epmatwp_ = nullptr, // nbndsub(k+q or 0), nmodes, nrr_q, nbndsub(k or R), nrr_k
                           *d_cufkk_ = nullptr,
                           *d_cufkq_ = nullptr,
                           *d_epwef_ = nullptr, // nbndsub(k+q), nmodes, nrr_q 
                           *d_epmatf_ = nullptr,
                           *d_epmatfl_ = nullptr,
                           *d_umn_ = nullptr,
                           *d_chw_ = nullptr,
                           *d_rdw_ = nullptr,
                           *d_cfac_ = nullptr,
                           *d_cfac_k_ = nullptr,
                           *d_cfac_q_ = nullptr,
                           *d_cfac_kq_ = nullptr,
                           *d_epmatf_reduce_ = nullptr,
                           *d_eptmp_ = nullptr;
    static cuDoubleComplex **d_cufkk_arr_ = nullptr,
                           **d_cufkq_arr_ = nullptr,
                           **d_epmatf_arr_ = nullptr,
                           **d_cufkk_ptr_ = nullptr,
                           **d_cufkq_ptr_ = nullptr,
                           **d_cuf_ptr_ = nullptr,
                           **d_eptmp_ptr_ = nullptr,
                           **d_epmatf_ptr_ = nullptr;
    static double *d_epmatf_dout;

    static int *d_valid_list_ = nullptr;

    static dim3 block_, grid_;

    static cublasHandle_t blasHandle_;
    static cusolverDnHandle_t solverHandle_;
    static syevjInfo_t syevj_params_ = NULL;
    static int *solverDevInfo_ = nullptr;
    static int lwork_;

    /* temporary variables */
    static double *d_ephwan2blochp_rdotk_q_ = nullptr;
    static cuDoubleComplex *d_q_cfac_ = nullptr,
                           *d_ephwan2blochp_cfac_ = nullptr,
                           *d_ephwan2blochp_eptmp_ = nullptr,
                           *d_ephwan2bloche_eptmp_ = nullptr,
                           *d_dynwan2bloch_chf_ = nullptr,
                           *d_hamwan2bloch_chf_ = nullptr,
                           *d_hamwan2bloch_chfkq_ = nullptr,
                           *d_ephwan2bloch_cfac_ = nullptr;

    static double *d_cfac_kq_rdotk_ = nullptr,
                  *d_cfac_kq_rdotk_q_ = nullptr;

    // debug
    static double *test_tmp = nullptr,
                  *test_tmp2 = nullptr,
                  *test_tmp3 = nullptr;

    // extern GPU_Timer timer_GPUr_;
    // extern CPU_Timer timer_CPUr_;
    static GPU_Timer timer_GPUr_;
    static CPU_Timer timer_CPUr_;

}

namespace device_funcs
{    
    static __device__ double cabs_(cuDoubleComplex a){
            double c = sqrt(a.x*a.x+a.y*a.y);
            return c;
    }

    static __device__ double c_angle(cuDoubleComplex a){
        double ag=atan2(a.y,a.x);
        return ag;
    }

    static __device__ int kplusq_kernel(int k, int q, int *NKgrid, int *NQgrid)
    {
        int veck[3],
            vecq[3],
            veckq[3];
        veck[0]=(k)/(NKgrid[1]*NKgrid[2]);
        veck[1]=(k)%(NKgrid[1]*NKgrid[2])/NKgrid[2];
        veck[2]=(k)%NKgrid[2];

        vecq[0]=(q)/(NQgrid[1]*NQgrid[2]);
        vecq[1]=(q)%(NQgrid[1]*NQgrid[2])/NQgrid[2];
        vecq[2]=(q)%NQgrid[2];

        vecq[0]=vecq[0]*(NKgrid[0]/NQgrid[0]);
        vecq[1]=vecq[1]*(NKgrid[1]/NQgrid[1]);
        vecq[2]=vecq[2]*(NKgrid[2]/NQgrid[2]);

        veckq[0]=(veck[0]+vecq[0])%NKgrid[0];
        veckq[1]=(veck[1]+vecq[1])%NKgrid[1];
        veckq[2]=(veck[2]+vecq[2])%NKgrid[2];

        int kplusq = veckq[0]*NKgrid[1]*NKgrid[2]+veckq[1]*NKgrid[2]+veckq[2];

        return kplusq;
    }

    static __inline__ __device__ double sigma_ags(double *v, double *G, double scalebroad, int *Ngrid){
        double sigma = -INFINITY;
        // double ngrid[3] = {10,10,10};
        double sig[3]={ 0.0, 0.0, 0.0 };
        for (int i = 0; i < 3; ++i)
        {
            sig[0] += G[0 + i] * (v[i]);
            sig[1] += G[3 + i] * (v[i]);
            sig[2] += G[6 + i] * (v[i]);
        }

        for (int i = 0; i < 3; ++i)
        {
            sigma = max(sigma, pow(sig[i] / double(Ngrid[i]), 2));
        }

        sigma = scalebroad * sqrt(sigma * 0.5);

        return sigma;
    }

    static __inline__ __device__ void matmul(double *v, double *v0, double *orth){
        for (int j=0; j<3; j++){
            v[j]=0.0;
            for (int i=0; i<3; i++){
                v[j] += v0[i] * orth[j+3*i];
            }
        }
    }

    static __inline__ __device__ double ddot3_device(double *vec1, double *vec2){
            double ddot=0.0;
            for (int i=0; i<3; i++){
                ddot += vec1[i] * vec2[i];
            }
            return ddot;
    }

    static __inline__ __device__ double Delta(double e, double sig, int NPTK){
        double wg = exp(-pow(e,2)/pow(sig,2))/sig*M_2_SQRTPI*0.5/double(NPTK);
        return wg;
    }    

    static __inline__ __device__ double Bose_dist(double e, double one_kbt){
        return 1.0/(exp(e*one_kbt)-1.0);
    }       

    static __inline__ __device__ double Fermi_dist(double e, double one_kbt){
        return 1.0/(exp(e*one_kbt)+1.0);
    }       

    static __device__ int find_box(long idx, int nbox, long *box){
        int this_box;
        for(int i=0;i<nbox;i++){
            if(idx < box[i]) {
                this_box = i;
                break;
            }
        }
        return this_box;
    }

    static __device__ double c_angle_(cuDoubleComplex a)
    {
        double ag=atan2(a.y,a.x);
        return ag;
    }

}

namespace polar_funcs
{
    static __global__ void dynmat_polar(int batchsize, int nrx1, int nrx2, int nrx3, int nmodes, int nat, double *bg,
                                        double *tau, double *xqc, ComplexD *dyn,  double *epsil, double *zeu, double gmax, double fac0)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;

        double g1,g2,g3;

        double qeq;
        double tpi = 2.0*M_PI;

        // ComplexD czero={0.0,0.0};
        ComplexD phase, facg;
        double fac;
        double zaq[3], zbq[3], zcq[3], fnat[3];

        if( idx<batchsize )
        {
            // printf("kernel : %d  %f , %f , %f \n", idx, xqc[idx*3+0],xqc[idx*3+1],xqc[idx*3+2]);

            for (int r1 = -nrx1; r1<=nrx1; r1++ ){
                for (int r2 = -nrx2; r2<=nrx2; r2++ ){
                    for (int r3 = -nrx3; r3<=nrx3; r3++ ){
                        g1 = bg[0]*r1 + bg[3]*r2 + bg[6] * r3 ;
                        g2 = bg[1]*r1 + bg[4]*r2 + bg[7] * r3 ;
                        g3 = bg[2]*r1 + bg[5]*r2 + bg[8] * r3 ;

                        qeq = (g1*(epsil[0]*g1+epsil[3]*g2+epsil[6]*g3 )+      
                            g2*(epsil[1]*g1+epsil[4]*g2+epsil[7]*g3 )+      
                            g3*(epsil[2]*g1+epsil[5]*g2+epsil[8]*g3 )); //*twopi/alat

                        // printf("kernel qeq : %d %d %d %d %g \n", idx, r1, r2, r3, qeq);
                        if (qeq > 1.0e-14 && qeq/4.0 < gmax ) {

                            fac = fac0*exp(-qeq/4.0)/qeq;

                            for (int iat = 0; iat<nat; iat++ ){
                                for(int idir=0;idir<3;idir++){
                                    zaq[idir]=g1*zeu[iat*9+3*idir]+g2*zeu[iat*9+3*idir+1]+g3*zeu[iat*9+3*idir+2];
                                    fnat[idir] = 0.0;
                                }
                                for (int jat = 0; jat<nat; jat++ ){
                                    double arg = tpi * ( g1 * ( tau[iat*3]-tau[jat*3]) +
                                                         g2 * ( tau[iat*3+1]-tau[jat*3+1]) +
                                                         g3 * ( tau[iat*3+2]-tau[jat*3+2]) );
                                    for(int idir=0; idir<3; idir++ ){
                                        zcq[idir]=g1*zeu[jat*9+3*idir]+g2*zeu[jat*9+3*idir+1]+g3*zeu[jat*9+3*idir+2];
                                        fnat[idir] += zcq[idir]*cos(arg) ;
                                    }
                                }
                                for (int i=0; i<3; i++){
                                    for (int j=0; j<3; j++){
                                        dyn[ (iat*3+i)+(iat*3+j)*nmodes + idx*nmodes*nmodes ] 
                                                                -=   fac  *  zaq[i] * fnat[j];
                                        // printf("kernel polar : %d %d %d %d %d, %f %f %f\n",
                                        //     r1, idx,iat,i,j,fac,zaq[i], fnat[j]);
                                    }
                                }
                            }
                        }
                        // /*
                        g1 = g1 + xqc[idx*3];
                        g2 = g2 + xqc[idx*3+1];
                        g3 = g3 + xqc[idx*3+2];

                        qeq = (g1*(epsil[0]*g1+epsil[3]*g2+epsil[6]*g3 )+      
                                g2*(epsil[1]*g1+epsil[4]*g2+epsil[7]*g3 )+      
                                g3*(epsil[2]*g1+epsil[5]*g2+epsil[8]*g3 )); //*twopi/alat
                        if (qeq > 1.0e-14 && qeq/4.0 < gmax ) {

                            fac = exp(-qeq/4.0)/qeq;

                            for (int jat = 0; jat<nat; jat++ ){
                                for(int idir=0;idir<3;idir++){
                                    zbq[idir]=g1*zeu[jat*9+3*idir]+g2*zeu[jat*9+3*idir+1]+g3*zeu[jat*9+3*idir+2];
                                }
                                for (int iat = 0; iat<nat; iat++ ){
                                    double arg = tpi * (    g1 * ( tau[iat*3]-tau[jat*3])     +
                                                            g2 * ( tau[iat*3+1]-tau[jat*3+1]) +
                                                            g3 * ( tau[iat*3+2]-tau[jat*3+2]) );
                                    for(int idir=0; idir<3; idir++ ){
                                        zaq[idir]=g1*zeu[iat*9+3*idir]+g2*zeu[iat*9+3*idir+1]+g3*zeu[iat*9+3*idir+2];
                                    }
                                    phase={cos(arg),sin(arg)};
                                    facg = fac * fac0 * phase; 
                                    for (int i=0; i<3; i++){
                                        for (int j=0; j<3; j++){
                                            dyn[ (iat*3+i)+(jat*3+j)*nmodes + idx*nmodes*nmodes ] 
                                                                        +=  facg * zaq[i] * zbq[j];
                                        }
                                    }
                                }
                            }
                        }
                        // */
                    }
                }
            }
            // printf("dynmat[%d]_11 =  %f\n", idx , dyn[ idx*nmodes*nmodes ].real());
            // printf("dynmat[%d]_11 = %f \n", idx, abs(dyn[idx*nmodes*nmodes]));
        }
    }
}

extern "C"{
    
    extern cublasFillMode_t char_to_cublas_fillmode(char _uplo);
    extern cublasOperation_t char_to_cublas_trans(char trans);

    extern __global__ void cryst_to_cart_global( double* bg, double* cryst, double* cart, int batchsize);
    extern __global__ void fc_massfac(cuDoubleComplex *rdw, double *mass, int *ityp, int nmodes, int nrr);

    extern __global__ void get_kpq_ene_npro(int nptk_q, int k, int *d_Eqindex_K, int *kpq, int *NKgrid, int *NQgrid,
                                     double *e3, double *el_energy, int nbnd, int Nlist_K);

    extern __global__ void omega_init_npro(int nptk_q, int *d_Eqindex_Q, double *e2, double radps2ev, double *ph_energy, int nmodes, int Nlist_Q);

    extern __global__ void ph_vel_init_npro(int nptk_q, double *d_ph_velocity, double **d_ph_velocity_ptr,
                                    double *d_ph_velocity_tmp, double **d_ph_velocity_tmp_ptr,
                                    double *d_Orthcar, double **d_Orthcar_ptr, int *d_Eqindex_Q,
                                    double *e2, double *ph_energy, int nmodes, int Nlist_Q);

    extern __global__ void vel_rotation_ptr_npro( int *kpq,
                                            double **d_Orthcar_ptr,
                                            double **d_el_velocity_ptr,
                                            double **d_el_velocity_tmp_ptr,
                                            double *d_Orthcar,
                                            double *d_el_velocity,
                                            double *d_el_velocity_tmp,
                                            int *d_Eqindex_K,
                                            int NPTK_K,
                                            int nbnd,
                                            int NPTK_Q);

    extern __global__ void count_npro(double *d_v2,
                                double *d_v3,
                                double e1,
                                double *e2,
                                double *e3,
                                double *rlattvec,
                                int *Ngrid,
                                int nmodes,
                                int nbnd,
                                double radps2ev,
                                double scalebroad,
                                int ismear_ecp,
                                double degauss,
                                double delta_mult,
                                int nptk_q,
                                int *valid_i,
                                double ph_cut);
    
    extern __global__ void countvalidph_npro( int *valid_i, int *validph, int NPTK);


    extern void _ind_k_to_xk(int *vec, double *xkx, int *ngrid, int k);

    extern __global__ void hermitianize_matrix_batched(int n, int batchSize, cuDoubleComplex *A);

    extern __global__ void dynmat_prep(ComplexD *dyn, double *mass, int *ityp, int nmodes, int batchSize);

    extern __global__ void cfac_batched(int nrr, double *irvec, int *ndegen, double *d_xkk, ComplexD *cfac, int batchsize);

    extern __global__ void iktoxxk_batch(int batchsize, double *xxk, int *ngrid, int ik);

    extern __global__ void istoxxk_batch(int batchsize, int *stateinterest, int *List, double *xxk, int *ngrid, int is_offset);

    extern __global__ void trans_epmat(double *d_epmatf_out,cuDoubleComplex *d_epmatf, double* wfqq, double eps, int batch_size, int nbnd, int nmodes, double ryd2ev);

    extern __global__ void trans_epmat_2(double *d_epmatf_out,cuDoubleComplex *d_epmatf, double* e2, int* Eqindex_Q, int Nlist_Q, int* valid_list, double radps2ry, double eps, int batch_size, int nbnd, int nmodes);


    extern __global__ void get_kpq_ene(int batchsize, int *d_Eqindex_K, int *d_Eqindex_Q,  int k, int q_offset, int *kpq, int *NKgrid,
                            int *NQgrid, double *e2, double *e3, double *el_energy, double *ph_energy, double radps2ev, int nbnd, int nmodes, int Nlist_K, int Nlist_Q);

    extern __global__ void set_vel_rotation_ptr( int *kpq,
                                          int q_offset,  
                                          double **d_Orthcar_ptr, 
                                          double **d_el_velocity_ptr, 
                                          double **d_el_velocity_tmp_ptr, 
                                          double *d_Orthcar, 
                                          double *d_el_velocity, 
                                          double *d_el_velocity_tmp, 
                                          double **d_Orthcar_ptr_q, 
                                          double **d_ph_velocity_ptr, 
                                          double **d_ph_velocity_tmp_ptr, 
                                          double *d_ph_velocity, 
                                          double *d_ph_velocity_tmp, 
                                          int *d_Eqindex_K,
                                          int *d_Eqindex_Q,
                                          int NPTK_K,
                                          int NPTK_Q,
                                          int nbnd,
                                          int nmodes,
                                          int batchsize);


    extern __global__ void check_validk_ags(double *d_v2,
                               double *d_v3,
                               double *e1,
                               double *e2,
                               double *e3,
                               double *rlattvec,
                               int *Ngrid,
                               int nmodes,
                               int nbnd,
                               double radps2ev,
                               double scalebroad,
                               double delta_mult,
                               double ph_cut,
                               int *valid_i,
                               int offset,
                               int batchSize);

    
    extern __global__ void check_validk_cgs(
                               double *e1,
                               double *e2,
                               double *e3,
                               int *Ngrid,
                               int nmodes,
                               int nbnd,
                               double sigma,
                               double delta_mult,
                               double ph_cut,
                               int *valid_i,
                               int offset,
                               int batchSize);

    extern __global__ void count_validk(int *valid_i,
                               int offset,
                               int batchsize);

    extern __global__ void iqtoxkxq_batch(int batchsize, int* NKgrid, int* NQgrid, int k, int q_offset, int* d_valid_list, double* xxq, double* xkq);

    extern __global__ void output_d_eptmp(char name,int offset, cuDoubleComplex *chf);

}
