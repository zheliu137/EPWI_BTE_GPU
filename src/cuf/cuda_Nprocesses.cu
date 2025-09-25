#include "cuda_settings.h"
// #include "cuda_timer.h"
// #include <cmath>
// #include "cuda_elphwann_namespace.h"

extern "C"
{   
    static __device__ int kplusq_kernel(int k, int q, int *NKgrid, int *NQgrid);

    __global__ void get_kpq_ene_npro(int nptk_q, int k, int *d_Eqindex_K, int *kpq, int *NKgrid, int *NQgrid,
                                     double *e3, double *el_energy, int nbnd, int Nlist_K)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if ( idx < nptk_q * nbnd)
        {
            int q    = idx/nbnd;
            int ibnd = idx%nbnd;
            kpq[q]  = kplusq_kernel( k, q, NKgrid, NQgrid );
            e3[idx] = el_energy[ d_Eqindex_K[ kpq[q] ] - 1 + ibnd * Nlist_K ];
        }
    }

    __global__ void omega_init_npro(int nptk_q, int *d_Eqindex_Q, double *e2, double radps2ev, double *ph_energy, int nmodes, int Nlist_Q)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if ( idx < nptk_q * nmodes)
        {
            int q   = idx/nmodes;
            int imode = idx%nmodes;
            e2[idx] = ph_energy[ d_Eqindex_Q[q]-1+imode*Nlist_Q ]*radps2ev;
        }
    }

    __global__ void ph_vel_init_npro(int nptk_q, double *d_ph_velocity, double **d_ph_velocity_ptr,
                                    double *d_ph_velocity_tmp, double **d_ph_velocity_tmp_ptr,
                                    double *d_Orthcar, double **d_Orthcar_ptr, int *d_Eqindex_Q,
                                    double *e2, double *ph_energy, int nmodes, int Nlist_Q)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if ( idx < nptk_q)
        {
            d_Orthcar_ptr[idx] = &d_Orthcar[ ( d_Eqindex_Q[ idx + nptk_q ] - 1 ) * 9 ];
            d_ph_velocity_ptr[idx] = &d_ph_velocity[ ( d_Eqindex_Q[idx ] - 1 ) * 3 * nmodes ];
            d_ph_velocity_tmp_ptr[idx] = &d_ph_velocity_tmp[ idx * 3 * nmodes ];
        }
    }

    __global__ void vel_rotation_ptr_npro( int *kpq,
                                          double **d_Orthcar_ptr,
                                          double **d_el_velocity_ptr,
                                          double **d_el_velocity_tmp_ptr,
                                          double *d_Orthcar,
                                          double *d_el_velocity,
                                          double *d_el_velocity_tmp,
                                          int *d_Eqindex_K,
                                          int NPTK_K,
                                          int nbnd,
                                          int NPTK_Q)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if ( idx < NPTK_Q )
        {
            d_Orthcar_ptr[idx] = &d_Orthcar[ ( d_Eqindex_K[ kpq[idx] + NPTK_K] - 1) * 9 ];
            d_el_velocity_ptr[idx] = &d_el_velocity[( d_Eqindex_K[ kpq[idx] ] - 1 ) * 3 * nbnd];
            d_el_velocity_tmp_ptr[idx] = &d_el_velocity_tmp[idx*3*nbnd];
        }
    }

    __global__ void count_npro(double *d_v2,
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
                               double ph_cut
                               )
                            //    int *valid_minus_i)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        // static int const one = 1;
        bool valid_gm;
        double sigma;

        if (idx < nptk_q ){
            valid_i[idx] = 0;
            for(int ib = 0;ib<nbnd;ib++){
                valid_gm=false;
                for(int im = 0;im<nmodes;im++){
                    if (e2[im+idx*nmodes] >= ph_cut){
                        if(ismear_ecp==0){
                            double sig[3] = { 0.0, 0.0, 0.0 };
                            sigma = -INFINITY;
                            for (int i = 0; i < 3; ++i)
                            {
                                sig[0] += rlattvec[0 + i] * (d_v2[ im * 3 + i + idx * 3 * nmodes] * radps2ev - d_v3[ib * 3 + i + idx*nbnd*3]);
                                sig[1] += rlattvec[3 + i] * (d_v2[ im * 3 + i + idx * 3 * nmodes] * radps2ev - d_v3[ib * 3 + i + idx*nbnd*3]);
                                sig[2] += rlattvec[6 + i] * (d_v2[ im * 3 + i + idx * 3 * nmodes] * radps2ev - d_v3[ib * 3 + i + idx*nbnd*3]);
                            }

                            for (int i = 0; i < 3; ++i)
                            {
                                sigma = max(sigma, pow(sig[i] / Ngrid[i], 2));
                            }
                            sigma = scalebroad * sqrt(sigma * 0.5);
                        }
                        else {
                            sigma = degauss;
                        }
                            // __syncthreads();

                        // printf("npro valid? %d %d %d : %f %f %f %f %f \n", idx, ib, im, ph_cut, e2[im+idx*nmodes], abs(e1 + e2[im+idx*nmodes] - e3[ib+idx*nbnd]), 
                        //              abs(e1 - e2[im+idx*nmodes] - e3[ib+idx*nbnd]),
                        //              2.0 * sigma);

                        if (abs(e1 + e2[im+idx*nmodes] - e3[ib+idx*nbnd]) <= delta_mult * sigma || 
                            abs(e1 - e2[im+idx*nmodes] - e3[ib+idx*nbnd]) <= delta_mult * sigma )
                        {
                            valid_gm=true;
                        }
                    }
                }
                // if(valid_gm)atomicAdd(&valid_i[idx],one);
                if(valid_gm)valid_i[idx]+=1;
            }
        }
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

    __global__ void countvalidph_npro( int *valid_i, int *validph, int NPTK)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < NPTK){
            if(valid_i[idx]){
                validph[idx] = 1;
            }            
        }
    }
}