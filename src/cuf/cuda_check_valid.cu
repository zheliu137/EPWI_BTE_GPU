
#include "cuda_settings.h"
// #include <cmath>
// #include "cuda_elphwann_namespace.h"

// using namespace cuda::std;
// using namespace cuda_elphwann_wannier;
extern "C"
{   
    static __device__ int kplusq_kernel(int k, int q, int *NKgrid, int *NQgrid);
    extern cublasOperation_t char_to_cublas_trans(char);

    // __global__ void omega_init_npro(int nptk_q, int *d_Eqindex_Q, double *e2, double radps2ev, double *ph_energy, int nmodes, int Nlist_Q)
    // {
    //     int idx = blockDim.x * blockIdx.x + threadIdx.x;
    //     if ( idx < nptk_q * nmodes)
    //     {
    //         int q   = idx/nmodes;
    //         int imode = idx%nmodes;
    //         e2[idx] = ph_energy[ d_Eqindex_Q[q]-1+imode*Nlist_Q ]*radps2ev;
    //     }
    // }

    __global__ void get_kpq_ene(int batchsize, int *d_Eqindex_K, int *d_Eqindex_Q,  int k, int q_offset, int *kpq, int *NKgrid,
                            int *NQgrid, double *e2, double *e3, double *el_energy, double *ph_energy,  double radps2ev, int nbnd, int nmodes, int Nlist_K, int Nlist_Q)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if ( idx < batchsize )
        {
            int q = idx + q_offset;
            kpq[idx] = kplusq_kernel(k, q, NKgrid, NQgrid);
            for ( int ibnd3 = 0; ibnd3 < nbnd; ibnd3++ )
            {
                e3[ibnd3+idx*nbnd] = el_energy[d_Eqindex_K[ kpq[idx] ] - 1 + ibnd3 * Nlist_K];
            }
            for ( int im = 0; im < nmodes; im++ )
            {
                e2[im+idx*nmodes] = ph_energy[d_Eqindex_Q[q] - 1 + im * Nlist_Q]*radps2ev;
            }
        }
    }

    __global__ void set_vel_rotation_ptr( int *kpq,
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
                                          int batchsize)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if ( idx < batchsize )
        {   
            int q = idx + q_offset;
            // d_Orthcar_ptr[idx] = &d_Orthcar[ ( d_Eqindex_K[ kpq[idx] - 1 + NPTK_K] - 1) * 9 ];
            d_Orthcar_ptr[idx] = &d_Orthcar[ ( d_Eqindex_K[ kpq[idx] + NPTK_K] - 1) * 9 ];
            // d_Orthcar_ptr[idx] = d_Orthcar;
            // d_el_velocity_ptr[idx] = &d_el_velocity[( d_Eqindex_K[ kpq[idx] - 1 ] - 1 ) * 3*nbnd];
            d_el_velocity_ptr[idx] = &d_el_velocity[( d_Eqindex_K[ kpq[idx] ] - 1 ) * 3*nbnd];
            // d_el_velocity_ptr[idx] = d_el_velocity;
            d_el_velocity_tmp_ptr[idx] = &d_el_velocity_tmp[idx*3*nbnd];
            // d_el_velocity_tmp_ptr[idx] = d_el_velocity;
            d_Orthcar_ptr_q[idx] = &d_Orthcar[ ( d_Eqindex_Q[ q + NPTK_Q] - 1) * 9 ];
            // d_Orthcar_ptr_q[idx] = &d_Orthcar[ ( d_Eqindex_Q[ q - 1 + NPTK_Q] - 1) * 9 ];
            // d_Orthcar_ptr_q[idx] = d_Orthcar;
            // d_ph_velocity_ptr[idx] = &d_ph_velocity[( d_Eqindex_Q[ kpq[idx] - 1 ] - 1 ) * 3*nmodes];
            d_ph_velocity_ptr[idx] = &d_ph_velocity[( d_Eqindex_Q[ kpq[idx] ] - 1 ) * 3*nmodes];
            // d_ph_velocity_ptr[idx] = d_el_velocity;
            d_ph_velocity_tmp_ptr[idx] = &d_ph_velocity_tmp[idx*3*nmodes];
            // d_ph_velocity_tmp_ptr[idx] = d_el_velocity;
            // printf("d_ph_velocity_tmp_ptr[%d] = %p\n",idx,d_el_velocity_tmp_ptr[idx]);
        }
    }

    __global__ void check_validk_ags(double *d_v2,
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
                               int batchsize)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        int idy = blockDim.y * blockIdx.y + threadIdx.y;
        int ib = idy/nmodes; // 
        int im = idy%nmodes; //

        if (idx < batchsize && idy < nbnd*nmodes && e2[im+idx*nmodes] >= ph_cut)
        {
            double sig[3] = { 0.0, 0.0, 0.0 };
            double sigma = -INFINITY;
            int const one = 1;

            for (int i = 0; i < 3; ++i)
            {
                sig[0] += rlattvec[0 + i] * (d_v2[ im * 3 + i + idx*nmodes*3 ] * radps2ev - d_v3[ib * 3 + i + idx*nbnd*3]);
                sig[1] += rlattvec[3 + i] * (d_v2[ im * 3 + i + idx*nmodes*3 ] * radps2ev - d_v3[ib * 3 + i + idx*nbnd*3]);
                sig[2] += rlattvec[6 + i] * (d_v2[ im * 3 + i + idx*nmodes*3 ] * radps2ev - d_v3[ib * 3 + i + idx*nbnd*3]);
            }

            for (int i = 0; i < 3; ++i)
            {
                sigma = max(sigma, pow(sig[i] / Ngrid[i], 2));
            }
            
            sigma = scalebroad * sqrt(sigma * 0.5);

            if (abs(*e1 + e2[im+idx*nmodes] - e3[ib+idx*nbnd]) <= delta_mult * sigma || abs(*e1 - e2[im+idx*nmodes] - e3[ib+idx*nbnd]) <= delta_mult * sigma)
            {   
                atomicAdd(&valid_i[idx],one);
            }
        }
    }
    
    __global__ void check_validk_cgs(
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
                                int batchsize)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        int idy = blockDim.y * blockIdx.y + threadIdx.y;
        int ib = idy/nmodes; // 
        int im = idy%nmodes; //

        if (idx < batchsize && idy < nbnd*nmodes && e2[im+idx*nmodes] >= ph_cut)
        // if (idx < batchsize && idy < nbnd*nmodes)
        // if (idx < batchsize && idy < nbnd*nmodes && e2[im] >= 1.0e-10)
        {
            // printf("check iq = %d, ibim = %d   e : %f %f %f %f %f \n", idx, idy, ph_cut, e2[im+idx*nmodes], abs(*e1 + e2[im+idx*nmodes] - e3[ib+idx*nbnd]), abs(*e1 - e2[im+idx*nmodes] - e3[ib+idx*nbnd]), 2.0 * sigma);
            int const one = 1;
            if (abs(*e1 + e2[im+idx*nmodes] - e3[ib+idx*nbnd]) <= delta_mult * sigma || abs(*e1 - e2[im+idx*nmodes] - e3[ib+idx*nbnd]) <= delta_mult * sigma)
            {   
                // printf("%d %d %d %f %f %f %f\n", idx, ib, im, *e1, e2[im+idx*nmodes], e3[ib+idx*nbnd], delta_mult*sigma);
                atomicAdd(&valid_i[idx],one);
            }
        }
    }

    __global__ void count_validk(int *valid_i,
                                int offset,
                                int batchsize)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < batchsize){
            // valid[idx] = valid_i[idx] != 0;
            valid_i[idx] = valid_i[idx]!= 0?idx+offset:-1;
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

        /* Check VALID */
    // __global__ void calc_v2_v3(double *d_v2,
    //                            double *d_v3,
    //                            double e1,
    //                            double *e2,
    //                            double *e3,
    //                            double *rlattvec,
    //                            int *Ngrid,
    //                            int nmodes,
    //                            int nbands,
    //                            double radps2ev,
    //                            double scalebroad,
    //                            bool *valid)
    // {
    //     int idx = blockDim.x * blockIdx.x + threadIdx.x;
    //     int idy = blockDim.y * blockIdx.y + threadIdx.y;
    //     if (idx < nmodes && idy < nbands && e2[idx] >= 1e-10)
    //     {
    //         double sig[3] = {0.0, 0.0, 0.0};
    //         double sigma = -INFINITY;
    //         for (int i = 0; i < 3; ++i)
    //         {
    //             sig[0] += rlattvec[0 + i] * (d_v2[idx * 3 + i] * radps2ev - d_v3[idy * 3 + i]);
    //             sig[1] += rlattvec[3 + i] * (d_v2[idx * 3 + i] * radps2ev - d_v3[idy * 3 + i]);
    //             sig[2] += rlattvec[6 + i] * (d_v2[idx * 3 + i] * radps2ev - d_v3[idy * 3 + i]);
    //         }
    //         for (int i = 0; i < 3; ++i)
    //         {
    //             sigma = max(sigma, pow(sig[i] / Ngrid[i], 2));
    //         }
            
    //         sigma = scalebroad * sqrt(sigma * 0.5);

    //         if (abs(e1 + e2[idx] - e3[idy]) <= 2.0 * sigma || abs(e1 - e2[idx] - e3[idy]) <= 2.0 * sigma)
    //         {   
    //             *valid = true;
    //         }
    //     }
    // }

}