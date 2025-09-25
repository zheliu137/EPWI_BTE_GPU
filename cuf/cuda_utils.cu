#include "cuda_settings.h"
// #include "cuda_timer.h"
// #include <cuda/std/complex>
// #include <cmath>
#include "cuda_elphwann_namespace.h"

using device_funcs::cabs_;
using device_funcs::kplusq_kernel;
// using cuda_elphwann_wannier::NQgrid_;
extern "C"
{   
    int kplusq_C(int k, int q, int* NKgrid, int* NQgrid)
    {
        int kplusq;
        int veck[3], vecq[3], veckq[3];

        veck[0] = int((k - 1) / double(NKgrid[1] * NKgrid[2]));
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

        kplusq = veckq[0] * NKgrid[1] * NKgrid[2] + veckq[1] * NKgrid[2] + veckq[2] + 1;

        return kplusq;
    }

    void _ind_k_to_xk(int *vec, double *xkx, int *ngrid, int k)
    {
        vec[0] = int((k - 1) / (ngrid[1] * ngrid[2]));
        vec[1] = int((k - 1) % (ngrid[1] * ngrid[2]) / ngrid[2]);
        vec[2] = (k - 1) % ngrid[2];
        for (int i = 0; i < 3; ++i)
        {
            xkx[i] = (double)vec[i] / (double)ngrid[i];
        }
    }

     __device__ void indk_to_xk(double *xkx, int *ngrid, int k)
    {   
        int vec[3];
        // static double xkx[3];
        // double xkx[3];
        vec[0] = int((k) / (ngrid[1] * ngrid[2]));
        vec[1] = int((k) % (ngrid[1] * ngrid[2]) / ngrid[2]);
        vec[2] = (k) % ngrid[2];
        for (int i = 0; i < 3; ++i)
        {
            xkx[i] = (double)vec[i] / (double)ngrid[i];
        }
        return;
    }

    __global__ void iktoxxk_batch(int batchsize, double *xxk, int *ngrid, int ik)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        int vec[3];
        if(idx < batchsize)
        {
            int k = ik + idx ;
            vec[0] = int((k-1) / (ngrid[1] * ngrid[2])) ;
            vec[1] = int((k-1) % (ngrid[1] * ngrid[2]) / ngrid[2]) ;
            vec[2] = (k-1) % ngrid[2] ;
            for (int i = 0; i < 3; ++i)
            {
                xxk[i+3*idx] = (double)vec[i] / (double)ngrid[i] ;
            }
        }
    }

    __global__ void iqtoxkxq_batch(int batchsize, int* NKgrid, int* NQgrid, int k, int q_offset, int* valid_list, double* xxq, double* xkq)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if(idx < batchsize)
        {
            int q = valid_list[q_offset + idx];
            indk_to_xk(xxq+3*idx, NQgrid,  q);

            int kpq = kplusq_kernel(k, q, NKgrid, NQgrid);
            indk_to_xk(xkq+3*idx, NKgrid,  kpq);
        }        
    }

    __global__ void istoxxk_batch(int batchsize, int *stateinterest, int *List, double *xxk, int *ngrid, int is_offset)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        int vec[3];
        if(idx < batchsize)
        {
            int k = List[stateinterest[(idx + is_offset)*2+1]-1];
            vec[0] = int((k - 1) / (ngrid[1] * ngrid[2])) ;
            vec[1] = int((k - 1) % (ngrid[1] * ngrid[2]) / ngrid[2]) ;
            vec[2] = (k - 1) % ngrid[2] ;
            for (int i = 0; i < 3; ++i)
            {
                xxk[i+3*idx] = (double)vec[i] / (double)ngrid[i] ;
            }
            // printf(" xk[%d] : %f %f %f \n", idx, xxk[3*idx], xxk[3*idx+1], xxk[3*idx+2]);
        }
    }    

    void _cryst_to_cart(int *_nvec, double *vec, double *trmat, int *_iflag)
    {
        int nvec = *_nvec;
        int iflag = *_iflag;
        double vau[3];
        for (int nv = 0; nv < nvec; ++nv)
        {
            if (iflag == 1)
            {
                for (int kpol = 0; kpol < 3; ++kpol)
                {
                    vau[kpol] = 0;
                    for (int i = 0; i < 3; ++i)
                    {
                        vau[kpol] += trmat[i * 3 + kpol] * vec[nv * 3 + i];
                    }
                }
            }
            else
            {
                for (int kpol = 0; kpol < 3; ++kpol)
                {
                    vau[kpol] = 0;
                    for (int i = 0; i < 3; ++i)
                    {
                        vau[kpol] += trmat[kpol * 3 + i] * vec[nv * 3 + i];
                    }
                }
            }
            for (int kpol = 0; kpol < 3; ++kpol)
            {
                vec[nv * 3 + kpol] = vau[kpol];
            }
        }
    }

    __global__ void cryst_to_cart_global( double* bg, double* cryst, double* cart, int batchsize)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if(idx<batchsize){
        for (int idir =0;idir<3;idir++){
            cart [idir+idx*3] = bg [idir] * cryst [idx*3] + bg [idir+3] 
                * cryst [1+idx*3] + bg [idir+6]  * cryst [2+idx*3];

            }
            // printf("kernel cryst : %d  %f , %f , %f \n", idx, cryst[idx*3+0],cryst[idx*3+1],cryst[idx*3+2]);
            // printf("kernel cart : %d  %f , %f , %f \n", idx, cart[idx*3+0],cart[idx*3+1],cart[idx*3+2]);
        }
    }


    __global__ void hermitianize_matrix_batched(int n, int batchSize, cuDoubleComplex *A)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        int idy = blockDim.y * blockIdx.y + threadIdx.y;
        if (idx < n*(n+1)/2 && idy < batchSize)
        {
            int row = ceil( sqrt(2 * idx + 2.25) - 1.5 );
            int column = idx - (row+1) * row / 2;
            int idx_1 = idy * n * n + row * n + column;
            int idx_2 = idy * n * n + column * n + row;
            A[idx_1] =
                cuCmul(
                    cuCadd(A[idx_1], cuConj(A[idx_2])),
                    {0.5, 0.0});
            A[idx_2] = cuConj(A[idx_1]);
        }
    }

    __global__ void cfac_batched(int nrr, double *irvec, int *ndegen, double *d_xkk, ComplexD *cfac, int batchsize)
    {
        int ir = blockDim.x * blockIdx.x + threadIdx.x;
        int ibatch = blockDim.y * blockIdx.y + threadIdx.y;
        if (ir < nrr && ibatch < batchsize)
        {
            double rdotk = 0.0;
            for (int i = 0; i < 3; ++i)
            {
                rdotk += d_xkk[3 * ibatch + i] * irvec[3 * ir + i];
            }
            rdotk *= M_PI * 2;
            cfac[ibatch * nrr + ir] = exp(ComplexD{0.0, 1.0} * rdotk) / (double)ndegen[ir];
        }
    }

    __global__ void cfac_batched_table(int *ik3, int *ir3, int *grid, ComplexD *cfac_table, int *ndegen, ComplexD *cfac, int nrr, int batchsize)
    {
        int ir = blockDim.x * blockIdx.x + threadIdx.x;
        int ibatch = blockDim.y * blockIdx.y + threadIdx.y;
        if (ir < nrr && ibatch < batchsize)
        {
            int irdk[3];
            for (int i = 0; i < 3; ++i)
            {
                irdk[i] = (ik3[3 * ibatch + i] * ir3[3 * ir + i])%grid[i];
            }
            cfac[ibatch * nrr + ir] = cfac_table[irdk[0]]*cfac_table[irdk[1]]*cfac_table[irdk[2]] / (double)ndegen[ir];
        }
    }

    __global__ void trans_epmat(double *d_epmatf_out,cuDoubleComplex *d_epmatf, double* wfqq, double eps, int batch_size, int nbnd, int nmodes, double ryd2ev){
        int ibatch = blockIdx.x * blockDim.x + threadIdx.x;
        int idy = blockIdx.y * blockDim.y + threadIdx.y;
        int imode = idy/nbnd;
        int ibnd = idy%nbnd;
        if(ibatch < batch_size && imode < nmodes){
            double g2_tmp = wfqq[ibatch*nmodes+imode] < eps/ryd2ev? 0.0 : 1.0;
            d_epmatf_out[ibnd + imode * nbnd + ibatch * nmodes * nbnd] = 
                g2_tmp*pow(cabs_(d_epmatf[ibnd + imode * nbnd + ibatch * nmodes * nbnd]),2)/(2.0*wfqq[imode+ibatch*nmodes])*ryd2ev*ryd2ev;
        }
    }

    __global__ void trans_epmat_2(double *d_epmatf_out,cuDoubleComplex *d_epmatf, double* e2, int* Eqindex_Q, int Nlist_Q, int* valid_list, double radps2ry, double eps, int batch_size, int nbnd, int nmodes){
        int ibatch = blockIdx.x * blockDim.x + threadIdx.x;
        int idy = blockIdx.y * blockDim.y + threadIdx.y;
        int imode = idy/nbnd;
        // int ibnd = idy%nbnd;
        if(ibatch < batch_size && imode < nmodes){
            int iqim = Eqindex_Q[ valid_list[ibatch]] - 1 ;//Eqindex_Q_[q[]];
            double wfqq = e2[iqim + imode*Nlist_Q];
            // double g2_tmp = wfqq < eps? 0.0 : 1.0;
            // d_epmatf_out[ibnd + imode * nbnd + ibatch * nmodes * nbnd] = 
                // g2_tmp*cabs_(d_epmatf[ibnd + imode * nbnd + ibatch * nmodes * nbnd]);
            // d_epmatf_out[ibnd + imode * nbnd + ibatch * nmodes * nbnd] = 
            //     g2_tmp*pow(cabs_(d_epmatf[ibnd + imode * nbnd + ibatch * nmodes * nbnd]),2)/(2.0*wfqq);
            printf( " ph0  [%d]  ind_q : [%d], wq[%d] :  %g \n", ibatch, iqim, imode, wfqq*radps2ry );
            // printf( " [%d]epmat[%d][%d] :  %g, g2_tmp : %g \n",ibnd,imode,ibatch, cabs_(d_epmatf[ibnd + imode * nbnd + ibatch * nmodes * nbnd]),g2_tmp );
        }
    }

    __global__ void init_ephwan2bloch_epmatf_batched_(int nbnd, int nrr, int nmodes, int *irvec, int *ndegen, double *d_xkk, ComplexD *cfac, ComplexD *epmatw, ComplexD *epmatf, int batchsize)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        int ibatch = blockDim.y * blockIdx.y + threadIdx.y;
        if (idx < nbnd * nbnd && ibatch < batchsize)
        {
            int ibnd = idx / nbnd;
            int jbnd = idx % nbnd;
            for (int ir = 0; ir < nrr; ++ir)
            {
                for (int imode = 0; imode < nmodes; ++imode)
                {
                    epmatf[imode * batchsize * nbnd * nbnd + ibatch * nbnd * nbnd + jbnd * nbnd + ibnd] += cfac[ibatch * nrr + ir] * epmatw[imode * nbnd * nbnd * nrr + ir * nbnd * nbnd + jbnd * nbnd + ibnd];
                }
            }
        }
    }

    __global__ void init_ephwan2bloch_epmatf_batched(int nbnd, int nrr, int nmodes, int *irvec, int *ndegen, double *d_xkk, ComplexD *cfac, ComplexD *epmatw, ComplexD *epmatf, int batchsize)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        int ibatch = blockDim.y * blockIdx.y + threadIdx.y;
        if (idx < nbnd * nbnd && ibatch < batchsize)
        {
            int ibnd = idx / nbnd;
            int jbnd = idx % nbnd;
            for (int ir = 0; ir < nrr; ++ir)
            {
                for (int imode = 0; imode < nmodes; ++imode)
                {
                    epmatf[ibatch * nmodes * nbnd * nbnd + imode * nbnd * nbnd + jbnd * nbnd + ibnd] += cfac[ibatch * nrr + ir] * epmatw[imode * nbnd * nbnd * nrr + ir * nbnd * nbnd + jbnd * nbnd + ibnd];
                }
            }
        }
    }

    __global__ void multiple_copy_cuf(ComplexD *cufkk, ComplexD *cufkk_, ComplexD *cufkq, ComplexD *cufkq_,
                                      int nbnd, int nmodes, int batch_size)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        int ibatch = blockDim.y * blockIdx.y + threadIdx.y;
        if (idx < nbnd * nbnd && ibatch < batch_size)
        {
            int ibnd = idx / nbnd;
            int jbnd = idx % nbnd;
            for (int imode = 0; imode < nmodes; ++imode)
            {
                // cufkk[ibatch * nmodes * nbnd * nbnd + imode * nbnd * nbnd + jbnd * nbnd + ibnd] = cufkk_[ibatch * nbnd * nbnd + jbnd * nbnd + ibnd];
                cufkk[imode * batch_size * nbnd * nbnd + ibatch * nbnd * nbnd + jbnd * nbnd + ibnd] = cufkk_[ibatch * nbnd * nbnd + jbnd * nbnd + ibnd];
                // cufkq[ibatch * nmodes * nbnd * nbnd + imode * nbnd * nbnd + jbnd * nbnd + ibnd] = cufkq_[ibatch * nbnd * nbnd + jbnd * nbnd + ibnd];
                cufkq[imode * batch_size * nbnd * nbnd + ibatch * nbnd * nbnd + jbnd * nbnd + ibnd] = cufkq_[ibatch * nbnd * nbnd + jbnd * nbnd + ibnd];
            }
        }
    }
    
    __global__ void fc_massfac(cuDoubleComplex *rdw, double *mass, int *ityp, int nmodes, int nrr)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        int idy = blockDim.y * blockIdx.y + threadIdx.y;
        if( idx<nmodes*nmodes && idy < nrr){
            int ia = idx/nmodes/3;
            int ib = (idx%nmodes)/3;
            rdw[idx+idy*nmodes*nmodes].x/=sqrt(mass[ityp[ib]-1]*mass[ityp[ia]-1]);
            rdw[idx+idy*nmodes*nmodes].y/=sqrt(mass[ityp[ib]-1]*mass[ityp[ia]-1]);
        }
    }

    __global__ void dynmat_prep(ComplexD *dyn, double *mass, int *ityp, int nmodes, int batchSize)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        int idy = blockDim.y * blockIdx.y + threadIdx.y;
        // if( idx<nmodes*nmodes && idy < batchSize){
        if( idx < nmodes*(nmodes+1)/2 && idy < batchSize)
        {
            int row = ceil( sqrt(2 * idx + 2.25) - 1.5 );
            int column = idx - (row+1) * row / 2;
            int idx_1 = idy * nmodes * nmodes + row * nmodes + column;
            int idx_2 = idy * nmodes * nmodes + column * nmodes + row;
            dyn[idx_1] = (dyn[idx_1] + conj(dyn[idx_2]))*0.5/sqrt(mass[ityp[row/3]-1]*mass[ityp[column/3]-1]);
            // dyn[idx_2] = cuConj(dyn[idx_1]);
            dyn[idx_2] = conj(dyn[idx_1]);
        }
    }

    __global__ void setting_array_kq(cuDoubleComplex *d_cufkq, cuDoubleComplex *d_eptmp, cuDoubleComplex *d_epmatf,
                                  cuDoubleComplex **d_cufkq_ptr, cuDoubleComplex **d_eptmp_ptr, cuDoubleComplex **d_epmatf_ptr,
                                  int batch_size, int nbnd, int nbnd_irrel, int nmodes)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < batch_size)
        {   
            d_cufkq_ptr[idx] = &d_cufkq[idx * nbnd * nbnd + nbnd_irrel*nbnd];
            d_eptmp_ptr[idx] = &d_eptmp[idx * nbnd*nmodes];
            d_epmatf_ptr[idx] = &d_epmatf[idx * nbnd*nmodes];
        }
    }

    __global__ void setting_array_q(cuDoubleComplex *d_cufq, cuDoubleComplex **d_cufq_ptr, int batch_size, int nbnd, int nmodes)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < batch_size)
        {   
            d_cufq_ptr[idx] = &d_cufq[idx * nbnd * nmodes * nmodes  ];
        }
    }

    __global__ void get_cfac_from_rdot(ComplexD *cfac, double *rdot, int *ndegen, int nrr, int batchSize)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        int bid = blockDim.y * blockIdx.y + threadIdx.y;
        if (idx < nrr && bid < batchSize)
        {
            cfac[bid * nrr + idx] = exp(ComplexD(0.0, 1.0) * rdot[bid * nrr + idx]) / (double)ndegen[idx];
        }
    }
    
}