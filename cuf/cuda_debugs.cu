
#include "cuda_settings.h"
// #include "cuda_timer.h"
// #include <cmath>
#include "cuda_elphwann_namespace.h"

using namespace device_funcs;

extern "C"
{   
    __global__ void output_chf(int nmodes, cuDoubleComplex *chf)
    {
        for (int i=0;i<nmodes;i++)
        {
            for (int j=0;j<nmodes;j++)
            {
                printf("chf_gpu : %f %f \n",chf[i*nmodes+j].x,chf[i*nmodes+j].y);
            }
        }
    }

    __global__ void output_tmp(int nbnd_eff, int nbnd, cuDoubleComplex **d_eptmp)
    {
        // for (int i=0;i<nbnd_eff;i++){
        for (int i=0;i<nbnd_eff;i++){
            // int j = 0;
            for (int j=0;j<nbnd;j++){
                printf("eptmp[0][%d,%d] = %20.10g\n", i, j, cabs_(d_eptmp[56+58][i+j*nbnd_eff]));
            }
            printf("----------------------------\n");
        }
    }

    __global__ void output_tmp2(int nbnd_eff, int nbnd, cuDoubleComplex **d_eptmp)
    {
        // for (int i=0;i<nbnd_eff;i++){
        for (int im=0;im<6;im++){
        for (int i=0;i<nbnd_eff;i++){
            // int j = 0;
            for (int j=0;j<nbnd;j++){
                printf("eptmp[%d][%d,%d] = %20.10g\n", im, i, j, cabs_(d_eptmp[im][i+j*nbnd_eff]));
            }
            printf("----------------------------\n");
        }
        }
    }
    
    __global__ void simple_matmul( int nbnd, int nbnd_irrel, int nbnd_eff, cuDoubleComplex *d_cufkq, cuDoubleComplex *epmatwef, int *d_StateInterest )
    {
        cuDoubleComplex eptmp;
        int idx = threadIdx.x+blockIdx.x*blockDim.x;
        int idy = threadIdx.y+blockIdx.y*blockDim.y;
        if(idx<nbnd_eff && idy<nbnd){
            eptmp.x=0.0;
            eptmp.y=0.0;
            // int i_band = d_StateInterest[idy]-1;
            for (int i=0;i<nbnd;i++){
                // eptmp[idx+idy*nbnd] = cuCadd(eptmp[idx+idy*nbnd],cuCmul(d_cufkq[idx+i*nbnd],epmatwef[i+idy*nbnd]));
                eptmp = cuCadd(eptmp,cuCmul( cuConj( d_cufkq[i+(idx+nbnd_irrel)*nbnd]), epmatwef[i+idy*nbnd]));
                // eptmp = cuCadd(eptmp,cuCmul( ( d_cufkq[(idx+nbnd_irrel)+i*nbnd]), epmatwef[i+(idy+i_band)*nbnd]));
                // eptmp = cuCadd(eptmp,cuCmul( ( d_cufkq[(idx)+i*nbnd]), epmatwef[i+(idy+nbnd_irrel)*nbnd]));
            }
            printf("matmul : ep[%d][%d] = %20.10g \n", idx,idy, cabs_(eptmp));
        }
    }

    __global__ void simple_matmul2( int nbnd, int nbnd_irrel, int nbnd_eff, cuDoubleComplex *d_eptmp, cuDoubleComplex *d_cufkk, int *d_StateInterest )
    {
        cuDoubleComplex eptmp;
        int idx = threadIdx.x+blockIdx.x*blockDim.x;
        int idy = threadIdx.y+blockIdx.y*blockDim.y;
        if(idx<nbnd_eff && idy<1){
            eptmp=make_cuDoubleComplex(double(0.0), double(0.0));
            int i_band = d_StateInterest[idy*2]-1;
            printf("iband = %d\n",i_band);
            for (int i=0;i<nbnd;i++){
                // eptmp[idx+idy*nbnd] = cuCadd(eptmp[idx+idy*nbnd],cuCmul(d_cufkq[idx+i*nbnd],epmatwef[i+idy*nbnd]));
                // eptmp = cuCadd(eptmp,cuCmul( cuConj( d_cufkq[i+(idx+nbnd_irrel)*nbnd]), epmatwef[i+idy*nbnd]));
                // eptmp = cuCadd(eptmp,cuCmul( ( d_cufkq[(idx+nbnd_irrel)+i*nbnd]), epmatwef[i+(idy+i_band)*nbnd]));
                // eptmp = cuCadd(eptmp,cuCmul( ( d_eptmp[(idx)+i*nbnd]), d_cufkk[i+(idy+nbnd_irrel+i_band)*nbnd]));
                printf("eptmp[%d][%d] = %20.10g, cufkk[%d][%d] = %20.10g \n", idx,i, cabs_(d_eptmp[(idx)+i*nbnd_eff]),i,idy+nbnd_irrel+i_band,cabs_(d_cufkk[i+(idy+nbnd_irrel+i_band)*nbnd]));
                eptmp = cuCadd(eptmp,cuCmul( ( d_eptmp[(idx)+i*nbnd_eff]), d_cufkk[i+(idy+nbnd_irrel+i_band)*nbnd]));
            }
            printf("matmul : ep[%d][%d] = %20.10g \n", idx,idy, cabs_(eptmp));
        }
    }
    
    __global__ void simple_matmul3(int *d_valid_list, int is, int nbnd, int nbnd_irrel, int nbnd_eff, cuDoubleComplex **d_eptmp, cuDoubleComplex **d_cufkk, int *d_StateInterest )
    {
        cuDoubleComplex eptmp;
        int idx = threadIdx.x+blockIdx.x*blockDim.x;
        int idy = threadIdx.y+blockIdx.y*blockDim.y;
        for (idx=0;idx<nbnd_eff;idx++){

        if(idx<nbnd_eff && idy==0){
            int idy_ = 56;
            int im = 1;
            eptmp=make_cuDoubleComplex(double(0.0), double(0.0));
            int i_band = d_StateInterest[d_valid_list[idy+is]*2]-1;
            // int i_band = d_StateInterest[idy_*2]-1;
            printf("iband = %d\n",i_band);
            for (int i=0;i<nbnd;i++){
                // eptmp[idx+idy*nbnd] = cuCadd(eptmp[idx+idy*nbnd],cuCmul(d_cufkq[idx+i*nbnd],epmatwef[i+idy*nbnd]));
                // eptmp = cuCadd(eptmp,cuCmul( cuConj( d_cufkq[i+(idx+nbnd_irrel)*nbnd]), epmatwef[i+idy*nbnd]));
                // eptmp = cuCadd(eptmp,cuCmul( ( d_cufkq[(idx+nbnd_irrel)+i*nbnd]), epmatwef[i+(idy+i_band)*nbnd]));
                // eptmp = cuCadd(eptmp,cuCmul( ( d_eptmp[(idx)+i*nbnd]), d_cufkk[i+(idy+nbnd_irrel+i_band)*nbnd]));
                printf("eptmp[%d][%d] = (%20.10g,%20.10g), cufkk[%d][%d] = (%20.10g,%20.10g) \n", idx,i, 
                cabs_(d_eptmp[idy_+im*58][(idx)+i*nbnd_eff]),c_angle_(d_eptmp[idy_+im*58][(idx)+i*nbnd_eff]),
                i,idy+nbnd_irrel+i_band,
                cabs_(d_cufkk[idy_+im*58][i+(idy+nbnd_irrel+i_band)*nbnd]),c_angle_(d_cufkk[idy_+im*58][i+(idy+nbnd_irrel+i_band)*nbnd]));
                eptmp = cuCadd(eptmp,cuCmul( ( d_eptmp[idy_+im*58][(idx)+i*nbnd_eff]), d_cufkk[idy_+im*58][i+(idy+nbnd_irrel+i_band)*nbnd]));
                printf("matmul now, ep[%d][%d] = (%20.10g,%20.10g) \n",idx,idy, cabs_(eptmp), c_angle_(eptmp));
            }
            printf("matmul : ep[%d][%d] = %20.10g \n", idx,idy, cabs_(eptmp));
        }
        }
    }    

    __global__ void output_gm(double *gm, int *indgm, int nbnd, int batchsize)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if(idx <batchsize)
        {
            for (int ib = 0; ib < nbnd; ib++ ){
                printf(" gm_output : %d %d %d %f\n", idx, ib, indgm[ib+idx*nbnd], gm[ib+idx*nbnd] );
            }
        }
    }

    __global__ void output_cond(double *elcond){
        for(int i=0;i<9;i++){
            printf("device elcond : %d %e\n", i, elcond[i]);
        }
    }

    __global__ void output_u(char s,int nbnd, cuDoubleComplex *u)
    {
        // for (int i=0;i<nbnd_eff;i++){
        for (int j=0;j<nbnd;j++){
            // int j = 0;
            for (int i=0;i<nbnd;i++){
                printf("%c u[%d,%d] = %20.10g\n", s, i, j, cabs_(u[i+j*nbnd]));
            }
            printf("----------------------------\n");
        }
    }

#ifdef DEBUG 
    __global__ void output_d_eptmp(char name,int offset, cuDoubleComplex *chf)
    {
        for (int i=0;i<60;i++)
        {
            printf("%c d_eptmp : %e\n", name, cabs_(chf[i*offset]));
            // printf("d_eptmp : %e %e\n", chf[i*offset].x,chf[i*offset].y);
        }
    }
#endif
    __global__ void output_real_image(char name, cuDoubleComplex *chf)
    {
        for (int i=0;i<100;i++)
        {
                printf("%c wq [%d] : %f %f  \n", name, i, chf[i].x, chf[i].y);
        }
    }

    __global__ void output_d_out(int offset, double *chf)
    {    
        for (int i=0;i<1;i++){
            printf("d_eptmp_out : %e\n", chf[i*offset]);
            // printf("d_eptmp : %e %e\n", chf[i*offset].x,chf[i*offset].y);
        }
    }

    __global__ void output_d_va(int *valid, int batch){
        for(int i = 0; i< batch ; i++){
            printf(" d_valid [%d] =  %d\n", i, valid[i]);
        }
    }

    
    __global__ void output_tmp3(int offsets, int nbnd, cuDoubleComplex *d_eptmp)
    {
        // for (int i=0;i<nbnd_eff;i++){
        for (int i=0;i<offsets;i++){
            // int j = 0;
            for (int j=0;j<nbnd;j++){
                // printf("cfac[%d,%d] = %20.10g\n", i, j, cabs_(d_eptmp[i+j*offsets]));
                printf("cfac[%d,%d] = %20.10g, %20.10g\n", i, j, d_eptmp[i+j*offsets].x, d_eptmp[i+j*offsets].y);
            }
            printf("----------------------------\n");
        }
    }

    __global__ void output_real(char key,int nbnd, int batchsize, double *d_eptmp)
    {
        for (int i=0;i<batchsize;i++){
            // int j = 0;
            for (int j=0;j<nbnd;j++){
                // printf("cfac[%d,%d] = %20.10g\n", i, j, cabs_(d_eptmp[i+j*offsets]));
                printf("%c i = %d; omega[%d] = %20.10g\n", key, i, j, sqrt(d_eptmp[j+i*nbnd]));
            }
            printf("----------------------------\n");
        }
    }

    __global__ void output_d_epmatl(int batchsize, int nmodes, int nbnd, ComplexD *epmat){
        for (int ibat=0; ibat< batchsize;ibat++){
            for (int im=0; im< nmodes;im++){
                for (int ib=0; ib< nbnd*nbnd;ib++){
                    printf("EPmat gpu final : %d %d %d %g\n", ibat, im, ib, 
                      abs(epmat[ibat*nmodes*nbnd*nbnd+im*nbnd*nbnd+ib]));
                }
            }
        }
    }

    void output_epmatl(int batchsize, int nmodes, int nbnd, ComplexD *epmat){
        for (int ibat=0; ibat< batchsize;ibat++){
            for (int im=0; im< nmodes;im++){
                for (int ib=0; ib< nbnd*nbnd;ib++){
                    printf("EPmat cpu get: %d %d %d %g\n", ibat, im, ib, 
                      abs(epmat[ibat*nmodes*nbnd*nbnd+im*nbnd*nbnd+ib]));
                }
            }
        }
    }

}
