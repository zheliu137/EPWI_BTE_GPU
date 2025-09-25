// #pragma once
#include "cuda_settings.h"
#include "cuda_timer.h"
#include <cstddef>
#include <cstdlib>
// #include <cuda/std/complex>
// #include <cmath>

// using namespace cuda_elphwann_wannier;

#define IDX2F(i, j, ld) ((((j - 1)) * (ld)) + ((i - 1)))

extern "C"
{

    static GPU_Timer timer_GPUr_;
    static CPU_Timer timer_CPUr_;
    __global__ void kpt_rotate(int NPTK, int nsymm, int *Ngrid, double *orth_rev, int *Eqindex, int *Eqsymm, int *d_symm_rev_idx)
    {
        int ik = blockDim.x * blockIdx.x + threadIdx.x;
        double veck[3];
        int intk[3];
        double cark[3];
        int Eqindex_tmp;

        if( ik<NPTK)
        {
            Eqindex[ik]=ik;
            Eqsymm[ik]=1; // the first symm. op. MUST be identity operation
            // kinIBZ[ik]=1;
            for (int isymm=0; isymm<nsymm; isymm++)
            {
                // this seems only work for nkf1=nkf2=nkf3
                veck[0]=(double)((ik)/(Ngrid[1]*Ngrid[2]));
                veck[1]=(double)((ik)%(Ngrid[1]*Ngrid[2])/Ngrid[2]);
                veck[2]=(double)((ik)%Ngrid[2]);

                for(int i=0; i<3; i++)
                {
                    cark[i] = veck[0]*orth_rev[i+9*isymm]+veck[1]*orth_rev[i+3+9*isymm]+veck[2]*orth_rev[i+6+9*isymm];
                }

                for(int i=0; i<3; i++)
                {
                    intk[i] = (((int)round(cark[i]))%Ngrid[i]+Ngrid[i])%Ngrid[i];
                }

                Eqindex_tmp = intk[0]*Ngrid[1]*Ngrid[2] + intk[1]*Ngrid[2] + intk[2];

                if(Eqindex_tmp<Eqindex[ik])
                {
                    // atomicAdd(&koutIBZ[ik],1);
                    // kinIBZ[ik] *= 0; 
                    // atomicAnd(&kinIBZ[ik],0);
                    Eqindex[ik]=Eqindex_tmp;
                    Eqsymm[ik]=isymm+1;
                }
            }
        }
    }

    __global__ void irr_counting(   
                                    int NPTK, 
                                    int batch, 
                                    int avg_nptk, 
                                    int residual_nptk, 
                                    int *Nlist, 
                                    int *List_tmp, 
                                    int *Eqindex 
                                )
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        int lower_bnd = avg_nptk*idx+min(idx,residual_nptk);
        int upper_bnd = avg_nptk*(idx+1)+min(idx+1,residual_nptk);
        if (idx<batch){
            Nlist[idx]=0;
            if (lower_bnd<NPTK)
            {
                for (int i = lower_bnd; i<upper_bnd; i++)
                {
                    if(Eqindex[i]==i)
                    {
                        List_tmp[Nlist[idx]+lower_bnd]=i+1;
                        Nlist[idx] += 1;
                    }
                }
            }
        }        
    }

    __global__ void irr_reduce(int NPTK, int batch, int avg_nptk, int residual_nptk, int *Nlist, int *naccum_list, int *List_tmp, int *List, int *List_hash)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        int lower_bnd = avg_nptk*idx+min(idx,residual_nptk);
        int upper_bnd = avg_nptk*(idx+1)+min(idx+1,residual_nptk);
        if (idx<batch)
        {
            for (int i = 0; i<Nlist[idx]; i++)
            {
                List[naccum_list[idx]+i] = List_tmp[i+lower_bnd];
                List_hash[List[naccum_list[idx]+i]] = naccum_list[idx] + i + 1;
            }        
        }
    }

    __global__ void eqidx_rearrange(int NPTK, int *List_hash, int *Eqindex, int *Eqindex_tmp, int *d_Eqsymm)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx<NPTK)
        {
            Eqindex[idx]=List_hash[Eqindex_tmp[idx]+1];
            Eqindex[idx+NPTK]=d_Eqsymm[idx];
        }
    }    

    void wedge_gpu_(int *Nlist, int *List, int *Eqindex, int *nsymm, int *Ngrid, double *orth_rev, int *symm_rev_idx)
    {
        long NPTK = Ngrid[0]*Ngrid[1]*Ngrid[2];
        int *d_Eqindex = nullptr, 
            *d_Ngrid = nullptr,
            *d_Eqindex_min = nullptr,
            *d_Eqsymm = nullptr;
        int *d_symm_rev_idx = nullptr;
        int *d_kinIBZ = nullptr,
            *d_List_tmp = nullptr,
            *d_Nlist_tmp = nullptr;
        double *d_orth_rev = nullptr;
        int *d_List = nullptr,
            *d_List_hash = nullptr;
        int *d_naccum_list = nullptr;
        int *Nlist_tmp = nullptr;
        int *naccum_list = nullptr;
        int batchsize = 10000;

        timer_GPUr_.start_clock("wedge_gpu");
        Nlist_tmp=(int *)malloc(batchsize*sizeof(int));
        naccum_list=(int *)malloc(batchsize*sizeof(int));

        CUDA_CHECK(cudaMalloc((void **)&d_Eqindex,  NPTK * *nsymm * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&d_Ngrid,  3 * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&d_Eqindex_min,  NPTK * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&d_Eqsymm,  NPTK * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&d_symm_rev_idx,  *nsymm * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&d_kinIBZ,  NPTK * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&d_List_tmp,  NPTK * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&d_Nlist_tmp,  batchsize * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&d_orth_rev,  *nsymm*9 * sizeof(double)));

        CUDA_CHECK(cudaMemcpy(d_Ngrid, Ngrid, 3 * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_orth_rev, orth_rev, *nsymm*9 * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_symm_rev_idx, symm_rev_idx, *nsymm * sizeof(int), cudaMemcpyHostToDevice));

        dim3 block(BLOCK_SIZE*BLOCK_SIZE);
        dim3 grid((NPTK + block.x - 1) / block.x);
        kpt_rotate<<<grid,block>>>(NPTK, *nsymm, d_Ngrid, d_orth_rev, d_Eqindex_min, d_Eqsymm, d_symm_rev_idx);

        int avg_nptk=NPTK/batchsize;
        int residual_nptk=NPTK-avg_nptk*batchsize;

        block=BLOCK_SIZE;
        grid=(batchsize + block.x - 1) / block.x;
        irr_counting<<<grid,block>>>(NPTK,batchsize,avg_nptk,residual_nptk,d_Nlist_tmp,d_List_tmp,d_Eqindex_min);

        CUDA_CHECK(cudaMemcpy(Nlist_tmp, d_Nlist_tmp, batchsize*sizeof(int), cudaMemcpyDeviceToHost));

        *Nlist = 0;
        for (int i=0;i<batchsize;i++){
            naccum_list[i] = *Nlist;
            *Nlist+=Nlist_tmp[i];
        }

        CUDA_CHECK(cudaMalloc((void **)&d_List,  *Nlist * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&d_List_hash,  NPTK * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&d_naccum_list,  batchsize * sizeof(int)));

        CUDA_CHECK(cudaMemcpy(d_naccum_list, naccum_list, batchsize * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());

        block=BLOCK_SIZE2;
        grid=(batchsize + block.x - 1) / block.x;
        irr_reduce<<<grid,block>>>(NPTK,batchsize,avg_nptk,residual_nptk, d_Nlist_tmp,d_naccum_list,d_List_tmp,d_List,d_List_hash);

        grid=(NPTK + block.x - 1) / block.x;
        eqidx_rearrange<<<grid,block>>>(NPTK, d_List_hash, d_Eqindex, d_Eqindex_min, d_Eqsymm);

        CUDA_CHECK(cudaMemcpy(List, d_List, *Nlist * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(Eqindex, d_Eqindex, NPTK * 2 * sizeof(int), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_Eqindex));
        CUDA_CHECK(cudaFree(d_Ngrid));
        CUDA_CHECK(cudaFree(d_Eqindex_min));
        CUDA_CHECK(cudaFree(d_Eqsymm));
        CUDA_CHECK(cudaFree(d_kinIBZ));
        CUDA_CHECK(cudaFree(d_List_tmp));
        CUDA_CHECK(cudaFree(d_Nlist_tmp));
        CUDA_CHECK(cudaFree(d_orth_rev));

        CUDA_CHECK(cudaFree(d_List));
        CUDA_CHECK(cudaFree(d_List_hash));
        CUDA_CHECK(cudaFree(d_naccum_list));

        free(Nlist_tmp);
        free(naccum_list);
        timer_GPUr_.stop_clock("wedge_gpu");
        timer_GPUr_.print_clock("wedge_gpu");
    }
}