#pragma once

/*
 * COMMON INCLUDES
 */

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <cublas_v2.h>
// #include "cublas_v2.h"
#include <cuda/std/complex>
// #include <complex>
#include "cusolverDn.h"

// using namespace std;
using namespace cuda::std;

/*
 * COMMON VARS
 */

#define GPUNUM 4
#define STREAM_NUM 16
#define BLOCK_SIZE 32
#define BLOCK_SIZE2 512
// #define DEBUG
#define DEBUG_CHECK

// typedef struct DComplex
// {
//     double x, y;
// } DComplex;

#define IDX2F(i, j, ld) ((((j - 1)) * (ld)) + ((i - 1)))

typedef complex<double> ComplexD;

const ComplexD cone_ = {1.0, 0.0}, czero_ = {0.0, 0.0};

// const double gm_eps = 1e-20;
const double gm_eps = 1e-20;
// #define gm_eps 1e-20;

/*
 * COMMOM FUNCTIONS
 */

#ifdef DEBUG_CHECK
#define CUSOLVER_CHECK(err) (HandlecusolverError(err, __FILE__, __LINE__))
#define CUDA_CHECK(err) (HandleError(err, __FILE__, __LINE__))
#define CUBLAS_CHECK(err) (HandleBlasError(err, __FILE__, __LINE__))
#else
#define CUSOLVER_CHECK(err) (err)
#define CUDA_CHECK(err) (err)
#define CUBLAS_CHECK(err) (err)
#endif

static void HandleBlasError(cublasStatus_t err, const char *file, int line)
{

    if (err != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "ERROR: %s in %s at line %d (error-code %d)\n",
                cublasGetStatusString(err), file, line, err);
        fflush(stdout);
        exit(-1);
    }
}

static void HandlecusolverError(cusolverStatus_t err, const char *file, int line)
{

    if (err != CUSOLVER_STATUS_SUCCESS)
    {
        fprintf(stderr, "ERROR: %d in %s at line %d, (error-code %d)\n",
                err, file, line, err);
        fflush(stdout);
        exit(-1);
    }
}

static void HandleError(cudaError_t err, const char *file, int line)
{

    if (err != cudaSuccess)
    {
        fprintf(stderr, "ERROR: %s in %s at line %d (error-code %d)\n",
                cudaGetErrorString(err), file, line, err);
        fflush(stdout);
        exit(-1);
    }
}

struct is_negative
{
__host__ __device__
bool operator()(const int x)
{
    return (x < 0);
}
};

struct is_zero_d
{
__host__ __device__
bool operator()(const double x)
{
    return (abs(x) < gm_eps);
}
};

struct is_zero_d0
{
__host__ __device__
bool operator()(const double x)
{
    return ( x == 0.0);
}
};

struct is_zero_i
{
__host__ __device__
bool operator()(const int x)
{
    return (x == 0);
}
};


struct is_mone_i
{
__host__ __device__
bool operator()(const int x)
{
    return (x == -1);
}
};


struct is_positive
{
__host__ __device__
bool operator()(const int x)
{
    return (x > 0);
}
};

