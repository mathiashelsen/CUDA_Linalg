#ifndef _BASIC_CUH
#define _BASIC_CUH

#include <cuda.h>
#include "Matrix.h"

__global__ void MatAddKernel(Matrix A, Matrix B, Matrix C);
__global__ void MatMulNaiveKernel(Matrix A, Matrix B, Matrix C);
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C);

#endif
